from associator import Association_condition, Combination_constraint
from trajectories import node_trajectory_base
from stat_funcs import statFunc
import numpy as np
from scipy.stats import norm

class bubble_trajectory(node_trajectory_base):
    def __init__(self, graph):
        super().__init__(graph)

    def extrapolate(self, t):
        time = self.time[-int(t > self.time[0])]
        a = 0.5
        dt = t - time
        return self.interpolate(time) + self.interpolate(t - a * dt/abs(dt), 1) * dt

    def _get_stats(self):
        velocities   = np.linalg.norm(self.displacements, axis = 1)/self.changes[:,0]
        self.mu_Vel     = np.average(velocities)
        self.sig_Vel    = np.std(velocities)
        self.mu_Area    = np.average((A := self.data[:,-2]))
        self.sig_Area   = np.std(A)

class bubble_trajectory_with_default_stats():
    def __init__(self, mu_Vel0, sig_Vel0, r_sig_Area0):
        self.mu_Vel0, self.sig_Vel0, self.r_sig_Area0 = mu_Vel0, sig_Vel0, r_sig_Area0

    def __call__(self, graph):
        trajectory = bubble_trajectory(graph)
        if len(trajectory) <= 2:
            trajectory.mu_Vel   = self.mu_Vel0
            trajectory.sig_Vel  = self.sig_Vel0
            trajectory.mu_Area  = np.average(trajectory.data[:,-2])
            trajectory.sig_Area = trajectory.mu_Area * self.r_sig_Area0
        else: trajectory._get_stats()
        return trajectory

class association_condition(Association_condition):
    def __init__(self,
                 max_displ_per_frame = 45,
                 radius_multiplyer = 2.5,
                 min_displacement = 30):

        def f(stop, start):
            if stop == start:                                                           return False

            dt = start.beginning[0] - stop.ending[0]
            dr = np.linalg.norm(start.beginning[2:4] - stop.ending[2:4])

            if   dt <= 0:                                                               return False
            elif dr > max_displ_per_frame * dt:                                         return False
            elif dr > (stop.ending[4]+start.beginning[4])/2 * radius_multiplyer * dt:   return False
            if dr < min_displacement * dt:                                              return True
            else:                                                                       return True

        super().__init__(f)

class combination_constraint(Combination_constraint):
    def __init__(self, upsilon, v_scaler = 10, max_a = 5):
        d_fi = lambda u, v: np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        def f(stops, starts):
            #If new or gone - defaults to True
            if len(stops) == 0 or len(starts) == 0: return True
            #Filters out quick changes in direction depending on velocity
            if len(stops) == 1 and len(starts) == 1:
                stop, start = stops[0], starts[0]
                dt = start.beginning[0] - stop.ending[0]
                mid_v = (start.positions[0,:] - stop.positions[-1,:])/dt
                if len(stop)  >= 2:
                    v = stop.displacements[-1,:]/stop.changes[-1,0]
                    acc = 2 * (mid_v - v)/(stop.changes[-1,0] + dt)
                    if np.linalg.norm(acc) > max_a: return False
                    if d_fi(v, mid_v) > (np.pi + 1e-6) * np.exp(-np.linalg.norm(v)/v_scaler): return False
                if len(start) >= 2:
                    v = start.displacements[0,:]/start.changes[0,0]
                    acc = 2 * (v - mid_v)/(start.changes[0,0] + dt)
                    if np.linalg.norm(acc) > max_a: return False
                    if d_fi(mid_v, v) > (np.pi + 1e-3) * np.exp(-np.linalg.norm(v)/v_scaler): return False
            #Area check
            S1, S2, sigs = 0, 0, 0
            for stop in stops:
                S1   += stop.mu_Area
                sigs += stop.mu_Area / stop.mu_Area
            for start in starts:
                S2   += start.mu_Area
                sigs += start.mu_Area / start.mu_Area
            sigs     /= len(stops) + len(starts)
            if abs(S2 - S1)/max(S2, S1) < upsilon * sigs:
                return True
            else: return False

        super().__init__(f)

class movement_func(statFunc):
    def __init__(self, sig_displacement, k):
        likelihood_displ = lambda dr, dt  : norm.pdf(dr, 0, sig_displacement * dt)/norm.pdf(0, 0, sig_displacement * dt)
        likelihood_S     = lambda dS, sigS: norm.pdf(dS, 0, sigS)/norm.pdf(0,0,sigS)

        def f(stop, start):
            stop, start = stop[0], start[0]

            t1  , t2    = stop.ending[0], start.beginning[0]
            dt = t2 - t1

            sig_S = (start.sig_Area + stop.sig_Area)/2
            dS    = start.mu_Area - stop.mu_Area
            b     = likelihood_S(dS, sig_S)

            try:
                p1 = stop(t1 + dt/2)
                try: p2 = start(t2 - dt/2)
                except:
                    p1 = stop(t2)
                    p2 = start.positions[0,:]
                finally: a = likelihood_displ(np.linalg.norm(p2 - p1), dt)

            except:
                p1 = stop.positions[-1,:]
                try:
                    p2 = start(t1)
                    a = likelihood_displ(np.linalg.norm(p2 - p1), dt)
                except:
                    p2 = start.positions[0,:]
                    dr      = np.linalg.norm(p2 - p1)
                    mu_d    = (start.mu_Vel + stop.mu_Vel)/2 * dt
                    sigma_d = (start.sig_Vel + stop.sig_Vel)/2 * dt
                    a       = norm.pdf(dr, mu_d, sigma_d)/norm.pdf(mu_d, mu_d, sigma_d)
            finally: return k * a + (1 - k) * b

        super().__init__(f, [1,1])


class split_merge_func(statFunc):
    def __init__(self, sig_displ, k, c, power = 3/2):
        likelihood_displ = lambda p1, p2, dt: np.divide(norm.pdf(np.linalg.norm(p2 - p1), 0, sig_displ*dt),norm.pdf(0, 0, sig_displ*dt))
        likelihood_S     = lambda dS, S_sig: norm.pdf(dS, 0, S_sig)/norm.pdf(0, 0, S_sig)
        f0 = lambda pos, Ss: np.array([np.dot(pos[:,i], Ss**power)/np.sum(Ss**power) for i in range(pos.shape[1])])

        c = int(c)
        def f(stops, starts):
            if c: #split
                trajectory   = stops[0]
                trajectories = starts
                t, p = trajectory.ending[0], trajectory.positions[-1,:]
                ts = [traject.beginning[0] for traject in trajectories]
            else: #merge
                trajectories = stops
                trajectory   = starts[0]
                t, p = trajectory.beginning[0], trajectory.positions[0,:]
                ts = [traject.ending[0] for traject in trajectories]

            positions = []
            dts = []
            Ss = []

            for traject, time in zip(trajectories, ts):
                try:
                    positions.append(traject(t))
                    Ss.append(traject.mu_Area)
                    dts.append(abs(time - t))
                except: pass

            if len(positions) != 0:
                p_predict = f0(np.array(positions), np.array(Ss))

                frac    = len(Ss)/len(trajectories)
                a       = likelihood_displ(p_predict, p, np.average(dts)) * frac
            else:
                a=0

            S       = np.sum(np.array([traject.mu_Area for traject in trajectories]))
            sig_S   = np.sum(np.array([tr.sig_Area for tr in trajectories]))

            dS      = trajectory.mu_Area - S
            S_sig   = (trajectory.sig_Area + sig_S)/2
            b       = likelihood_S(dS, S_sig)

            return k * a + (1 - k) * b


        super().__init__(f, [['n', 1],[1, 'n']][c])


class exit_entry_func(statFunc):
    def __init__(self, a, b, c, axis = 1):
        f0 = lambda x: 1/(1+np.exp(a*(x-b)))
        def f(stop, start):
            if c:
                trajectory, dt = start[0], -1
                t = trajectory.beginning[0] + dt
                try:    y = trajectory(t)[axis]
                except: y = trajectory.positions[0, axis] + trajectory.mu_Vel * dt
            else:
                trajectory, dt = stop[0], 1
                t = trajectory.ending[0] + dt
                try:    y = trajectory(t)[axis]
                except: y = trajectory.positions[-1, axis]  + trajectory.mu_Vel * dt
            return f0(y)

        super().__init__(f, [1 - c, c])
