from associator import Association_condition, Combination_constraint
from trajectories import node_trajectory_base
from stat_funcs import statFunc
import numpy as np
from scipy.stats import norm

class particle_trajectory(node_trajectory_base):
    mu_Vel0 = 0
    sig_Vel0 = 100
    def __init__(self, graph):
        super().__init__(graph)
        self._get_stats()

    def extrapolate(self, t):
        time = self.time[-int(t > self.time[0])]
        a = 0.5
        dt = t - time
        return self.interpolate(time) + self.interpolate(t - a * dt/abs(dt), 1) * dt

    def _get_stats(self):
        if len(self) <= 2:
            self.mu_Vel   = self.mu_Vel0
            self.sig_Vel  = self.sig_Vel0
        else:
            velocities   = np.linalg.norm(self.displacements, axis = 1)/self.changes[:,0]
            self.mu_Vel    = np.average(velocities)
            self.sig_Vel   = np.std(velocities)

class association_condition(Association_condition):
    def __init__(self, Soi = 45):

        def f(stop, start):
            if stop == start:                                                               return False

            dt = start.beginning[0] - stop.ending[0]
            dr = np.linalg.norm(start.beginning[2:4] - stop.ending[2:4])

            if dt <= 0:                                                                      return False
            if dr > Soi * dt:                                                               return False
            else:                                                                           return True

        super().__init__(f)

class combination_constraint(Combination_constraint):
    def __init__(self, v_scaler, max_a):
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
            return True
        super().__init__(f)

class movement_func(statFunc):
    def __init__(self, sig_displacement, sig_acc, k_v, w1, w2):
        likelihood_displ = lambda dr, dt  : norm.pdf(dr, 0, sig_displacement * dt)/norm.pdf(0, 0, sig_displacement * dt)
        likelihood_accel = lambda acc: norm.pds(acc, 0, sig_acceleration)/norm.pds(0, 0, sig_acceleration)
        likelihood_angle = lambda v, phi: norm.pdf(phi, 0, np.pi * np.exp(-1/vk * np.linalg.norm(v1)))/norm.pdf(0, 0, np.pi * np.exp(-1/k_v * np.linalg.norm(v)))
        def f(stop, start):
            stop, start = stop[0], start[0]

            t1  , t2    = stop.ending[0], start.beginning[0]
            dt = t2 - t1
            vk = (start.positions[0,:] - stop.positions[-1,:])/dt
            try:
                p1 = stop(t1 + dt/2)
                v1 = stop.velocities[-1,:]
                dt1 = stop.changes[-1,0]
                phi1 = np.arccos(np.dot(v1, vk)/(np.linalg.norm(v1)*np.linalg.norm(vk)))
                acc_1= 2*(np.linalg.norm(vk-v1)/ (dt + dt1))
                b1 = likelihood_angle(v1, phi1)
                c1 = likelihood_accel(acc_1)
                try:
                    p2 = start(t2 - dt/2)
                    v2 = start.velocities[0,:]
                    dt2 = start.changes[0,0]
                    phi2 = np.arccos(np.dot(v2, vk)/(np.linalg.norm(v2)*np.linalg.norm(vk)))
                    acc_2= 2*(np.linalg.norm(v2-vk)/ (dt + dt2))
                    b2 = likelihood_angle(vk, phi2)
                    c2 = likelihood_accel(acc_2)
                    b = b1 * b2
                    c = c1 * c2

                except:
                    p1 = stop(t2)
                    p2 = start.positions[0,:]
                    c = c1**2
                finally: a = likelihood_displ(np.linalg.norm(p2 - p1), dt)

            except:
                p1 = stop.positions[-1,:]
                try:
                    p2 = start(t1)
                    a = likelihood_displ(np.linalg.norm(p2 - p1), dt)
                    phi2 = np.arccos(np.dot(v2, vk)/(np.linalg.norm(v2)*np.linalg.norm(vk)))
                    acc_2= 2*(np.linalg.norm(v2-vk)/ (dt + dt2))
                    b2 = likelihood_angle(vk, phi2)
                    c2 = likelihood_accel(acc_2)
                    c = c2
                    b = b2**2
                except:
                    p2 = start.positions[0,:]
                    dr      = np.linalg.norm(p2 - p1)
                    mu_d    = (start.mu_Vel + stop.mu_Vel)/2 * dt
                    sigma_d = (start.sig_Vel + stop.sig_Vel)/2 * dt
                    a       = norm.pdf(dr, mu_d, sigma_d)/norm.pdf(mu_d, mu_d, sigma_d)
                    b, c = a**2, a**2
            finally: return w1*a + (1-w1)*(w2 * b + (1-w2) * c)

        super().__init__(f, [1,1])

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
