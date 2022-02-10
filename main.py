import matplotlib.pyplot as plt
import numpy as np

from custom_methods import *
del(Association_condition, Combination_constraint)
from tracer import Tracer as tTracer
from associator import Associator as aAssociator

import glob
import os

plt.rcParams['figure.dpi'] = 500
np.set_printoptions(suppress=True)
#%%
# drive = 'C:\\'
# w_dir = drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
w_dir = r'C:\Users\User\Documents\Zvejnieks\ParticleProcessing\Particles\for_tracing'
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 5
J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = os.getcwd() + sub_dirs[J][1:]
print(sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 8
Sig_acceleration            = 15
Velocity_scaler             = 50
Weight_movement1            = 0.7
Weight_movement2            = 0.2
move = movement_func(Sig_displacement_movement, Sig_acceleration, Velocity_scaler, Weight_movement1, Weight_movement2)

A                   = 0.01
Boundary            = 7
Width               = 424
exitt   = exit_entry_func(A, Boundary, 1, 0)
entry   = exit_entry_func(-A, Width - Boundary, 0, 0)

stat_funcs = [move, exitt, entry]
#%%
SoiMax = 7
SoiVelScaler = 200
Extrap_w = 0.3

asc_condition  = association_condition(SoiMax, SoiVelScaler, Extrap_w)

Velocity_scaler_constr  = 200
Max_acceleration        = 6
comb_constr = combination_constraint(Velocity_scaler_constr, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr, max_k = 1)
#%%
Mu_Vel0 = 10
Sig_Vel0 = 3
Vel_thresh = 0
Sig_mul = 10
Smoother = 30
Extrapolation_w = Extrap_w

particle_trajectory.smoother = Smoother
particle_trajectory.mu_Vel0 = Mu_Vel0
particle_trajectory.sig_Vel0 = Sig_Vel0
particle_trajectory.vel_thresh = Vel_thresh
particle_trajectory.sig_mul = Sig_mul
particle_trajectory.extrapolation_w = Extrapolation_w
#%%
Max_occlusion = 2
Quantile = 0.5
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
for name, value in zip(parameters.keys(), parameters.values()): print(name, ' :', value)
tracer = tTracer(aSSociator, stat_funcs, particle_trajectory, Max_occlusion, Quantile, sub_dir)
#%%
print('Likelihood :', tracer.get_total_log_likelihood())
string = '/search_'
tracer.dump_data(string, memory = 15, smallest_trajectories = 5)
import json
with open(sub_dir + '/Tracer Output' +string +str(len(tracer.trajectories)) + '_' + str(tracer.get_total_log_likelihood()) + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
