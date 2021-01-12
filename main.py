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
drive = 'C:\\'
w_dir = drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 20

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = os.getcwd() + sub_dirs[J][1:]
print(sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 6
Sig_acceleration            = 8
Velocity_scaler             = 6
Weight_movement1            = 0.6
Weight_movement2            = 0.5
move = movement_func(Sig_displacement_movement, Sig_acceleration, Velocity_scaler, Weight_movement1, Weight_movement2)

A                   = 0.04
Boundary            = 15
Width               = 480
exitt   = exit_entry_func(A, Boundary, 1, 0)
entry   = exit_entry_func(-A, Width - Boundary, 0, 0)

stat_funcs = [move, exitt, entry]
#%%
Soi = 10
asc_condition  = association_condition(Soi)

Velocity_scaler_constr  = 10
Max_acceleration        = 8
comb_constr = combination_constraint(Velocity_scaler_constr, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr, max_k = 1)
#%%
Mu_Vel0     = 22
Sig_Vel0    = 8
particle_trajectory.Mu_Vel0 = Mu_Vel0
particle_trajectory.Sig_Vel0 = Sig_Vel0
#%%
Max_occlusion = 1
Quantile = 0.4
#%%
tracer = tTracer(aSSociator, stat_funcs, particle_trajectory, Max_occlusion, Quantile, sub_dir)
#%%
indx = 18
string = '/'+'test_new_constr_%i_'%indx+str(Max_occlusion)
#%%
tracer.dump_data(string, 15, 10)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
