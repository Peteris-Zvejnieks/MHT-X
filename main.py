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
I = 2

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 0.01
Weight_movement             = 1
move = movement_func(Sig_displacement_movement, Weight_movement)

A                   = 50
Boundary            = 0.025
Height              = 0.15
exitt   = exit_entry_func(A, Boundary, 1, 0)
entry   = exit_entry_func(-A, Height - Boundary, 0, 0)

Sig_displacement_movement_split_merge   = 0.01 #@param {type:"slider", min:0, max:150}
Weight_split_merge                      = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
merge  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 0)
split  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 1)

stat_funcs = [move, exitt, entry, split, merge]
#%%
Max_displ_per_frame = 0.008
Radius_multlplyer   = 4
Min_displacement    = 0.001
asc_condition  = association_condition(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon                 = 4
Velocity_coefficient    = 10
Max_acceleration        = 5
comb_constr = combination_constraint(Upsilon, Velocity_coefficient, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr)
#%%
Mu_Vel0     = 0.0025 #@param {type:"slider", min:0, max:100}
Sig_Vel0    = 0.01 #@param {type:"slider", min:0, max:100}
R_sig_Area0 = 0.5 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
trajectory_stats = bubble_trajectory_with_default_stats(Mu_Vel0, Sig_Vel0, R_sig_Area0)
#%%
Max_occlusion = 1
Quantile = 0.05
#%%
tracer = tTracer(aSSociator, stat_funcs, trajectory_stats, Max_occlusion, Quantile, sub_dir)
#%%
indx = 69
string = '/'+'test_new_constr_%i_'%indx+str(Max_occlusion)
#%%
tracer.dump_data(string, 15, 1)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
