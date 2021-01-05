
import matplotlib.pyplot as plt
import numpy as np

from custom_methods import *
from trajectories import node_trajectory_with_stats
from tracer import Tracer as tTracer

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
I = 12

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 40
Weight_movement             = 0.5
move    = movement_func(Sig_displacement_movement, Weight_movement)

A                   = 0.1
Boundary            = 20
Height              = 1208
exit     = exit_entry_func(A, Boundary, 1, 0)
entry    = exit_entry_func(-A, width - Boundary, 0, 0)

Sig_displacement_movement_split_merge   = 64 #@param {type:"slider", min:0, max:150}
Weight_split_merge                      = 0.6 #@param {type:"slider", min:0, max:1, step:0.01}
merge  = multi_bubble_likelihood_func(Sig_displacement2, K2, 0)
split  = multi_bubble_likelihood_func(Sig_displacement2, K2, 1)

stat_funcs = [move, exit, entry, split, merge]
#%%
Max_displ_per_frame = 300
Radius_multlplyer   = 4
Min_displacement    = 40
asc_condition  = association_condition(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon                 = 3
Velocity_coefficient    = 30
Max_acceleration        = 5
comb_constr = combination_constraint(Upsilon, Velocity_coefficient, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr)
#%%
Mu_Vel0     = 100 #@param {type:"slider", min:0, max:100}
Sig_Vel0    = 6 #@param {type:"slider", min:0, max:100}
R_sig_Area0 = 1 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
trajectory_stats = bubble_trajectory_with_default_stats(Mu_Vel0, Sig_Vel0, R_sig_Area0)
#%%
Max_occlusion = 3
Quantile = 0.5
#%%
tracer = tTracer(aSSociator, stat_funcs, trajectory_stats, Max_occlusion, Quantile, sub_dir)
#%%
indx = 33
string = '/'+'test_new_constr_%i_'%indx+str(Max_occlusion)
#%%
tracer.dump_data(string, 15, 5)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
