import matplotlib.pyplot as plt
import numpy as np

from custom_methods import *
del(Association_condition, Combination_constraint)
from tracer import Tracer as tTracer
from tracer import unzip_images
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
images = unzip_images('%s\\Compressed Data\\Shapes.zip'%sub_dir)
#%%
I = 6

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 30
Weight_movement             = 0.3
move = movement_func(Sig_displacement_movement, Weight_movement)

A                   = 0.1
Boundary            = 80
Height              = images[0].shape[0]
exitt   = exit_entry_func(A, Boundary, 1, 1)
entry   = exit_entry_func(-A, Height - Boundary, 0, 1)

Sig_displacement_movement_split_merge   = 30
Weight_split_merge                      = 0.7
Power = 3/2
merge  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 0, Power)
split  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 1, Power)

stat_funcs = [move, exitt, entry, split, merge]
#%%
Max_displ_per_frame = 200
Radius_multlplyer   = 6
Min_displacement    = 60
asc_condition  = association_condition(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon                 = 0.6
Velocity_coefficient    = 300
Max_acceleration        = 132
comb_constr = combination_constraint(Upsilon, Velocity_coefficient, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr)
#%%
Mu_Vel0     = 20
Sig_Vel0    = 30
R_sig_Volume0 = 1
trajectory_stats = bubble_trajectory_with_default_stats(Mu_Vel0, Sig_Vel0, R_sig_Volume0)
#%%
Max_occlusion = 3
Quantile = 0.01
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
