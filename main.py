import os
w_dir = r"C:\Users\User\Documents\Zvejnieks\MHT-X_"
os.chdir(w_dir)

import matplotlib.pyplot as plt
import numpy as np

from custom_methods import *
del(Association_condition, Combination_constraint)
from tracer import Tracer as tTracer
from tracer import unzip_images
from associator import Associator as aAssociator

import glob

plt.rcParams['figure.dpi'] = 500
np.set_printoptions(suppress=True)
#%%
w_dir = r"C:\Users\User\Documents\Zvejnieks\Alex_tracing"
os.chdir(w_dir)
case_paths = sorted(glob.glob('./*'))
case_paths = [path for path in case_paths if os.path.isdir(path)]
#%%
I = 1
case_path = case_paths[I]
images = unzip_images('%s\\images.zip'%case_path)
del I
#%%
Sig_displacement_movement   = 130
Weight_movement             = 0.7
move = movement_func(Sig_displacement_movement, Weight_movement)

A                   = 1e-2
Boundary            = 50
Width              = images[0].shape[1]
exitt   = exit_entry_func(A, Boundary, 1, 0)
entry   = exit_entry_func(-A, Width - Boundary, 0, 0)

Sig_displacement_movement_split_merge   = 100
Weight_split_merge                      = 0.9
Power = 3/2
merge  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 0, Power)
split  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 1, Power)

stat_funcs = [move, exitt, entry, split, merge]

Max_displ_per_frame = 200
Radius_multlplyer   = 2
Min_displacement    = 0
asc_condition  = association_condition(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon                 = 5
Velocity_coefficient    = 80
Max_acceleration        = 80
comb_constr = combination_constraint(Velocity_coefficient, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr)

Mu_Vel0     = 50
Sig_Vel0    = 30
R_sig_Area0 = 1.5
R_sig_Volume0 = 1.5

bubble_trajectory.mu_Vel0 = Mu_Vel0
bubble_trajectory.sig_Vel0 = Sig_Vel0
bubble_trajectory.r_sig_Area0 = R_sig_Area0
bubble_trajectory.r_sig_Volume0 = R_sig_Volume0

Max_occlusion = 2
Quantile = 0.01
#%%
tracer = tTracer(aSSociator, stat_funcs, bubble_trajectory, Max_occlusion, Quantile, case_path)

parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
for name, value in zip(parameters.keys(), parameters.values()):
    print(name, ' :', value)
print('Likelihood :', tracer.get_total_log_likelihood())
string = '/search_'
tracer.dump_data(string, images, 25, 1)
string += '_' + str(tracer.get_total_log_likelihood())
import json
with open(case_path + '/Tracer Output' +string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
