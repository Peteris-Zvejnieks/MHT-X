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
#%%
I = 38

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
images = unzip_images('%s\\Compressed Data\\Shapes.zip'%sub_dir)
del(I, J)
#%%
Sig_displacement_movement   = 130
Weight_movement             = 0.7
move = movement_func(Sig_displacement_movement, Weight_movement)

A                   = 0.01
Boundary            = 100
Height              = images[0].shape[0]
exitt   = exit_entry_func(A, Boundary, 1, 1)
entry   = exit_entry_func(-A, Height - Boundary, 0, 1)

Sig_displacement_movement_split_merge   = 150
Weight_split_merge                      = 0.9
Power = 3/2
merge  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 0, Power)
split  = split_merge_func(Sig_displacement_movement_split_merge, Weight_split_merge, 1, Power)

stat_funcs = [move, exitt, entry, split, merge]

Max_displ_per_frame = 400
Radius_multlplyer   = 5
Min_displacement    = 30
asc_condition  = association_condition(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon                 = 1.2
Velocity_coefficient    = 80
Max_acceleration        = 80
comb_constr = combination_constraint(Upsilon, Velocity_coefficient, Max_acceleration)

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
tracer = tTracer(aSSociator, stat_funcs, bubble_trajectory, Max_occlusion, Quantile, sub_dir)

parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
for name, value in zip(parameters.keys(), parameters.values()):
    print(name, ' :', value)
print('Likelihood :', tracer.get_total_log_likelihood())
string = '/search_'
tracer.dump_data(string, images, 15, 1)
string += '_' + str(tracer.get_total_log_likelihood())
import json
with open(sub_dir + '/Tracer Output' +string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
