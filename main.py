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
w_dir = r'D:\Zvejnieks\Particles\for_tracing'
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 15
J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = os.getcwd() + sub_dirs[J][1:]
print(sub_dir)
del(I, J)
#%%
### Particle tracking 
## eqs 9 - 12
# I mainly play with the weights at this point
Sig_displacement_movement   = 8 # sigma_pos
Sig_acceleration            = 15 # simga_a
Velocity_scaler             = 50 # lambda
Weight_movement1            = 0.7 # beta_1
Weight_movement2            = 0.2 # beta_2
move = movement_func(Sig_displacement_movement, Sig_acceleration, Velocity_scaler, Weight_movement1, Weight_movement2)

### MHT-X tracking 
## eq 21

# Shouln't really be tinkered with, this is fine
A                   = 0.01 # a

# Shouln't really be tinkered with, this is fine
Boundary            = 7 # not in definition

path = sub_dir + '\\Compressed Data\\Shapes.zip'
import zipfile, io
from PIL import Image as img
with zipfile.ZipFile(path) as zp:
    names = zp.namelist()
    try:    names.remove(*[x for x in names if len(x.split('/')[-1]) == 0 or x.split(',')[-1] == 'ini'])
    except: pass
    image = img.open(io.BytesIO(zp.read(names[0])))

# Make this as large as is the wish for the sequence
Width               = image.width # not stricly in definition
# Boundary and Width make up b
exitt   = exit_entry_func(A, Boundary, 1, 0)
entry   = exit_entry_func(-A, Width - Boundary, 0, 0)

stat_funcs = [move, exitt, entry]
#%%
### Particle tracking 
## eq 8

# This is the most important one, ideally as big as could be, but can result in situations 
# that can not be resoved till the heat death of the universe.
SoiMax = 7              # R_max

# The bigger the lower the dropoff of R_max with respect to velocity
SoiVelScaler = 200      # lambda

# This can be played around with freely, 0 - all hope on PIV, 1 - all hope on splines
Extrap_w = 0.0         # alpha

asc_condition  = association_condition(SoiMax, SoiVelScaler, Extrap_w)

### MHT-X 
## eq 11
Velocity_scaler_constr  = 200 # lambda

## eq 12
# In my experience acts as a limmiter for jumps
Max_acceleration        = 6 # a_c
comb_constr = combination_constraint(Velocity_scaler_constr, Max_acceleration)

aSSociator = aAssociator(asc_condition, comb_constr, max_k = 1)
#%%
## Just some default values for when no information is not available and such
## these params should not matter too much when enough context has been generated
## as it turns out, lots of it is legacy, kek

# Should stay at 0, this is practically legact at this point
Vel_thresh = 0 # if velocity lower than this, replace it with default

# Legacy, kicks in only when thershold kicks in
Mu_Vel0 = 10  # default absolute velocity

# Velocity sigma is legacy, nothing uses it, nvm this
Sig_mul = 10 # multiplyer for velocity variance

# kicks in only when there are not enough points, is not multiplied by Sig_mul
Sig_Vel0 = 3 # default absolute velocity variance


# can and will have large impact on spline extrapolation, tightness of the splines
'''
Verbatim from documentation
    A smoothing condition. The amount of smoothness is determined by satisfying the conditions: 
    sum((w * (y - g))**2,axis=0) <= s, where g(x) is the smoothed interpolation of (x,y). 
    The user can use s to control the trade-off between closeness and smoothness of fit. 
    Larger s means more smoothing while smaller values of s indicate less smoothing. 
    Recommended values of s depend on the weights, w. 
    If the weights represent the inverse of the standard-deviation of y, 
    then a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)), 
    where m is the number of data points in x, y, and w.
'''
# Reading this I'm starting to think that maybe it should not be constant
Smoother = 30 # a parameter for splines, a smoothing constant

# This can be played around with freely, 0 - all hope on PIV, 1 - all hope on splines
# I just keep it the same as the other for consistnecy
Extrapolation_w = Extrap_w 

particle_trajectory.smoother = Smoother
particle_trajectory.mu_Vel0 = Mu_Vel0
particle_trajectory.sig_Vel0 = Sig_Vel0
particle_trajectory.vel_thresh = Vel_thresh
particle_trajectory.sig_mul = Sig_mul
particle_trajectory.extrapolation_w = Extrapolation_w
#%%
## these should be understandable
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
