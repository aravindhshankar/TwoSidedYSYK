import sys
import os
if not os.path.exists('../Sources'):
	print("Error - Path to sources directory not found")
sys.path.insert(1,'../Sources')
import h5py

import numpy as np
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *

M = int(2**18)
T = int(2**12)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2]-t[1]
grid_flag = testingscripts.RealGridValidator(omega,t,M,T,dt,dw)
print(omega[-1])
tot_freq_grid_points = int(2**12)
omega_max = 1
omega_min = -1*omega_max
idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
comp_omega_slice = slice(idx_min,idx_max,skip)
comp_omega = omega[comp_omega_slice]

print("# of points on original grid = ", idx_max - idx_min)
print(omega_min, omega_max)
print((omega_max-omega_min)/(dw*tot_freq_grid_points), skip)
print(comp_omega[0],comp_omega[-1], "Total points on final grid = ", len(comp_omega))



# Your Python dictionary
my_dict = {'omega': comp_omega}

# Specify the path to the HDF5 file
file_path = 'test_h5.h5'

dict2h5(my_dict, file_path, verbose=True)

loaded_dict = h52dict(file_path, verbose = True)
loaded_omega = loaded_dict['omega']

np.testing.assert_almost_equal(my_dict['omega'], loaded_dict['omega'], 5, 'Failed')

print(type(loaded_omega), loaded_omega[0])



