import sys
import os 
if not os.path.exists('./Sources'):
	print("Error - Path to Sources directory not found ")
sys.path.insert(1,'./Sources')

import numpy as np
from matplotlib import pyplot as plt
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *
from scipy.signal import find_peaks

# savename = 'default_savename'
# path_to_output = './Outputs'
# path_to_dump  = './Dump/redoCWH'
path_to_loadfile = './DUMP/redoCWH/'
# path_to_loadfile = './Outputs/'
# loadfile = 'cSYKbeta200kappa0_05.h5.npy'
# loadfile = 'cSYKbeta_M18T121000kappa0_01.npy'
loadfile = 'cSYK_M18T12beta1000kappa0.01.npy'
# loadfile = 'cSYKbeta1000kappa0_01.npy'

if not os.path.exists(path_to_loadfile):
	raise(Exception('Load directory does not exist'))
	exit(1)


# savefile = os.path.join(path_to_output, savename+'.h5')
# savefile_dump = os.path.join(path_to_dump, savename+'.npy')

GDRomega, GODRomega = np.load(os.path.join(path_to_loadfile,loadfile))

fft_check = testingscripts.realtimeFFT_validator() # Should return True

DUMP = True
PLOTTING = True


J = 1.
#beta = 100.
# beta = 1./(2e-4)
#beta = 1./(5e-5)
# beta = 200
beta = 1000
mu = 0. 
kappa = 0.01
# kappa = 0.05
ITERMAX = 10000

M = int(2**18) #number of points in the grid
T = int(2**14) #upper cut-off fot the time
#M = int(2**16)
#T = int(2**10)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
err = 1e-6
eta = dw*2.1
#delta = 0.420374134464041
delta = 0.25

peak_idxlist = find_peaks(-np.imag(GDRomega)[M:],prominence=0.1)[0]
print(omega[M:][peak_idxlist])

c = omega[M:][peak_idxlist][0] / delta
print('predicted peaks = ', c * (np.arange(4) + delta))

if PLOTTING == True:
	fig,ax = plt.subplots(1)
	ax.plot(omega,-np.imag(GDRomega),'.-')
	ax.set_xlabel(r'$\omega$')
	ax.set_ylabel(r'$-Im G^R(\omega)$')

	for peak_idx in peak_idxlist:
		ax.axvline(omega[M:][peak_idx],ls='--')
	plt.show()
