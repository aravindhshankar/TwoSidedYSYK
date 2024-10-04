import sys
#############
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
path_to_loadfile = './Dump/redoCWH/'
# path_to_loadfile = './Outputs/'
# loadfile = 'cSYKbeta200kappa0_05.h5.npy'
# loadfile = 'cSYKbeta_M18T121000kappa0_01.npy'
# loadfile = 'cSYK_M18T12beta1000kappa0.01.npy'
# loadfile = 'GRM19T14kappa0_01beta1000.npy'
# loadfile = 'GRM20T17kappa0_01beta1000.npy'
# loadfile = 'GRM20T17beta20J1_0kappa0_01.npy'
# loadfile = 'cSYKbeta1000kappa0_01.npy'

if not os.path.exists(path_to_loadfile):
	raise(Exception('Load directory does not exist'))
	exit(1)


# savefile = os.path.join(path_to_output, savename+'.h5')
# savefile_dump = os.path.join(path_to_dump, savename+'.npy')

fft_check = testingscripts.realtimeFFT_validator() # Should return True

DUMP = True
PLOTTING = True


J = 1.
#beta = 100.
# beta = 1./(2e-4)
#beta = 1./(5e-5)
# beta = 200
# beta = 1000
# beta = 6000
beta = 1000
mu = 0. 
kappa = 0.01
# kappa = 0.05
ITERMAX = 10000

M = int(2**16) #number of points in the grid
# M = int(2**18) #number of points in the grid
# T = int(2**15) #upper cut-off fot the time
T = int(2**12)
#T = int(2**10)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
err = 1e-6
eta = dw*2.1
#delta = 0.420374134464041
delta = 0.25

print(f'omega max = {omega[-1]}')

# savename = 'GR'
savename = 'cSYKWH'
savefile = savename
savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
# savefile += 'beta' + str((round(beta*100))/100.) 
savefile += 'beta' + str(beta) 
savefile += 'J' + str(J)
savefile += 'kappa' + f'{kappa:.3}'
savefile = savefile.replace('.','_') 
savefiledump = savefile + '.npy' 
savefileoutput = savefile + '.h5'

# GDRomega, GODRomega = np.load(os.path.join(path_to_loadfile,loadfile))
GDRomega, GODRomega = np.load(os.path.join(path_to_loadfile,savefiledump))



# peak_idxlist = find_peaks(-np.imag(GDRomega)[M:],prominence=0.1)[0]
# print(omega[M:][peak_idxlist])

# c = omega[M:][peak_idxlist][0] / delta
# print('predicted peaks = ', c * (np.arange(4) + delta))

if PLOTTING == True:
	fig,ax = plt.subplots(1)
	ax.plot(omega[M:M+10000:10],-np.imag(GDRomega[M:M+10000:10]),'.-')
	ax.set_xlabel(r'$\omega$')
	ax.set_ylabel(r'$-Im G^R(\omega)$')

	# for peak_idx in peak_idxlist:
	# 	ax.axvline(omega[M:][peak_idx],ls='--')
	plt.show()
