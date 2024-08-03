import numpy as np 
from matplotlib import pyplot as plt
import sys
import os
from scipy.signal import find_peaks

if not os.path.exists('./Sources'):
	print("Error - Path to sources directory not found")
sys.path.insert(1,'./Sources')
from SYK_fft import *
import h5py
import matplotlib.pyplot as plt


# path_to_outfile = './Dump/RTWHDumpfiles/'
path_to_outfile = './Dump/RTGapscaling/'
# path_to_BH = './Dump/RTWHDumpfiles0_00/'

# outfile = 'RTgapM16T12beta200g0_5lamb0_06.npy'
# BH_outfile = 'l_00M16T12beta200g0_5lamb0_0.npy'

M=int(2**16)
T=2**12
beta = 200
temp=1./beta
g = 0.5
# lamb = 0.06
J = 0.
# GDomega,GODomega,DDomega,DODomega = np.load(savepath)
lamb_start = 0.01
target_lamb = 0.04
lamblooplist = np.arange(lamb_start,target_lamb+1e-10, 0.001)
lambsavelist = lamblooplist
delta = 0.420374134464041

omega,t  = RealGridMaker(M,T)
dt = t[2]-t[1]
dw = omega[2]-omega[1]
print(f'grid spacing = {dw}')
gaplist = np.zeros_like(lambsavelist)





for ival, lamb in enumerate(lamblooplist):
	savefile = 'RTgap'
	savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
	# savefile += 'beta' + str((round(beta*100))/100.) 
	savefile += 'beta' + str(beta) 
	savefile += 'g' + str(g)
	savefile += 'lamb' + f'{lamb:.3}'
	savefile = savefile.replace('.','_') 
	savefiledump = savefile + '.npy'
	outfile = savefiledump

	savepath = os.path.join(path_to_outfile, outfile)
	if not os.path.exists(savepath):
		raise(Exception(f"WH Output file {outfile} not found"))

	loaded = np.array(np.load(savepath))
	GDomega,GODomega,DDomega,DODomega = loaded
	rhoGD = -np.imag(GDomega)
	rhoGOD = -np.imag(GODomega)
	rhoDD = -np.imag(DDomega)
	rhoDOD = -np.imag(DODomega)

	# peaks = find_peaks(rhoGD,prominence=1)[0]
	# # print(peaks)
	# peakvals = omega[peaks]
	# print(peakvals)

	peakmax = omega[M + np.argmax(rhoGD[M:])]
	gaplist[ival] = peakmax
	# print(peakmax)


print('Exited from loop after loading all data')

gradslope = np.gradient(gaplist,lamblooplist)
m1,c1 = np.polyfit(np.log(np.abs(lamblooplist)), np.log(gaplist),1)
slope_expect = 1./(2-2*delta)
print(f'Expected Slope = {slope_expect:.4}')
print(f'mean gradient = {np.mean(gradslope):.4}')
print(f'Slope from fit = {m1:.4}')

fig,ax = plt.subplots(1)
ax.loglog(lamblooplist,gaplist,'.-')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Energy gap from spectral function')
ax.loglog(lamblooplist, np.exp(c1)*np.abs(lamblooplist)**m1, label=f'Fit with slope {m1:.03f}')
ax.axvline(1./beta,ls='--',c='gray',label='Temperature')
ax.legend()



# np.save('RTgaplistExtended.npy', np.array([lamblooplist, gaplist]))

plt.show()