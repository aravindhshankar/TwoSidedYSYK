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


path_to_outfile = './Dump/RTWHDumpfiles/'
#outfile = 'Y_WH_2153063.h5'
#path_to_outfile = './Outputs/RTWH/NFLstart'
#outfile = 'NFL10M16T12beta1000g0_5lamb0_01.h5'
outfile = 'RTWHlocalM18T15beta20g0_5lamb0_01.npy'
savepath = os.path.join(path_to_outfile, outfile)

if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


M=int(2**18)
T=2**15
beta = 20.
g = 0.05
lamb = 0.01
J = 0.
# GDomega,GODomega,DDomega,DODomega = np.load(savepath)
loaded = np.load(savepath)
print(len(loaded))
print(len(loaded[2]), 2*M)

omega,t  = RealGridMaker(M,T)




# rhoGD = -np.imag(GDomega)
# rhoGOD = -np.imag(GODomega)
# rhoDD = -np.imag(DDomega)
# rhoDOD = -np.imag(DODomega)

# fig, ax = plt.subplots(2,2)
# titlestring = 'beta = ' + str(beta) + ' lamb = ' + str(lamb) + ' J = ' + str(J)
# titlestring += ' Log2M = ' + str(np.log2(M))
# titlestring += ' g = ' + str(g)
# fig.suptitle(titlestring)

# ax[0,0].plot(omega, rhoGD)
# ax[0,0].set_xlim(-5,5)
# ax[0,0].set_title(r'rho GD')

# ax[0,1].plot(omega, rhoGOD)
# #ax[0,1].set_xlim(-1,1)
# ax[0,1].set_title(r'rho GOD')

# ax[1,0].plot(omega, rhoDD)
# #ax[1,0].set_xlim(-1,1)
# ax[1,0].set_title(r'rho DD')

# ax[1,1].plot(omega, rhoDOD)
# #ax[1,1].set_xlim(-1,1)
# ax[1,1].set_title(r'rho DOD')

# ############# log log plot ###################
# delta = 0.420374134464041
# fig, ax = plt.subplots(1)
# start = len(omega)//2 + 1 
# stop = start + 500
# temp_idx = np.argmin(np.abs(temp - omega))
# fitslice = slice(temp_idx+40, temp_idx + 50)
# #fitslice = slice(start+25, start + 35)

# functoplot = rhoGD
# m,c = np.polyfit(np.log(np.abs(omega[fitslice])), np.log(functoplot[fitslice]),1)
# print(f'slope of fit = {m:.03f}')
# print('2 Delta - 1 = ', 2*delta-1)

# ax.loglog(omega[start:stop], functoplot[start:stop],'p',label = 'numerics rhoGDomega')
# ax.loglog(omega[start:stop], np.exp(c)*np.abs(omega[start:stop])**m, label=f'Fit with slope {m:.03f}')
# ax.axvline((temp,),ls='--')
# #ax.set_ylim(1e-1,1e1)
# ax.set_xlabel(r'$\omega$')
# ax.set_ylabel(r'$-\Im{GD(\omega)}$')
# #ax.set_aspect('equal', adjustable='box')
# #ax.axis('square')
# ax.legend()

# plt.show()



fig,ax  = plt.subplots(1)
to_plot = np.imag(loaded[0])
ax.plot(omega,to_plot)
peakpoints = find_peaks(to_plot,prominence=0.001)[0]
peakvals = [omega[peak] for peak in peakpoints]
print(f'peakvals = {peakvals}')
print(f'diffs = {np.diff(peakvals)}')
for peak in peakvals:
    ax.axvline(peak,ls='--',c='gray')
plt.show()