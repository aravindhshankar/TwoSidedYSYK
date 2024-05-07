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
path_to_outfile = './Dump/RTWHDumpfiles0_05'
path_to_BH = './Dump/RTWHDumpfiles0_00'
#outfile = 'Y_WH_2153063.h5'
#path_to_outfile = './Outputs/RTWH/NFLstart'
#outfile = 'NFL10M16T12beta1000g0_5lamb0_01.h5'
# outfile = 'RTWHlocalM18T15beta20g0_5lamb0_01.npy'
BH_outfile = 'l_00M16T12beta200g0_5lamb0_0.npy'
outfile = 'l_05M16T12beta199g0_5lamb0_05.npy'
savepath = os.path.join(path_to_outfile, outfile)
BHsavepath = os.path.join(path_to_BH, BH_outfile)
if not os.path.exists(savepath):
	raise(Exception("WH Output file not found"))
if not os.path.exists(BHsavepath):
	raise(Exception("BH Output file not found"))


M=int(2**16)
T=2**12
beta = 200.
temp=1./beta
g = 0.5
lamb = 0.05
J = 0.
# GDomega,GODomega,DDomega,DODomega = np.load(savepath)
loaded = np.array(np.load(savepath))
loaded_BH = np.array(np.load(BHsavepath))

# print(len(loaded))
# print(len(loaded[2]), 2*M)
delta = 0.420374134464041
expo = 1./(2-2*delta)
lambexpo = lamb**expo

omega,t  = RealGridMaker(M,T)

idx = 0
fig,ax  = plt.subplots(1)
to_plot = -np.imag(loaded[idx])
BH_to_plot = -np.imag(loaded_BH[idx])

# to_plot -= min(to_plot)
# BH_to_plot -= min(BH_to_plot)

deltarho = to_plot / BH_to_plot

# xaxis = omega
xaxis = omega/lambexpo
ax.set_xlabel(r'$\frac{\omega}{\lambda^{1/(2-2\Delta)}}$')
ax.plot(xaxis,to_plot)
ax.plot(xaxis,BH_to_plot,'--')
ax.plot(xaxis,deltarho)
# ax.set_ylim(0,10)
# ax.set_xlim(-5,5)
ax.set_xlim(-20,50)
print(f'min BH = {min(BH_to_plot)}')
print(f'min WH = {min(to_plot)}')
peakpoints = find_peaks(deltarho,prominence=1.)[0]
peakvals = [xaxis[peak] for peak in peakpoints if peak>=M]
print(f'peakvals = {peakvals}')
print(f'diffs = {np.diff(peakvals)}')
# ax.axvline(g**(2./3), ls= '--',c='blue',label=r'$g^{2/3}$')
# ax.axvline(1./beta,ls='--',c='magenta',label=r'temperature')
ax.axvline(g**(2./3)/lambexpo, ls= '--',c='blue',label=r'$g^{2/3}$')
ax.axvline(temp/lambexpo,ls='--',c='magenta',label=r'temperature')
for peak in peakvals:
    ax.axvline(peak,ls='--',c='gray')
ax.legend()
ax.set_xticks(np.arange(-20,50,1))

# GDomega,GODomega,DDomega,DODomega = loaded_BH
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
# delta = 0.420374134464041
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























plt.show()