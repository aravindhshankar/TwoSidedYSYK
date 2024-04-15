import sys
import os
if not os.path.exists('./Sources'):
	print("Error - Path to sources directory not found")
sys.path.insert(1,'./Sources')
import h5py
import matplotlib.pyplot as plt


import numpy as np
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *

path_to_outfile = './Outputs/RTWH/'
#outfile = 'Y_WH_2153063.h5'
#path_to_outfile = './Outputs/RTWH/NFLstart'
#outfile = 'NFL10M16T12beta1000g0_5lamb0_01.h5'
outfile = 'RTWHlocalM18T14beta50g0_5lamb0_01.h5'
savepath = os.path.join(path_to_outfile, outfile)

if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


data = h52dict(savepath, verbose = True)
print(data.keys())

dw = np.pi/data['T']
temp = 1./data['beta']

print(f'dw = {dw}:.4f', f'temp = {temp}:.4f')
print(f'temp/dw = {temp/dw}:.2f')

titlestring = 'beta = ' + str(data['beta']) + ' lamb = ' + str(data['lamb']) + ' J = ' + str(data['J'])
titlestring += ' Log2M = ' + str(np.log2(data['M']))
titlestring += ' g = ' + str(data['g'])

fig, ax = plt.subplots(2,2)

fig.suptitle(titlestring)

ax[0,0].plot(data['omega'], data['rhoGD'])
ax[0,0].set_xlim(-0.5,0.5)
ax[0,0].set_title(r'rho GD')

ax[0,1].plot(data['omega'], data['rhoGOD'])
#ax[0,1].set_xlim(-1,1)
ax[0,1].set_title(r'rho GOD')

ax[1,0].plot(data['omega'], data['rhoDD'])
#ax[1,0].set_xlim(-1,1)
ax[1,0].set_title(r'rho DD')

ax[1,1].plot(data['omega'], data['rhoDOD'])
#ax[1,1].set_xlim(-1,1)
ax[1,1].set_title(r'rho DOD')

############# log log plot ###################
delta = 0.420374134464041
fig, ax = plt.subplots(1)
start = len(data['omega'])//2 + 1 
stop = start + 200
temp_idx = np.argmin(np.abs(temp - data['omega']))
fitslice = slice(temp_idx+40, temp_idx + 50)
#fitslice = slice(start+25, start + 35)

functoplot = data['rhoGD']
m,c = np.polyfit(np.log(np.abs(data['omega'][fitslice])), np.log(functoplot[fitslice]),1)
print(f'slope of fit = {m:.03f}')
print('2 Delta - 1 = ', 2*delta-1)

ax.loglog(data['omega'][start:stop], functoplot[start:stop],'p',label = 'numerics rhoGDomega')
ax.loglog(data['omega'][start:stop], np.exp(c)*np.abs(data['omega'][start:stop])**m, label=f'Fit with slope {m:.03f}')
ax.axvline((temp,),ls='--')
#ax.set_ylim(1e-1,1e1)
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$-\Im{GD(\omega)}$')
#ax.set_aspect('equal', adjustable='box')
#ax.axis('square')
ax.legend()

plt.show()