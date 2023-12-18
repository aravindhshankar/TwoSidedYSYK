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

path_to_outfile = './Outputs'
outfile = 'Y_WH_2153063.h5'
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

fig, ax = plt.subplots(2,2)

fig.suptitle(titlestring)

ax[0,0].plot(data['omega'], data['rhoGD'])
ax[0,0].set_xlim(-0.1,0.1)
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

plt.show()




