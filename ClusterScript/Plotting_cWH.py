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
outfile = 'cSYK_WH_2165159.h5'
savepath = os.path.join(path_to_outfile, outfile)

if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


data = h52dict(savepath, verbose = True)
print(data.keys())

dw = np.pi/data['T']
temp = 1./data['beta']

titlestring = 'beta = ' + str(data['beta']) + ' mu = ' + str(data['mu'])
titlestring += ' Log2M = ' + str(np.log2(data['M'])) + 'kappa = ' + str(data['kappa'])


fig, (ax1,ax2) = plt.subplots(1,2)

fig.suptitle(titlestring)
xval  = data['omega']/(data['kappa']**(2./3.))

ax1.plot(data['omega'], data['rhoGD'],'.-')
ax1.set_xlabel('omega')
ax1.set_ylabel('rhoGD')
#ax1.set_xlim(-20,60)
ax1.set_xlim(-0.5,0.5)

ax2.plot(data['omega'], data['rhoGOD'],'.-')
#ax2.set_xlim(-20,60)
ax2.set_xlim(-0.5,0.5)
ax2.set_xlabel('omega')
ax2.set_ylabel('rhoGOD')

plt.show()

