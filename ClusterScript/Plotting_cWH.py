import sys
import os
if not os.path.exists('./Sources'):
	print("Error - Path to sources directory not found")
	raise(Exception("Error - Path to sources directory not found"))
else:
	sys.path.insert(1,'./Sources')
import h5py
import matplotlib.pyplot as plt


import numpy as np
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *

path_to_outfile = './Outputs'
#outfile = 'cSYK_WH_2332886.h5'
#outfile = 'testingStephconv.h5'
outfile = 'StephcSYK_WH_2335508.h5'
savepath = os.path.join(path_to_outfile, outfile)

BHpath = os.path.join(path_to_outfile,'StephcSYK_WH_2336401.h5')

if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


data = h52dict(savepath, verbose = True)
#data = h52dict(BHpath, verbose = True)
BHdata = h52dict(BHpath, verbose = True)
print(data.keys())

dw = np.pi/data['T']
temp = 1./data['beta']

titlestring = 'beta = ' + str(data['beta']) + ' mu = ' + str(data['mu'])
titlestring += ' Log2M = ' + str(np.log2(data['M'])) + 'kappa = ' + str(data['kappa'])


fig, (ax1,ax2) = plt.subplots(1,2)

fig.suptitle(titlestring)
xval  = data['omega']/(data['kappa']**(2./3.))

#ax1.plot(data['omega'], BHdata['rhoGD'],'--')
ax1.plot(data['omega'], data['rhoGD']/BHdata['rhoGD'],'.-')
ax1.set_xlabel('omega')
#ax1.set_ylabel('rhoGD')
ax1.set_ylabel(r'$\delta \rho_{GD}$')
#ax1.set_xlim(-10,50)
ax1.set_xlim(-0.1,0.1)

ax2.plot(data['omega'], data['rhoGOD'],'.-')
#ax2.set_xlim(-1,1)
ax2.set_xlim(-0.05,0.05)
ax2.set_xlabel('omega')
ax2.set_ylabel('rhoGOD')

plt.show()

