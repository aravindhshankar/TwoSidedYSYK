import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

if not os.path.exists('../Dump/SupCondYSYKImagDumpfiles'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
else:
    path_to_dump = '../Dump/SupCondYSYKImagDumpfiles'


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


DUMP = False

Nbig = int(2**14)
err = 1e-5
#err = 1e-2
ITERMAX = 5000

global beta

beta = 1.
mu = 0.0
g = 0.5
r = 1.

target_beta = 500

kappa = 1.
beta_step = 1


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)

GAP = 0.1
#Gtau = Gfreetau
Dtau = Dfreetau
Gtau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Ftau = Freq2TimeF(GAP/((1j*omega)**2 + mu**2),Nbig,beta)



fig, ax = plt.subplots(3)

titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)
ax[0].plot(tau/beta, np.real(Gtau), 'r', label = 'Gtau')
#ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
ax[0].plot(tau/beta, np.real(Gfreetau), 'g-.', label = 'FreeGtau' )
# ax[0].set_ylim(-1,1)
ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0].set_ylabel(r'$\Re{G(\tau)}$')
ax[0].legend()

ax[1].plot(tau/beta, np.real(Dtau), 'r', label = 'Dtau')
#ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
ax[1].plot(tau/beta, np.real(Dfreetau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1].set_ylabel(r'$\Re{D(\tau)}$')
ax[1].legend()

ax[2].plot(tau/beta, np.real(Ftau), 'r--', label = 'Real Ftau')
ax[2].plot(tau/beta, np.imag(Ftau), 'b', label = 'Imag Ftau')
#ax[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[2].set_ylim(-1,1)
ax[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[2].set_ylabel(r'$\Re{F(\tau)}$')
ax[2].legend()

plt.show()




























