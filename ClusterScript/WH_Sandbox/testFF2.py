import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


if not os.path.exists('../Dump/WHYSYKImagDumpfiles'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
else:
    path_to_dump = '../Dump/WHYSYKImagDumpfiles'


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_rolling_YSYKWH 
#import time


######### TODO: IMPLEMENT THE BLOODY PLOTTER FOR THE OFF_DIAG ELEMENTS ##########



Nbig = int(2**14)
err = 1e-6
#err = 1e-2
ITERMAX = 5000

global beta

beta_start = 1.
beta = beta_start
mu = 0.0
g = 0.0
r = 1.

#lamb = 0.005
#lamb = 10 #FF2
lamb = 10.
J = 0
#J = np.sqrt(lamb)
#J = 0.0001
#J = 0.


# g = np.sqrt(10**3)
# r = (10)**2

kappa = 1.
beta_step = 1

num = 1.1 

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')

Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
#omegar2 = ret_omegar2(g,beta)


#GDtau, DDtau = Gfreetau, Dfreetau

GDtau = Freq2TimeF((1j*omega+mu)/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
DDtau = Freq2TimeB((nu**2+r)/((nu**2 + r)**2 - J**2), Nbig, beta)
DODtau = Freq2TimeB(-J/((nu**2 + r)**2 - J**2), Nbig, beta)

# load_file = 'Nbig14beta1000_0lamb0_05J0_05g0_5r1_0.npy'
# #load_file = 'Nbig14beta1000_0lamb0_001J0_001g0_5r1_0.npy'
# GDtau, GODtau, DDtau, DODtau = np.load(os.path.join(path_to_dump,load_file))


fig, ax = plt.subplots(2,2)


Gconftau = Gfreetau # FF1
Dconftau = Dfreetau 

titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
titlestring += r' $\lambda$ = ' + str(lamb) + r' J = ' + str(J)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)
ax[0,0].plot(tau/beta, np.real(GDtau), 'r', label = 'numerics GDtau')
ax[0,0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'FF1 GDtau' )
#ax[0,0].set_ylim(-0.5,0.1)
ax[0,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0,0].set_ylabel(r'$\Re{GD(\tau)}$')
ax[0,0].legend()

ax[0,1].plot(tau/beta, np.real(GODtau), 'r', label = 'numerics Real GODtau')
ax[0,1].plot(tau/beta, np.imag(GODtau), 'k', label = 'numerics imag GODtau')
#ax[0,1].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[0,1].set_ylim(-1,1)
ax[0,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0,1].set_ylabel(r'$\Re{GOD(\tau)}$')
ax[0,1].legend()

ax[1,0].plot(tau/beta, np.real(DDtau), 'r', label = 'numerics DDtau')
ax[1,0].plot(tau/beta, np.real(Dconftau), 'b--', label = 'FF1 DDtau' )
#ax[1,0].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1,0].set_ylim(0,1)
ax[1,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1,0].set_ylabel(r'$\Re{DD(\tau)}$')
ax[1,0].legend()

ax[1,1].plot(tau/beta, np.real(DODtau), 'r', label = 'numerics real DODtau')
ax[1,1].plot(tau/beta, np.imag(DODtau), 'k', label = 'numerics imag DODtau')
#ax[1,1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
#ax[1,1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1,1].set_ylim(0,1)
ax[1,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1,1].set_ylabel(r'$\Re{DOD(\tau)}$')
ax[1,1].legend()


##### Log Linear ####


fig,ax = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

startT, stopT, stepT  = 1, 2000, 100
plotsliceT = slice(startT,stopT,stepT)

fitsliceT = slice(startT+500, startT + 600)
#fitslice = slice(start+25, start + 35)
functoplotT = np.abs(np.real(GDtau))
mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
print(f'slope of fit = {mT:.03f}')
# print('2 Delta  = ', 2*delta)

ax[0,0].semilogy(tau[plotsliceT], np.abs(np.real(GDtau[plotsliceT])),'p',label = 'numerics GDtau')
#ax[0,0].semilogy(tau[plotsliceT], conf_fit_GD[plotsliceT],'k--',label = 'ES power law')
#ax[0,0].semilogy(tau[startT:], -np.imag(Gconf[startT:]),'m.',label = 'ES solution')
#ax[0,0].semilogy(tau[startT:], alt_conf_fit_G[startT:],'g--', label = 'alt power law')
#ax[0,0].set_xlim(tau[startT]/2,tau[startT+15])
ax[0,0].semilogy(tau[plotsliceT], np.exp(mT*tau[plotsliceT] + cT), label=f'Fit with slope {mT:.03f}')
#ax[0,0].set_ylim(1e-1,1e1)
ax[0,0].set_xlabel(r'$\tau$')
ax[0,0].set_ylabel(r'$-\Re G(\tau)$')
#ax[0,0].set_aspect('equal', adjustable='box')
#ax[0,0].axis('square')
ax[0,0].legend()
ax[0,0].set_yscale('log')


ax[1,0].semilogy(tau[plotsliceT], np.abs(np.real(DDtau[plotsliceT])),'p',label='numerics DDtau')
#ax[1,0].semilogy(tau[startB:stopB], conf_fit_DD,'k--',label = 'ES power law')
#ax[1,0].semilogy(tau[startB:], np.real(Dconf[startB:]),'m.',label = 'ES solution')
#ax[1,0].semilogy(tau[startB:], alt_conf_fit_D,'g--', label = 'alt power law')
#ax[1,0].set_xlim(tau[startB]/2,tau[startB+15])
#ax[1,0].set_ylim(5e-1,100)
ax[1,0].set_xlabel(r'$\nu_n/g^2$')
ax[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
#ax[1,0].set_aspect('equal', adjustable='box')
ax[1,0].legend()

plt.show()










