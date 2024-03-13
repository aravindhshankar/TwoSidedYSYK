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
#import time


Nbig = int(2**14)
err = 1e-5

global beta

beta = 1000.0
mu = 0.0
g = 0.5
r = 1.

kappa = 1.
lamb = 0.05
J = 0.05

savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
savefile += 'lamb' + str(lamb) + 'J' + str(J)
savefile += 'g' + str(g) + 'r' + str(r)
savefile = savefile.replace('.','_') 
savefile += '.npy'

try :
    #plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
    plotfile = os.path.join(path_to_dump, savefile)
except FileNotFoundError: 
    print("INPUT FILE NOT FOUND")
    exit(1)

print('savefile = ', savefile)

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)

#Gtau = Gfreetau
#Dtau = Dfreetau

GDtau, GODtau, DDtau, DODtau = np.load(plotfile)
assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

GDomega = Time2FreqF(GDtau, Nbig, beta)
GODomega = Time2FreqF(GODtau, Nbig, beta)
DDomega = Time2FreqB(DDtau, Nbig, beta)
DODomega = Time2FreqB(DODtau, Nbig, beta)


################## PLOTTING ######################
print(beta), print(tau[-1])
Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
FreeDtau = DfreeImagtau(tau,r,beta)


fig, ax = plt.subplots(2,2)

titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
titlestring += r' $\lambda$ = ' + str(lamb) + r' J = ' + str(J)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)
ax[0,0].plot(tau/beta, np.real(GDtau), 'r', label = 'numerics GDtau')
ax[0,0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[0,0].set_ylim(-1,1)
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
ax[1,0].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
ax[1,0].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
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

#fig.suptitle(r'$\beta$ = ', beta)
#plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')
#plt.show()






############### POWER LAW PLOT #####################

start, stop = Nbig//2, Nbig//2 + 100
startB, stopB = Nbig//2 + 1 , Nbig//2 + 101
delta = 0.420374134464041
alt_delta = 0.116902  

fitGD_val = -np.imag(GDomega[start+0])*(g**2)
#fitGD_val = -np.imag(Gconf[start:stop])*(g**2)
conf_fit_GD = 1 * np.abs(omega/(g**2))**(2*delta - 1)
conf_fit_GD = conf_fit_GD/conf_fit_GD[start] * fitGD_val

fitDD_val = np.real(DDomega[startB])*(g**2)
#fitDD_val = np.real(Dconf[startB:stopB])
conf_fit_DD = 1 * np.abs(nu[startB:stopB]/(g**2))**(1-4*delta)
conf_fit_DD = conf_fit_DD/conf_fit_DD[0] * fitDD_val



fig,ax = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

fitslice = slice(start+0, start + 15)
#fitslice = slice(start+25, start + 35)
functoplot = -np.imag(GDomega)*(g**2)
m,c = np.polyfit(np.log(np.abs(omega[fitslice])/(g**2)), np.log(functoplot[fitslice]),1)
print(f'slope of fit = {m:.03f}')
print('2 Delta - 1 = ', 2*delta-1)

ax[0,0].loglog(omega[start:stop]/(g**2), -np.imag(GDomega[start:stop])*(g**2),'p',label = 'numerics GDomega')
ax[0,0].loglog(omega[start:stop]/(g**2), conf_fit_GD[start:stop],'k--',label = 'ES power law')
#ax[0,0].loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
#ax[0,0].loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
#ax[0,0].set_xlim(omega[start]/2,omega[start+15])
ax[0,0].loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
#ax[0,0].set_ylim(1e-1,1e1)
ax[0,0].set_xlabel(r'$\omega_n/g^2$')
ax[0,0].set_ylabel(r'$-g^2\,\Im{GD(\omega_n)}$')
#ax[0,0].set_aspect('equal', adjustable='box')
#ax[0,0].axis('square')
ax[0,0].legend()


ax[1,0].loglog(nu[startB:stopB]/(g**2), np.real(DDomega[startB:stopB])*(g**2),'p',label='numerics')
ax[1,0].loglog(nu[startB:stopB]/(g**2), conf_fit_DD,'k--',label = 'ES power law')
#ax[1,0].loglog(nu[startB:]/(g**2), np.real(Dconf[startB:]),'m.',label = 'ES solution')
#ax[1,0].loglog(nu[startB:]/(g**2), alt_conf_fit_D,'g--', label = 'alt power law')
#ax[1,0].set_xlim(nu[startB]/2,nu[startB+15])
#ax[1,0].set_ylim(5e-1,100)
ax[1,0].set_xlabel(r'$\nu_n/g^2$')
ax[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
#ax[1,0].set_aspect('equal', adjustable='box')
ax[1,0].legend()



###################### Log-Linear Plot ###############################


fig,ax = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

startT, stopT  = 1, 2000

fitsliceT = slice(startT, startT + 10)
#fitslice = slice(start+25, start + 35)
functoplotT = np.abs(np.real(GDtau))
mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
print(f'slope of fit = {mT:.03f}')
# print('2 Delta  = ', 2*delta)

ax[0,0].semilogy(tau[startT:stopT], np.abs(np.real(GDtau[startT:stopT])),'p',label = 'numerics GDtau')
#ax[0,0].semilogy(tau[startT:stopT], conf_fit_GD[startT:stopT],'k--',label = 'ES power law')
#ax[0,0].semilogy(tau[startT:], -np.imag(Gconf[startT:]),'m.',label = 'ES solution')
#ax[0,0].semilogy(tau[startT:], alt_conf_fit_G[startT:],'g--', label = 'alt power law')
#ax[0,0].set_xlim(tau[startT]/2,tau[startT+15])
ax[0,0].semilogy(tau[startT:stopT], np.exp(mT*tau[startT:stopT] + cT), label=f'Fit with slope {mT:.03f}')
#ax[0,0].set_ylim(1e-1,1e1)
ax[0,0].set_xlabel(r'$\tau$')
ax[0,0].set_ylabel(r'$-\Re G(\tau)$')
#ax[0,0].set_aspect('equal', adjustable='box')
#ax[0,0].axis('square')
ax[0,0].legend()
ax[0,0].set_yscale('log')


ax[1,0].semilogy(tau[startT:stopT], np.abs(np.real(DDtau[startT:stopT])),'p',label='numerics DDtau')
#ax[1,0].semilogy(tau[startB:stopB], conf_fit_DD,'k--',label = 'ES power law')
#ax[1,0].semilogy(tau[startB:], np.real(Dconf[startB:]),'m.',label = 'ES solution')
#ax[1,0].semilogy(tau[startB:], alt_conf_fit_D,'g--', label = 'alt power law')
#ax[1,0].set_xlim(tau[startB]/2,tau[startB+15])
#ax[1,0].set_ylim(5e-1,100)
ax[1,0].set_xlabel(r'$\nu_n/g^2$')
ax[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
#ax[1,0].set_aspect('equal', adjustable='box')
ax[1,0].legend()





#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


















