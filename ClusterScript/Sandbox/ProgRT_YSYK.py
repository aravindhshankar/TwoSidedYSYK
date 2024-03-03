import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time
from YSYK_iterator import RE_YSYK_iterator
import testingscripts
assert testingscripts.realtimeFFT_validator(), "FT_Testing failed" # Should return True

DUMP = True

M = int(2**13) #number of points in the grid
T = 2**9 #upper cut-off for the time
err = 1e-5
#err = 1e-2

omega,t  = RealGridMaker(M,T)
dw = omega[2] - omega[1]
dt = t[2] - t[1]

delta = 0.420374134464041
ITERMAX = 5000
global beta

mu = 0.0
g = 0.5
r = 1.
kappa = 1.
eta = dw*2.1

beta_start = 1.
beta = beta_start
target_beta = 100.
beta_step = 0.1


#Gtau,Dtau = np.load('temp.npy')
#assert len(Gtau) == Nbig, 'Improperly loaded starting guess'
GRomega = 1/(omega + 1j*eta + mu)
#DRomega = -1/(-1.0*(omega + 1j*eta)**2 + r) # modified
DRomega = 1/(-1.0*(omega + 1j*eta)**2 + r)
grid = [M,omega,t]
pars = [g,mu,r]
while(beta < target_beta):
    #beta_step = 0.01 if (beta<1) else 1
    GRomega, DRomega = RE_YSYK_iterator(GRomega,DRomega,grid,pars,beta,err=err,ITERMAX=ITERMAX,eta = eta,verbose=True) 
   
    if DUMP == True and int(beta) % 10 == 0 :
        savefile = 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
        savefile += 'beta' + str(beta) 
        savefile += 'g' + str(g).replace('.','_') + 'r' + str(r) + '.npy'  
        np.save(savefile, np.array([GRomega,DRomega])) 
        print(savefile)
    print("##### Finished beta = ", beta, "############")
    beta = beta + beta_step





GRt = (0.5/np.pi) * freq2time(GRomega,M,dt)
DRt = (0.5/np.pi) * freq2time(DRomega,M,dt)

temp = 1./beta
Tstar = g**2 * np.sqrt(r)

################## PLOTTING ######################
#np.save('beta10kN14g0_5r1x0_01.npy', np.array([Gtau,Dtau])) 
print('beta = ', beta)
fig, ax = plt.subplots(2)

ax[0].plot(2*np.pi*t/beta, np.real(GRt), label = r'numerics Re(GR(t))')
ax[0].plot(2*np.pi*t/beta, np.imag(GRt), label = r'numerics Im(GR(t))')
#ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[0].set_ylim(-1,1)
ax[0].set_xlim(-10,100)
ax[0].set_xlabel(r'$2\pi t/\beta$',labelpad = 0)
ax[0].set_ylabel(r'${G^R(t)}$')
ax[0].legend()

ax[1].plot(2*np.pi*t/beta, np.real(DRt),'-',label = 'numerics Re(DR(t))')
ax[1].plot(2*np.pi*t/beta, np.imag(DRt), label = 'numerics Im(DR(t))')
#ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
#ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$2\pi t/\beta$',labelpad = 0)
ax[1].set_ylabel(r'${D^R(t)}$')
#ax[1].set_xlim(0,beta/(2*np.pi))
ax[1].set_xlim(-10,10)
ax[1].legend()

print('eta = ', eta)

#fig.suptitle(r'$\beta$ = ', beta)
#plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')
plt.show()



############### Spectral functions plot ####################

rhoG, rhoD = -1.0*np.imag(GRomega), -1.0*np.imag(DRomega)
#rhoG, rhoD = -1.0*np.imag(GRomega), 1.0*np.imag(DRomega)
fig, ax = plt.subplots(2)
titlestring = r'$\beta$ = ' + str(beta) + r', temp/dw = ' + f'{(temp/dw):.2f}' + r', $g = $' + str(g)
fig.suptitle(titlestring)
omegar2 = ret_omegar2(g,beta)

match_omega = 0.5
match_point = M + int(np.floor(match_omega/dw))
om_th = np.sqrt(omegar2)
#om_th = 1/beta
#match_coeff = rhoD[match_point]*(np.abs(omega[match_point])**(4*delta-1))
#match_rhoD = match_coeff * np.abs(omega)**(1-4*delta)
match_coeff = rhoD[match_point]*(np.abs(omega[match_point] - om_th + 1j*eta )**(4*delta-1))
match_rhoD = match_coeff * np.abs(omega-om_th)**(1-4*delta)

ax[0].plot(omega, rhoG, 'r', label = r'numerics $\rho_G(\omega)$')
#ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[0].set_ylim(-1,1)
ax[0].set_xlim(-0.1,0.1)
ax[0].set_xlabel(r'$\omega$',labelpad = 0)
ax[0].set_ylabel(r'$-\Im{G^R(\omega)}$')
ax[0].legend()

ax[1].plot(omega, rhoD, 'r', label = r'numerics $\rho_D(\omega)$')
#ax[1].plot(omega,-np.imag(thermalfreeboson),label = 'free boson with thermal mass')
#ax[1].plot(omega,match_rhoD, c = 'k', ls = '--', label = r'$c |\omega - \omega_r|^{1-4\Delta}$')
#ax[1].plot(omega, np.imag(DfreeRealomega(omega,r,eta=1./beta)), ls = '--', label = 'free boson with bare mass')

#ax[1].set_ylim(-0.2,1.)
ax[1].set_ylim(-2,2)
ax[1].set_xlabel(r'$\omega$',labelpad = 0)
ax[1].set_ylabel(r'$-\Im{D^R(\omega)}$')
ax[1].set_xlim(-0.02,1.5)
ax[1].legend()
ax[1].plot(omega, np.zeros_like(omega),ls = '--', c = 'gray')
ax[1].axvline([0], ls = '--', c = 'gray')
ax[1].axvline([1/beta], ls = '--', c = 'gray')
ax[1].axvline([eta], ls = '--', c = 'blue')
#ax[1].axvline([Tstar], ls = '--', c = 'green')
ax[1].axvline(om_th, ls = '--', c = 'orange')
ax[1].axvline(np.sqrt(r), ls = '--', c = 'magenta')
#ax[1].axvline([omega[peakrhoD[0][-1]]], ls = '--', c = 'black')
ax[1].text(1/beta+0.002,0.6, r'$T$',rotation=90)
ax[1].text(eta+0.001,0.4, r'$\eta$',rotation=90)
ax[1].text(om_th+0.002,-0.1, r'$\omega_r$',rotation=90)
#ax[1].text(Tstar+0.002,0.4, r'$T^{*}$',rotation=90)
ax[1].text(np.sqrt(r)+0.002,-0.1, r'$\omega_0$',rotation=90)

#plt.savefig('KoenraadEmails/RealTimeConvBosonWithMR.pdf')
#plt.savefig('KoenraadEmails/RealTimeConvBosonWithoutMR.pdf')
plt.show()
print(DRomega[-1])




################ POWER LAW PLOT #####################

# powD = 1. - 4*Delta
delta = 0.420374134464041
start,stop = M+1, M+10000

avg = 0.5*(np.real(GRomega)- np.imag(GRomega))

fitG_val = -np.imag(GRomega[start])
conf_fit_G = 1 * np.abs(omega)**(2*delta - 1)
conf_fit_G = conf_fit_G/conf_fit_G[start] * fitG_val


fitD_val = -np.imag(DRomega[start])
conf_fit_D = 1 * np.abs(omega[start:stop])**(1-4*delta)
conf_fit_D = conf_fit_D/conf_fit_D[0] * fitD_val

#meet_idx= np.argmin(abs(np.real(GRomega)+np.imag(GRomega)))
meet_idx = omega_idx(temp,dw,M)

#fitslice = slice(meet_idx, meet_idx + 15)
#fitslice = slice(start+10, start + 20)
#fitslice = slice(start+25, start + 35)
fitslice = slice(meet_idx - 10, meet_idx+10)
print(omega[meet_idx])
#functoplot = -np.imag(GRomega)
functoplot = avg
m,c = np.polyfit(np.log(np.abs(omega[fitslice])), np.log(functoplot[fitslice]),1)
print(f'slope of fit = {m:.03f}')
print('2Delta - 1 = ', 2*delta-1)

# fitD_val = np.abs(DRomega[start])
# conf_fit_D = 1 * np.abs(omega[start:stop]+1j*eta)**(1-4*delta)
# conf_fit_D = conf_fit_D/conf_fit_D[0] * fitD_val

fig,(ax1,ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{M}$ = ' + str(np.log2(M)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

ax1.loglog(np.abs(omega[start:stop]), -np.imag(GRomega[start:stop]),'p',label = '-Im G')
ax1.loglog(np.abs(omega[start:stop]), np.real(GRomega[start:stop]),'p',label = 'Re G')
ax1.loglog(np.abs(omega[start:stop]), avg[start:stop],'p',label = 'avg')

#ax1.loglog(omega[start:stop], np.abs(GRomega[start:stop]),'p',label = 'numerics')
ax1.loglog(np.abs(omega[start:stop]), conf_fit_G[start:stop],'k--',label = 'ES power law')
ax1.loglog(np.abs(omega[start:stop]),np.exp(c)*np.abs(omega[start:stop])**m, label=f'Fit with slope {m:.03f}')
#ax1.set_xlim(1e-3,1e-1)
ax1.axvline([temp],ls = '-.', c= 'gray',label = 'temperature')
ax1.axvline([g**(2/3)],ls = '--', c= 'purple',label = r'$g^{2/3}$')
ax1.axvline([np.sqrt(r)],ls = '--', c= 'magenta',label = r'$\omega_0$')
ax1.axvline([omega[meet_idx]],ls = '--', label = 'fit omega')

ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$-\,\Im{G(\omega)}$')
#ax1.set_ylabel(r'$|G(\omega)|$')
#ax1.set_aspect('equal', adjustable='box')
#ax1.axis('square')
ax1.legend(loc = 'lower left')
#ax1.text(0.05,100,f'slope of fit = {m:.03f}')

ax2.loglog(np.abs(omega[start:stop]), -np.imag(DRomega[start:stop]),'p',label='numerics')
#ax2.loglog(omega[start:stop], np.abs(DRomega[start:stop]),'p',label='numerics')
#ax2.loglog(np.abs(omega[start:stop]+1j*eta), conf_fit_D,'k--',label = 'ES power law')

ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'$-\Im{D(\omega)}$',labelpad = None)
#ax2.set_ylabel(r'$|D(\omega)|$',labelpad = None)
#ax2.set_aspect('equal', adjustable='box')
ax2.legend(loc = 'upper right')

#plt.savefig('KoenraadEmails/PowerLawRealTimeConvBosonWithMR.pdf')
#plt.savefig('KoenraadEmails/PowerLawRealTimeConvBosonWithoutMR.pdf')dEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


