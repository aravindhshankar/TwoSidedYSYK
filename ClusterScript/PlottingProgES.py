#### Plotting functionality to be added for data from cluster
#### WARNING : INCOMPLETE FILE


import sys
import os 
if not os.path.exists('./Sources'):
    print("Error - Path to Sources directory not found ")
    raise(Exception("Error - Path to Sources directory not found "))
sys.path.insert(1,'./Sources')

import numpy as np
from matplotlib import pyplot as plt
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *

#Gtau, Dtau = np.load('./Outputs/ProgESOutputs/default_savenameNbig16beta20g0_5r1.0.npy')


path_to_outfile = './Outputs/ProgESOutputs'
#outfile = 'cSYK_WH_2332886.h5'
outfile = 'ProgES_2364560Nbig22beta200g0_5r1_0.h5'
#outfile = 'StephcSYK_WH_2335479.h5'
savepath = os.path.join(path_to_outfile, outfile)


if not os.path.exists(savepath):
	raise(Exception("Output file not found"))


data = h52dict(savepath, verbose = True)
#data = h52dict(BHpath, verbose = True)

print(data.keys())

Nbig = data['Nbig']
# r = data['r']
r = 1.
g = data['g']


fig, ax = plt.subplots(2)
tau = data['tau']
Gtau = data['Gtau']
Dtau = data['Dtau']
beta = data['beta']

ax[0].plot(tau/beta, np.real(Gtau), 'r', label = 'numerics Gtau')
#ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
ax[0].set_ylim(-1,1)
ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0].set_ylabel(r'$\Re{G(\tau)}$')
ax[0].legend()

ax[1].plot(tau/beta, np.real(Dtau), 'r', label = 'numerics Dtau')
#ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
#ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1].set_ylabel(r'$\Re{D(\tau)}$')
ax[1].legend()

titlestring = r'$\beta$ = ' + str(data['beta']) 
titlestring += r' $\log_2$N = ' + str(np.log2(data['Nbig']))
titlestring += r' $g$ = ' + str(data['g'])
titlestring += r' $\omega_0$ = ' + str(np.sqrt(r))
fig.suptitle(titlestring)
plt.savefig('../KoenraadEmails/28FebProgES_beta200.pdf',bbox_inches='tight')
plt.show()






# ################## PLOTTING ######################
# print(beta), print(tau[-1])
# Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
# Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
# FreeDtau = DfreeImagtau(tau,r,beta)

# fig, ax = plt.subplots(2)

# ax[0].plot(tau/beta, np.real(Gtau), 'r', label = 'numerics Gtau')
# ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
# ax[0].set_ylim(-1,1)
# ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
# ax[0].set_ylabel(r'$\Re{G(\tau)}$')
# ax[0].legend()

# ax[1].plot(tau/beta, np.real(Dtau), 'r', label = 'numerics Dtau')
# ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
# ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
# #ax[1].set_ylim(0,1)
# ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
# ax[1].set_ylabel(r'$\Re{D(\tau)}$')
# ax[1].legend()

# #fig.suptitle(r'$\beta$ = ', beta)
# #plt.savefig('../../KoenraadEmails/WithMR_imagtime.pdf',bbox_inches='tight')
# plt.show()





# ################ POWER LAW PLOT #####################

omega = data['omega']
nu = data['nu']
Gomega = data['Gomega']
Domega = data['Domega']

print(omega[0:10])
print('2pi/beta = ', 2*np.pi/beta)
print('dw = ', omega[1]-omega[0])



start, stop = 0, 0 + 100
startB, stopB = 1 , 1+ 100
delta = 0.420374134464041
alt_delta = 0.116902  

fitG_val = -np.imag(Gomega[start])*(g**2)
#fitG_val = -np.imag(Gconf[start:stop])*(g**2)
conf_fit_G = 1 * np.abs(omega/(g**2))**(2*delta - 1)
conf_fit_G = conf_fit_G/conf_fit_G[start] * fitG_val
alt_conf_fit_G = fitG_val * np.abs(omega/(g**2))**(2*alt_delta - 1)

fitD_val = np.real(Domega[startB])*(g**2)
#fitD_val = np.real(Dconf[startB:stopB])
conf_fit_D = 1 * np.abs(nu[startB:stopB]/(g**2))**(1-4*delta)
conf_fit_D = conf_fit_D/conf_fit_D[0] * fitD_val
alt_conf_fit_D = 1 * np.abs(nu[startB]/(g**2))**(1-4*alt_delta)


fig,(ax1,ax2) = plt.subplots(1,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

fitslice = slice(start+0, start + 5)
#fitslice = slice(start+25, start + 35)
functoplot = -np.imag(Gomega)*(g**2)
m,c = np.polyfit(np.log(np.abs(omega[fitslice])/(g**2)), np.log(functoplot[fitslice]),1)
print(f'slope of fit = {m:.03f}')
#print('2 Delta - 1 = ', 2*delta-1)

ax1.loglog(omega[start:stop]/(g**2), -np.imag(Gomega[start:stop])*(g**2),'p',label = 'numerics')
ax1.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
#ax1.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
#ax1.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
#ax1.set_xlim(omega[start]/2,omega[start+15])
ax1.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
#ax1.set_ylim(1e-1,1e1)
ax1.set_xlabel(r'$\omega_n/g^2$')
ax1.set_ylabel(r'$-g^2\,\Im{G(\omega_n)}$')
ax1.set_aspect('equal', adjustable='box')
#ax1.axis('square')
ax1.legend()


ax2.loglog(nu[startB:stopB]/(g**2), np.real(Domega[startB:stopB])*(g**2),'p',label='numerics')
ax2.loglog(nu[startB:stopB]/(g**2), conf_fit_D,'k--',label = 'ES power law')
#ax2.loglog(nu[startB:]/(g**2), np.real(Dconf[startB:]),'m.',label = 'ES solution')
#ax2.loglog(nu[startB:]/(g**2), alt_conf_fit_D,'g--', label = 'alt power law')
#ax2.set_xlim(nu[startB]/2,nu[startB+15])
#ax2.set_ylim(5e-1,100)
ax2.set_xlabel(r'$\nu_n/g^2$')
ax2.set_ylabel(r'$g^2\,\Re{D(\nu_n)}$',labelpad = None)
ax2.set_aspect('equal', adjustable='box')
ax2.legend()

plt.savefig('../KoenraadEmails/28Feblowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('KoenraadEmails/ImagFreqpowerlaw_withMR.pdf', bbox_inches = 'tight')
plt.show()
