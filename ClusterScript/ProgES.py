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


savename = 'default_savename'
path_to_output = './Outputs'
if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created")

path_to_subfolder = './Outputs/ProgESOutputs' 
if not os.path.exists(path_to_subfolder):
    os.makedirs(path_to_subfolder)
    print("Subfolder ProgESOutputs created")

if len(sys.argv) > 1: 
    savename = str(sys.argv[1])

compressed_savefile = os.path.join(path_to_subfolder, savename+'.h5')
docstring = ' rhoLL = -ImG, rhoLR = 1j*ReG '


##########################################

Nbig = int(2**18)
#err = 1e-4
err = 1e-5
ITERMAX = 200

global beta

beta_start = 1
beta = beta_start
mu = 0.0
g = 0.5
r = 1.

target_beta = 200

# g = np.sqrt(10**3)
# r = (10)**2

kappa = 1.



omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)

#Gtau = Gfreetau
#Dtau = Dfreetau

#Gtau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
#Dtau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)

Gtau = -0.5*np.ones(Nbig)
Dtau = 1.0*np.ones(Nbig)

#Gtau,Dtau = np.load('temp.npy')
assert len(Gtau) == Nbig, 'Improperly loaded starting guess'

for beta in range(beta_start, target_beta+1, 1):
    itern = 0
    diff = 1.
    diffG = 1.
    diffD = 1.
    x = 0.5
    xG = 0.5
    xD = 0.5

    print("##### NOW beta = ", beta, "############\n")

    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')

    while(diff>err and itern < ITERMAX):
        itern+=1
        diffold = 1.0*diff
        diffoldG = 1.0*diffG
        diffoldD = 1.0*diffD
        
        oldGtau = 1.0*Gtau
        oldDtau = 1.0*Dtau
        
        if itern == 1:
            oldGomega = Time2FreqF(oldGtau,Nbig,beta)
            oldDomega = Time2FreqB(oldDtau,Nbig,beta)
        else:
            oldGomega = 1.0*Gomega
            oldDomega = 1.0*Domega
        
        Sigmatau = 1.0 * kappa * (g**2) * Dtau * Gtau
        Pitau = 2.0 * g**2 * Gtau * Gtau[::-1] #KMS G(-tau) = -G(beta-tau)
        
        Sigmaomega = Time2FreqF(Sigmatau,Nbig,beta)
        Piomega =  Time2FreqB(Pitau,Nbig,beta)
        # if itern < 15 : 
        #     Piomega[Nbig//2] = 1.0*r - omegar2
        #Piomega[Nbig//2] = 1.0*r - omegar2
        
        
        Gomega = xG*(1./(1j*omega + mu - Sigmaomega)) + (1-xG)*oldGomega
        Domega = xD*(1./(nu**2 + r - Piomega)) + (1-xD)*oldDomega

        Gtau = Freq2TimeF(Gomega,Nbig,beta)
        Dtau = Freq2TimeB(Domega,Nbig,beta)

        
        if itern>0:
            diffG = np.sqrt(np.sum((np.abs(Gtau-oldGtau))**2)) #changed
            diffD = np.sqrt(np.sum((np.abs(Dtau-oldDtau))**2))
            #diff = np.max([diffG,diffD])
            diff = 0.5*(diffG+diffD)
            diffG, diffD = diff, diff
            
            if diffG>diffoldG:
                xG/=2.
            if diffD>diffoldD:
                xD/=2.
            #print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD)

    if beta % 100 == 0 :
        savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        savefile += 'g' + str(g).replace('.','_') + 'r' + str(r) + '.npy'  
        np.save(savefile, np.array([Gtau,Dtau])) 
        print(savefile)

################## PLOTTING ######################
#np.save('temp.npy', np.array([Gtau,Dtau])) 
print(beta), print(tau[-1])
Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
FreeDtau = DfreeImagtau(tau,r,beta)

fig, ax = plt.subplots(2)

ax[0].plot(tau/beta, np.real(Gtau), 'r', label = 'numerics Gtau')
ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
ax[0].set_ylim(-1,1)
ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0].set_ylabel(r'$\Re{G(\tau)}$')
ax[0].legend()

ax[1].plot(tau/beta, np.real(Dtau), 'r', label = 'numerics Dtau')
ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1].set_ylabel(r'$\Re{D(\tau)}$')
ax[1].legend()

#fig.suptitle(r'$\beta$ = ', beta)
#plt.savefig('../../KoenraadEmails/WithMR_imagtime.pdf',bbox_inches='tight')
plt.show()






################ POWER LAW PLOT #####################

start, stop = Nbig//2, Nbig//2 + 100
startB, stopB = Nbig//2 + 1 , Nbig//2 + 101
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
titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
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

#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('KoenraadEmails/ImagFreqpowerlaw_withMR.pdf', bbox_inches = 'tight')
plt.show()


