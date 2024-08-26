import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

path_to_loadfile = '../Dump/OnesideMET'
if not os.path.exists(path_to_loadfile):
    print('INCORRECT LOAD DIRECTORY SPECIFIED!!!!')
    print(path_to_loadfile)
    print('exiting.....')
    exit(1)

path_to_dump = '../Dump/OnesideInclSup'
if not os.path.exists(path_to_dump):
    print(f'path to Dump {path_to_dump} does not exist, making it now ....')
    os.makedirs(path_to_dump)



from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


DUMP = False
print("Here")

Nbig = int(2**14)
err = 1e-12
#err = 1e-2
ITERMAX = 15000

global beta

betalooplist = np.arange(1,501,1)
beta = betalooplist[0]
mu = 0.0
g = 0.5
r = 1.

#target_beta = 101.
target_beta = beta + 1 

kappa = 1.
beta_step = 1


Ftau = (1+1j)*np.ones(Nbig)


for beta in betalooplist:
    loadfile = 'OnesideMET'
    loadfile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    loadfile += 'g' + str(g).replace('.','_') + 'r' + str(r) + '.npy'  
    loadfile = loadfile.replace('.','_') 
    loadfile += '.npy'

    try:
        Gtau,Dtau = np.load(os.path.join(path_to_loadfile,loadfile))
    except FileNotFoundError:
        print("Filename: ", savefile)
        print("Input File not found!!! Exiting.......")
        exit(1)


    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')


    Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
    Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
    delta = 0.420374134464041
    omegar2 = ret_omegar2(g,beta)


    if np.sum(np.abs(Ftau[:10])) < 1e-2:
        Ftau = (1+1j)*np.ones_like(Gtau)

    #Gtau,Dtau = np.load('temp.npy')
    assert len(Gtau) == Nbig, 'Improperly loaded starting guess'

        itern = 0
        diff = err*1.1
        x = 0.01

        beta_step = 1 if (beta>130) else 1

        omega = ImagGridMaker(Nbig,beta,'fermion')
        nu = ImagGridMaker(Nbig,beta,'boson')
        tau = ImagGridMaker(Nbig,beta,'tau')
        diff = 1.
        iterni=0
        while(diff>err and itern < ITERMAX):
            itern+=1
            iterni += 1 
            diffold = 1.0*diff
            
            oldGtau = 1.0*Gtau
            oldDtau = 1.0*Dtau
            oldFtau = 1.0*Ftau
            
            if iterni == 1:
                oldGomega = Time2FreqF(oldGtau,Nbig,beta)
                oldDomega = Time2FreqB(oldDtau,Nbig,beta)
                oldFomega = Time2FreqF(oldFtau,Nbig,beta)
            else:
                oldGomega = 1.0*Gomega
                oldDomega = 1.0*Domega
                oldFomega = 1.0*Fomega
            
            Sigmatau = 1.0 * kappa * (g**2) * Dtau * Gtau
            #Pitau = 2.0 * g**2 * (Gtau * Gtau[::-1] - Ftau * Ftau[::-1]) #KMS G(-tau) = -G(beta-tau) , VIS
            Pitau = -2.0 * g**2 * (-1.* Gtau * Gtau[::-1] - np.conj(Ftau) * Ftau)#KMS G(-tau) = -G(beta-tau), me
            Phitau = -1.0 * kappa * (g**2) * Dtau * Ftau
            
            Sigmaomega = Time2FreqF(Sigmatau,Nbig,beta)
            Piomega =  Time2FreqB(Pitau,Nbig,beta)
            Phiomega = Time2FreqF(Phitau,Nbig,beta)
            
            detGmat = (1j*omega + mu - Sigmaomega) * (1j*omega - mu + np.conj(Sigmaomega)) - (np.abs(Phiomega))**2
            
            Gomega = x*((1j*omega - mu +np.conj(Sigmaomega))/(detGmat)) + (1-x)*oldGomega
            Fomega = x*((Phiomega)/(detGmat)) + (1-x)*oldFomega
            Domega = x*(1./(nu**2 + r - Piomega)) + (1-x)*oldDomega

            Gtau = Freq2TimeF(Gomega,Nbig,beta)
            Dtau = Freq2TimeB(Domega,Nbig,beta)
            Ftau = Freq2TimeF(Fomega,Nbig,beta)

            
            if iterni>0:
                # diffG = np.sqrt(np.sum((np.abs(Gtau-oldGtau))**2)) #changed
                # diffD = np.sqrt(np.sum((np.abs(Dtau-oldDtau))**2))
                diffG = np.sum((np.abs(Gtau-oldGtau))**2)#changed
                diffD = np.sum((np.abs(Dtau-oldDtau))**2)
                diffF = np.sum((np.abs(Ftau-oldFtau))**2)

                diff = 0.33*(diffG+diffD+diffF)

        if DUMP == True:
            savefile = 'OnesideSUP'
            savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
            savefile += 'g' + str(g) + 'r' + str(r)
            savefile = savefile.replace('.','_') 
            savefile += '.npy'
            np.save(os.path.join(path_to_dump, savefile), np.array([Gtau,Dtau,Ftau])) 
        print("##### Finished beta = ", beta, "############")
        print(f"F(tau = 0+) = {Ftau[0]:.4}")
        print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n',flush=True)


################## PLOTTING ######################
print(beta), print(tau[-1])
Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
FreeDtau = DfreeImagtau(tau,r,beta)

fig, ax = plt.subplots(3)

titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)
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

ax[2].plot(tau/beta, np.real(Ftau), 'r--', label = 'numerics Real Ftau')
ax[2].plot(tau/beta, np.imag(Ftau), 'b', label = 'numerics Imag Ftau')
#ax[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
#ax[2].set_ylim(-1,1)
ax[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[2].set_ylabel(r'$\Re{F(\tau)}$')
ax[2].legend()

#fig.suptitle(r'$\beta$ = ', beta)
#plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')






################ POWER LAW PLOT #####################

start, stop = Nbig//2, Nbig//2 + 100
startB, stopB = Nbig//2 + 1 , Nbig//2 + 101
delta = 0.420374134464041
alt_delta = 0.116902  

fitG_val = -np.imag(Gomega[start+0])*(g**2)
#fitG_val = -np.imag(Gconf[start:stop])*(g**2)
conf_fit_G = 1 * np.abs(omega/(g**2))**(2*delta - 1)
conf_fit_G = conf_fit_G/conf_fit_G[start] * fitG_val
alt_conf_fit_G = fitG_val * np.abs(omega/(g**2))**(2*alt_delta - 1)

fitD_val = np.real(Domega[startB])*(g**2)
#fitD_val = np.real(Dconf[startB:stopB])
conf_fit_D = 1 * np.abs(nu[startB:stopB]/(g**2))**(1-4*delta)
conf_fit_D = conf_fit_D/conf_fit_D[0] * fitD_val
alt_conf_fit_D = 1 * np.abs(nu[startB]/(g**2))**(1-4*alt_delta)


fig,(ax1,ax2,ax3) = plt.subplots(1,3)
#fig.set_figwidth(10)
titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

fitslice = slice(start+0, start + 15)
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
#ax1.set_aspect('equal', adjustable='box')
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
#ax2.set_aspect('equal', adjustable='box')
ax2.legend()


ax3.loglog(omega[start:stop]/(g**2), np.abs(np.imag(Fomega[start:stop])*(g**2)),'p',label = 'numerics imag Fomega')
ax3.loglog(omega[start:stop]/(g**2), np.abs(np.real(Fomega[start:stop])*(g**2)),'p',label = 'numerics real Fomega')
#ax3.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
#ax3.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
#ax3.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
#ax3.set_xlim(omega[start]/2,omega[start+15])
#ax3.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
#ax3.set_ylim(1e-1,1e1)
ax3.set_xlabel(r'$\omega_n/g^2$')
ax3.set_ylabel(r'$-g^2\,-\Im{F(\omega_n)}$')
#ax3.set_aspect('equal', adjustable='box')
#ax3.axis('square')
ax3.legend()

#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


