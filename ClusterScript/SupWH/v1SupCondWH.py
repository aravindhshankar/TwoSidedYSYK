import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

path_to_dump = '../Dump/SupCondWHImagDumpfiles'

if not os.path.exists(path_to_dump):
    print("Error - Path to Dump directory not found")
    print("Creating Dump directory : ", path_to_dump)
    os.makedirs(path_to_dump)
    #raise Exception("Error - Path to Dump directory not found ")


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


DUMP = True
PLOTTING = False

Nbig = int(2**14)
err = 1e-12
#err = 1e-2
ITERMAX = 5000

global beta

beta_start = 1
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
alpha = 0.
lamb = 0.05
# lamb = 0.0
J = 0.0

target_beta = 120.

kappa = 1.
beta_step = 1


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


# Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
# Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
# FreeDtau = DfreeImagtau(tau,r,beta)
Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)

#Gtau = Gfreetau
DDtau = Dfreetau
# Gtau = np.zeros_like(Dtau)
GDtau = Gfreetau
FDtau = GDtau.copy()
# Ftau = np.ones_like(Dtau)
# Ftau = np.zeros_like(Dtau)

DODtau = np.zeros_like(DDtau)
GODtau = np.zeros_like(GDtau)
FODtau = np.zeros_like(FDtau)

assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

while(beta <= target_beta):
    itern = 0
    diff = err*1.1
    x = 0.01

    beta_step = 1 if (beta>130) else 1

    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')
    diff = 1.
    while(diff>err and itern < ITERMAX):
        itern+=1
        diffold = 1.0*diff
        
        oldGDtau = 1.0*GDtau
        oldDDtau = 1.0*DDtau
        oldFDtau = 1.0*FDtau
        oldGODtau = 1.0*GODtau
        oldDODtau = 1.0*DODtau
        oldFODtau = 1.0*FODtau
        
        if itern == 1:
            oldGDomega = Time2FreqF(oldGDtau,Nbig,beta)
            oldDDomega = Time2FreqB(oldDDtau,Nbig,beta)
            oldFDomega = Time2FreqF(oldFDtau,Nbig,beta)
            oldGODomega = Time2FreqF(oldGODtau,Nbig,beta)
            oldDODomega = Time2FreqB(oldDODtau,Nbig,beta)
            oldFODomega = Time2FreqF(oldFODtau,Nbig,beta)
        else:
            oldGDomega = 1.0*GDomega
            oldDDomega = 1.0*DDomega
            oldFDomega = 1.0*FDomega
            oldGODomega = 1.0*GODomega
            oldDODomega = 1.0*DODomega
            oldFODomega = 1.0*FODomega
        
        SigmaDtau = 1.0 * kappa * (g**2) * DDtau * GDtau
        #Pitau = 2.0 * g**2 * (Gtau * Gtau[::-1] - Ftau * Ftau[::-1]) #KMS G(-tau) = -G(beta-tau) , VIS
        PiDtau = -2.0 * g**2 * (-1.* GDtau * GDtau[::-1] - (1-alpha) * np.conj(FDtau) * FDtau)#KMS G(-tau) = -G(beta-tau), me
        PhiDtau = -1.0 * (1-alpha) * kappa * (g**2) * DDtau * FDtau
        SigmaODtau = 1.0 * kappa * (g**2) * DODtau * GODtau
        #Pitau = 2.0 * g**2 * (Gtau * Gtau[::-1] - Ftau * Ftau[::-1]) #KMS G(-tau) = -G(beta-tau) , VIS
        PiODtau = -2.0 * g**2 * (-1.* GODtau * GODtau[::-1] - (1-alpha) * np.conj(FODtau) * FODtau)#KMS G(-tau) = -G(beta-tau), me
        PhiODtau = -1.0 * (1-alpha) * kappa * (g**2) * DODtau * FODtau
        
        SigmaDomega = Time2FreqF(SigmaDtau,Nbig,beta)
        PiDomega =  Time2FreqB(PiDtau,Nbig,beta)
        PhiDomega = Time2FreqF(PhiDtau,Nbig,beta)
        SigmaODomega = Time2FreqF(SigmaODtau,Nbig,beta)
        PiODomega =  Time2FreqB(PiODtau,Nbig,beta)
        PhiODomega = Time2FreqF(PhiODtau,Nbig,beta)

        # detGmat = (1j*omega + mu - Sigmaomega) * (1j*omega - mu + np.conj(Sigmaomega)) - (np.abs(Phiomega))**2
        detD = (nu**2 + r - PiDomega)**2 - (J - PiODomega)**2
        
        GDomega = 0.5*x*(-((lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))/(-((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega)))) + (-lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb - 1j*omega)**2 - (SigmaDomega + SigmaODomega)*(np.conj(SigmaDomega) + np.conj(SigmaODomega)) - (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)) + 2*(1j*lamb + omega)*np.imag(SigmaDomega + SigmaODomega))) + (1-x)*oldGDomega

        FDomega = 0.5*x*((-PhiDomega + PhiODomega)/(-((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) + (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) - np.conj(SigmaODomega)) - (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldFDomega

        DDomega = x*((nu**2 + r - PiDomega)/(detD)) + (1-x)*oldDDomega

        GODomega = 0.5*x*((lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))/(-((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) + (-lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb - 1j*omega)**2 - (SigmaDomega + SigmaODomega)*(np.conj(SigmaDomega) + np.conj(SigmaODomega)) - (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)) + 2*(1j*lamb + omega)*np.imag(SigmaDomega + SigmaODomega))) + (1-x)*oldGODomega

        FODomega = 0.5*x*((PhiDomega - PhiODomega)/(-((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) - np.conj(SigmaODomega))) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) + (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) - np.conj(SigmaODomega)) - (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldFODomega

        DODomega = x*(-1.*(J- PiODomega)/(detD)) + (1-x)*oldDODomega


        GDtau = Freq2TimeF(GDomega,Nbig,beta)
        DDtau = Freq2TimeB(DDomega,Nbig,beta)
        FDtau = Freq2TimeF(FDomega,Nbig,beta)
        GODtau = Freq2TimeF(GODomega,Nbig,beta)
        DODtau = Freq2TimeB(DODomega,Nbig,beta)
        FODtau = Freq2TimeF(FODomega,Nbig,beta)


        diffGD = np.sum((np.abs(GDtau-oldGDtau))**2)#changed
        diffDD = np.sum((np.abs(DDtau-oldDDtau))**2)
        diffFD = np.sum((np.abs(FDtau-oldFDtau))**2)

        diff = 0.33*(diffGD+diffDD+diffFD)

    if DUMP == True and beta % 10 == 0 :
        savefile = 'MET'
        savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        savefile += 'g' + str(g) + 'r' + str(r)
        savefile += 'lamb' + f'{lamb:.3}'
        savefile = savefile.replace('.','_') 
        savefile += '.npy'
        np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,DDtau,FDtau,GODtau,DODtau,FODtau])) 
        print(savefile)
    print("##### Finished beta = ", beta, "############")
    print(f"FD(tau = 0+) = {FDtau[0]:.4}")
    print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n')
    beta = beta + beta_step



beta = beta-beta_step

################## PLOTTING ######################
if PLOTTING == False:
    print("Simulation Finished, exiting Without PLOTTING ........")
    exit(0)




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

