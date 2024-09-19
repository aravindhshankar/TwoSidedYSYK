import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


if not os.path.exists('../Dump'):
    print("error- Path to Dump directory not found ")
    exit(1)

path_to_dump = '../Dump/OneSideYSYKg2/'
if not os.path.exists(path_to_dump):
    print('Creating Target directory ', path_to_dump)
    os.makedirs(path_to_dump)

from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


Nbig = int(2**14)
x = 0.001
# err = 1e-6
err = 1e-3 * (x**2)
# beta = 100
mu = 0.0
g = 2
#r = 0.0 + 1e-8 * 1j
r = 1.
#r = g**2 * beta/2 + 0.001
#r = 1e-8
kappa = 1.

def Single_YSYK_anneal_temp(betalist,GFtaus,Nbig,g,r,mu,kappa,x=[0.001,0.001],err = err, DUMP=False,path_to_dump=None,savelist=None,calcfe=False,verbose=True):
    Gtau, Dtau = GFtaus
    xG, xD = x
    for beta in betalist:
        omega = ImagGridMaker(Nbig,beta,'fermion')
        nu = ImagGridMaker(Nbig,beta,'boson')
        # tau = ImagGridMaker(Nbig,beta,'tau')
        itern = 0
        diff = 1.
        diffG = 1.
        diffD = 1.
        assert len(Gtau) == Nbig, 'Improperly loaded starting guess'
        print(f"Started beta = {beta}") if verbose == True else None
        while(diff>err):
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
            
            
            Gomega = xG*(1./(1j*omega + mu - Sigmaomega)) + (1-xG)*oldGomega
            Domega = xD*(1./((nu**2 + r) - Piomega)) + (1-xD)*oldDomega

        
            #Gtau = Freq2TimeF(Gomega - (1./(1j*omega)),Nbig,beta) - 0.5
            #Dtau = Freq2TimeB(Domega - (1./(nu**2+r)),Nbig,beta) + DfreeImagtau(tau,r,beta)
            Gtau = Freq2TimeF(Gomega,Nbig,beta)
            Dtau = Freq2TimeB(Domega,Nbig,beta)
            
            diffG = np.sum((np.abs(Gtau-oldGtau))**2) #changed
            diffD = np.sum((np.abs(Dtau-oldDtau))**2)
            # diff = np.max([diffG,diffD])
            diff = 0.5 * (diffG + diffD)
            
            # if diffG>diffoldG:
            #     xG/=2.
            # if diffD>diffoldD:
            #     xD/=2.
            #print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD, end = '\r')
            # print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD) if verbose == True else None


        
        print(" Finished beta = ", beta, " with itern = ", itern, " , diff = ", diff, " , x = ", xG) if verbose == True else None

        if DUMP == True and np.isclose(beta,savelist).any():
            savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
            # savefile += 'lamb' + str(lamb) + 'J' + str(J)
            savefile += 'g' + str(g) + 'r' + str(r)
            savefile = savefile.replace('.','_') 
            savefile += '.npy'
            np.save(os.path.join(path_to_dump, savefile), np.array([Gtau,Dtau])) 
            print(savefile) if verbose == True else None
    return [Gtau,Dtau]


###################### CALCULATION ###########################

# Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
# Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
# # Dfreetau = Freq2TimeB(-1./(nu**2 + r),Nbig,beta)
# delta = 0.420374134464041
# omegar2 = ret_omegar2(g,beta)
# Gtau_powerlaw = -1.0*np.sign(tau)*(np.pi/np.abs(beta*np.sin(np.pi*tau/beta))) ** (2*delta)
# Dtau_powerlaw =  1.0*(np.pi/np.abs(beta*np.sin(np.pi*tau/beta))) ** (2 - 4*delta)

# Gtau = Gfreetau
# Dtau = Dfreetau
#Gtau = Gtau_powerlaw * (-0.5/Gtau_powerlaw[0])
#Dtau = Dtau_powerlaw * (DfreeImagtau(tau,r,beta)[0]/Dtau_powerlaw[0])
target_beta = 1000
beta_step = 1
beta_start = 1


betalist = np.arange(beta_start, target_beta + beta_step, beta_step)

omega = ImagGridMaker(Nbig,beta_start,'fermion')
nu = ImagGridMaker(Nbig,beta_start,'boson')
tau = ImagGridMaker(Nbig,beta_start,'tau')

Gtau = -0.5*np.ones(Nbig)
Dtau = 1.0*np.ones(Nbig)
Gtau = Freq2TimeF(GconfImag(omega,g,beta_start),Nbig,beta_start)
Dtau = Freq2TimeB(DconfImag(nu,g,beta_start),Nbig,beta_start)
GFtaus = [Gtau,Dtau]

Gtau, Dtau = Single_YSYK_anneal_temp(betalist,GFtaus,Nbig,g,r,mu,kappa,x=[0.001,0.001],err=err,DUMP=True,path_to_dump=path_to_dump,savelist=betalist,calcfe=False,verbose=True)
beta = betalist[-1]
omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')
Gomega = Time2FreqF(Gtau,Nbig,beta)
Domega = Time2FreqB(Dtau,Nbig,beta)
################## PLOTTING ######################

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

#plt.savefig('KoenraadEmails/WithMR_imagtime.pdf',bbox_inches='tight')
# plt.show()

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

ax1.loglog(omega[start:stop]/(g**2), -np.imag(Gomega[start:stop])*(g**2),'p',label = 'numerics')
ax1.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
#ax1.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
#ax1.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
#ax1.set_xlim(omega[start]/2,omega[start+15])
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

#plt.savefig('lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('KoenraadEmails/ImagFreqpowerlaw_withMR.pdf', bbox_inches = 'tight')
plt.show()


