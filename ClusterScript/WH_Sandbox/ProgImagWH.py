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


DUMP = True
print("DUMP = ", DUMP)

Nbig = int(2**14)
err = 1e-5
#err = 1e-2
ITERMAX = 500

global beta

beta_start = 1.
beta = beta_start
mu = 0.0
g = 0.5
r = 1.

lamb = 0.05
J = 0.05

target_beta = 10001

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
omegar2 = ret_omegar2(g,beta)

GDtau, GODtau = Gfreetau, np.zeros_like(Gfreetau)
DDtau, DODtau = Dfreetau, np.zeros_like(Dfreetau)

#Gtau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
#Dtau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)

# Gtau = -0.5*np.ones(Nbig)
# Dtau = 1.0*np.ones(Nbig)

#Gtau,Dtau = np.load('temp.npy')
assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

while(beta < target_beta):
    # Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
    # Gfreebetaplus = Freq2TimeF(1./(1j*omega + mu),Nbig,beta-beta_step)
    # err  = 0.1*np.sum(np.abs(Gfreetau - Gfreebetaplus)**2)
    itern = 0
    diff = err*1.1
    diffG = 1.
    diffD = 1.
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

        oldGDtau, oldGODtau = 1.0*GDtau, 1.0*GODtau
        oldDDtau, oldDODtau = 1.0*DDtau, 1.0*DODtau
        
        if iterni == 1:
            oldGDomega,oldGODomega = Time2FreqF(oldGDtau,Nbig,beta),Time2FreqF(oldGODtau,Nbig,beta)
            oldDDomega, oldDODomega = Time2FreqB(oldDDtau,Nbig,beta),Time2FreqB(oldDODtau,Nbig,beta)
        else:
            oldGDomega, oldGODomega = 1.0*GDomega, 1.0*GODomega
            oldDDomega, oldDODomega = 1.0*DDomega, 1.0*DODomega
        
        SigmaDtau = 1.0 * kappa * (g**2) * DDtau * GDtau
        SigmaODtau = 1.0 * kappa * (g**2) * DODtau * GODtau
        PiDtau = 2.0 * g**2 * GDtau * GDtau[::-1] #KMS G(-tau) = -G(beta-tau)
        PiODtau = 2.0 * g**2 * GODtau * GODtau[::-1] #KMS G(-tau) = -G(beta-tau)
        
        SigmaDomega, SigmaODomega = Time2FreqF(SigmaDtau,Nbig,beta),Time2FreqF(SigmaODtau,Nbig,beta)
        PiDomega, PiODomega =  Time2FreqB(PiDtau,Nbig,beta), Time2FreqB(PiODtau,Nbig,beta)
        # if itern < 15 : 
        #     Piomega[Nbig//2] = 1.0*r - omegar2
        #Piomega[Nbig//2] = 1.0*r - omegar2
        
        detG = (1j*omega+mu-SigmaDomega)**2 - (lamb - SigmaODomega)**2
        detD = (nu**2 + r - PiDomega)**2 - (J - PiODomega)**2
        GDomega = x*((1j*omega + mu - SigmaDomega)/(detG)) + (1-x)*oldGDomega
        GODomega = x*(-1.*(lamb- SigmaODomega)/(detG)) + (1-x)*oldGODomega
        DDomega = x*((nu**2 + r - PiDomega)/(detD)) + (1-x)*oldDDomega
        DODomega = x*(-1.*(J- PiODomega)/(detD)) + (1-x)*oldDODomega

        GDtau = Freq2TimeF(GDomega,Nbig,beta)
        GODtau = Freq2TimeF(GODomega,Nbig,beta)
        DDtau = Freq2TimeB(DDomega,Nbig,beta)
        DODtau = Freq2TimeB(DODomega,Nbig,beta)

        
        if iterni>0:
            # diffG = np.sqrt(np.sum((np.abs(Gtau-oldGtau))**2)) #changed
            # diffD = np.sqrt(np.sum((np.abs(Dtau-oldDtau))**2))
            diffGD = np.sum((np.abs(GDtau-oldGDtau))**2)#changed
            #diffD = np.sum((np.abs(Dtau-oldDtau))**2)
            #diff = np.max([diffG,diffD])
            #diff = 0.5*(diffG+diffD)
            diff = diffGD
            #diffG, diffD = diff, diff
            
            # if diffG<diffoldG and xG < 1./num:
            #     xG *= num
            # if diffD<diffoldD and xD < 1./num:
            #     xD *= num
            # if diffG>diffoldG and xG > num * err :
            #     xG /= num
            # if diffD>diffoldD and xD > num * err :
            #     xD /= num


            #print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD)
    if DUMP == True and beta in [50,100,500,1000,2000,5000,10000]:
        savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        savefile += 'lamb' + str(lamb) + 'J' + str(J)
        savefile += 'g' + str(g) + 'r' + str(r)
        savefile = savefile.replace('.','_') 
        savefile += '.npy'
        np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,GODtau,DDtau,DODtau])) 
        print(savefile)
    print("##### Finished beta = ", beta, "############")
    #print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n')
    print("diff = ", diff,' , itern = ',itern)
    beta = beta + beta_step

################## PLOTTING ######################
#np.save('beta10kN14g0_5r1x0_01.npy', np.array([Gtau,Dtau])) 
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


# ax[1,0].semilogy(tau[startB:stopB], np.real(DDomega[startB:stopB]),'p',label='numerics')
# #ax[1,0].semilogy(tau[startB:stopB], conf_fit_DD,'k--',label = 'ES power law')
# #ax[1,0].semilogy(tau[startB:], np.real(Dconf[startB:]),'m.',label = 'ES solution')
# #ax[1,0].semilogy(tau[startB:], alt_conf_fit_D,'g--', label = 'alt power law')
# #ax[1,0].set_xlim(tau[startB]/2,tau[startB+15])
# #ax[1,0].set_ylim(5e-1,100)
# ax[1,0].set_xlabel(r'$\nu_n/g^2$')
# ax[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
# #ax[1,0].set_aspect('equal', adjustable='box')
# ax[1,0].legend()





#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()

