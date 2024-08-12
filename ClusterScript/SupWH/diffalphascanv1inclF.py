############### Actually Using v2 convention ###################


import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

# path_to_dump = '../Dump/SupCondWHImagDumpfiles'
# path_to_dump = '../Dump/LOWTEMP_lamb_anneal_dumpfiles'
path_to_loadfile = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd/'
path_to_dump = '../Dump/l_05Supalpha0_1/'
# path_to_dump = '../Dump/l_05Supalpha0_2/'

if not os.path.exists(path_to_loadfile):
    print("Error - Path to Dump directory not found")
    print("expected path: ", path_to_loadfile)
    # print("Creating Dump directory : ", path_to_dump)
    #os.makedirs(path_to_dump)
    raise Exception("Error - Path to Load directory not found ")
    exit(1)

if not os.path.exists(path_to_dump):
    print("Error - Path to Dump directory not found")
    print("expected path: ", path_to_dump)
    print("Creating Dump directory : ", path_to_dump)
    os.makedirs(path_to_dump)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


DUMP = True
PLOTTING = False

Nbig = int(2**14)
# Nbig = int(2**16)
err = 1e-12
#err = 1e-2
ITERMAX = 50000

global beta

# beta_start = 5000
beta_start = 1
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
alpha = 0.1
# lamb = 0.05
lamb = 0.05
#J = 0.0
J = 0

# target_beta = 40.
target_beta = 101

kappa = 1.
beta_step = 1

betalooplist = np.arange(beta_start,target_beta)
betasavelist = betalooplist

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


FDtau = (1+1j)*np.ones(Nbig)
FODtau = (1-1j)*np.ones(Nbig)





for beta in betalooplist:
    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')
    ################# LOADING STEP ##########################
    # savefile = 'MET'
    # savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    # savefile += 'g' + str(g) + 'r' + str(r)
    # savefile += 'lamb' + f'{lamb:.3}'
    # savefile = savefile.replace('.','_') 
    # savefile += '.npy'

    savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    savefile += 'lamb' + str(lamb) + 'J' + str(J)
    savefile += 'g' + str(g) + 'r' + str(r)
    savefile = savefile.replace('.','_') 
    savefile += '.npy'

    try:
        # GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
        GDtau,GODtau,DDtau,DODtau = np.load(os.path.join(path_to_loadfile, savefile)) 
    except FileNotFoundError:
        print(savefile, " not found")
        exit(1)

    ##########################################################

    assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

    #Include anomalous propagators
    if np.sum(np.abs(FDtau[:20])) < 1e-2:
        FDtau = (1+1j)*np.ones_like(GDtau)
        FODtau = (1-1j)*np.ones_like(GODtau)
    itern = 0
    diff = err*1.1
    # x = 0.01
    x = 0.001

    beta_step = 1 if (beta>130) else 1

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
        
        GDomega = 0.5*x*((lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) - (lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldGDomega

        FDomega = 0.5*x*((-PhiDomega + PhiODomega)/((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) - (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldFDomega

        DDomega = x*((nu**2 + r - PiDomega)/(detD)) + (1-x)*oldDDomega

        GODomega = 0.5*x*(-((lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega)))) - (lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega))/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldGODomega

        FODomega = 0.5*x*((PhiDomega - PhiODomega)/((lamb - SigmaDomega + SigmaODomega + 1j*omega)*(lamb - 1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega - PhiODomega)*(np.conj(PhiDomega) - np.conj(PhiODomega))) - (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - 1j*omega)*(lamb + 1j*omega + np.conj(SigmaDomega) + np.conj(SigmaODomega)) + (PhiDomega + PhiODomega)*(np.conj(PhiDomega) + np.conj(PhiODomega)))) + (1-x)*oldFODomega

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

    if DUMP == True and np.isclose(betasavelist,beta).any() :
        betaval = betasavelist[np.isclose(betasavelist,beta)][0]
        savefile = 'SUP'
        savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(betaval) 
        savefile += 'g' + str(g) + 'r' + str(r)
        savefile += 'lamb' + f'{lamb:.3}'
        savefile += 'alpha' + f'{alpha:.2}'
        savefile = savefile.replace('.','_') 
        savefile += '.npy'
        np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,DDtau,FDtau,GODtau,DODtau,FODtau])) 
        print(savefile)
    print("##### Finished beta = ", beta, "############")
    print(f"FD(tau = 0+) = {FDtau[0]:.4}")
    print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n')

# np.testing.assert_almost_equal(beta,target_beta)



################## PLOTTING ######################
if PLOTTING == False:
    print("Simulation Finished, exiting Without PLOTTING ........")
    exit(0)




# print(f"Simulation with inclusion of superconductivity ended with FDtau[0] = {FDtau[0]:.4}, FDomega[0] = {FDomega[Nbig//2]:.4}")
# tanph = np.imag(FDomega[Nbig//2]) / np.real(FDomega[Nbig//2])
# phaseangle = np.arctan(tanph) * 180/ np.pi
# print(f"The phase angle of each superconductor is {phaseangle:.4f} degrees")
# BCSgap = 1j*omega * FDomega/ GDomega 
# fig,ax = plt.subplots(1)
# ax.plot(omega, BCSgap.real,label='Re$\\Delta$') 
# ax.plot(omega,BCSgap.imag, label='Im$\\Delta$')
# ax.set_xlabel(r'$\omega_n$')
# ax.set_ylabel(r'$\Delta$')
# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# ax.set_title(titlestring)
# BCSphaseangle = np.arctan(BCSgap[Nbig//2].imag / BCSgap[Nbig//2].real) 
# BCSphaseangle = BCSphaseangle * 180 / np.pi
# print(f"BCS gap phase angle is = {BCSphaseangle:.4f} degrees")

print(beta), print(tau[-1])
Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
FreeDtau = DfreeImagtau(tau,r,beta)

fig, ax = plt.subplots(3)

titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)
ax[0].plot(tau/beta, np.real(GDtau), 'r', label = 'numerics Gtau')
ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
ax[0].set_ylim(-1,1)
ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0].set_ylabel(r'$\Re{G(\tau)}$')
ax[0].legend()

ax[1].plot(tau/beta, np.real(DDtau), 'r', label = 'numerics Dtau')
ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1].set_ylabel(r'$\Re{D(\tau)}$')
ax[1].legend()

ax[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
ax[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
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

fitG_val = -np.imag(GDomega[start+0])*(g**2)
#fitG_val = -np.imag(Gconf[start:stop])*(g**2)
conf_fit_G = 1 * np.abs(omega/(g**2))**(2*delta - 1)
conf_fit_G = conf_fit_G/conf_fit_G[start] * fitG_val
alt_conf_fit_G = fitG_val * np.abs(omega/(g**2))**(2*alt_delta - 1)

fitD_val = np.real(DDomega[startB])*(g**2)
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
functoplot = -np.imag(GDomega)*(g**2)
m,c = np.polyfit(np.log(np.abs(omega[fitslice])/(g**2)), np.log(functoplot[fitslice]),1)
print(f'slope of fit = {m:.03f}')
#print('2 Delta - 1 = ', 2*delta-1)

ax1.loglog(omega[start:stop]/(g**2), -np.imag(GDomega[start:stop])*(g**2),'p',label = 'numerics')
ax1.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
#ax1.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
#ax1.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
#ax1.set_xlim(omega[start]/2,omega[start+15])
ax1.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
#ax1.set_ylim(1e-1,1e1)
ax1.set_xlabel(r'$\omega_n/g^2$')
ax1.set_ylabel(r'$-g^2\,\Im{GD(\omega_n)}$')
#ax1.set_aspect('equal', adjustable='box')
#ax1.axis('square')
ax1.legend()


ax2.loglog(nu[startB:stopB]/(g**2), np.real(DDomega[startB:stopB])*(g**2),'p',label='numerics')
ax2.loglog(nu[startB:stopB]/(g**2), conf_fit_D,'k--',label = 'ES power law')
#ax2.loglog(nu[startB:]/(g**2), np.real(Dconf[startB:]),'m.',label = 'ES solution')
#ax2.loglog(nu[startB:]/(g**2), alt_conf_fit_D,'g--', label = 'alt power law')
#ax2.set_xlim(nu[startB]/2,nu[startB+15])
#ax2.set_ylim(5e-1,100)
ax2.set_xlabel(r'$\nu_n/g^2$')
ax2.set_ylabel(r'$g^2\,\Re{D(\nu_n)}$',labelpad = None)
#ax2.set_aspect('equal', adjustable='box')
ax2.legend()


ax3.loglog(omega[start:stop]/(g**2), np.abs(np.imag(FDomega[start:stop])*(g**2)),'p',label = 'numerics imag Fomega')
ax3.loglog(omega[start:stop]/(g**2), np.abs(np.real(FDomega[start:stop])*(g**2)),'p',label = 'numerics real Fomega')
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


###################### Log-Linear Plot ###############################


fig,ax = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

startT, stopT  = 0, 5000

fitsliceT = slice(startT+4500, startT + 4600)
#fitslice = slice(start+25, start + 35)
functoplotT = np.abs(np.real(GDtau))
mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
print(f'tau/beta at start of fit = {(tau[fitsliceT][0]/beta):.3f}')
print(f'slope of fit = {mT:.03f}')
# print('2 Delta  = ', 2*delta)

ax[0,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GDtau[startT:stopT])),'p',label = 'numerics GDtau')
#ax[0,0].semilogy(tau[startT:stopT], conf_fit_GD[startT:stopT],'k--',label = 'ES power law')
#ax[0,0].semilogy(tau[startT:], -np.imag(Gconf[startT:]),'m.',label = 'ES solution')
#ax[0,0].semilogy(tau[startT:], alt_conf_fit_G[startT:],'g--', label = 'alt power law')
#ax[0,0].set_xlim(tau[startT]/2,tau[startT+15])
ax[0,0].semilogy(tau[startT:stopT]/beta, np.exp(mT*tau[startT:stopT] + cT), label=f'Fit with slope {mT:.03f}')
#ax[0,0].set_ylim(1e-1,1e1)
ax[0,0].set_xlabel(r'$\tau/\beta$')
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
ax[1,0].set_xlabel(r'$\tau$')
ax[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
#ax[1,0].set_aspect('equal', adjustable='box')
ax[1,0].legend()





#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


