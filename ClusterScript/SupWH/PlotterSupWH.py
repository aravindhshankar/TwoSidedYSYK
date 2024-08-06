import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

# path_to_dump = '../Dump/SupCondWHImagDumpfiles'
path_to_dump = '../Dump/l_05Sup/'

if not os.path.exists(path_to_dump):
    print("Error - Path to Dump directory not found")
    print("expected path: ", path_to_dump)
    # print("Creating Dump directory : ", path_to_dump)
    #os.makedirs(path_to_dump)
    raise Exception("Error - Path to Dump directory not found ")


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time


DUMP = False
PLOTTING = True

# Nbig = int(2**14)
Nbig = int(2**14)
err = 1e-12
#err = 1e-2
ITERMAX = 15000

global beta

mu = 0.0
g = 0.5
r = 1.
alpha = 0.
lamb = 0.05
# lamb = 0.01
#J = 0.0
J = 0

# betalist = [25,42,54,80,99]
betalist = [25,42,54,80,99]

kappa = 1.






############## CREATING FIGS ######################################
figLL,(ax1,ax2,ax3) = plt.subplots(1,3)
#fig.set_figwidth(10)
# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# figLL.suptitle(titlestring)
figLL.tight_layout(pad=2)

fig, ax = plt.subplots(3)

# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# fig.suptitle(titlestring)
fig.tight_layout(pad=2)

figSL,axSL = plt.subplots(2,2)
#figSL.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# figSL.suptitle(titlestring)
figSL.tight_layout(pad=2)




############### EVENT LOOP STARTS ##############################


for i, beta in enumerate(betalist): 
    col = 'C'+str(i)
    lab = f'beta = {beta}'
    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')

    Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
    Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
    delta = 0.420374134464041
    omegar2 = ret_omegar2(g,beta)

    ################# LOADING STEP ##########################
    # savefile = 'MET'
    # savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    # savefile += 'g' + str(g) + 'r' + str(r)
    # savefile += 'lamb' + f'{lamb:.3}'
    # savefile = savefile.replace('.','_') 
    # savefile += '.npy'


    savefile = 'SUP'
    savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    savefile += 'g' + str(g) + 'r' + str(r)
    savefile += 'lamb' + f'{lamb:.3}'
    # savefile += 'alpha' + f'{alpha:.2}'
    savefile = savefile.replace('.','_') 
    savefile += '.npy'

    try:
        # GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
        GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
    except FileNotFoundError:
        print(savefile, " not found")
        exit(1)

    ##########################################################

    assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

    GDomega = Time2FreqF(GDtau,Nbig,beta)
    FDomega = Time2FreqF(FDtau,Nbig,beta)
    GODomega = Time2FreqF(GODtau,Nbig,beta)
    FODomega = Time2FreqF(FODtau,Nbig,beta)
    DDomega = Time2FreqB(DDtau,Nbig,beta)
    DODomega = Time2FreqB(DODtau,Nbig,beta)










    # print(beta), print(tau[-1])
    Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
    Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
    FreeDtau = DfreeImagtau(tau,r,beta)

    ax[0].plot(tau/beta, np.real(GDtau), c = col, label = lab)
    ax[0].plot(tau/beta, np.real(Gconftau), c = col, ls='--' )
    ax[0].set_ylim(-1,1)
    ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[0].set_ylabel(r'$\Re{G(\tau)}$')
    ax[0].legend()

    ax[1].plot(tau/beta, np.real(DDtau), c=col, label = lab)
    ax[1].plot(tau/beta, np.real(Dconftau), c=col, ls='--' )
    # ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #ax[1].set_ylim(0,1)
    ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[1].set_ylabel(r'$\Re{D(\tau)}$')
    ax[1].legend()

    # ax[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
    # ax[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
    ax[2].plot(tau/beta, (np.abs(FDtau)), c=col, label = lab)
    #ax[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #ax[2].set_ylim(-1,1)
    ax[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # ax[2].set_ylabel(r'$\Re{F(\tau)}$')
    ax[2].set_ylabel(r'$|F(\tau)|$')
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



    fitslice = slice(start+0, start + 15)
    #fitslice = slice(start+25, start + 35)
    functoplot = -np.imag(GDomega)*(g**2)
    m,c = np.polyfit(np.log(np.abs(omega[fitslice])/(g**2)), np.log(functoplot[fitslice]),1)
    print(f'slope of fit = {m:.03f}')
    #print('2 Delta - 1 = ', 2*delta-1)

    ax1.loglog(omega[start:stop]/(g**2), -np.imag(GDomega[start:stop])*(g**2),'p',c=col,label = lab)
    ax1.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--')
    # ax1.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
    #ax1.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
    #ax1.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
    #ax1.set_xlim(omega[start]/2,omega[start+15])
    ax1.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, c=col, label=f'Fit with slope {m:.03f}')
    #ax1.set_ylim(1e-1,1e1)
    ax1.set_xlabel(r'$\omega_n/g^2$')
    ax1.set_ylabel(r'$-g^2\,\Im{GD(\omega_n)}$')
    #ax1.set_aspect('equal', adjustable='box')
    #ax1.axis('square')
    ax1.legend()


    ax2.loglog(nu[startB:stopB]/(g**2), np.real(DDomega[startB:stopB])*(g**2),'p',c=col,label=lab)
    ax2.loglog(nu[startB:stopB]/(g**2), conf_fit_D,'k--')
    # ax2.loglog(nu[startB:stopB]/(g**2), conf_fit_D,'k--',label = 'ES power law')
    #ax2.loglog(nu[startB:]/(g**2), np.real(Dconf[startB:]),'m.',label = 'ES solution')
    #ax2.loglog(nu[startB:]/(g**2), alt_conf_fit_D,'g--', label = 'alt power law')
    #ax2.set_xlim(nu[startB]/2,nu[startB+15])
    #ax2.set_ylim(5e-1,100)
    ax2.set_xlabel(r'$\nu_n/g^2$')
    ax2.set_ylabel(r'$g^2\,\Re{D(\nu_n)}$',labelpad = None)
    #ax2.set_aspect('equal', adjustable='box')
    ax2.legend()


    # ax3.loglog(omega[start:stop]/(g**2), np.abs(np.imag(FDomega[start:stop])*(g**2)),'p',label = 'numerics imag Fomega')
    # ax3.loglog(omega[start:stop]/(g**2), np.abs(np.real(FDomega[start:stop])*(g**2)),'p',label = 'numerics real Fomega')
    ax3.loglog(omega[start:stop]/(g**2), (np.abs(FDomega[start:stop]))*(g**2),'p',c=col,label = lab)
    #ax3.loglog(omega[start:stop]/(g**2), conf_fit_G[start:stop],'k--',label = 'ES power law')
    #ax3.loglog(omega[start:]/(g**2), -np.imag(Gconf[start:])*(g**2),'m.',label = 'ES solution')
    #ax3.loglog(omega[start:]/(g**2), alt_conf_fit_G[start:],'g--', label = 'alt power law')
    #ax3.set_xlim(omega[start]/2,omega[start+15])
    #ax3.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
    #ax3.set_ylim(1e-1,1e1)
    ax3.set_xlabel(r'$\omega_n/g^2$')
    # ax3.set_ylabel(r'$-g^2\,-\Im{F(\omega_n)}$')
    ax3.set_ylabel(r'$-g^2\,|F(\omega_n)|$')
    #ax3.set_aspect('equal', adjustable='box')
    #ax3.axis('square')
    ax3.legend()


    ###################### Log-Linear Plot ###############################



    startT, stopT  = 0, 5000

    fitsliceT = slice(startT+4500, startT + 4600)
    #fitslice = slice(start+25, start + 35)
    functoplotT = np.abs(np.real(GDtau))
    mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
    print(f'tau/beta at start of fit = {(tau[fitsliceT][0]/beta):.3f}')
    print(f'slope of fit = {mT:.03f}')
    # print('2 Delta  = ', 2*delta)

    axSL[0,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GDtau[startT:stopT])),'p',c=col,label = lab)
    #axSL[0,0].semilogy(tau[startT:stopT], conf_fit_GD[startT:stopT],'k--',label = 'ES power law')
    #axSL[0,0].semilogy(tau[startT:], -np.imag(Gconf[startT:]),'m.',label = 'ES solution')
    #axSL[0,0].semilogy(tau[startT:], alt_conf_fit_G[startT:],'g--', label = 'alt power law')
    #axSL[0,0].set_xlim(tau[startT]/2,tau[startT+15])
    axSL[0,0].semilogy(tau[startT:stopT]/beta, np.exp(mT*tau[startT:stopT] + cT), c=col,label=f'Fit with slope {mT:.03f}')
    #axSL[0,0].set_ylim(1e-1,1e1)
    axSL[0,0].set_xlabel(r'$\tau/\beta$')
    axSL[0,0].set_ylabel(r'$-\Re G(\tau)$')
    #axSL[0,0].set_aspect('equal', adjustable='box')
    #axSL[0,0].axis('square')
    axSL[0,0].legend()
    axSL[0,0].set_yscale('log')


    axSL[1,0].semilogy(tau[startT:stopT], np.abs(np.real(DDtau[startT:stopT])),'p',c=col,label=lab)
    #axSL[1,0].semilogy(tau[startB:stopB], conf_fit_DD,'k--',label = 'ES power law')
    #axSL[1,0].semilogy(tau[startB:], np.real(Dconf[startB:]),'m.',label = 'ES solution')
    #axSL[1,0].semilogy(tau[startB:], alt_conf_fit_D,'g--', label = 'alt power law')
    #axSL[1,0].set_xlim(tau[startB]/2,tau[startB+15])
    #axSL[1,0].set_ylim(5e-1,100)
    axSL[1,0].set_xlabel(r'$\tau$')
    axSL[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
    #axSL[1,0].set_aspect('equal', adjustable='box')
    axSL[1,0].legend()



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
    # # titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
    # # ax.set_title(titlestring)
    # BCSphaseangle = np.arctan(BCSgap[Nbig//2].imag / BCSgap[Nbig//2].real) 
    # BCSphaseangle = BCSphaseangle * 180 / np.pi
    # print(f"BCS gap phase angle is = {BCSphaseangle:.4f} degrees")


#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


