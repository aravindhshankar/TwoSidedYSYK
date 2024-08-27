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

path_to_dump = '../Dump/RTOneSideSup/'
if not os.path.exists(path_to_dump):
    print("ERROR dump directory does not exist", path_to_dump)
    exit(1)


betalist = [15,42,99]


M = int(2**16) #number of points in the grid
T = 2**12 #upper cut-off for the time
err = 1e-5
#err = 1e-2

omega,t  = RealGridMaker(M,T)
dw = omega[2] - omega[1]
dt = t[2] - t[1]

print("dw = ", dw)
print("dt = ", dt)

delta = 0.420374134464041
ITERMAX = 5000
global beta

mu = 0.0
g = 0.5
r = 1.
kappa = 1.
eta = dw*2.1

figT, axT = plt.subplots(2)
fig, ax = plt.subplots(3)
# titlestring = r'$\beta$ = ' + str(beta) + r', temp/dw = ' + f'{(temp/dw):.2f}' + r', $g = $' + str(g)
# fig.suptitle(titlestring)


for i, beta in enumerate(betalist):
    try:
        savefile = 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
        savefile += 'beta' + str((round(beta*100))/100.) 
        savefile += 'g' + str(g) + 'r' + str(r) 
        savefile = savefile.replace('.','_') 
        savefile +=  '.npy' 
        GRomega,DRomega,FRomega = np.load(os.path.join(path_to_dump,savefile)) 
    except FileNotFoundError:
        print("**********PATH TO INPUT FILE NOT FOUND!***************")
        exit(1)
    #assert len(Gtau) == Nbig, 'Improperly loaded starting guess'

    col = 'C' + str(i)
    lab = r'$\beta$ = ' + str(beta)
    GRt = (0.5/np.pi) * freq2time(GRomega,M,dt)
    DRt = (0.5/np.pi) * freq2time(DRomega,M,dt)

    temp = 1./beta
    Tstar = g**2 * np.sqrt(r)

    ################## PLOTTING ######################
    #np.save('beta10kN14g0_5r1x0_01.npy', np.array([Gtau,Dtau])) 
    axT[0].plot(2*np.pi*t/beta, np.real(GRt), c=col,label=lab)
    axT[0].plot(2*np.pi*t/beta, np.imag(GRt), c=col,ls='--')
    #axT[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #axT[0].set_ylim(-1,1)
    axT[0].set_xlim(-10,100)
    axT[0].set_xlabel(r'$2\pi t/\beta$',labelpad = 0)
    axT[0].set_ylabel(r'${G^R(t)}$')
    axT[0].legend()

    axT[1].plot(2*np.pi*t/beta, np.real(DRt),c=col,label=lab)
    axT[1].plot(2*np.pi*t/beta, np.imag(DRt), c=col,ls='--')
    #axT[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
    #axT[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #axT[1].set_ylim(0,1)
    axT[1].set_xlabel(r'$2\pi t/\beta$',labelpad = 0)
    axT[1].set_ylabel(r'${D^R(t)}$')
    #axT[1].set_xlim(0,beta/(2*np.pi))
    axT[1].set_xlim(-10,10)
    axT[1].legend()


    #fig.suptitle(r'$\beta$ = ', beta)
    #plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')
    #plt.show()



    ############### Spectral functions plot ####################
    rhoG, rhoD, rhoF = -1.0*np.imag(GRomega), -1.0*np.imag(DRomega), -1.0*np.imag(FRomega)
    #rhoG, rhoD = -1.0*np.imag(GRomega), 1.0*np.imag(DRomega)
    print(f'minimum of rhoG = {min(rhoG)}')
    omegar2 = ret_omegar2(g,beta)

    match_omega = 0.5
    match_point = M + int(np.floor(match_omega/dw))
    om_th = np.sqrt(omegar2)
    #om_th = 1/beta
    #match_coeff = rhoD[match_point]*(np.abs(omega[match_point])**(4*delta-1))
    #match_rhoD = match_coeff * np.abs(omega)**(1-4*delta)
    match_coeff = rhoD[match_point]*(np.abs(omega[match_point] - om_th + 1j*eta )**(4*delta-1))
    match_rhoD = match_coeff * np.abs(omega-om_th)**(1-4*delta)

    ax[0].plot(omega, rhoG, c=col, label = lab)
    # ax[0].plot(omega, np.real(GRomega),ls='--',c=col)
    #ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #ax[0].set_ylim(-1,1)
    ax[0].set_xlim(-5,5)
    ax[0].set_xlabel(r'$\omega$',labelpad = 0)
    ax[0].set_ylabel(r'$-\Im{G^R(\omega)}$')
    ax[0].legend()

    ax[1].plot(omega, rhoD, c=col, label = lab)
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

    ax[2].plot(omega,rhoF, c=col, label = lab)
    ax[2].set_xlabel(r'$\omega$')
    ax[2].set_ylabel(r'$\rho_F(\omega)$')






#plt.savefig('KoenraadEmails/PowerLawRealTimeConvBosonWithMR.pdf')
#plt.savefig('KoenraadEmails/PowerLawRealTimeConvBosonWithoutMR.pdf')dEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


