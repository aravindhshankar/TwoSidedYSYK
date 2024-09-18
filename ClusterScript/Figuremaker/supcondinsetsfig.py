import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

# path_to_dump = '../Dump/SupCondWHImagDumpfiles'
path_to_dump = '../Dump/l_05Sup/'
path_to_dump = '../Dump/l_05SupHIGH/'
# path_to_dump = '../Dump/l_05Supalpha0_1/'
# path_to_dump = '../Dump/l1Sup/'
# path_to_dump = '../Dump/lambannealSup'

path_to_oneside = '../Dump/OnesideInclSup'
path_to_metal = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd'

if not os.path.exists(path_to_oneside):
    print(f'path to onside {path_to_oneside} not found!')
    exit(1)



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
from Insethelpers import add_subplot_axes


plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.titlesize'] = '10'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams['axes.labelsize'] = '10'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
plt.rcParams['lines.markersize'] = '2'
plt.rcParams['lines.linewidth'] = '0.5'
plt.rcParams['axes.formatter.limits'] = '-2,2'

# plt.rcParams['figure.figsize'] = '8,7'

DUMP = False
PLOTTING = True

# Nbig = int(2**16)
Nbig = int(2**14)
err = 1e-12
#err = 1e-2
ITERMAX = 15000

global beta

mu = 0.0
g = 0.5
r = 1.
alpha = 0.
# alpha = 0.1
# lamb = 0.002
lamb = 0.05
# lamb = 1.0
#J = 0.0
J = 0

# betalist = [25,42,54,80,99]
betalist = [25,42,99]
# betalist = [25,42,54,73,80,99]
# betalist = [20,25,31,42,54,73,80,84,99]
# betalist = [25,50,80,190]
# betalist = [2000,]

kappa = 1.

############## CREATING FIGS ######################################
fig, ax = plt.subplots(1,3)
fig.set_figwidth(3.25*2)
# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# axinset = fig.add_axes([left, bottom, width, height])
rect = [0.2,0.2,0.7,0.7]
rect = [0.2,0.6,0.3,0.3]
# axinset0 = add_subplot_axes(ax[0],[0.1,0.1,0.2,0.2])
# axinset1 = add_subplot_axes(ax[1],[0,0,0.2,0.2])
# axinset2 = add_subplot_axes(ax[2],[0.6,0.1,0.2,0.2])
# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# fig.suptitle(titlestring)
fig.tight_layout(pad=2.5)



figOD, axOD = plt.subplots(1,3)
# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# axinset = fig.add_axes([left, bottom, width, height])
rect = [0.2,0.2,0.7,0.7]
rect = [0.2,0.6,0.3,0.3]
# axinset0 = add_subplot_axes(axOD[0],[0,0.1,0.2,0.2])
# axinset1 = add_subplot_axes(axOD[1],[0,0,0.2,0.2])
# axinset2 = add_subplot_axes(axOD[2],[0.6,0.1,0.2,0.2])
# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# fig.suptitle(titlestring)
fig.tight_layout(pad=2.5)


figmet, axmet = plt.subplots(2,2)


figdiff, axdiff = plt.subplots(1,3)

############### EVENT LOOP STARTS ##############################
for i, beta in enumerate(betalist): 
    col = 'C'+str(i)
    lab = r'$\beta = $' + f'{beta} ('
    if beta > 32:
        lab += 'SC'
    if beta > 62: 
        lab += 'WH'
    else:
        lab += 'BH'
    lab += ')'
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
    # savefile += 'g' + str(g) + 'r' + st$r(r)
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

    onesidefile = 'OnesideSUP'
    onesidefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
    onesidefile += 'g' + str(g) + 'r' + str(r)
    onesidefile = onesidefile.replace('.','_') 
    onesidefile += '.npy'

    try:
        # GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
        GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
    except FileNotFoundError:
        print(savefile, " not found")
        exit(1)


    try:
        # GDtau,DDtau,FDtau,GODtau,DODtau,FODtau = np.load(os.path.join(path_to_dump, savefile)) 
        Gtau,Dtau,Ftau= np.load(os.path.join(path_to_oneside, onesidefile)) 
    except FileNotFoundError:
        print(onesidefile, " not found")
        exit(1)
    ##########################################################

    assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

    GDomega = Time2FreqF(GDtau,Nbig,beta)
    FDomega = Time2FreqF(FDtau,Nbig,beta)
    GODomega = Time2FreqF(GODtau,Nbig,beta)
    FODomega = Time2FreqF(FODtau,Nbig,beta)
    DDomega = Time2FreqB(DDtau,Nbig,beta)
    DODomega = Time2FreqB(DODtau,Nbig,beta)


    skip = 10
    startT, stopT  = 0, Nbig//2 - 1
    llplotslice = slice(startT,stopT,skip)
    Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
    Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
    FreeDtau = DfreeImagtau(tau,r,beta)


    ################## NON SUPERCONDUCTING STATE #################
    try :
        metfile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        metfile += 'lamb' + str(lamb) + 'J' + str(J)
        metfile += 'g' + str(g) + 'r' + str(r)
        metfile = metfile.replace('.','_') 
        metfile += '.npy'
        metGDtau, metGODtau, metDDtau, metDODtau = np.load(os.path.join(path_to_metal, metfile))
        metGDtau = -metGODtau
    except FileNotFoundError: 
        print('Filename : ', savefile)
        print("INPUT FILE NOT FOUND") 
        exit(1)
    axmet[0,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(metGDtau[llplotslice])),'.',c = col, label = lab)
    # axmet[0,0].semilogy(tau[llplotslice]/beta, np.exp(mT*tau[llplotslice] + cT), label=f'Fit with slope {mT:.03f}')
    axmet[0,0].set_xlabel(r'$\tau/\beta$')
    axmet[0,0].set_ylabel(r'$-\Re G_{d}(\tau)$')
    axmet[0,0].set_yscale('log')


    axmet[0,1].semilogy(tau[llplotslice]/beta, np.abs(np.real(metGDtau[llplotslice])),'.',c = col, label = lab)
    axmet[0,1].set_xlabel(r'$\tau/\beta$')
    axmet[0,1].set_ylabel(r'$-\Re G_{od}(\tau)$')
    axmet[0,1].set_yscale('log')

    axmet[1,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(metDDtau[llplotslice])),'.',c=col,label=lab)
    axmet[1,0].set_xlabel(r'$\tau/\beta$')
    axmet[1,0].set_ylabel(r'$g^2\,\Re{D_{d}(\nu_n)}$',labelpad = None)

    axmet[1,1].semilogy(tau[llplotslice]/beta, np.abs(np.real(metDODtau[llplotslice])),'.',c=col,label=lab)
    axmet[1,1].set_xlabel(r'$\tau/\beta$')
    axmet[1,1].set_ylabel(r'$g^2\,\Re{D_{od}(\nu_n)}$',labelpad = None)

    #################### Superconducting state diagonals ###########
    ax[0].semilogy(tau[llplotslice]/beta, np.abs(np.real(GDtau[llplotslice])),c=col,label=lab)
    ax[0].semilogy(tau[llplotslice]/beta, np.abs(np.real(Gtau[llplotslice])),c=col,ls='--')
    fitslicemet1 = slice(np.argmin(np.abs(tau/beta - 0.3)),np.argmin(np.abs(tau/beta - 0.4)))
    
    if beta > 60:
        SUPm1,SUPlogc1 = np.polyfit(tau[fitslicemet1]/beta,np.log(np.abs(np.real(Gtau))[fitslicemet1]),1)
        Supc1 = np.exp(SUPlogc1)
        ax[0].semilogy(tau[llplotslice]/beta, Supc1*np.exp(SUPm1*tau[llplotslice]/beta),c='k',label = f'Gap $\\Delta = $ {np.abs(SUPm1/beta):.4}',ls='--')
    # axinset0.plot(tau[llplotslice]/beta, np.real(GDtau[llplotslice]), c = col, label = lab)
    # axinset0.plot(tau[llplotslice]/beta, np.real(Gtau[llplotslice]), c = col, ls='--' )
    # axinset0.plot(tau[llplotslice]/beta, np.real(Gconftau[llplotslice]), c = col, ls='--' )
    # ax[0].set_ylim(-1,1)
    ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[0].set_ylabel(r'$|\Re{G_{d}(\tau)}|$')
    # ax[0].legend(framealpha = 0.0)


    # axinset1.plot(tau[llplotslice]/beta, np.real(DDtau[llplotslice]), c=col, label = lab)
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dtau[llplotslice]), c=col, ls='--' )
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dconftau[llplotslice]), c=col, ls='--' )
    ax[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(DDtau[llplotslice])),c=col,label=lab)
    ax[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(Dtau[llplotslice])),c=col,ls='--')

    # axinset[1].semilogy(tau/beta, np.real(Dconftau), c=col, ls='--' )
    # ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #ax[1].set_ylim(0,1)
    ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[1].set_ylabel(r'$|\Re{D_{d}(\tau)}|$')
    # ax[1].legend()


    # ax[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
    # ax[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FDtau[llplotslice])), c=col, label = lab)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FODtau[llplotslice])), ls='--', c=col)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(Ftau[llplotslice])), ls='--', c=col)
    ax[2].semilogy(tau[llplotslice]/beta, np.abs(FDtau[llplotslice]),c=col,label=lab)
    ax[2].semilogy(tau[llplotslice]/beta, np.abs(Ftau[llplotslice]),c=col,ls='--')

    #ax[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #ax[2].set_ylim(-1,1)
    ax[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # ax[2].set_ylabel(r'$\Re{F(\tau)}$')
    ax[2].set_ylabel(r'$|F_{d}(\tau)|$')
    # ax[2].legend()

    # #fig.suptitle(r'$\beta$ = ', beta)
    # #plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')

    ################################# OFF DIAGONALS ##################################################

    axOD[0].semilogy(tau[llplotslice]/beta, np.abs(np.real(GODtau[llplotslice])),c=col,label=lab)
    axOD[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    axOD[0].set_ylabel(r'$|\Re{G_{od}(\tau)}|$')
    axOD[0].legend()


    # axinset1.plot(tau[llplotslice]/beta, np.real(DDtau[llplotslice]), c=col, label = lab)
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dtau[llplotslice]), c=col, ls='--' )
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dconftau[llplotslice]), c=col, ls='--' )
    axOD[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(DODtau[llplotslice])),c=col,label=lab)
    # axOD[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(Dtau[llplotslice])),c=col,ls='--')

    # axinset[1].semilogy(tau/beta, np.real(Dconftau), c=col, ls='--' )
    # axOD[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #axOD[1].set_ylim(0,1)
    axOD[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    axOD[1].set_ylabel(r'$|\Re{D_{od}(\tau)}|$')
    axOD[1].legend()


    # axOD[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
    # axOD[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FDtau[llplotslice])), c=col, label = lab)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FODtau[llplotslice])), ls='--', c=col)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(Ftau[llplotslice])), ls='--', c=col)
    axOD[2].semilogy(tau[llplotslice]/beta, np.abs(FODtau[llplotslice]),c=col,label=lab)
    # axOD[2].semilogy(tau[llplotslice]/beta, np.abs(Ftau[llplotslice]),c=col,ls='--')

    #axOD[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #axOD[2].set_ylim(-1,1)
    axOD[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # axOD[2].set_ylabel(r'$\Re{F(\tau)}$')
    axOD[2].set_ylabel(r'$|F_{od}(\tau)|$')
    axOD[2].legend()


    
   

    ##### DIFFS ########
    diffsG = np.abs(np.real(GDtau-Gtau))
    if beta > 62:
        fitslice0 = slice(np.argmin(np.abs(tau/beta - 0.3)),np.argmin(np.abs(tau/beta - 0.4)))
        m0,logc0 = np.polyfit(tau[fitslice0]/beta,np.log(diffsG[fitslice0]),1)
        c0 = np.exp(logc0)
        axdiff[0].semilogy(tau[llplotslice]/beta, c0*np.exp(m0*tau[llplotslice]/beta),c=col,label = f'fit with slope {m0/beta:.4}',ls='--')

        metm0,metlogc0 = np.polyfit(tau[fitslice0]/beta,np.log(np.abs(np.real(metGDtau))[fitslice0]),1)
        metc0 = np.exp(metlogc0)
        axmet[0,0].semilogy(tau[llplotslice]/beta, metc0*np.exp(metm0*tau[llplotslice]/beta),c=col,label = f'fit with slope {metm0/beta:.4}',ls='--')

    # axdiff[0].semilogy(tau[llplotslice]/beta, np.abs(np.real(Gtau[llplotslice])),c=col,ls='--')
    # axinset0.plot(tau[llplotslice]/beta, np.real(GDtau[llplotslice]), c = col, label = lab)
    # axinset0.plot(tau[llplotslice]/beta, np.real(Gtau[llplotslice]), c = col, ls='--' )
    # axinset0.plot(tau[llplotslice]/beta, np.real(Gconftau[llplotslice]), c = col, ls='--' )
    axdiff[0].semilogy(tau[llplotslice]/beta, diffsG[llplotslice],c=col,label=lab)
    # axdiff[0].set_ylim(-1,1)
    axdiff[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    axdiff[0].set_ylabel(r'$|\Re{G_{d}(\tau)}|$')
    axdiff[0].legend(framealpha = 0.0)

    # axinset1.plot(tau[llplotslice]/beta, np.real(DDtau[llplotslice]-Dtau[llplotslice]), c=col, label = lab)
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dtau[llplotslice]), c=col, ls='--' )
    # axinset1.plot(tau[llplotslice]/beta, np.real(Dconftau[llplotslice]), c=col, ls='--' )
    # axdiff[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(DDtau[llplotslice])),c=col,label=lab)
    diffsD = np.abs(np.real(DDtau-Dtau))
    if beta > 62:
        fitslice1 = slice(np.argmin(np.abs(tau/beta - 0.2)),np.argmin(np.abs(tau/beta - 0.3)))
        fitslice1 = slice(np.argmin(np.abs(tau/beta - 0.3)),np.argmin(np.abs(tau/beta - 0.35)))
        m1,logc1 = np.polyfit(tau[fitslice1]/beta,np.log(diffsD[fitslice1]),1)
        c1 = np.exp(logc1)
        axdiff[1].semilogy(tau[llplotslice]/beta, c1*np.exp(m1*tau[llplotslice]/beta),c=col,label=f'fit with slope {m1/beta:.4}',ls='--')

        metm1,metlogc1 = np.polyfit(tau[fitslice1]/beta,np.log(np.abs(np.real(metDDtau))[fitslice1]),1)
        metc1 = np.exp(metlogc1)
        axmet[1,0].semilogy(tau[llplotslice]/beta, metc1*np.exp(metm1*tau[llplotslice]/beta),c=col,label = f'fit with slope {metm1/beta:.4}',ls='--')

    axdiff[1].semilogy(tau[llplotslice]/beta, diffsD[llplotslice],c=col,label=lab)

    # axinset[1].semilogy(tau/beta, np.real(Dconftau), c=col, ls='--' )
    # axdiff[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #axdiff[1].set_ylim(0,1)
    axdiff[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    axdiff[1].set_ylabel(r'$|\Re{D_{d}(\tau)}|$')
    axdiff[1].legend(framealpha=0.0)


    # axdiff[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
    # axdiff[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FDtau[llplotslice]-Ftau[llplotslice])), c=col, label = lab)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(FODtau[llplotslice])), ls='--', c=col)
    # axinset2.plot(tau[llplotslice]/beta, (np.abs(Ftau[llplotslice])), ls='--', c=col)
    diffsF = np.abs(FDtau-Ftau)
    if beta > 62:
        fitslice2 = slice(np.argmin(np.abs(tau/beta - 0.3)),np.argmin(np.abs(tau/beta - 0.4)))
        m2,logc2 = np.polyfit(tau[fitslice2]/beta,np.log(diffsF[fitslice2]),1)
        c2 = np.exp(logc2)
        axdiff[2].semilogy(tau[llplotslice]/beta, c2*np.exp(m2*tau[llplotslice]/beta),c=col,label=f'fit with slope {m2/beta:.4}',ls='--')

    axdiff[2].semilogy(tau[llplotslice]/beta, np.abs(FDtau[llplotslice])-np.abs(Ftau[llplotslice]),c=col,label=lab)
    # axdiff[2].semilogy(tau[llplotslice]/beta, np.abs(Ftau[llplotslice]),c=col,ls='--')
    #axdiff[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #axdiff[2].set_ylim(-1,1)
    axdiff[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # axdiff[2].set_ylabel(r'$\Re{F(\tau)}$')
    axdiff[2].set_ylabel(r'$|F_{d}(\tau)|$')
    axdiff[2].legend(framealpha=0.0)

    axmet[1,1].legend()
    axmet[1,0].legend()
    axmet[0,0].legend()
    axmet[0,1].legend()






handles, labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5,-0.25),frameon=True,fancybox=True,borderaxespad=2, bbox_transform=ax[1].transAxes)
fig.suptitle(r"Superconducting Green's functions for $\lambda=0.05$")
fig.savefig('SupCondFigs.pdf', bbox_inches='tight')







    # figFE.savefig('../../KoenraadEmails/FreeEnergyOscillationSUP.pdf', bbox_inches = 'tight')
    # figFE.savefig('../../KoenraadEmails/JosephsonCurrent.pdf', bbox_inches = 'tight')

# fig.savefig('insetsSupGFs.pdf',bbox_inches='tight')

#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
# plt.show()


