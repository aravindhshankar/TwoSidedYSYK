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
from matplot_fmt_pi.ticker import MultiplePi
from Insethelpers import add_subplot_axes
#import time


plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.titlesize'] = '8'
plt.rcParams['axes.titlesize'] = '8'
plt.rcParams['axes.labelsize'] = '8'
plt.rcParams['figure.figsize'] = f'{3.25*2/3},{3.25*2/3}'
# plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1'
plt.rcParams['lines.linewidth'] = '1'



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

betalist = [25,42,54,80,99]
# betalist = [25,42,54,73,80,99]
# betalist = [20,25,31,42,54,73,80,84,99]
# betalist = [25,50,80,190]
# betalist = [2000,]

kappa = 1.



############## CREATING FIGS ######################################





figFE, axFEs = plt.subplots(nrows=1,ncols=2)
axFE, axFE2 = axFEs
axFE.set_aspect('equal',adjustable='box')
axFE2.set_aspect('equal', adjustable='box')
# left, bottom, width, height = [0.15, 0.15, 0.1, 0.1]
# axFE2 = figFE.add_axes([left, bottom, width, height])
# figFE2, axFE2 = plt.subplots(1)
# figFE2.tight_layout(pad=2)

axFE.tick_params(axis='x',  pad=0)
axFE2.tick_params(axis='x', pad=0)

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
    ##################### Calculation step ###################
    assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

    GDomega = Time2FreqF(GDtau,Nbig,beta)
    FDomega = Time2FreqF(FDtau,Nbig,beta)
    GODomega = Time2FreqF(GODtau,Nbig,beta)
    FODomega = Time2FreqF(FODtau,Nbig,beta)
    DDomega = Time2FreqB(DDtau,Nbig,beta)
    DODomega = Time2FreqB(DODtau,Nbig,beta)




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

    skip = 10
    startT, stopT  = 0, Nbig//2 - 1
    llplotslice = slice(startT,stopT,skip)
    Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
    Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
    FreeDtau = DfreeImagtau(tau,r,beta)




    thetalist = np.linspace(0,2*np.pi,100)
    retFE = lambda theta : np.sum(-np.log(lamb**4 + ((SigmaDomega + SigmaODomega - 1j*omega)*(-1j*omega - np.conj(SigmaDomega) - np.conj(SigmaODomega)) - PhiDomega*np.conj(PhiDomega))*((SigmaDomega - SigmaODomega - 1j*omega)*(-1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) - PhiDomega*np.conj(PhiDomega)) - lamb**2*(SigmaDomega**2 - 4j*SigmaDomega*omega - 2*omega**2 + np.conj(SigmaDomega)**2 + 4j*omega*np.real(SigmaDomega) - 4*np.real(SigmaODomega)**2) + 2*lamb*(lamb*(np.abs(SigmaODomega)**2 + np.abs(PhiDomega)**2)*np.cos(2*theta) + np.cos(theta)*(SigmaODomega*np.abs(SigmaODomega)**2 - 2j*SigmaODomega*omega*np.conj(SigmaDomega) - SigmaODomega*np.conj(SigmaDomega)**2 - SigmaDomega*(SigmaDomega - 2j*omega)*np.conj(SigmaODomega) + SigmaODomega*np.conj(SigmaODomega)**2 + 2*(lamb**2 + omega**2 + np.abs(PhiDomega)**2)*np.real(SigmaODomega)))))
    normaln = -np.sum(np.log(omega**4))
    FEsumangle = np.array([retFE(theta) - normaln for theta in thetalist]) 
    FEsumangle -= np.mean(FEsumangle)


    ##################### Plotting Step #################
    axFE.plot(thetalist, (1./beta) * np.gradient(FEsumangle,thetalist), ls ='-', c=col,label=lab)


    axFE2.plot(thetalist, (1./beta) * FEsumangle, c=col,ls = '-',label=lab)

    # pi_controller = MultiplePi(2) # For pi/2 and multiples


    # axFE.set_title(r'phase dependent part of the free energy')
    axFE2.set_title(r'Phase dependent part of the free energy')
    axFE.set_title(r'Josephson Current')
    axFE.set_xlabel(r'$\theta$')
    axFE2.set_xlabel(r'$\theta$')
    # axFE2.legend()
    axFE.xaxis.set_major_locator(MultiplePi(2).locator())
    axFE.xaxis.set_major_formatter(MultiplePi(2).formatter())
    axFE2.xaxis.set_major_locator(MultiplePi(2).locator())
    axFE2.xaxis.set_major_formatter(MultiplePi(2).formatter())
    # figFE.savefig('../../KoenraadEmails/FreeEnergyOscillationSUP.pdf', bbox_inches = 'tight')
    # figFE.savefig('../../KoenraadEmails/JosephsonCurrent.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')


handles, labels = axFE.get_legend_handles_labels()
# figFE.legend(handles, labels, ncol=len(labels))

figFE.tight_layout()
# figFE.savefig('../Figuremaker/JosephsonCurrent.pdf',bbox_inches='tight')
# figFE2.savefig('../Figuremaker/PhaseDepFreeEnergy.pdf',bbox_inches='tight')


plt.show()


