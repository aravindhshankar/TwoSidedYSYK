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
#import time


plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
# plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '11'
plt.rcParams['figure.figsize'] = '8,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1'
plt.rcParams['axes.labelsize'] = '13'



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

# def add_subplot_axes(ax,rect,axisbg='w'):
def add_subplot_axes(ax,rect,facecolor='w'): # matplotlib 2.0+
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    # subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax






############## CREATING FIGS ######################################
fig, ax = plt.subplots(1,3)
# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# axinset = fig.add_axes([left, bottom, width, height])
rect = [0.2,0.2,0.7,0.7]
rect = [0.2,0.6,0.3,0.3]
axinset0 = add_subplot_axes(ax[0],[0,0.1,0.2,0.2])
axinset1 = add_subplot_axes(ax[1],[0,0,0.2,0.2])
axinset2 = add_subplot_axes(ax[2],[0.6,0.1,0.2,0.2])
# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# fig.suptitle(titlestring)
fig.tight_layout(pad=2.5)

figSL,axSL = plt.subplots(2,2)
#figSL.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
# figSL.suptitle(titlestring)
figSL.tight_layout(pad=2)



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

    
    skip = 10
    startT, stopT  = 0, Nbig//2 - 1
    llplotslice = slice(startT,stopT,skip)
    Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
    Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
    FreeDtau = DfreeImagtau(tau,r,beta)

    ax[0].semilogy(tau[llplotslice]/beta, np.abs(np.real(GDtau[llplotslice])),c=col,label=lab)
    axinset0.plot(tau[llplotslice]/beta, np.real(GDtau[llplotslice]), c = col, label = lab)
    axinset0.plot(tau[llplotslice]/beta, np.real(Gconftau[llplotslice]), c = col, ls='--' )
    # ax[0].set_ylim(-1,1)
    ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[0].set_ylabel(r'$|\Re{G_{d}(\tau)}|$')
    ax[0].legend(framealpha = 0.0)


    axinset1.plot(tau[llplotslice]/beta, np.real(DDtau[llplotslice]), c=col, label = lab)
    axinset1.plot(tau[llplotslice]/beta, np.real(Dconftau[llplotslice]), c=col, ls='--' )
    ax[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(DDtau[llplotslice])),c=col,label=lab)

    # axinset[1].semilogy(tau/beta, np.real(Dconftau), c=col, ls='--' )
    # ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
    #ax[1].set_ylim(0,1)
    ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    ax[1].set_ylabel(r'$|\Re{D_{d}(\tau)}|$')
    ax[1].legend()


    # ax[2].plot(tau/beta, np.real(FDtau), 'r--', label = 'numerics Real Ftau')
    # ax[2].plot(tau/beta, np.imag(FDtau), 'b', label = 'numerics Imag Ftau')
    axinset2.plot(tau[llplotslice]/beta, (np.abs(FDtau[llplotslice])), c=col, label = lab)
    axinset2.plot(tau[llplotslice]/beta, (np.abs(FODtau[llplotslice])), ls='--', c=col)
    ax[2].semilogy(tau[llplotslice]/beta, np.abs(FDtau[llplotslice]),c=col,label=lab)

    #ax[2].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
    #ax[2].set_ylim(-1,1)
    ax[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # ax[2].set_ylabel(r'$\Re{F(\tau)}$')
    ax[2].set_ylabel(r'$|F_{d}(\tau)|$')
    ax[2].legend()

    #fig.suptitle(r'$\beta$ = ', beta)
    #plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')






    
    ###################### Log-Linear Plot ###############################



    # startT, stopT  = 0, 5000

    # fitsliceT = slice(startT+4500, startT + 4600)
    # #fitslice = slice(start+25, start + 35)
    # functoplotT = np.abs(np.real(GDtau))
    # mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
    # print(f'tau/beta at start of fit = {(tau[fitsliceT][0]/beta):.3f}')
    # print(f'slope of fit = {mT:.03f}')
    # # print('2 Delta  = ', 2*delta)

    axSL[0,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GDtau[startT:stopT])),'p',c=col,label = lab)
    # axSL[0,0].semilogy(tau[startT:stopT]/beta, np.exp(mT*tau[startT:stopT] + cT), c=col,label=f'Fit with slope {mT:.03f}')
    #axSL[0,0].set_ylim(1e-1,1e1)
    axSL[0,0].set_xlabel(r'$\tau/\beta$')
    axSL[0,0].set_ylabel(r'$-\Re G(\tau)$')
    #axSL[0,0].set_aspect('equal', adjustable='box')
    #axSL[0,0].axis('square')
    axSL[0,0].legend()
    axSL[0,0].set_yscale('log')


    axSL[1,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(DDtau[startT:stopT])),'p',c=col,label=lab)
    axSL[1,0].set_xlabel(r'$\tau/beta$')
    axSL[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
    #axSL[1,0].set_aspect('equal', adjustable='box')
    axSL[1,0].legend()




    # figFE.savefig('../../KoenraadEmails/FreeEnergyOscillationSUP.pdf', bbox_inches = 'tight')
    # figFE.savefig('../../KoenraadEmails/JosephsonCurrent.pdf', bbox_inches = 'tight')
fig.savefig('insetsSupGFs.pdf',bbox_inches='tight')
#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()

