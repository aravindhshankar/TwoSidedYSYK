import sys ###
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
plt.rcParams['figure.titlesize'] = '8'
plt.rcParams['axes.titlesize'] = '8'
plt.rcParams['axes.labelsize'] = '8'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
plt.rcParams['lines.markersize'] = '2'
plt.rcParams['lines.linewidth'] = '0.5'
plt.rcParams['axes.formatter.limits'] = '-2,2'
# plt.tick_params(axis='both', labelsize=5)
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
# betalist = [20,25,31,42,54,73,80,99]
# betalist = [25,50,80,190]
# betalist = [2000,]

kappa = 1.

############## CREATING FIGS ######################################
# figmet, axmet = plt.subplots(2,2)
figSUP, axSUP = plt.subplots(1,3, tight_layout=True)


figdiffG, axdiffG = plt.subplots(1,2,tight_layout=True)
axdiffG[0].set_box_aspect(aspect=1)
axdiffG[1].set_box_aspect(aspect=1)
figdiffG.set_figwidth(3.25)
# figdiffD.set_figwidth(3.25)
figdiffG.tight_layout()


for axi in axdiffG:
    axi.tick_params(axis='both', labelsize=5,pad=0.0)


axdiffG[0].set_xlabel(r'$\tau/\beta$',labelpad = -3)
# axdiffG[0].set_ylabel(r'$|\Delta G_d|$',labelpad=-4)
# axdiffG[0].set_ylabel(r'$|\Re G^{\mathrm{sup}}_d - \Re G^{\mathrm{sup}}_{\mathrm{one-side}}|$',labelpad=-10)
axdiffG[0].set_title(r'$|\Re G^{\mathrm{sup}}_d - \Re G^{\mathrm{sup}}_{\mathrm{one-side}}|$')
# figdiffG.suptitle('Subtracting two sided and one sided superconducting solutions compared to two sided metallic solution',y=1.001)
axdiffG[1].set_xlabel(r'$\tau/\beta$',labelpad = -3)
# axdiffG[1].set_ylabel(r'$|\Re{G^{\mathrm{met}}_{d}(\tau)}|$',labelpad=-4)
axdiffG[1].set_title(r'$|\Re{G^{\mathrm{non-sup}}_{d}(\tau)}|$')
# axdiffG[0].yaxis.set_label_coords(-0.1, 0.55)
# axdiffG[1].yaxis.set_label_coords(-0.1, 0.5)


# figdiffF, axdiffF = plt.subplots(1)
figdiffF, (axdiffF, axGratio) = plt.subplots(1,2)
figdiffF.set_figwidth(3.25)
figdiffF.tight_layout()
axdiffF.tick_params(axis='both', labelsize=5,pad=0.0)

axdiffF.set_box_aspect(aspect=1)
axGratio.set_box_aspect(aspect=1)
# figGratio, axGratio = plt.subplots(1)
# figGratio.set_figwidth(3.25)
# figGratio.tight_layout()
axGratio.tick_params(axis='both', labelsize=5,pad=0.0)

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
        metGODtau = -metGODtau
    except FileNotFoundError: 
        print('Filename : ', savefile)
        print("INPUT FILE NOT FOUND") 
        exit(1)
    # axmet[0,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(metGDtau[llplotslice])),'.',c = col, label = lab)
    # # axmet[0,0].semilogy(tau[llplotslice]/beta, np.exp(mT*tau[llplotslice] + cT), label=f'Fit with slope {mT:.03f}')
    # axmet[0,0].set_xlabel(r'$\tau/\beta$')
    # axmet[0,0].set_ylabel(r'$-\Re G_{d}(\tau)$')
    # axmet[0,0].set_yscale('log')


    # axmet[0,1].semilogy(tau[llplotslice]/beta, np.abs(np.real(metGDtau[llplotslice])),'.',c = col, label = lab)
    # axmet[0,1].set_xlabel(r'$\tau/\beta$')
    # axmet[0,1].set_ylabel(r'$-\Re G_{od}(\tau)$')
    # axmet[0,1].set_yscale('log')

    # axmet[1,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(metDDtau[llplotslice])),'.',c=col,label=lab)
    # axmet[1,0].set_xlabel(r'$\tau/\beta$')
    # axmet[1,0].set_ylabel(r'$g^2\,\Re{D_{d}(\nu_n)}$',labelpad = None)

    # axmet[1,1].semilogy(tau[llplotslice]/beta, np.abs(np.real(metDODtau[llplotslice])),'.',c=col,label=lab)
    # axmet[1,1].set_xlabel(r'$\tau/\beta$')
    # axmet[1,1].set_ylabel(r'$g^2\,\Re{D_{od}(\nu_n)}$',labelpad = None)

    
    
   

    ##### DIFFS ########
    Gratio = (np.abs(np.real(GDtau-Gtau)))/(np.abs(np.real(Gtau)))
    diffsG = (np.abs(np.real(GDtau-Gtau)))
    plotGtau = np.abs(np.real(Gtau))
    plotDtau = np.abs(np.real(Dtau))
    plotFtau = np.abs(Ftau)
    if beta > 62:
        fitslice0 = slice(np.argmin(np.abs(tau/beta - 0.3)),np.argmin(np.abs(tau/beta - 0.4)))
        fitslice0 = slice(np.argmin(np.abs(tau/beta - 0.2)),np.argmin(np.abs(tau/beta - 0.4)))
        m0,logc0 = np.polyfit(tau[fitslice0]/beta,np.log(diffsG[fitslice0]),1)
        c0 = np.exp(logc0)
        print(f'diff coeff =  {c0}')
        axdiffG[0].semilogy(tau[llplotslice]/beta, c0*np.exp(m0*tau[llplotslice]/beta),c='r',label = f'fit with slope {m0/beta:.4}',ls='--')

        SUPm0,SUPlogc0 = np.polyfit(tau[fitslice0]/beta,np.log(plotGtau[fitslice0]),1)
        SUPc0 = np.exp(SUPlogc0)
        print(f'SUPoneside coeff = {SUPc0}')
        axSUP[0].semilogy(tau[llplotslice]/beta, SUPc0*np.exp(SUPm0*tau[llplotslice]/beta),c='r',label = f'fit with slope {SUPm0/beta:.4}',ls='--')

        GRm0,GRlogc0 = np.polyfit(tau[fitslice0]/beta,np.log(Gratio[fitslice0]),1)
        GRc0 = np.exp(GRlogc0)
        axGratio.semilogy(tau[llplotslice]/beta, GRc0*np.exp(GRm0*tau[llplotslice]/beta),c='r',label = f'fit with slope {GRm0/beta:.4}',ls='--')

        metm0,metlogc0 = np.polyfit(tau[fitslice0]/beta,np.log(np.abs(np.real(metGDtau))[fitslice0]),1)
        metc0 = np.exp(metlogc0)
        print(f'metWH coeff = {metc0}')
        axdiffG[1].semilogy(tau[llplotslice]/beta, metc0*np.exp(metm0*tau[llplotslice]/beta),c='b',label = f'fit with slope {metm0/beta:.4}',ls='--')

    axdiffG[0].semilogy(tau[llplotslice]/beta, diffsG[llplotslice],c=col,label=lab)
    # axdiffG[0].legend(framealpha = 0.0)
    axSUP[0].semilogy(tau[llplotslice]/beta, plotGtau[llplotslice],c=col,label=lab)
    axSUP[1].semilogy(tau[llplotslice]/beta, plotDtau[llplotslice],c=col,label=lab)
    axSUP[2].semilogy(tau[llplotslice]/beta, plotFtau[llplotslice],c=col,label=lab)

    axGratio.semilogy(tau[llplotslice]/beta, Gratio[llplotslice],c=col,label=lab)
    axdiffG[1].semilogy(tau[llplotslice]/beta, np.abs(np.real(metGDtau))[llplotslice],c=col,label=lab)
    # axdiffG[1].legend(framealpha = 0.0)


    diffsF = np.abs(np.abs(FDtau)-np.abs(Ftau))
    Fratio = diffsF/np.abs(Ftau)
    axdiffF.semilogy(tau[llplotslice]/beta, Fratio[llplotslice],c=col,label=lab)
    axdiffF.set_xlabel(r'$\tau/\beta$',labelpad = 0)
    # axdiffF.set_ylabel(r'$||F(\tau)| -|F_{\mathrm{one side}}(\tau)||/|F_{\mathrm{one side}}(\tau)|$')
    # axdiffF.set_title(r'$||F(\tau)| -|F_{\mathrm{one side}}(\tau)||/|F_{\mathrm{one side}}(\tau)|$')
    axdiffF.set_title(r'$\frac{||F| -|F_{\mathrm{one side}}||}{|F_{\mathrm{one side}}|}$')
    # axdiffF.legend(framealpha=0.0)

for axi in axSUP:
    axi.legend(framealpha=0.0)

# for axi in axdiffG:
#     axi.legend(framealpha=0.0)

# for axi in axdiffD:
#     axi.legend(framealpha=0.0)
    
axSUP[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSUP[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSUP[2].set_xlabel(r'$\tau/\beta$',labelpad = 0)

axSUP[0].set_ylabel(r'$|\Re G(\tau)|$',labelpad = 0)
axSUP[1].set_ylabel(r'$|\Re D(\tau)|$',labelpad = -2)
axSUP[2].set_ylabel(r'$|F(\tau)|$',labelpad = -2)

figSUP.suptitle('One sided Greens functions Including superconductivity')

axGratio.set_xlabel(r'$\tau/\beta$',labelpad = 0)
# Gratio = (np.abs(np.real(GDtau-Gtau)))/(np.abs(np.real(Gtau)))
# axGratio.set_title(r'$|\Re G^{\mathrm{sup}}_d(\tau) - \Re G^{\mathrm{sup}}_{\mathrm{one side}}(\tau)|/|\Re G^{\mathrm{sup}}_{\mathrm{one side}}(\tau)|$')
axGratio.set_title(r'$\frac{|\Re G^{\mathrm{sup}}_d - \Re G^{\mathrm{sup}}_{\mathrm{one side}}|}{|\Re G^{\mathrm{sup}}_{\mathrm{one side}}|}$',y=1.01)
# axGratio.legend(framealpha=0.0)

# figGratio.suptitle('Comparison of relative magnitudes of one and two side superconducting Greens functions',y=1.001)
# figGratio.savefig('diffsDETAIL_figures/Gratio.pdf',bbox_inches='tight')

# figSUP.savefig('diffsDETAIL_figures/OnesidedInclSUP.pdf')
# handles, labels = axdiffG[0].get_legend_handles_labels()
# handles1, labels1 = axdiffG[1].get_legend_handles_labels()
# joined_handles = handles.copy()
# joined_labels = labels.copy()
# for i,label in enumerate(labels1):
#     if label not in joined_labels:
#         joined_labels.append(label)
#         joined_handles.append(handles1[i])

# lgd = figdiffG.legend(joined_handles, joined_labels, ncol=len(joined_labels)//2 , loc="lower center", bbox_to_anchor=(1.2,-0.15),frameon=True,fancybox=True,borderaxespad=2, bbox_transform=axdiffG[0].transAxes)


handles, labels = axdiffG[0].get_legend_handles_labels()
handles1, labels1 = axdiffG[1].get_legend_handles_labels()
joined_handles = handles.copy()
joined_labels = labels.copy()
for i,label in enumerate(labels1):
    if label not in joined_labels:
        joined_labels.append(label)
        joined_handles.append(handles1[i])
lgd = figdiffG.legend(joined_handles, joined_labels, ncol=len(joined_labels)//2 , loc="lower center", bbox_to_anchor=(1.2,-1),frameon=True,fancybox=True,borderaxespad=2, bbox_transform=axdiffG[0].transAxes)

# list(set(x).symmetric_difference(set(f)))

# figdiffG.savefig('diffsDETAIL_figures/diffG.pdf',bbox_inches='tight')

handles, labels = axdiffF.get_legend_handles_labels()
handles1, labels1 = axGratio.get_legend_handles_labels()
joined_handles = handles.copy()
joined_labels = labels.copy()
for i,label in enumerate(labels1):
    if label not in joined_labels:
        joined_labels.append(label)
        joined_handles.append(handles1[i])
lgd = figdiffF.legend(joined_handles, joined_labels, ncol=len(joined_labels)//2 , loc="lower center", bbox_to_anchor=(1.2,-0.8),frameon=True,fancybox=True,borderaxespad=2, bbox_transform=axdiffF.transAxes, fontsize=8)


# fig.suptitle(r"Superconducting Green's functions for $\lambda=0.05$")
# fig.savefig('SupCondFigs.pdf', bbox_inches='tight')

# figdiffF.savefig('ratioFs.pdf')
# figdiffF.savefig('diffsDETAIL_figures/ratioFsv2.pdf',bbox_inches='tight')






plt.show()


