import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

# path_to_dump = '../Dump/SupCondWHImagDumpfiles'
# path_to_dump = '../Dump/l_05Sup/'
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
from collectconvergedbetas import ret_converged_betas
#import time


plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.titlesize'] = '10'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams['axes.labelsize'] = '10'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
# plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '0.8'
plt.rcParams['lines.markersize'] = '3'
# plt.rcParams['axes.formatter.limits'] = '-2,2'
# plt.rcParams['text.usetex'] = 'False'



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
# alpha = 0.2
# alpha = 0.1
# lamb = 0.001
lamb = 0.05
# lamb = 1.0
#J = 0.0
J = 0

# betalist = [25,42,54,80,99]
man_exclude = np.array([10,22,24,27,30,31])
betalist = ret_converged_betas(filename='NEWsupl05.out',ITERMAX=10000,man_exclude=man_exclude) #filename already the default one
# betalist = ret_converged_betas(filename='alpha0_1.out',ITERMAX=50000) 
# CritCurrlist = np.zeros_like(betalist, dtype=np.float64)
FElist = np.zeros_like(betalist, dtype=np.float64)
# betalist = [25,50,80,190]
# betalist = [2000,]

kappa = 1.
fig, ax = plt.subplots(1)
fig.set_figwidth(3.25*2/3)
fig.tight_layout()
ax.set_box_aspect(aspect=1)
ax.tick_params(axis='both', labelsize=8)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='x',  pad=1)
ax.tick_params(axis='x', pad=1)

thetalist = np.linspace(0,2*np.pi,100)
for i, beta in enumerate(betalist): 
    col = 'C'+str(i)
    lab = r'$\beta = $' + f'{beta}'
    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')

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


    retFE = lambda theta : np.sum(-np.log(lamb**4 + ((SigmaDomega + SigmaODomega - 1j*omega)*(-1j*omega - np.conj(SigmaDomega) - np.conj(SigmaODomega)) - PhiDomega*np.conj(PhiDomega))*((SigmaDomega - SigmaODomega - 1j*omega)*(-1j*omega - np.conj(SigmaDomega) + np.conj(SigmaODomega)) - PhiDomega*np.conj(PhiDomega)) - lamb**2*(SigmaDomega**2 - 4j*SigmaDomega*omega - 2*omega**2 + np.conj(SigmaDomega)**2 + 4j*omega*np.real(SigmaDomega) - 4*np.real(SigmaODomega)**2) + 2*lamb*(lamb*(np.abs(SigmaODomega)**2 + np.abs(PhiDomega)**2)*np.cos(2*theta) + np.cos(theta)*(SigmaODomega*np.abs(SigmaODomega)**2 - 2j*SigmaODomega*omega*np.conj(SigmaDomega) - SigmaODomega*np.conj(SigmaDomega)**2 - SigmaDomega*(SigmaDomega - 2j*omega)*np.conj(SigmaODomega) + SigmaODomega*np.conj(SigmaODomega)**2 + 2*(lamb**2 + omega**2 + np.abs(PhiDomega)**2)*np.real(SigmaODomega)))))

    # normaln = -np.sum(np.log(omega**4))
    # FEsumangle = np.array([retFE(theta) - normaln for theta in thetalist]) 
    # FEsumangle -= np.mean(FEsumangle)
    # FEsumangle = np.real(FEsumangle)
    # JosephsonCurrent = (1./beta) * np.gradient(FEsumangle,thetalist)
    # CritCurrent = np.max(JosephsonCurrent)
    # CritCurrlist[i] = CritCurrent
    detD0inv = (nu**2+ r)**2 
    Sf = retFE(0) + np.sum(np.log(omega**4)) - 4*np.log(2) #ret FE is - ln det 
    Sd = 0.5*kappa*np.sum(np.log(((nu**2+r-PiDomega)**2 - (J-PiODomega)**2)/(detD0inv)))
    Slm = 2*kappa*np.sum(DDomega*PiDomega + DODomega*PiODomega)
    # Sb0 = 0.5*(np.sqrt(r)*beta + 2*np.log(1- np.exp(-1.0*beta*np.sqrt(r)))) #From Valentinis, Inkof, Schmalian
    Sb0 = -(0.5*np.sqrt(r)*beta - np.log(1- np.exp(-1.0*beta*np.sqrt(r)))) #From Valentinis, Inkof, Schmalian
    Fe = np.real(Sf + Sd + Slm + Sb0)/beta
    FElist[i] = Fe
    




Tlist = 1./betalist
gradTF = np.gradient(FElist,Tlist)

# axFE.plot(thetalist, (1./beta) * np.gradient(FEsumangle,thetalist), ls ='-', c=col,label=lab)
# ax.plot(1./betalist, CritCurrlist)
ax.axvline(1./62,ls='--',label=r'$T_{WH}$',c='C4')
# ax.axvline(1./33,ls='--',label=r'$T_{c}')
ax.axvline(1./35,ls='--',label=r'$T_{c}$',c='C3')
ax.plot(1./betalist, FElist,'.-')
# ax.plot(Tlist,gradTF,'.-')
# ax.set_xlabel(r'$\beta$')
ax.set_xlabel(r'$T$',labelpad=-1)
ax.set_title(r'Free Energy',loc='right',pad=-2)
ax.set_xscale('log')
ax.xaxis.set_label_coords(0.5,-0.01)
ax.set_xlim(1./108,0.21)
ax.set_ylim(-1.22,-0.8)
# ax.set_ylabel(r'$FE$')
ax.legend(loc=(0.5,0.1))
# fig.savefig('../Figuremaker/SUP_free_energy.pdf',bbox_inches='tight')


fig2,ax2 = plt.subplots(1)
fig2.set_figwidth(3.25*2/3)
fig2.tight_layout()
ax2.set_box_aspect(aspect=1)
ax2.tick_params(axis='both', labelsize=8)
ax2.tick_params(axis='y', pad=1)
ax2.tick_params(axis='x',  pad=1)
ax2.tick_params(axis='x', pad=1)

ax2.set_xlabel(r'$T$',labelpad=-1)
ax2.set_title(r'$\partial_T F$',loc='right',pad=-1.8)
ax2.set_xscale('log')
ax2.xaxis.set_label_coords(0.5,-0.01)
ax2.set_xlim(1./108,0.21)
ax2.set_ylim(-1.49,0)

gradTfrombeta = -1.0 * betalist**2 * np.gradient(FElist,betalist)


ax2.plot(Tlist,gradTF,'.-')
# ax2.plot(Tlist,gradTfrombeta,':')
ax2.axvline(1./62,ls='--',label=r'$T_{WH}$',c='C4')
ax2.axvline(1./35,ls='--',label=r'$T_{c}$',c='C3')
ax2.legend(framealpha=0)



# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
left, bottom, width, height = [0.50, 0.32, 0.3, 0.3]
ax2inset = fig2.add_axes([left, bottom, width, height])

ax2inset.set_box_aspect(aspect=1)
ax2inset.plot(Tlist,gradTF,'.-')
ax2inset.set_xlim(1./75,1./55)
ax2inset.set_ylim(-0.1,0)
ax2inset.set_xscale('log')
# ax2inset.set_xlabel(r'$\tau/\beta$',fontsize=7)
# ax2inset.set_ylabel(r'$|G_d(\tau)|$',fontsize=7)
# ax2inset.yaxis.set_label_coords(-0.3,0.5)
ax2inset.tick_params(which='major', length=1.5, width=0.4, direction="in", right=True, top=True,labelsize=0,pad=0.1)
ax2inset.tick_params(which='minor', length=1, width=0.2, direction="in", right=True, top=True,labelsize=0,pad=0.1)
# ax2inset.set_title(titlestring,fontsize=7)
ax2inset.axvline(1./62,ls='--',label=r'$T_{WH}$',c='C4')
ax2inset.axvline(1./35,ls='--',label=r'$T_{c}$',c='C3')
ax2inset.text(1./72, -0.02, r'$T_{WH}$',fontsize=7,c='C4',alpha=1)



fig2.savefig('partialTFSUP.pdf',bbox_inches='tight')
# plt.show()