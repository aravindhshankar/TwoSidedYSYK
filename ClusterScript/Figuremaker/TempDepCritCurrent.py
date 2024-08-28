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
from scipy.interpolate import Akima1DInterpolator,CubicSpline,PchipInterpolator
#import time


plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.titlesize'] = '10'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams['axes.labelsize'] = '10'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
# plt.rcParams['lines.markersize'] = '6'
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
# alpha = 0.2
# alpha = 0.1
# lamb = 0.001
lamb = 0.05
# lamb = 1.0
#J = 0.0
J = 0

man_exclude = np.array([10,22,24,27,30,31])
# betalist = [25,42,54,80,99]
betalist = ret_converged_betas(filename='../SupWH/NEWsupl05.out',ITERMAX=10000,man_exclude = man_exclude) #filename already the default one
# betalist = ret_converged_betas(filename='alpha0_1.out',ITERMAX=50000) 
CritCurrlist = np.zeros_like(betalist, dtype=np.float64)
# betalist = [25,50,80,190]
# betalist = [2000,]

kappa = 1.
fig, ax = plt.subplots(1,constrained_layout=True)
fig.set_figwidth(3.25*1*2/3)
ax.set_box_aspect(aspect=1)
ax.tick_params(axis='both', labelsize=7)
ax.tick_params(axis='y', pad=2)
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

    normaln = -np.sum(np.log(omega**4))
    FEsumangle = np.array([retFE(theta) - normaln for theta in thetalist]) 
    FEsumangle -= np.mean(FEsumangle)
    FEsumangle = np.real(FEsumangle)
    JosephsonCurrent = (1./beta) * np.gradient(FEsumangle,thetalist)
    CritCurrent = np.max(JosephsonCurrent)
    CritCurrlist[i] = CritCurrent




# ax.plot(thetalist, (1./beta) * np.gradient(FEsumangle,thetalist), ls ='-', c=col,label=lab)
ax.plot(1./betalist, CritCurrlist, '.-')
ax.axvline(1./62,ls='--',c='C1',label =r'$T_{WH}$')
ax.axvline(1./32,ls='--',c='C2',label = r'$T_c$')
ax.plot(1./betalist, np.zeros_like(CritCurrlist), ls='--', c='gray')
# ax.plot(betalist, CritCurrlist,'.-')
# ax.set_xlabel(r'$\beta$')

stopidx = np.argmin(np.abs(betalist - 46))
startidx = np.argmin(np.abs(betalist - 34))
interslice = slice(stopidx,startidx,-1) #strictly increasing order
predictslice = slice(stopidx, startidx-3, -1 )
Tlist = 1./betalist
print(Tlist[interslice], list[interslice])

# interpol = Akima1DInterpolator(x=Tlist[interslice],y=CritCurrlist[interslice],method='makima',extrapolate=True)
# interpol = PchipInterpolator(x=Tlist[interslice],y=CritCurrlist[interslice],extrapolate=True)
interpol = CubicSpline(x=Tlist[interslice],y=CritCurrlist[interslice],extrapolate=True)

# ax.plot(Tlist[interslice], interpol(Tlist[interslice]),ls='--',c='gray')
ax.plot(Tlist[predictslice], interpol(Tlist[predictslice]),ls='--',c='C3',label='extrapolation')
print(interpol(Tlist[predictslice]))
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$I_c$')
ax.set_xlim(0,0.121)
ax.set_ylim(0,)
ax.set_title(r'Critical current',pad=-8,loc='right')
# ax.legend(fontsize=16)

handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, ncol=len(labels)//2+1, loc="lower center", bbox_to_anchor=(0.5,-0.6),frameon=True,fancybox=True,borderaxespad=4, bbox_transform=ax.transAxes)


# fig.tight_layout()

print(fig.get_size_inches())
fig.savefig('TempDepCritCurrent.pdf', bbox_inches='tight')
plt.show()