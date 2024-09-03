import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


if not os.path.exists('../Dump/'):
	print("Error - Path to Dump directory not found ")
	raise Exception("Error - Path to Dump directory not found ")
	exit(1)
else:
	# path_to_dump_temp_fwd = '../Dump/lamb_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/temp_anneal_dumpfiles/'
	path_to_dump_temp_fwd = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd/'
	path_to_dump_temp_rev = '../Dump/zoom_xshift_temp_anneal_dumpfiles/rev/'
	# path_to_dump_temp_fwd = '../Dump/24Aprzoom_x0_01_temp_anneal_dumpfiles/fwd/'
	# path_to_dump_temp_rev = '../Dump/24Aprzoom_x0_01_temp_anneal_dumpfiles/rev/'
	if not os.path.exists(path_to_dump_temp_fwd):
		raise Exception('Generate Data first! Path to lamb dump not found')
		exit(1)
	if not os.path.exists(path_to_dump_temp_rev):
		raise Exception('Generate Data first! Path to temp dump not found')
		exit(1)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_YSYKWH 
from annealers import anneal_temp, anneal_lamb


plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
plt.rcParams['figure.figsize'] = '8 ,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1.6'
plt.rcParams['axes.labelsize'] = '16'
plt.rcParams['axes.titlesize'] = '16'

plt.rcParams['legend.fontsize'] = '12'





def calcFE_met(GFs, freq_grids, Nbig, beta, g=0.5, r=1., mu=0,lamb=0.,kappa=1.,J=0,alpha=1):
	GDtau, GODtau, DDtau, DODtau = GFs
	FDtau = np.zeros_like(GDtau)
	FODtau = np.zeros_like(GDtau)
	omega,nu = freq_grids

	np.testing.assert_almost_equal(omega[2] - omega[1], 2*np.pi/beta)
	np.testing.assert_almost_equal(nu[2] - nu[1], 2*np.pi/beta)
	np.testing.assert_equal(Nbig, len(DDtau))
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

	PhiDomega = np.zeros_like(SigmaDomega)
	PhiODomega = np.zeros_like(SigmaODomega)
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
	return Fe





PLOTTING = False
Nbig = int(2**14)

beta_start = 20
beta_start = 2
# target_beta = 2001
target_beta = 101
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
J = 0
kappa = 1.
beta_step = 1
# betasavelist = np.array([50,100,500,1000,5000,10000])
# lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])
betasavelist = np.arange(beta_start,target_beta)
# lamb = lamblooplist[0]
lambsavelist = (0.05,)

FEstempfwd = np.zeros((len(betasavelist), ))
FEstemprev= np.zeros((len(betasavelist),))

#load both the dumpfiles files 
for i, beta in enumerate(betasavelist):
	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	freq_grids = [omega,nu]
	for j, lamb in enumerate(lambsavelist):
		savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
		savefile += 'lamb' + str(lamb) + 'J' + str(J)
		savefile += 'g' + str(g) + 'r' + str(r)
		savefile = savefile.replace('.','_') 
		savefile += '.npy'
		#print('savefile = ', savefile)
		try :
			#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
			plotfiletempfwd = os.path.join(path_to_dump_temp_fwd, savefile)
			GFstempfwd = np.load(plotfiletempfwd)
			GFstempfwd[1] = -1.0*GFstempfwd[1]
		except FileNotFoundError: 
			print("TEMP FWD ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)
		try :
			#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
			plotfiletemprev = os.path.join(path_to_dump_temp_rev, savefile)
			GFstemprev = np.load(plotfiletemprev)
			GFstemprev[1] = -GFstemprev[1]
		except FileNotFoundError: 
			print("TEMP REV ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)


		FEstempfwd[i] = calcFE_met(GFstempfwd, freq_grids, Nbig, beta, g=0.5, r=1, mu=0, lamb=lamb,kappa=1,J=0)
		FEstemprev[i] = calcFE_met(GFstemprev, freq_grids, Nbig, beta, g=0.5, r=1, mu=0, lamb=lamb,kappa=1,J=0)


residuals = FEstempfwd-FEstemprev
# print(np.array2string(residuals,precision=4,floatmode='fixed'))

############# Fit of FE in different phases #################
# F0 = 40
F0 = 0 
# # F0 = np.min(FEstempfwd)
nflslice = slice(15,20)
# whslice = slice(90,99)
mbh, cbh = np.polyfit(1./betasavelist[nflslice], FEstempfwd[nflslice] + F0 , 1)
# mwh, cwh = np.polyfit(1./betasavelist[whslice], FEstempfwd[whslice] + F0 , 1)

pbh,qbh,rbh = np.polyfit(1./betasavelist[nflslice], FEstempfwd[nflslice] + F0 , 2)






xaxis = 1./betasavelist

lambi = 0
lamb = lambsavelist[lambi]
fig, ax = plt.subplots(1)
# ax.plot(1./betasavelist, FEstempfwd - F0,'.--', label='temp annealed fwd')
# ax.plot(1./betasavelist, FEstemprev - F0,'.--', label='temp annealed rev' )
ax.plot(1./betasavelist, FEstempfwd - F0,'.--', label='Annealing from high to low temperature')
ax.plot(1./betasavelist, FEstemprev - F0,'.--', label='Annealing from low to high temperature' )
# ax.axvline(lamb,ls='--')
ax.axvline(1/62, ls = '--', c='grey')

# ax.plot(xaxis, mbh*xaxis + cbh, ls = '--', label = f'nflFit with intercept {cbh:.6}')
# ax.plot(xaxis, pbh*xaxis**2 + qbh*xaxis + rbh, ls = '--', label = f'nfl parabolic Fit with intercept {rbh:.6}')
# ax.plot(xaxis, mwh*xaxis + cwh, ls = '--', label = f'whFit with intercept {cwh:.6}')

ax.legend(loc='upper right')
ax.set_xlabel('temperature T')
ax.set_ylabel('Free energy')
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_title(r'$\lambda$ = ' + str(lamb))
# ax.set_xscale('log')
# ax2 = ax.twiny()
# ax2.plot(betasavelist, FEstempfwd, '--', c= 'k', alpha = 0.1)
# ax2 = ax.secondary_xaxis('top', functions =(lambda x: 1/x, lambda x : 1/x))
# ax2.set_xlabel('Inverse temperature $\\beta$')
# ax2.set_ticks(betasavelist)
# ax.set_xlim(0.005,0.03)
# ax.set_ylim(-0.01,0.06)
Tlist = 1./betasavelist
# ax3 = ax.twinx()
# # ax3.plot(1./betasavelist, -1.*(betasavelist**2) * np.gradient(FEstempfwd,betasavelist), '.-', c = 'k', label=r'Gradient $\frac{dF}{dT}$')
# ax3.plot(Tlist, np.gradient(FEstempfwd,Tlist), '.-', c = 'k', label=r'Gradient $\frac{dF}{dT}$')
# ax3.set_ylabel(r'$\frac{dF}{dT}$')
# ax3.legend()


fig.tight_layout()
# plt.savefig('../Figuremaker/butterflyplotmetal.pdf',bbox_inches='tight')




plt.show()



