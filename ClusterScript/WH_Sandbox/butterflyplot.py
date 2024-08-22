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


PLOTTING = False
Nbig = int(2**14)

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

		
		
		impose_saddle = False
		FEstempfwd[i] = free_energy_YSYKWH(GFstempfwd, freq_grids, Nbig, beta, g, r, mu, kappa,lamb,J,impose_saddle = impose_saddle)
		FEstemprev[i] = free_energy_YSYKWH(GFstemprev, freq_grids, Nbig, beta, g, r, mu, kappa, lamb,J,impose_saddle = impose_saddle )


residuals = FEstempfwd-FEstemprev
# print(np.array2string(residuals,precision=4,floatmode='fixed'))

############# Fit of FE in different phases #################
# F0 = 40
F0 = 0 
# F0 = np.min(FEstempfwd)
nflslice = slice(48,58)
whslice = slice(90,99)
mbh, cbh = np.polyfit(1./betasavelist[nflslice], FEstempfwd[nflslice] + F0 , 1)
mwh, cwh = np.polyfit(1./betasavelist[whslice], FEstempfwd[whslice] + F0 , 1)

pbh,qbh,rbh = np.polyfit(1./betasavelist[nflslice], FEstempfwd[nflslice] + F0 , 2)






xaxis = 1./betasavelist

lambi = 0
lamb = lambsavelist[lambi]
fig, ax = plt.subplots(1)
ax.plot(1./betasavelist, FEstempfwd - F0,'.--', label='temp annealed fwd')
ax.plot(1./betasavelist, FEstemprev - F0,'.--', label='temp annealed rev' )
ax.axvline(lamb,ls='--')
ax.axvline(1/62, ls = '--', c='grey')

# ax.plot(xaxis, mbh*xaxis + cbh, ls = '--', label = f'nflFit with intercept {cbh:.6}')
# ax.plot(xaxis, pbh*xaxis**2 + qbh*xaxis + rbh, ls = '--', label = f'nfl parabolic Fit with intercept {rbh:.6}')
# ax.plot(xaxis, mwh*xaxis + cwh, ls = '--', label = f'whFit with intercept {cwh:.6}')

ax.legend()
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

# ax3 = ax.twinx()
# ax3.plot(1./betasavelist, -1.*(betasavelist**2) * np.gradient(FEstempfwd,betasavelist), '.-', c = 'k', label=r'Gradient $\frac{dF}{dT}$')
# ax3.set_ylabel(r'$\frac{dF}{dT}$')
# ax3.legend()


fig.tight_layout()


plt.show()




