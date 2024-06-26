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
	# path_to_dump_lamb = '../Dump/lamb_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/temp_anneal_dumpfiles/'
	path_to_dump_lamb = '../Dump/xshift_lamb_anneal_dumpfiles/'
	path_to_dump_temp = '../Dump/23Aprzoom_xshift_temp_anneal_dumpfiles/fwd' 
	if not os.path.exists(path_to_dump_lamb):
		raise Exception('Generate Data first! Path to lamb dump not found')
		exit(1)
	if not os.path.exists(path_to_dump_temp):
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

beta_start = 1 
target_beta = 970
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
J = 0
kappa = 1.
beta_step = 1
# betasavelist = np.array([50,100,500,1000,5000,10000])
# lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])
betasavelist = np.array([10,20,50,100,150,200,300,500,700,1000])
lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])

lamblooplist = np.arange(1,0.001 - 1e-10,-0.001)
lamb = lamblooplist[0]
lamb = 0.005

FEstemp = np.zeros((len(betasavelist),len(lambsavelist)))
FEslamb = np.zeros((len(betasavelist),len(lambsavelist)))

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
			plotfiletemp = os.path.join(path_to_dump_temp, savefile)
			GFstemp = np.load(plotfiletemp)
		except FileNotFoundError: 
			print("TEMP ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)
		try :
			#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
			plotfilelamb = os.path.join(path_to_dump_lamb, savefile)
			GFslamb = np.load(plotfilelamb)
		except FileNotFoundError: 
			print("LAMB ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)

		
		
		impose_saddle = False
		FEslamb[i,j] = free_energy_YSYKWH(GFslamb, freq_grids, Nbig, beta, g, r, mu, kappa,lamb,J,impose_saddle = impose_saddle)
		FEstemp[i,j] = free_energy_YSYKWH(GFstemp, freq_grids, Nbig, beta, g, r, mu, kappa, lamb,J,impose_saddle = impose_saddle )


residuals = FEstemp-FEslamb
# print(np.array2string(residuals,precision=4,floatmode='fixed'))

lambi = 1
lamb = lambsavelist[lambi]
fig, ax = plt.subplots(1)
ax.plot(1./betasavelist, FEstemp[:,lambi],'.-', label='temp annealed')
ax.plot(1./betasavelist, FEslamb[:,lambi],'.-', label='lamb annealed' )
ax.legend()
ax.set_xlabel('temperature T')
ax.set_ylabel('Free energy')
ax.set_title(r'$\lambda$ = ' + str(lamb))

plt.show()






