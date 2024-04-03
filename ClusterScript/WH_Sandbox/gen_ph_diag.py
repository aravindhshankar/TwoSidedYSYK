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
	path_to_dump_lamb = '../Dump/lamb_anneal_dumpfiles/'
	path_to_dump_temp = '../Dump/temp_anneal_dumpfiles/'
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
DUMP = True
Nbig = int(2**14)

beta_start = 1 
target_beta = 100001
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
J = 0
kappa = 1.
beta_step = 1
betasavelist = np.array([50,100,500,1000,5000,10000])
lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])
lamblooplist = np.arange(1,0.001 - 1e-10,-0.001)
lamb = lamblooplist[0]

FEstemp = np.zeros(len(betasavelist),len(lambsavelist))
FEslamb = np.zeros(len(betasavelist),len(lambsavelist))

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
		except FileNotFoundError: 
			print("TEMP ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)
		try :
			#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
			plotfilelamb = os.path.join(path_to_dump_lamb, savefile)
		except FileNotFoundError: 
			print("LAMB ANNEAL INPUT FILE NOT FOUND")
			print(f'lamb = {lamb}, beta = {beta}')
			exit(1)

		GFstemp = np.load(plotfiletemp)
		GFslamb = np.load(plotfilelamb)
		FEslamb[i,j] = free_energy_YSYKWH(GFslamb, freq_grids, Nbig, beta, g, r, mu, kappa)
		FEstemp[i,j] = free_energy_YSYKWH(GFstemp, freq_grids, Nbig, beta, g, r, mu, kappa)




fig,ax = plt.subplots(1)
fig.suptitle('Phase diagram')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\lambda$')
for i, beta in enumerate(betasavelist):
	for j, lamb in enumerate(lambsavelist):
		if FEslamb[i,j] < FEstemp[i,j]: #WH
			ax.scatter(beta,lamb,c='r',marker='*')
		elif FEslamb[i,j] > FEstemp[i,j]: #NFL
			ax.scatter(beta,lamb,c='b',marker='^')
		else:
			ax.scatter(beta,lamb,c='g',marker='o')

plt.show()



















