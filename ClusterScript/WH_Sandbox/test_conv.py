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


def calcerr(GFs,Nbig, beta, g, r, mu, kappa,lamb,J,freq_grids=None):
	'''
	GFs are GFtaus as loaded from dump file
	'''
	try:
		omega,nu = freq_grids
	except ValueError:
		omega = ImagGridMaker(Nbig,beta,'fermion')
		nu = ImagGridMaker(Nbig,beta,'boson')

	
	GDtau, GODtau, DDtau, DODtau = GFs
	SigmaDtau = kappa * g**2 * GDtau * DDtau
	SigmaODtau = kappa * g**2 * GODtau * DODtau
	PiDtau = 2 * g**2 * GDtau * GDtau[::-1]
	PiODtau = 2 * g**2 * GODtau * GODtau[::-1]

	SigmaDomega, SigmaODomega = Time2FreqF(SigmaDtau,Nbig,beta),Time2FreqF(SigmaODtau,Nbig,beta)
	PiDomega, PiODomega =  Time2FreqB(PiDtau,Nbig,beta), Time2FreqB(PiODtau,Nbig,beta)
	
	detG = (1j*omega+mu-SigmaDomega)**2 - (lamb - SigmaODomega)**2
	detD = (nu**2 + r - PiDomega)**2 - (J - PiODomega)**2
	GDomega = ((1j*omega + mu - SigmaDomega)/(detG)) 
	GODomega = (-1.*(lamb- SigmaODomega)/(detG)) 
	DDomega = ((nu**2 + r - PiDomega)/(detD))
	DODomega = (-1.*(J- PiODomega)/(detD)) 

	loadedGDomega = Time2FreqF(GDtau, Nbig, beta)
	loadedGODomega = Time2FreqF(GODtau, Nbig, beta)
	loadedDDomega = Time2FreqB(DDtau, Nbig, beta)
	loadedDODomega = Time2FreqB(DODtau, Nbig, beta)

	twonorm = np.sum(np.abs(GDomega-loadedGDomega)**2) + np.sum(np.abs(GODomega-loadedGODomega)**2)
	twonorm += np.sum(np.abs(DDomega-loadedDDomega)**2) + np.sum(np.abs(DODomega-loadedDODomega)**2)
	return twonorm


def main():
	PLOTTING = False
	Nbig = int(2**14)

	beta_start = 1 
	target_beta = 10001
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

	FEstemp = np.zeros((len(betasavelist),len(lambsavelist)))
	FEslamb = np.zeros((len(betasavelist),len(lambsavelist)))
	ERRtemp = np.zeros((len(betasavelist),len(lambsavelist)))
	ERRlamb = np.zeros((len(betasavelist),len(lambsavelist)))

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

			GFstemp = np.load(plotfiletemp) #GFtaus
			GFslamb = np.load(plotfilelamb)
			impose_saddle = True
			FEslamb[i,j] = free_energy_YSYKWH(GFslamb, freq_grids, Nbig, beta, g, r, mu, kappa,lamb,J,impose_saddle = impose_saddle)
			FEstemp[i,j] = free_energy_YSYKWH(GFstemp, freq_grids, Nbig, beta, g, r, mu, kappa, lamb,J,impose_saddle = impose_saddle)
			ERRlamb[i,j] = calcerr(GFslamb,Nbig, beta, g, r, mu, kappa,lamb,J,freq_grids=freq_grids)
			ERRtemp[i,j] = calcerr(GFstemp,Nbig, beta, g, r, mu, kappa,lamb,J,freq_grids=freq_grids)


	# residuals = FEstemp-FEslamb
	# print(np.array2string(residuals,precision=4,floatmode='fixed'))
	print('ERRlamb')
	print(ERRlamb)
	print('ERRtemp')
	print(ERRtemp)


	fig,ax = plt.subplots(1)
	fig.suptitle('Phase diagram')
	ax.set_ylabel(r'$T$')
	ax.set_xlabel(r'$\lambda$')
	for i, beta in enumerate(betasavelist):
		temp = 1./beta 
		for j, lamb in enumerate(lambsavelist):
			if FEslamb[i,j] < FEstemp[i,j]: #WH
				ax.scatter(lamb,temp,c='r',marker='*',label='WH')
			elif FEslamb[i,j] > FEstemp[i,j]: #NFL
				ax.scatter(lamb,temp,c='b',marker='^',label='NFL')
			else:
				ax.scatter(lamb,temp,c='g',marker='o')

	handles, labels = ax.get_legend_handles_labels()
	unique_labels = list(set(labels))
	unique_handles = [handles[labels.index(label)] for label in unique_labels]
	ax.legend(unique_handles,unique_labels)
	#ax.legend()
	plt.show()



if __name__ == '__main__':
	main()















