import sys ####
import os 

from SYK_fft import *
import numpy as np

#from ConformalAnalytical import *
#import time


def free_energy_YSYKWH(GFs, freq_grids, Nbig, beta, g, r, mu, kappa, lamb, J, impose_saddle=True):
	'''
	Used to calculate free energy after loading Gtaus from file
	Signature : free_energy_YSYKWH(GFs, freq_grids, Nbig, beta, g, r, mu, kappa)
	GDtau, GODtau, DDtau, DODtau = GFs
	omega,nu = freq_grids

	'''
	GDtau, GODtau, DDtau, DODtau = GFs
	omega,nu = freq_grids

	np.testing.assert_almost_equal(omega[2] - omega[1], 2*np.pi/beta)
	np.testing.assert_almost_equal(nu[2] - nu[1], 2*np.pi/beta)
	np.testing.assert_equal(Nbig, len(DDtau))

	DDomega = Time2FreqB(DDtau, Nbig, beta)
	DODomega = Time2FreqB(DODtau, Nbig, beta)

	PiDtau = 2.0 * g**2 * GDtau * GDtau[::-1] 
	PiODtau = 2.0 * g**2 * GODtau * GODtau[::-1] 
	PiDomega = Freq2TimeB(PiDtau,Nbig,beta) 
	PiODomega = Freq2TimeB(PiODtau,Nbig,beta) 
	GDomega = Time2FreqF(GDtau, Nbig, beta)
	GODomega = Time2FreqF(GODtau, Nbig, beta)	
	if impose_saddle == True:
		detGinv = 1./(GDomega**2 - GODomega**2) #Was + in earlier version of code: mistake!
		detDinv = 1./(DDomega**2 - DODomega**2)
	else: 
		SigmaDomega = Time2FreqF(g**2 * kappa* GDtau * DDtau, Nbig,beta)
		SigmaODomega = Time2FreqF(g**2 * kappa * GODtau * DODtau, Nbig, beta)
		# detGinv = (1j*omega+mu-SigmaDomega)**2 - (lamb-SigmaODomega)**2
		detGinv = (1j*omega+mu-SigmaDomega)**2 - (lamb+SigmaODomega)**2
		detDinv = (nu**2+r-PiDomega)**2 - (J - PiODomega)**2

	# detG0inv = (1j*omega+mu)**2 - (lamb)**2
	# detD0inv = (nu**2+r)**2 - (J)**2
	detG0inv = (1j*omega+mu)**2 
	detD0inv = (nu**2+ r)**2 
	detGinv = detGinv/detG0inv
	detDinv = detDinv/detD0inv
	# free_energy = 2*np.log(2)-np.sum(np.log(detGinv))
	free_energy = -2*np.log(2)-np.sum(np.log(detGinv))
	free_energy += 0.5*kappa*np.sum(np.log(detDinv)) 
	free_energy += 1.0*kappa*(np.sum(DDomega*PiDomega) + np.sum(DODomega*PiODomega)) #changed
	free_energy += 0.5*(np.sqrt(r)*beta + 2*np.log(1- np.exp(-1.0*beta*np.sqrt(r)))) #From Valentinis, Inkof, Schmalian
	# free_energy += -2*(beta**2)*kappa*(g**2)/Nbig * (np.sum(DDtau*GDtau*GDtau[::-1])+np.sum(DODtau*GODtau*GODtau[::-1]))
	# free_energy += -2*(np.sum(GDomega*SigmaDomega) + np.sum(GODomega*SigmaODomega))
	free_energy = np.real(free_energy) / beta
	# free_energy = free_energy.real 

	return free_energy

def free_energy_rolling_YSYKWH(GFs,BSEs,freq_grids,Nbig,beta,g,r,mu,kappa,tests=True):
	'''
	Here GFs are frequency green functions, SEs are frequency bosonic self energies
	NOTE: GDomega,GODomega,DDomega,DODomega = GFs
	PiDomega,PiODomega = SEs
	omega,nu = freq_grids
	'''
	GDomega,GODomega,DDomega,DODomega = GFs
	PiDomega,PiODomega = BSEs
	omega,nu = freq_grids

	if tests == True:
		np.testing.assert_almost_equal(omega[2] - omega[1], 2*np.pi/beta)
		np.testing.assert_almost_equal(nu[2] - nu[1], 2*np.pi/beta)
		np.testing.assert_equal(Nbig, len(PiDomega))
		np.testing.assert_equal(Nbig, len(GDomega))
		
	detGinv = 1./(GDomega**2 - GODomega**2)
	detDinv = 1./(DDomega**2 - DODomega**2)

	free_energy = 2*np.log(2)-np.sum(np.log(detGinv/((1j*omega + mu)**2)))
	free_energy += 0.5*kappa*np.sum(np.log((detDinv)/((nu**2+r)**2))) 
	free_energy += np.sum(DDomega*PiDomega) + np.sum(DODomega*PiODomega)
	free_energy = free_energy.real / beta

	return free_energy



def FUNCTIONALfree_energy_YSYKWH(GFs, freq_grids, Nbig, beta, g, r, mu, kappa, lamb, J, impose_saddle=False):
	'''
	Used to calculate free energy after loading Gtaus from file
	Signature : free_energy_YSYKWH(GFs, freq_grids, Nbig, beta, g, r, mu, kappa)
	GDtau, GODtau, DDtau, DODtau = GFs
	omega,nu = freq_grids

	'''
	GDtau, GODtau, DDtau, DODtau = GFs
	omega,nu = freq_grids

	np.testing.assert_almost_equal(omega[2] - omega[1], 2*np.pi/beta)
	np.testing.assert_almost_equal(nu[2] - nu[1], 2*np.pi/beta)
	np.testing.assert_equal(Nbig, len(DDtau))

	DDomega = Time2FreqB(DDtau, Nbig, beta)
	DODomega = Time2FreqB(DODtau, Nbig, beta)

	PiDtau = 2.0 * g**2 * GDtau * GDtau[::-1] 
	PiODtau = 2.0 * g**2 * GODtau * GODtau[::-1] 
	PiDomega = Freq2TimeB(PiDtau,Nbig,beta) 
	PiODomega = Freq2TimeB(PiODtau,Nbig,beta) 

	if impose_saddle == True:
		GDomega = Time2FreqF(GDtau, Nbig, beta)
		GODomega = Time2FreqF(GODtau, Nbig, beta)
		detGinv = 1./(GDomega**2 - GODomega**2) #Was + in earlier version of code: mistake!
		detDinv = 1./(DDomega**2 - DODomega**2)
	else: 
		SigmaDomega = Time2FreqF(g**2 * kappa* GDtau * DDtau, Nbig,beta)
		SigmaODomega = Time2FreqF(g**2 * kappa * GODtau * DODtau, Nbig, beta)
		detGinv = (1j*omega+mu-SigmaDomega)**2 - (lamb-SigmaODomega)**2
		detDinv = (nu**2+r-PiDomega)**2 - (J - PiODomega)**2

	detGinv = detGinv.real
	detDinv = detDinv.real
	free_energy = 2*np.log(2)-np.sum(np.log(detGinv/((1j*omega + mu)**2)))
	free_energy += 0.5*kappa*np.sum(np.log((detDinv)/((nu**2+r)**2))) 
	free_energy += 1.0*kappa*(np.sum(DDomega*PiDomega) + np.sum(DODomega*PiODomega)) #changed
	free_energy = free_energy.real / beta

	return free_energy




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
























######################## tests ##########################


def free_energy_test1():
	Gomega = 1./(-1j*omega - mu)
	mats_sum = -np.sum(np.log(Gomega**2))
	log2val = np.log(2)
	reg_mats_sum = -2*np.sum(np.log(Gomega))
	print('beta = ', beta)
	print('range of |omega| is from ', omega[0], ' to ', omega[-1], ' with dw = ', omega[2]-omega[1])
	print('mast_sum = ', mats_sum)
	print('regularized mats_sum = ', reg_mats_sum)
	print('log2 = ', log2val)
	print(np.sum(np.log(1j*omega)))
	print(omega[Nbig//2 - 2 : Nbig//2 + 2 ])

	for cutoff in 2**np.arange(1,10, dtype = int):
		chopslice = slice(Nbig//2 - cutoff, Nbig//2 + cutoff)
		print("cutoff ", cutoff, "    :     " ,-2*np.sum(np.log(Gomega[chopslice])))

def free_energy_test2():
	Nbig = int(2**14)
	err = 1e-5
	#err = 1e-2

	beta_start = 1000
	beta = beta_start
	#mu = 0.0
	mu = 1e-6
	g = 0.5
	r = 1.

	target_beta = 50.

	kappa = 1.
	beta_step = 1
	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')


	if not os.path.exists('../Dump/WHYSYKImagDumpfiles'):
		print("Error - Path to Dump directory not found ")
		raise Exception("Error - Path to Dump directory not found ")
	else:
		path_to_dump = '../Dump/WHYSYKImagDumpfiles'
	   
	try :
		plotfile = os.path.join(path_to_dump, 'Nbig14beta1000_0lamb0_05J0_05g0_5r1_0.npy')
		#plotfile = os.path.join(path_to_dump, savefile)
	except FileNotFoundError: 
		print("INPUT FILE NOT FOUND")
		exit(1)
	GDtau, GODtau, DDtau, DODtau = np.load(plotfile)
	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

	GDomega = Time2FreqF(GDtau, Nbig, beta)
	GODomega = Time2FreqF(GODtau, Nbig, beta)
	DDomega = Time2FreqB(DDtau, Nbig, beta)
	DODomega = Time2FreqB(DODtau, Nbig, beta)	

	# GDomega = 1./(1j*omega+mu)
	# GODomega = np.zeros_like(GDomega)
	# DDomega = 1./(nu**2 + r)
	# DODomega = np.zeros_like(DDomega)


	detGinv = 1./(GDomega**2 - GODomega**2)
	detDinv = 1./(DDomega**2 - DODomega**2)



	fs = []
	fs.append(np.sum(np.log(detGinv/((1j*omega + mu)**2))))
	fs.append(np.sum(np.log(detGinv)))
	fs.append(np.sum(np.log((1j*omega)**2)))
	#print(fs)

	bs = []
	bs.append(np.sum(np.log((detDinv)/((nu**2+r**2)**2))))
	bs.append(np.sum(np.log(detDinv)))
	bs.append(np.sum(np.log((nu**2 + r**2)**2)))
	#print(bs)

	PiDtau = 2.0 * g**2 * GDtau * GDtau[::-1] 
	PiODtau = 2.0 * g**2 * GODtau * GODtau[::-1] 
	PiDomega = Freq2TimeB(PiDtau,Nbig,beta) 
	PiODomega = Freq2TimeB(PiODtau,Nbig,beta) 


	f_arr = [2*np.log(2) , -1*fs[0].real , 0.5*bs[0].real] 
	f_arr.append(np.sum(DDomega*PiDomega))
	f_arr.append(np.sum(DODomega*PiODomega))
	print(f_arr)
	f = np.sum(f_arr)/beta
	print('total free energy = ', f)

	free_energy = 2*np.log(2)-np.sum(np.log(detGinv/((1j*omega + mu)**2)))
	free_energy += 0.5*kappa*np.sum(np.log((detDinv)/((nu**2+r**2)**2))) 
	free_energy += np.sum(DDomega*PiDomega) + np.sum(DODomega*PiODomega)
	free_energy = free_energy.real / beta
	print('direct calculated free energy = ', free_energy)
	print(np.testing.assert_almost_equal(f,free_energy), np.testing.assert_almost_equal(2,2))

	GFs = [GDtau, GODtau, DDtau, DODtau]
	freq_grids = [omega,nu]
	fromfunc = free_energy_YSYKWH(GFs, Nbig, beta, g, r, mu, kappa, freq_grids)

	print('from function = ', fromfunc)
	print(np.testing.assert_almost_equal(fromfunc,f), np.testing.assert_almost_equal(2,2))



if __name__ == '__main__':
	free_energy_test2()























