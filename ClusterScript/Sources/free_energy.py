import sys
import os 

from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
#from ConformalAnalytical import *
#import time


Nbig = int(2**14)
err = 1e-5
#err = 1e-2

global beta

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


def test1():
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

def test2():
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

	detGinv = 1./(GDomega**2 + GODomega**2)
	detDinv = 1./(DDomega**2 + DODomega**2)

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

	



if __name__ == '__main__':
	test2()























