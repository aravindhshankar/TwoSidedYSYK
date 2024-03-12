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

beta_start = 100
beta = beta_start
mu = 0.0
#mu = 1e-6
g = 0.5
r = 1.

target_beta = 50.

kappa = 1.
beta_step = 1


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


def main():
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






if __name__ == '__main__':
	main()























