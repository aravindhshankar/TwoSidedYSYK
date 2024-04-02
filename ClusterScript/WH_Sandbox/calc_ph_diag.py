#### TODO: make this after 
# 1) Fixing plotter 
# 2) test loop in phase space between FF1 and FF2
#

import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


if not os.path.exists('../Dump/WHYSYKImagDumpfiles'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
else:
    path_to_dump = '../Dump/WHYSYKImagDumpfiles'


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_rolling_YSYKWH 
from annealers import anneal_temp

DUMP = False
Nbig = int(2**14)

#global beta

beta_start = 1 
target_beta = 1000
betas = [beta_start, target_beta]

beta = beta_start
mu = 0.0
g = 0.5
r = 1.
lamb = 0.05
J = 0


print("############ Started : target beta = , ", target_beta, " #############")

# g = np.sqrt(10**3)
# r = (10)**2

kappa = 1.
beta_step = 1

num = 1.1 

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')

Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)


GDtau, DDtau = Gfreetau, Dfreetau
GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
DODtau = Freq2TimeB(-J/(nu**2 + r)**2 - J**2, Nbig, beta)





























