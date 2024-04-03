import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


# Make 2 directories one for NFL, one for WH, dump GFs there 
if not os.path.exists('../Dump/'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
    exit(1)
else:
	path_to_dump_lamb = '../Dump/lamb_anneal_dumpfiles/'
	path_to_dump_temp = '../Dump/temp_anneal_dumpfiles/'
	if not os.path.exists(path_to_dump_lamb):
		print("Making directory for lamb dump")
		os.mkdir(path_to_dump_lamb)
	if not os.path.exists(path_to_dump_temp):
		print("Making directory for temp dump")
		os.mkdir(path_to_dump_temp)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_rolling_YSYKWH 
from annealers import anneal_temp, anneal_lamb


### TODO: Implement parallization

PLOTTING = False
DUMP = True
Nbig = int(2**14)

beta_start = 1 
target_beta = 100001
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
lamb = 0.05
J = 0
kappa = 1.
beta_step = 1
betasavelist = [50,100,500,1000,5000,10000]
lambsavelist = [0.1,0.05,0.01,0.005,0.001]
lamblooplist = np.arange(1,0.001 - 1e-10,-0.001)

### Setting up initial conditions for temp annealer
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
GFtaus = [GDtau,GODtau,DDtau,DODtau]


#Step 1 : anneal in temperature for all lambs
for lamb in lambsavelist:
	_,_,_,_,_ = anneal_temp(target_beta,GFtaus,Nbig,beta_start,beta_step,
					g,r,mu,lamb,J,kappa,
						DUMP=DUMP,path_to_dump=path_to_dump_temp,savelist=betasavelist,
								calcfe=False,verbose=True)



beta = target_beta


#Step2: anneal in lamb for all betas


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')




freq_grids = [omega,nu]
fe = free_energy_YSYKWH(GFs, freq_grids, Nbig, beta, g, r, mu, kappa)
print(f'Free energy = {fe}')


if not PLOTTING:
	print("cal_ph_diag.py exiting without plotting")
	exit(0)































































