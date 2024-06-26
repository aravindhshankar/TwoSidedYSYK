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
	path_to_dump_J = '../Dump/LOWTEMP_J_anneal_dumpfiles/'
	if not os.path.exists(path_to_dump_J):
		print("Making directory for J dump")
		os.mkdir(path_to_dump_J)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_rolling_YSYKWH 
from annealers import anneal_temp, anneal_lamb, anneal_J



PLOTTING = False
DUMP = True
Nbig = int(2**14)

beta_start = 1 
target_beta = 2000
beta = beta_start
mu = 0.0
g = 0.5
r = 1.
lamb = 0
# J = 0.05
J = 10.
kappa = 1.
beta_step = 1
# betasavelist = [50,100,500,1000,5000,10000]
betasavelist = [target_beta,]
Jlooplist = np.arange(1-1e-10,0.001 - 1e-10,-0.001)
# Jsavelist = [0.1,0.05,0.01,0.005,0.001]
# Jsavelist = np.arange(0.01,0.001 - 1e-10,-0.001)
Jsavelist = Jlooplist

# J = Jlooplist[0]


### Setting up initial conditions for temp annealer
# omega = ImagGridMaker(Nbig,beta,'fermion')
# nu = ImagGridMaker(Nbig,beta,'boson')
# tau = ImagGridMaker(Nbig,beta,'tau')
# Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
# Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
# delta = 0.420374134464041
# omegar2 = ret_omegar2(g,beta)

# GDtau, DDtau = Gfreetau, Dfreetau
# GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
# DODtau = Freq2TimeB(-J/((nu**2 + r)**2 - J**2), Nbig, beta)
# GFtaus = [GDtau,GODtau,DDtau,DODtau]


# #Step 1 : anneal in temperature for all lambs
# for lamb in Jsavelist:
# 	_,_,_,_,_ = anneal_temp(target_beta,GFtaus,Nbig,beta_start,beta_step,
# 					g,r,mu,lamb,J,kappa,
# 						DUMP=DUMP,path_to_dump=path_to_dump_temp,savelist=betasavelist,
# 								calcfe=False,verbose=True)



beta = target_beta


#Step2: anneal in J for all betas
for beta in betasavelist:
	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')
	GDtau = Freq2TimeF((1j*omega + mu)/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
	DDtau = Freq2TimeB((nu**2+r)/((nu**2 + r)**2 - J**2), Nbig, beta)
	GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
	DODtau = Freq2TimeB(-J/((nu**2 + r)**2 - J**2), Nbig, beta)
	GFtaus = [GDtau,GODtau,DDtau,DODtau]
	_,_,_,_,_ = anneal_J(Jlooplist,GFtaus,Nbig,g,r,mu,beta,lamb,kappa,
						DUMP=DUMP,path_to_dump=path_to_dump_J,savelist=Jsavelist,
							calcfe=False,verbose=True)






if not PLOTTING:
	print("PLOTTING TURNED OFF")
	exit(0)