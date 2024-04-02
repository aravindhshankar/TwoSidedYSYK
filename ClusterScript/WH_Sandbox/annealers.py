#### Annealing down in temperature starting from any high temperature 
#### Always passes through the NFL before entering WH phase => NOT GOOD!  
#### Only way to reach WH is to start at low temp FF2 and anneal to small lamb at fixed temp

import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_rolling_YSYKWH 
#import time

err = 1e-6
ITERMAX = 5000


def anneal_temp(GFtaus,Nbig,betas,g,r,mu,lamb,J,kappa,DUMP=False,path_to_dump=None,calcfe=True,verbose=True):
	if verbose: print("############ Started : target beta = , ", target_beta, " #############")
	GDtau,GODtau,DDtau,DODtau = GFtaus
	beta_start, target_beta = betas
	beta = beta_start

	fe_list = [] #could be slow - try prealloacating for speed later


	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

	while(beta < target_beta):
	    itern = 0
	    diff = err*1.1
	    diffG = 1.
	    diffD = 1.
	    x = 0.01
	    beta_step = 1 if (beta>=500) else 1

	    omega = ImagGridMaker(Nbig,beta,'fermion')
	    nu = ImagGridMaker(Nbig,beta,'boson')
	    tau = ImagGridMaker(Nbig,beta,'tau')
	    
	    diff = 1.
	    iterni=0
	    while(diff>err and itern < ITERMAX):
	        itern+=1
	        iterni += 1 

	        oldGDtau, oldGODtau = 1.0*GDtau, 1.0*GODtau
	        oldDDtau, oldDODtau = 1.0*DDtau, 1.0*DODtau
	        
	        if iterni == 1:
	            oldGDomega,oldGODomega = Time2FreqF(oldGDtau,Nbig,beta),Time2FreqF(oldGODtau,Nbig,beta)
	            oldDDomega, oldDODomega = Time2FreqB(oldDDtau,Nbig,beta),Time2FreqB(oldDODtau,Nbig,beta)
	        else:
	            oldGDomega, oldGODomega = 1.0*GDomega, 1.0*GODomega
	            oldDDomega, oldDODomega = 1.0*DDomega, 1.0*DODomega
	        
	        SigmaDtau = 1.0 * kappa * (g**2) * DDtau * GDtau
	        SigmaODtau = 1.0 * kappa * (g**2) * DODtau * GODtau
	        PiDtau = 2.0 * g**2 * GDtau * GDtau[::-1] #KMS G(-tau) = -G(beta-tau)
	        PiODtau = 2.0 * g**2 * GODtau * GODtau[::-1] #KMS G(-tau) = -G(beta-tau)
	        
	        SigmaDomega, SigmaODomega = Time2FreqF(SigmaDtau,Nbig,beta),Time2FreqF(SigmaODtau,Nbig,beta)
	        PiDomega, PiODomega =  Time2FreqB(PiDtau,Nbig,beta), Time2FreqB(PiODtau,Nbig,beta)
	        
	        detG = (1j*omega+mu-SigmaDomega)**2 - (lamb - SigmaODomega)**2
	        detD = (nu**2 + r - PiDomega)**2 - (J - PiODomega)**2
	        GDomega = x*((1j*omega + mu - SigmaDomega)/(detG)) + (1-x)*oldGDomega
	        GODomega = x*(-1.*(lamb- SigmaODomega)/(detG)) + (1-x)*oldGODomega
	        DDomega = x*((nu**2 + r - PiDomega)/(detD)) + (1-x)*oldDDomega
	        DODomega = x*(-1.*(J- PiODomega)/(detD)) + (1-x)*oldDODomega

	        GDtau = Freq2TimeF(GDomega,Nbig,beta)
	        GODtau = Freq2TimeF(GODomega,Nbig,beta)
	        DDtau = Freq2TimeB(DDomega,Nbig,beta)
	        DODtau = Freq2TimeB(DODomega,Nbig,beta)

	    
	        diffGD = np.sum((np.abs(GDtau-oldGDtau))**2) 
	        diffDOD = np.sum((np.abs(DODtau-oldDODtau))**2)
	        diff = 0.5*(diffGD+diffDOD) # less stringent error metric - faster code
	        # diffGD = np.sqrt(np.sum(np.abs(SigmaDtau - 1.0*kappa*(g**2)*DDtau*GDtau)**2))
	        # diffDOD = np.sqrt(np.sum(np.abs(PiODtau - 2.0 * g**2 * GODtau * GODtau[::-1])**2))
	        # diff = 0.5*(diffGD+diffDOD)

	    if calcfe == True:
		    GFs = [GDomega,GODomega,DDomega,DODomega]
		    BSEs = [PiDomega,PiODomega]
		    freq_grids = [omega,nu] #used for free energy calculation
		    fe = free_energy_rolling_YSYKWH(GFs,BSEs,freq_grids,Nbig,beta,g,r,mu,kappa)
		    fe_list += [fe]

	    if DUMP == True and beta in [50,100,500,1000,2000,5000,10000,50000,100000]:
	        savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	        savefile += 'lamb' + str(lamb) + 'J' + str(J)
	        savefile += 'g' + str(g) + 'r' + str(r)
	        savefile = savefile.replace('.','_') 
	        savefile += '.npy'
	        np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,GODtau,DDtau,DODtau])) 
	        if verbose==True: print(savefile)

	    if verbose == True :
	    	print("##### Finished beta = ", beta, "############")
	    	#print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n')
	    	print("diff = ", diff,' , itern = ',itern, " , free energy = ", fe)
	    beta = beta + beta_step

    return GDtau,GODtau,DDtau,DODtau,fe_list







	

















def anneal_lamb():
	return None















