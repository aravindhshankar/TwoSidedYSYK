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

# err = 1e-10
# err = 1e-8
err = 1e-60
ITERMAX = 50000


def anneal_temp(target_beta,GFtaus,Nbig,beta_start,beta_step,g,r,mu,lamb,J,kappa,DUMP=False,path_to_dump=None,savelist=None,calcfe=False,verbose=True):
	if verbose: 
		print("############ Started : target beta = , ", target_beta, " #############")
	GDtau,GODtau,DDtau,DODtau = GFtaus
	beta = beta_start

	fe_list = [] #could be slow - try prealloacating for speed later
	fe = 0.0

	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

	while(beta != target_beta):
		itern = 0
		diff = err*1.1
		diffG = 1.
		diffD = 1.
		x = 0.01
		# x = 0.5 if beta < 5 else 0.1
		#beta_step = 1 if (beta>=500) else 1

		omega = ImagGridMaker(Nbig,beta,'fermion')
		nu = ImagGridMaker(Nbig,beta,'boson')
		tau = ImagGridMaker(Nbig,beta,'tau')
		
		diff = 1.
		iterni=0
		conv_flag = False
		# while(conv_flag == False): 
		while(diff>err and itern < ITERMAX): 
			diffold = diff
			# if diff < 10*err :
			# 	x = 1.
			if itern == ITERMAX - 1: 
				print(f"WARNING : CONVERGENCE NOT REACHED FOR BETA = {beta}, LAMB = {lamb} in TEMP ANNEAL")
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
			# if (diff>err or np.isclose(x,1.)==False) and itern < ITERMAX: #means you need to iterate more 
			# 	if diff > diffold  :
			# 		x = max(x*0.5, 0.01)
			# 	elif diff < diffold:
			# 		x = min(x*1.1, 1)
				# if diff < 1e-8 and x > 0.5:
				# 	x = 1.
				# if x > 0.9:
				# 	x = 1.
				# if diff < err: 
				# 	x = 1.
			# else:
			# 	conv_flag = True
			# if diff < 1e-8 and x > 0.5:
			# 	x = 1.
			# if x > 0.9:
			# 	x = 1.
			# if diff < err: 
			# 	x = 1.

		if calcfe == True:
			GFs = [GDomega,GODomega,DDomega,DODomega]
			BSEs = [PiDomega,PiODomega]
			freq_grids = [omega,nu] #used for free energy calculation
			fe = free_energy_rolling_YSYKWH(GFs,BSEs,freq_grids,Nbig,beta,g,r,mu,kappa)
			fe_list += [fe]

		if DUMP == True and np.isclose(beta,savelist).any():
			savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
			savefile += 'lamb' + str(lamb) + 'J' + str(J)
			savefile += 'g' + str(g) + 'r' + str(r)
			savefile = savefile.replace('.','_') 
			savefile += '.npy'
			np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,GODtau,DDtau,DODtau])) 
			if verbose==True: 
				print(savefile)

		if verbose == True :
			print("##### Finished beta = ", beta, "############")
			#print("end x = ", x, " , end diff = ", diff,' , end itern = ',itern, '\n')
			print(f'diff = {diff:.5}, itern = {itern}, free energy = {fe:.5}, x = {x:.5}',flush=True)
		beta = beta + beta_step

	beta -= beta_step 
	return GDtau,GODtau,DDtau,DODtau,fe_list






def anneal_lamb(lamb_list,GFtaus,Nbig,g,r,mu,beta,J,kappa,DUMP=False,path_to_dump=None,savelist=None,calcfe=False,verbose=True):
	if verbose: 
		print("############ Started : target lamb = , ", lamb_list[-1], " #############")
	GDtau,GODtau,DDtau,DODtau = GFtaus

	fe_list = [] #could be slow - try prealloacating for speed later
	fe = 0.0

	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'
	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')

	for lamb in lamb_list:
		itern = 0
		diff = err*1.1
		diffG = 1.
		diffD = 1.
		# x = 0.01
		# x = 0.5 if beta < 10 else 0.01
		# x = 0.5 if beta < 5 else 1.
		# x = 0.5 if beta < 5 else 0.01
		x = 0.01
		diff = 1.
		iterni=0
		conv_flag = False
		# while(conv_flag == False):
		while(diff>err and itern < ITERMAX):
			# if diff < 10*err and itern > 2:
			# 	x = 1.
			diffold = diff
			if itern == ITERMAX - 1: 
				print(f"WARNING : CONVERGENCE NOT REACHED FOR BETA = {beta}, LAMB = {lamb} in LAMB ANNEAL")
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
			# if (diff>err or np.isclose(x,1.)==False) and itern < ITERMAX:
			# 	if diff > diffold and x*0.5 > 0.01 :
			# 		x *= 0.5
			# 	elif diff < diffold and x*1.1 <= 1. :
			# 		x *= 1.1
			# 	if diff < 1e-8 and x > 0.5:
			# 		x = 1.
			# 	if x > 0.9:
			# 		x = 1.
			# 	if diff < err: 
			# 		x = 1.
			# else: 
			# 	conv_flag = True
			# if diff < 1e-8 and x > 0.5:
			# 	x = 1.
			# if x > 0.9:
			# 	x = 1.
			# if diff < err: 
			# 	x = 1.

		if calcfe == True:
			GFs = [GDomega,GODomega,DDomega,DODomega]
			BSEs = [PiDomega,PiODomega]
			freq_grids = [omega,nu] #used for free energy calculation
			fe = free_energy_rolling_YSYKWH(GFs,BSEs,freq_grids,Nbig,beta,g,r,mu,kappa)
			fe_list += [fe]

		if DUMP == True and np.isclose(savelist,lamb).any():
			lambval = savelist[np.isclose(savelist,lamb)][0]
			savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
			savefile += 'lamb' + f'{lambval:.3}' + 'J' + str(J)
			savefile += 'g' + str(g) + 'r' + str(r)
			savefile = savefile.replace('.','_') 
			savefile += '.npy'
			np.save(os.path.join(path_to_dump, savefile), np.array([GDtau,GODtau,DDtau,DODtau])) 
			if verbose==True: 
				print(savefile)

		if verbose == True :
			print(f"##### Finished lamb =  {lamb} ############")
			print(f'diff = {diff:.5}, itern = {itern}, free energy = {fe:.5}, x = {x:.3}',flush=True)
	
	return GDtau,GODtau,DDtau,DODtau,fe_list


################################### Tests ###########################################

def test_anneal_temp():
	Nbig = int(2**14)
	beta_start = 1 
	target_beta = int(1e4)
	beta = beta_start
	mu = 0.0
	g = 0.5
	r = 1.
	lamb = 0.005
	J = 0
	kappa = 1.
	DUMP = False
	beta_step = 1 
	savelist = [50,100,500,1000,5000,10000]
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

	GDtau,GODtau,DDtau,DODtau,fe_list = anneal_temp(target_beta,GFtaus,Nbig,beta_start,beta_step,g,r,mu,lamb,J,kappa,
											DUMP=DUMP,path_to_dump=None,calcfe=False,verbose=True)

	beta = target_beta
	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')

	Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
	Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
	FreeDtau = DfreeImagtau(tau,r,beta)

	fig, ax = plt.subplots(2,2)

	titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
	titlestring += r' $\lambda$ = ' + str(lamb) + r' J = ' + str(J)
	fig.suptitle(titlestring)
	fig.tight_layout(pad=2)
	ax[0,0].plot(tau/beta, np.real(GDtau), 'r', label = 'numerics GDtau')
	ax[0,0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
	#ax[0,0].set_ylim(-1,1)
	ax[0,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,0].set_ylabel(r'$\Re{GD(\tau)}$')
	ax[0,0].legend()

	ax[0,1].plot(tau/beta, np.real(GODtau), 'r', label = 'numerics Real GODtau')
	ax[0,1].plot(tau/beta, np.imag(GODtau), 'k', label = 'numerics imag GODtau')
	#ax[0,1].set_ylim(-1,1)
	ax[0,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,1].set_ylabel(r'$\Re{GOD(\tau)}$')
	ax[0,1].legend()

	ax[1,0].plot(tau/beta, np.real(DDtau), 'r', label = 'numerics DDtau')
	ax[1,0].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
	ax[1,0].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
	#ax[1,0].set_ylim(0,1)
	ax[1,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,0].set_ylabel(r'$\Re{DD(\tau)}$')
	ax[1,0].legend()

	ax[1,1].plot(tau/beta, np.real(DODtau), 'r', label = 'numerics real DODtau')
	ax[1,1].plot(tau/beta, np.imag(DODtau), 'k', label = 'numerics imag DODtau')
	#ax[1,1].set_ylim(0,1)
	ax[1,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,1].set_ylabel(r'$\Re{DOD(\tau)}$')
	ax[1,1].legend()

	fig,ax = plt.subplots(1)
	fig.suptitle('Free energy as a function of temp')
	# ax.plot(np.arange(beta_start,target_beta,beta_step), fe_list)
	ax.plot(np.arange(beta_start,target_beta+1), fe_list)
	ax.set_ylabel('Free energy')
	ax.set_xlabel(r'$\beta')

	plt.show()



def test_anneal_lamb():
	Nbig = int(2**14)
	beta = int(1e3)
	mu = 0.0
	g = 0.5
	r = 1.
	#lamb = 0.005
	lamb_list = np.arange(1,0.001 - 1e-10,-0.001) # IT would be be nice if this can be rounded 
	lamb = lamb_list[0]
	savelist = []
	J = 0
	kappa = 1.
	DUMP = False
	calcfe = True

	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')

	Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
	Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
	delta = 0.420374134464041
	omegar2 = ret_omegar2(g,beta)

	#GDtau, DDtau = Gfreetau, Dfreetau
	GDtau = Freq2TimeF((1j*omega + mu)/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
	DDtau = Freq2TimeB((nu**2+r)/((nu**2 + r)**2 - J**2), Nbig, beta)
	GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
	DODtau = Freq2TimeB(-J/(nu**2 + r)**2 - J**2, Nbig, beta)
	GFtaus = [GDtau,GODtau,DDtau,DODtau]

	GDtau,GODtau,DDtau,DODtau,fe_list = anneal_lamb(lamb_list,GFtaus,Nbig,g,r,mu,beta,J,kappa,
						DUMP=False,path_to_dump=None,savelist=None,calcfe=calcfe,verbose=True)

	lamb = lamb_list[-1]
	Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
	Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
	FreeDtau = DfreeImagtau(tau,r,beta)

	fig, ax = plt.subplots(2,2)

	titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
	titlestring += r' $\lambda$ = ' + str(lamb) + r' J = ' + str(J)
	fig.suptitle(titlestring)
	fig.tight_layout(pad=2)
	ax[0,0].plot(tau/beta, np.real(GDtau), 'r', label = 'numerics GDtau')
	ax[0,0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
	#ax[0,0].set_ylim(-1,1)
	ax[0,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,0].set_ylabel(r'$\Re{GD(\tau)}$')
	ax[0,0].legend()

	ax[0,1].plot(tau/beta, np.real(GODtau), 'r', label = 'numerics Real GODtau')
	ax[0,1].plot(tau/beta, np.imag(GODtau), 'k', label = 'numerics imag GODtau')
	#ax[0,1].set_ylim(-1,1)
	ax[0,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,1].set_ylabel(r'$\Re{GOD(\tau)}$')
	ax[0,1].legend()

	ax[1,0].plot(tau/beta, np.real(DDtau), 'r', label = 'numerics DDtau')
	ax[1,0].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
	ax[1,0].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
	#ax[1,0].set_ylim(0,1)
	ax[1,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,0].set_ylabel(r'$\Re{DD(\tau)}$')
	ax[1,0].legend()

	ax[1,1].plot(tau/beta, np.real(DODtau), 'r', label = 'numerics real DODtau')
	ax[1,1].plot(tau/beta, np.imag(DODtau), 'k', label = 'numerics imag DODtau')
	#ax[1,1].set_ylim(0,1)
	ax[1,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,1].set_ylabel(r'$\Re{DOD(\tau)}$')
	ax[1,1].legend()

	if calcfe:
		fig,ax = plt.subplots(1)
		fig.suptitle('Free energy as a function of temp')
		# ax.plot(np.arange(beta_start,target_beta,beta_step), fe_list)
		ax.plot(lamb_list, fe_list)
		ax.set_ylabel('Free energy')
		ax.set_xlabel(r'$\lambda')

	plt.show()




if __name__ == '__main__': 
	test_anneal_temp()
	# test_anneal_lamb()









