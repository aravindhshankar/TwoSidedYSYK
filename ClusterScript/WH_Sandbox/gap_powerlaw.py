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
	path_to_dump_lamb = '../Dump/gap_powerlawx01_lamb_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/zoom_xshift_temp_anneal_dumpfiles/rev'
	if not os.path.exists(path_to_dump_lamb):
		print("Making directory for lamb dump")
		os.mkdir(path_to_dump_lamb)
		# print('Input File not found')
		# exit(1)
	# if not os.path.exists(path_to_dump_temp):
	# 	print("Making directory for temp dump")
	# 	os.mkdir(path_to_dump_temp)
	# 	# print('Input File not found')
	# 	# exit(1)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_YSYKWH 
from annealers import anneal_temp, anneal_lamb


calc = False
delta = 0.420374134464041

if calc == True:
	### TODO: Implement parallization
	path_to_dump = path_to_dump_lamb
	PLOTTING = False
	DUMP = True
	calcfe = False
	verbose = True


	global diff

	Nbig = int(2**14)

	beta_start = 1 
	target_beta = 1001
	beta = beta_start
	mu = 0.0
	g = 0.5
	r = 1.
	J = 0
	kappa = 1.
	beta_step = 1
	err = 1e-8
	ITERMAX = 2000


	betasavelist = np.array([100,])
	# lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])
	lambsavelist = np.linspace(0.001,1.,1000)
	lambsavelist = lambsavelist[::-1]
	lamblooplist = lambsavelist
	beta = betasavelist[0]


	omegar2 = ret_omegar2(g,beta)

	# GDtau, DDtau = Gfreetau, Dfreetau
	# GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
	# DODtau = Freq2TimeB(-J/(nu**2 + r)**2 - J**2, Nbig, beta)
	# GFtaus = [GDtau,GODtau,DDtau,DODtau]

	gaplist = []
	# GFtaus = [GDtau,GODtau,DDtau,DODtau]
	startT, stopT  = 0, 5000
	fitsliceT = slice(startT+4500, startT + 4600)
	# startT, stopT  = Nbig//2 - 60, Nbig//2 - 30
	# fitsliceT = slice(startT, stopT)
	#Step2: anneal in lamb for all betas
	for beta in betasavelist:
		lamb = lamblooplist[0]
		if verbose: 
				print("############ Started : target lamb = , ", lambsavelist[-1], " #############")
			# GDtau,GODtau,DDtau,DODtau = GFtaus

		fe_list = [] #could be slow - try prealloacating for speed later
		fe = 0.0

		# assert len(GDtau) == Nbig, 'Improperly loaded starting guess'
		omega = ImagGridMaker(Nbig,beta,'fermion')
		nu = ImagGridMaker(Nbig,beta,'boson')
		tau = ImagGridMaker(Nbig,beta,'tau')
		GDtau = Freq2TimeF((1j*omega + mu)/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
		DDtau = Freq2TimeB((nu**2+r)/((nu**2 + r)**2 - J**2), Nbig, beta)
		GODtau = Freq2TimeF(-lamb/((1j*omega+mu)**2 - lamb**2), Nbig, beta)
		DODtau = Freq2TimeB(-J/(nu**2 + r)**2 - J**2, Nbig, beta)

		for lamb in lamblooplist:
			itern = 0
			diff = err*1.1
			diffG = 1.
			diffD = 1.
			# x = 0.01
			# x = 0.5 if beta < 10 else 0.01
			# x = 0.5 if beta < 5 else 1.
			# x = 0.5 if beta < 5 else 0.01
			# x = 0.1
			# diff = 1.
			iterni=0
			conv_flag = False
			# while(conv_flag == False):
			for x in [0.1,]:
				diff = 1.
				# err = 1e-6 if x < 0.5 else 1e-8
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

			if DUMP == True and np.isclose(lambsavelist,lamb).any():
				lambval = lambsavelist[np.isclose(lambsavelist,lamb)][0]
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
				print(f'diff = {diff:.5}, itern = {itern}, free energy = {fe:.5}, x = {x:.3}')

			functoplotT = np.abs(np.real(GDtau))
			mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
			gaplist += [-1.*mT]


	gaplist = np.array(gaplist)
	np.save('beta100lambgaplist.npy',np.array([lamblooplist,gaplist]))





else: #if calc == False:
	try:
		lamblooplist, gaplist = np.load('beta100lambgaplist.npy')
	except FileNotFoundError: 
		print('Gaplist not found!')
		exit(1)


lamblooplist = lamblooplist[::-1]
gaplist = gaplist[::-1]


m1,c1 = np.polyfit(np.log(np.abs(lamblooplist[50:60])), np.log(gaplist[50:60]),1)
m2,c2 = np.polyfit(np.log(np.abs(lamblooplist[6:20])), np.log(gaplist[6:20]),1)
gradslope = np.gradient(gaplist,lamblooplist)

beta = 100.
g= 0.5
slope_expect = 1./(2-2*delta)
print(f'Expected Slope = {slope_expect:.4}')
# print(f'Fit slope = {m}:.4')
fig,ax = plt.subplots(1)
ax.loglog(lamblooplist,gaplist,'.-')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('mass from exponential fit')
ax.loglog(lamblooplist, np.exp(c1)*np.abs(lamblooplist)**m1, label=f'Fit with slope {m1:.03f}')
ax.loglog(lamblooplist, np.exp(c2)*np.abs(lamblooplist)**m2, label=f'Fit with slope {m2:.03f}')
ax.axvline(1./beta,ls='--',c='gray',label='Temperature')
ax.axvline(g**(2./3.),ls='--',c='green',label=r'$g^{2/3}$')
ax.legend()

ax3 = ax.twinx()
ax3.loglog(lamblooplist, np.abs(gradslope), '.-', c = 'k', label=r'abs(Gradient)')
ax3.set_ylabel(r'grad slope with lambda')
ax3.legend()


fig.tight_layout()
fig.suptitle(f'beta = 100')

plt.show()


















