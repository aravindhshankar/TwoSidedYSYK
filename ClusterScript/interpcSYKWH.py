import sys
import os 
if not os.path.exists('./Sources'):
	print("Error - Path to Sources directory not found ")
sys.path.insert(1,'./Sources')

import numpy as np
from matplotlib import pyplot as plt
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator 

savename = 'default_savename'
path_to_output = './Outputs/redoCWH'
path_to_dump  = './Dump/redoCWH'

if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
	print("Outputs directory created")

if not os.path.exists(path_to_dump):
	os.makedirs(path_to_dump)
	print("Dump directory created, ", path_to_dump)


if len(sys.argv) > 1:
	savename = str(sys.argv[1])

# savefile = os.path.join(path_to_output, savename+'.h5')
# savefile_dump = os.path.join(path_to_dump, savename+'.npy')


fft_check = testingscripts.realtimeFFT_validator() # Should return True

DUMP = True
PLOTTING = True

##################

def rhotosigma(rhoG,M,dt,t,omega,J,beta,kappa,delta=1e-6):
	'''
	c-SYK rho to sigma
	'''
	eta = np.pi/(M*dt)*(0.001)
	rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
	rhoFpp = (1/np.pi)*freq2time(rhoG * fermidirac(beta*omega),M,dt)
	rhoFpm = (1/np.pi)*freq2time(rhoG * fermidirac(-1.*beta*omega),M,dt)
	rhoFmp = (1/np.pi)*freq2time(rhoGrev * fermidirac(beta*(omega)),M,dt)
	rhoFmm = (1/np.pi)*freq2time(rhoGrev * fermidirac(-1.*beta*omega),M,dt)
	
	argSigma = (rhoFpp*rhoFpp*rhoFmp + rhoFpm*rhoFpm*rhoFmm) * np.heaviside(t,1)
	# argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.heaviside(t,1)
	Sigma = -1j*(J**2)* time2freq(argSigma,M,dt)

	return Sigma

###################

J = 1.
#beta = 100.
# beta = 1./(2e-4)
#beta = 1./(5e-5)
# beta = 500
# betalist = [20,50,100,200,500]
# betalist = [1000,2000,5000]
# betalist = [1000,2000,5000]
# betalist = [5000,6000,7000,10000,12000,20000]
betalist = [6000,6000]
mu = 0. 
# kappa = 0.05
kappa = 0.01
ITERMAX = 1000
# ITERMAX = 5

M = int(2**16) #number of points in the grid
T = int(2**12) #upper cut-off fot the time
# M = int(2**20) #number of points in the grid
# T = int(2**16) #upper cut-off fot the time
#M = int(2**16)
#T = int(2**10)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
# err = 1e-5
err = 1e-6
# err = 1e-7
eta = dw*2.1
#delta = 0.420374134464041
delta = 0.25

print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
# print("dw/temp = ", f'{dw*beta:.4f}')
print("flag fft_check = ", fft_check)
print("grid_flag = ", grid_flag)

## State varriables go into .out file
print("######## State Variables ################")
print("J = ", J)
print("mu = ", mu)
print("kappa = ", kappa)
# print("beta = ", beta)
print("log_2 M = ", np.log2(M))
print("eta = ", eta)
print("T = ", T)
print("err = ", err)
print("######## End of State variables #########")


	
def RE_wormhole_cSYK_iterator(GDRomega,GODRomega,J,mu,kappa,beta,omega,t,M,dt,ITERMAX = 1000, eta=1e-6,verbose=True):
	itern = 0
	diff = 1
	# x = 0.5
	# x = 0.01
	# x = 0.001
	# x = 0.01 if beta <= 100 else 0.001
	x = 0.01 
	diffseries = []
	# xGD, xGOD = (0.5,0.5)
	xGD, xGOD = (x,x)
	diffGD, diffGOD= (1.,1.)
	conv_flag = True
	
	while (diff>err and itern<ITERMAX and conv_flag): 
		itern += 1 
		diffoldGD,diffoldGOD= (diffGD,diffGOD)
		GDRoldomega,GODRoldomega= (1.0*GDRomega, 1.0*GODRomega)

		rhoGD = -1.0*np.imag(GDRomega)
		rhoGOD = -1.0*np.imag(GODRomega)

		SigmaDomega= rhotosigma(rhoGD,M,dt,t,omega,J,beta,kappa,delta=eta)
		# SigmaODomega= -1.0*rhotosigma(rhoGOD,M,dt,t,omega,J,beta,kappa,delta=eta)
		SigmaODomega= 1.0*rhotosigma(rhoGOD,M,dt,t,omega,J,beta,kappa,delta=eta)
		
		detGmat = (omega+1j*eta - mu - SigmaDomega)**2 - (kappa + SigmaODomega)**2
	
		GDRomega = xGD*((omega+1j*eta - mu - SigmaDomega)/detGmat) + (1-xGD)*GDRoldomega
		GODRomega = xGOD*((kappa + SigmaODomega)/detGmat) + (1-xGOD)*GODRoldomega


		diffGD = np. sqrt(np.sum((np.abs(GDRomega-GDRoldomega))**2)) #changed
		diffGOD = np. sqrt(np.sum((np.abs(GODRomega-GODRoldomega))**2)) 

		diff = 0.5*(diffGD+diffGOD)
		# diffGD,diffGOD = diff,diff
		# diffseries += [diff]

		# if diffGD>diffoldGD:
		# 	xGD/=2.
		# if diffGOD>diffoldGOD:
		# 	xGOD/=2.
		if verbose:
			print("itern = ",itern, " , diff = ", diff , " , x = ", x)
		# if itern>30:
		# 	conv_flag = testingscripts.diff_checker(diffseries, tol = 1e-4, periods = 5)
			

	return (GDRomega,GODRomega)


#####################

def main():

	# fig,ax = plt.subplots(1)
	# ax.set_xlabel(r'$\omega$')
	# ax.set_ylabel(r'$-Im G^R(\omega)$')
	# ax.set_xlim(-5,5)

	load_flag = True
	#################Data Compression################

	tot_freq_grid_points = int(2**14)
	omega_max = 5
	omega_min = -1*omega_max
	idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
	skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
	comp_omega_slice = slice(idx_min,idx_max,skip)
	#comp_omega = omega[comp_omega_slice]


	###### INITIALIZATION ###########
	if load_flag == False:
		GDRomega = (omega + 1j*eta + mu)/ (omega+1j*eta - mu )**2 - (kappa)**2
		#GODRomega = np.zeros_like(omega)
		# GODRomega = 1j*eta*np.ones_like(omega)
		GODRomega = kappa / (omega+1j*eta - mu )**2 - (kappa)**2

	###### EVENT LOOP STARTS ############
	for beta in betalist:
		# ITERMAX = 10000 if beta <= 100 else 1000 if beta < 250 else 100
		# ITERMAX = 10000 if beta <= 100 else 10000 if beta < 250 else 10000
		savefile = 'cSYKWH'
		savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
		# savefile += 'beta' + str((round(beta*100))/100.) 
		savefile += 'beta' + str(beta) 
		savefile += 'J' + str(J)
		savefile += 'kappa' + f'{kappa:.3}'
		savefile = savefile.replace('.','_') 
		savefiledump = savefile + '.npy' 
		savefileoutput = savefile + '.h5'

		if load_flag == True:
			GDRomega,GODRomega = np.load(os.path.join(path_to_dump,savefiledump))
			load_flag = False
		else:
			newM = int(2**22) #number of points in the grid
			newT = int(2**18) #upper cut-off fot the time
			newomega, newt = RealGridMaker(newM,newT)
			newdt = newt[2]-newt[1]
			newdw = newomega[2]-newomega[1]
			neweta = newdw * 2.1
			newGDRomega = PchipInterpolator(omega, GDRomega)(newomega)
			newGODRomega = PchipInterpolator(omega, GODRomega)(newomega)
			GDRomega = newGDRomega
			GODRomega = newGODRomega
			print(f'new dw = {newdw}')
			print(f'temp = {1./beta}')
			print(f'temp/dw = {1./(beta*newdw)}')
			for iters in np.arange(10):
				# GDRomega,GODRomega = RE_wormhole_cSYK_iterator(newGDRomega,newGODRomega,J,mu,kappa,beta,newomega,newt,newM,newdt,ITERMAX = ITERMAX, eta=neweta,verbose=True)
				GDRomega,GODRomega = RE_wormhole_cSYK_iterator(GDRomega,GODRomega,J,mu,kappa,beta,newomega,newt,newM,newdt,ITERMAX = ITERMAX, eta=neweta,verbose=True)
				if DUMP == True:
					newsavefile = 'INTERPcSYKWH'
					newsavefile += 'M' + str(int(np.log2(newM))) + 'T' + str(int(np.log2(newT))) 
					# savefile += 'beta' + str((round(beta*100))/100.) 
					newsavefile += 'beta' + str(beta) 
					newsavefile += 'J' + str(J)
					newsavefile += 'kappa' + f'{kappa:.3}'
					newsavefile = newsavefile.replace('.','_') 
					newsavefiledump = newsavefile + '.npy' 
					newsavefileoutput = newsavefile + '.h5'
					np.save(os.path.join(path_to_dump, newsavefiledump),[GDRomega,GODRomega])

			



				###########Data Writing############ 
				print("\n###########Data Writing############")
				dictionary = {
				"J": J,
				"mu": mu,
				"beta": beta,
				"kappa": kappa,
				"M": newM, 
				"T": newT,
				"omega": newomega[comp_omega_slice],
				"rhoGD": -np.imag(GDRomega[comp_omega_slice]),
				"rhoGOD": -np.imag(GODRomega[comp_omega_slice]),
				"compressed": True, 
				"eta": eta
				}
				
				dict2h5(dictionary, os.path.join(path_to_output, newsavefileoutput), verbose=True) 

				# peak_idxlist = find_peaks(-np.imag(GDRomega)[M:],prominence=0.1)[0]
				# print(omega[M:][peak_idxlist])

				# c = omega[M:][peak_idxlist][0] / delta
				# print('predicted peaks = ', c * (np.arange(4) + delta))

			# if PLOTTING == True:
			# 	ax.plot(newomega,-np.imag(newGDRomega),'.-',label=f'$\beta = $ {beta}')
			# 	# for peak_idx in peak_idxlist:
			# 	# 	ax.axvline(omega[M:][peak_idx],ls='--')

	# plt.show()


	print(f"*********Program exited successfully *********")


if __name__ == "__main__":
	main()


































