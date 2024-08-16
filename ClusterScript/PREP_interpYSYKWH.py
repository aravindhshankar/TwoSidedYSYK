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

from YSYK_iterator import RE_WHYSYK_iterator



DUMP = True
PLOTTING = True



# savename = 'default_savename'
path_to_output = './Outputs/redoYWH'
path_to_dump  = './Dump/redoYWH'

if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
	print("Outputs directory created")
if not os.path.exists(path_to_dump):
	os.makedirs(path_to_dump)
	print("Dump directory created")

# if len(sys.argv) > 1:
# 	savename = str(sys.argv[1])

# savefile = os.path.join(path_to_output, savename+'.h5')

fft_check = testingscripts.realtimeFFT_validator() # Should return True

##################

#rhotosigma already exists in ConformalAnalytical.py

###################

g = 0.5
#beta = 100.
# beta = 1./(2e-4)
# beta = 200
betalist = [200,500,1000,2000,5000,6000,10000]
#beta = 1./(5e-5)
mu = 0. 
r = 1.
# lamb = 10./beta
# J = 20./beta
# lamb = 0.001
lamb = 0.01
J = 0

# M = int(2**16) #number of points in the grid
# T = int(2**12) #upper cut-off fot the time
M = int(2**16) #number of points in the grid
T = int(2**12) #upper cut-off fot the time
#M = int(2**16)
#T = int(2**10)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
err = 1e-7
# eta = dw*10.
eta = dw*2.1
#delta = 0.420374134464041
ITERMAX = 10000
# ITERMAX = 25000





print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
# print("dw/temp = ", f'{dw*beta:.4f}')
print("flag fft_check = ", fft_check)
print("grid_flag = ", grid_flag)

## State varriables go into .out file
print("######## State Variables ################")
print("g = ", g)
print("mu = ", mu)
print("r = ", r)
print("lamb= ", lamb)
print("J = ", J)
# print("beta = ", beta)
print("log_2 M = ", np.log2(M))
print("eta = ", eta)
print("T = ", T)
print("err = ", err)
print("######## End of State variables #########")



# omegar2 = ret_omegar2(g,beta)
	


def main():
	# neglamb = -1.0 * lamb #this passed to iterators
	load_flag = True

	grid = [M,omega,t]
	pars = [g,mu,r]
	############# SETTING UP FIGURE ############
	fig,ax = plt.subplots(1)
	ax.set_xlabel(r'$\omega$')
	ax.set_ylabel(r'$-Im G_d^R(\omega)$')
	ax.set_xlim(-5,5)


	#################Data Compression################
	tot_freq_grid_points = int(2**14)
	omega_max = 5
	omega_min = -1*omega_max
	idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
	skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
	comp_omega_slice = slice(idx_min,idx_max,skip)
	#comp_omega = omega[comp_omega_slice]

	######## INITIALIZATION ############################
	if load_flag == False:
		raise(Exception("SUPPORT NOT PROVIDED YET: LOAD STARTING STATE"))
		exit(1)
		detG0mat = (omega+1j*eta + mu)**2 - (lamb)**2
		GDRomega = (omega+1j*eta + mu)/detG0mat
		GODRomega = (lamb)/detG0mat
		DDRomega = 1/(r -1.0*(omega + 1j*eta)**2) 
		DODRomega = np.zeros_like(DDRomega)
	
	########### EVENT LOOP STARTS ##############	
	for beta in betalist:
		savefile = 'YWH'
		savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
		savefile += 'beta' + str(beta) 
		savefile += 'g' + str(g)
		savefile += 'lamb' + str(lamb)
		savefile = savefile.replace('.','_') 
		savefiledump = savefile + '.npy' 
		savefileoutput = savefile + '.h5'

		x = 0.001

		if load_flag == True:
			try:
				load_flag = False
				GFs = np.load(os.path.join(path_to_dump,savefiledump))
				GFs[1] = -1.0*GFs[1]
			except FileNotFoundError:
				print('INPUT FILE NOT FOUND!!!!!!!!!!')
				print(os.path.join(path_to_dump,savefiledump))
				exit(1)
		else:
			# GFs = [GDRomega, GODRomega, DDRomega, DODRomega]
			# GDRomega,GODRomega,DDRomega,DODRomega = RE_wormhole_YSYK_iterator(GDRomega,GODRomega,DDRomega,DODRomega,g,lamb,J,beta,eta=1e-6,verbose=True)
			GFs, INFO = RE_WHYSYK_iterator(GFs,grid,pars,beta,lamb,J,err=err,x=x,ITERMAX=ITERMAX,eta = eta,verbose=True,diffcheck=False) 

			if DUMP == True:
				np.save(os.path.join(path_to_dump,savefiledump), GFs)

			GDRomega, GODRomega, DDRomega, DODRomega = GFs

			###########Data Writing############ 
			print("\n###########Data Writing############")
			dictionary = {
			   "g": g,
			   "mu": mu,
			   "beta": beta,
			   "r": r,
			   "lamb": lamb,
			   "J": J,
			   "M": M, 
			   "T": T,
			   "omega": omega[comp_omega_slice],
			   "rhoGD": -np.imag(GDRomega[comp_omega_slice]),
			   "rhoGOD": -np.imag(GODRomega[comp_omega_slice]),
			   "rhoDD": -np.imag(DDRomega[comp_omega_slice]),
			   "rhoDOD": -np.imag(DODRomega[comp_omega_slice]), 
			   "compressed": True, 
			   "eta": eta
			}
				
			dict2h5(dictionary, savefileoutput, verbose=True)

			if PLOTTING == True:
				ax.plot(omega,-np.imag(GDRomega),'.-',label=f'$\beta = $ {beta}')


	print("Exited from event loop")
	plt.show()


	print(f"*********Program exited successfully *********")


if __name__ == "__main__":
	main()


























