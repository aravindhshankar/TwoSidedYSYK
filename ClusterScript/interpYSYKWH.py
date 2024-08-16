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
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator 



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

if len(sys.argv) > 1:
	savename = str(sys.argv[1])

# savefile = os.path.join(path_to_output, savename+'.h5')

fft_check = testingscripts.realtimeFFT_validator() # Should return True

##################

#rhotosigma already exists in ConformalAnalytical.py

###################

g = 0.5
#beta = 100.
# beta = 1./(2e-4)
betalist = [750,750]
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
err = 1e-5
# eta = dw*10.
eta = dw*2.1
#delta = 0.420374134464041
ITERMAX = 100
# ITERMAX = 25000


# savefile = savename
# savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
# # savefile += 'beta' + str((round(beta*100))/100.) 
# savefile += 'beta' + str(beta) 
# savefile += 'g' + str(g)
# savefile += 'lamb' + str(lamb)
# savefile = savefile.replace('.','_') 
# savefiledump = savefile + '.npy' 
# savefileoutput = savefile + '.h5'

x = 0.01





	


def main():
	load_flag = True
	########### EVENT LOOP STARTS ##############	
	print(f'ITERMAX = {ITERMAX}')
	for beta in betalist:
		savefile = 'YWH'
		savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
		savefile += 'beta' + str(beta) 
		savefile += 'g' + str(g)
		savefile += 'lamb' + str(lamb)
		savefile = savefile.replace('.','_') 
		savefiledump = savefile + '.npy' 
		savefileoutput = savefile + '.h5'

		x = 0.01

		if load_flag == True:
			try:
				load_flag = False
				GFs = np.load(os.path.join(path_to_dump,savefiledump))
				GDRomega, GODRomega, DDRomega, DODRomega = GFs
				# GFs[1] = -1.0*GFs[1]
			except FileNotFoundError:
				print('INPUT FILE NOT FOUND!!!!!!!!!!')
				print(os.path.join(path_to_dump,savefiledump))
				exit(1)
		else:
			# GFs = [GDRomega, GODRomega, DDRomega, DODRomega]
			# GDRomega,GODRomega,DDRomega,DODRomega = RE_wormhole_YSYK_iterator(GDRomega,GODRomega,DDRomega,DODRomega,g,lamb,J,beta,eta=1e-6,verbose=True)
			newM = int(2**20) #number of points in the grid
			newT = int(2**16) #upper cut-off fot the time
			newomega, newt = RealGridMaker(newM,newT)
			newdt = newt[2]-newt[1]
			newdw = newomega[2]-newomega[1]
			neweta = newdw * 2.1
			newgrid = [newM,newomega,newt]
			pars = [g,mu,r]

			newGDRomega = PchipInterpolator(omega, GDRomega)(newomega)
			newGODRomega = PchipInterpolator(omega, GODRomega)(newomega)
			newDDRomega = PchipInterpolator(omega, DDRomega)(newomega)
			newDODRomega = PchipInterpolator(omega, DODRomega)(newomega)
			GDRomega = newGDRomega
			GODRomega = newGODRomega
			DDRomega = newDDRomega
			DODRomega = newDODRomega
			print(f'new dw = {newdw}')
			print(f'temp = {1./beta}')
			print(f'temp/dw = {1./(beta*newdw)}')
			GFs = [GDRomega,GODRomega,DDRomega,DODRomega]
			for iters in np.arange(100):
				GFs, INFO = RE_WHYSYK_iterator(GFs,newgrid,pars,beta,lamb,J,err=err,x=x,ITERMAX=ITERMAX,eta = neweta,verbose=True,diffcheck=False) 

				if DUMP == True:
					savefile = 'YWH'
					savefile += 'M' + str(int(np.log2(newM))) + 'T' + str(int(np.log2(newT))) 
					savefile += 'beta' + str(beta) 
					savefile += 'g' + str(g)
					savefile += 'lamb' + str(lamb)
					savefile = savefile.replace('.','_') 
					savefiledump = savefile + '.npy' 
					savefileoutput = savefile + '.h5'
					np.save(os.path.join(path_to_dump, savefiledump),GFs)


				GDRomega, GODRomega, DDRomega, DODRomega = GFs

			# ###########Data Writing############ 
			# print("\n###########Data Writing############")
			# dictionary = {
			#    "g": g,
			#    "mu": mu,
			#    "beta": beta,
			#    "r": r,
			#    "lamb": lamb,
			#    "J": J,
			#    "M": M, 
			#    "T": T,
			#    "omega": omega[comp_omega_slice],
			#    "rhoGD": -np.imag(GDRomega[comp_omega_slice]),
			#    "rhoGOD": -np.imag(GODRomega[comp_omega_slice]),
			#    "rhoDD": -np.imag(DDRomega[comp_omega_slice]),
			#    "rhoDOD": -np.imag(DODRomega[comp_omega_slice]), 
			#    "compressed": True, 
			#    "eta": eta
			# }
				
			# dict2h5(dictionary, savefileoutput, verbose=True)

	print(f"*********Program exited successfully *********")


if __name__ == "__main__":
	main()


























