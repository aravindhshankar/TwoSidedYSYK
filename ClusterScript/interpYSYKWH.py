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



savename = 'default_savename'
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

savefile = os.path.join(path_to_output, savename+'.h5')

fft_check = testingscripts.realtimeFFT_validator() # Should return True

##################

#rhotosigma already exists in ConformalAnalytical.py

###################

g = 0.5
#beta = 100.
# beta = 1./(2e-4)
beta = 200
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
ITERMAX = 5
# ITERMAX = 25000


savefile = savename
savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
# savefile += 'beta' + str((round(beta*100))/100.) 
savefile += 'beta' + str(beta) 
savefile += 'g' + str(g)
savefile += 'lamb' + str(lamb)
savefile = savefile.replace('.','_') 
savefiledump = savefile + '.npy' 
savefileoutput = savefile + '.h5'

x = 0.01

print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
print("dw/temp = ", f'{dw*beta:.4f}')
print("flag fft_check = ", fft_check)
print("grid_flag = ", grid_flag)

## State varriables go into .out file
print("######## State Variables ################")
print("g = ", g)
print("mu = ", mu)
print("r = ", r)
print("lamb= ", lamb)
print("J = ", J)
print("beta = ", beta)
print("log_2 M = ", np.log2(M))
print("eta = ", eta)
print("T = ", T)
print("err = ", err)
print("######## End of State variables #########")



omegar2 = ret_omegar2(g,beta)
	
# def RE_wormhole_YSYK_iterator(GDRomega,GODRomega,DDRomega,DODRomega,g,lamb,J,beta,eta=1e-6,verbose=True):
# 	itern = 0
# 	diff = 1
# 	x = 0.5
# 	diffseries = []
# 	xGD, xGOD, xDD, xDOD = (0.5,0.5,0.5,0.5)
# 	diffGD, diffGOD, diffDD, diffDOD = (1.,1.,1.,1.)
# 	conv_flag = True
	
# 	while (diff>err and itern<150 and conv_flag): 
# 		itern += 1 
# 		diffoldGD,diffoldDD,diffoldGOD,diffoldDOD = (diffGD,diffDD,diffGOD,diffDOD)
# 		GDRoldomega,DDRoldomega,GODRoldomega,DODRoldomega = (1.0*GDRomega, 1.0*DDRomega, 1.0*GODRomega, 1.0*DODRomega)

# 		rhoGD = -1.0*np.imag(GDRomega)
# 		rhoDD = -1.0*np.imag(DDRomega)
# 		rhoGOD = -1.0*np.imag(GODRomega)
# 		rhoDOD = -1.0*np.imag(DODRomega)

# 		SigmaDomega,PiDomega = Davrhotosigma(rhoGD,rhoDD,M,dt,t,omega,g,beta,kappa=1,delta=eta)
# 		SigmaODomega,PiODomega = Davrhotosigma(rhoGOD,rhoDOD,M,dt,t,omega,g,beta,kappa=1,delta=eta)
# 		####PiDOmega[M] = -1.0*r - omegar2 - eta**2

# 	#   if itern < 10 : 
# 	#   PiDomega[M] = -1.0*r - omegar2 - eta**2
# 	#	PiODomega[M] = 0
		
# 		detGmat = (omega+1j*eta + mu - SigmaDomega)**2 - (lamb - SigmaODomega)**2
# 		detDmat = ((omega+1j*eta)**2 - r - PiDomega)**2 - (J - PiODomega)**2
	
# 		GDRomega = xGD*((omega+1j*eta + mu - SigmaDomega)/detGmat) + (1-xGD)*GDRoldomega
# 		GODRomega = xGOD*(-1.0*(lamb - SigmaODomega)/detGmat) + (1-xGOD)*GODRoldomega
# 		DDRomega = xDD*(((omega+1j*eta)**2 - r - PiDomega)/detDmat) + (1-xDD)*DDRoldomega
# 		DODRomega = xDOD*(-1.0*(J - PiODomega)/detDmat) + (1-xDOD)*DODRoldomega

# 		if itern > 15 :
# 		    eta=dw*0.01

# 		diffGD = np. sqrt(np.sum((np.abs(GDRomega-GDRoldomega))**2)) #changed
# 		diffDD = np. sqrt(np.sum((np.abs(DDRomega-DDRoldomega))**2))
# 		diffGOD = np. sqrt(np.sum((np.abs(GODRomega-GODRoldomega))**2)) 
# 		diffDOD = np. sqrt(np.sum((np.abs(DODRomega-DODRoldomega))**2))

# 		diff = 0.25*(diffGD+diffDD+diffGOD+diffDOD)
# 		diffGD,diffDD,diffGOD,diffDOD = diff,diff,diff,diff
# 		diffseries += [diff]

# 		if diffGD>diffoldGD:
# 			xGD/=2.
# 		if diffGOD>diffoldGOD:
# 			xGOD/=2.
# 		if diffDD>diffoldDD:
# 			xDD/=2.
# 		if diffDOD>diffoldDOD:
# 			xDOD/=2.
# 		if verbose:
# 			print("itern = ",itern, " , diff = ", diffGD, diffDOD, " , x = ", xGOD, xDD)
# 		if itern>30:
# 			conv_flag = testingscripts.diff_checker(diffseries, tol = 1e-4, periods = 5)
			

# 	return (GDRomega,GODRomega,DDRomega,DODRomega)



#####################

def main():

	# GDRomega = 1/(omega + 1j*eta + mu)
	# GODRomega = np.zeros_like(omega)
	# DDRomega = -1/(-1.0*(omega + 1j*eta)**2 + r) 
	# DODRomega = np.zeros_like(omega)
	neglamb = -1.0 * lamb #this passed to iterators
	detG0mat = (omega+1j*eta + mu)**2 - (lamb)**2

	GDRomega = (omega+1j*eta + mu)/detG0mat
	GODRomega = (lamb)/detG0mat
	DDRomega = 1/(r -1.0*(omega + 1j*eta)**2) 
	DODRomega = np.zeros_like(DDRomega)


	GFs = [GDRomega, GODRomega, DDRomega, DODRomega]
	grid = [M,omega,t]
	pars = [g,mu,r]
	# GDRomega,GODRomega,DDRomega,DODRomega = RE_wormhole_YSYK_iterator(GDRomega,GODRomega,DDRomega,DODRomega,g,lamb,J,beta,eta=1e-6,verbose=True)
	GFs, INFO = RE_WHYSYK_iterator(GFs,grid,pars,beta,neglamb,J,err=err,x=0.01,ITERMAX=ITERMAX,eta = eta,verbose=True,diffcheck=False) 

	if DUMP == True:
		np.save(os.path.join(path_to_dump,savefiledump), GFs)

	GDRomega, GODRomega, DDRomega, DODRomega = GFs
	#GDRt = (0.5/np.pi) * freq2time(GDRomega,M,dt)
	#DDRt = (0.5/np.pi) * freq2time(DDRomega,M,dt)
	#GODRt = (0.5/np.pi) * freq2time(GODRomega,M,dt)
	#DODRt = (0.5/np.pi) * freq2time(DODRomega,M,dt)
	# GDRt = (0.5/np.pi) * freq2time(GDRomega - GfreeRealomega(omega,mu,eta),M,dt) + GfreeRealt(t,mu,eta)
	# DDRt = (0.5/np.pi) * freq2time(DDRomega - DfreeRealomega(omega,r,eta),M,dt) + DfreeRealt(t,r,eta)
	# GODRt = (0.5/np.pi) * freq2time(GODRomega - GfreeRealomega(omega,mu,eta),M,dt) + GfreeRealt(t,mu,eta)
	# DODRt = (0.5/np.pi) * freq2time(DODRomega - DfreeRealomega(omega,r,eta),M,dt) + DfreeRealt(t,r,eta)

	#################Data Compression################

	tot_freq_grid_points = int(2**14)
	omega_max = 5
	omega_min = -1*omega_max
	idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
	skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
	comp_omega_slice = slice(idx_min,idx_max,skip)
	#comp_omega = omega[comp_omega_slice]



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
	print(f"*********Program exited successfully *********")


if __name__ == "__main__":
	main()


























