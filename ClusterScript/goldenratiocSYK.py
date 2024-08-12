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
from scipy.linalg import norm
from functools import partial
import time

savename = 'default_savename'
path_to_output = './Outputs'
path_to_dump  = './Dump/redoCWH'

if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
	print("Outputs directory created")

if len(sys.argv) > 1:
	savename = str(sys.argv[1])

savefile = os.path.join(path_to_output, savename+'.h5')
savefile_dump = os.path.join(path_to_dump, savename+'.npy')


fft_check = testingscripts.realtimeFFT_validator() # Should return True

DUMP = True
PLOTTING = True

##################
def fixed_point_egraal(T, x0, err, ITERMAX = 2000, phi=1.5, output=False):
	"""
	Golden Ratio Algorithm for the problem x = Tx

	T is the operator
	x0 is the starting point

	"""
	
	JF = lambda x: norm(x)
	
	x, x_ = x0.copy(), x0.copy()
	tau = 1. / phi + 1. / phi**2

	F = lambda x: x - T(x)
	la = 1
	step_list = [la]
	th = 1
	Fx = F(x0)
	values = [JF(Fx)]
	res = 1.
	itern = 0
	while res > err and itern < ITERMAX:
		itern += 1 
		x1 = x_ - la * Fx
		Fx1 = F(x1)

		n1 = norm(x1 - x)**2
		n2 = norm(Fx1 - Fx)**2
		la1 = min(tau * la, 0.25 * phi * th / la * (n1 / n2))
		x_ = ((phi - 1) * x1 + x_) / phi
		th = phi * la1 / la
		x, la, Fx = x1, la1, Fx1
		res = JF(Fx)
		values.append(res)
		step_list.append(la1)

	return values, x, step_list

#############
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
beta = 1000
mu = 0. 
# kappa = 0.05
kappa = 0.01
ITERMAX = 10000

# M = int(2**16) #number of points in the grid
# T = int(2**12) #upper cut-off fot the time
M = int(2**19) #number of points in the grid
T = int(2**14) #upper cut-off fot the time
#M = int(2**16)
#T = int(2**10)
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
err = 1e-2
eta = dw*2.1
#delta = 0.420374134464041
delta = 0.25

print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
print("dw/temp = ", f'{dw*beta:.4f}')
print("flag fft_check = ", fft_check)
print("grid_flag = ", grid_flag)

## State varriables go into .out file
print("######## State Variables ################")
print("J = ", J)
print("mu = ", mu)
print("kappa = ", kappa)
print("beta = ", beta)
print("log_2 M = ", np.log2(M))
print("eta = ", eta)
print("T = ", T)
print("err = ", err)
print("######## End of State variables #########")


	
def RE_wormhole_cSYK_STEP(Gomegas,J,mu,kappa,beta,eta=1e-6):

	GDRomega, GODRomega = Gomegas
	rhoGD = -1.0*np.imag(GDRomega)
	rhoGOD = -1.0*np.imag(GODRomega)

	SigmaDomega= rhotosigma(rhoGD,M,dt,t,omega,J,beta,kappa,delta=eta)
	# SigmaODomega= -1.0*rhotosigma(rhoGOD,M,dt,t,omega,J,beta,kappa,delta=eta)
	SigmaODomega= 1.0*rhotosigma(rhoGOD,M,dt,t,omega,J,beta,kappa,delta=eta)

	detGmat = (omega+1j*eta - mu - SigmaDomega)**2 - (kappa + SigmaODomega)**2

	TGDRomega = (omega+1j*eta - mu - SigmaDomega)/detGmat
	TGODRomega = (kappa + SigmaODomega)/detGmat

	return np.array((TGDRomega,TGODRomega))




GDRomega = (omega + 1j*eta + mu)/ (omega+1j*eta - mu )**2 - (kappa)**2
#GODRomega = np.zeros_like(omega)
# GODRomega = 1j*eta*np.ones_like(omega)
GODRomega = kappa / (omega+1j*eta - mu )**2 - (kappa)**2
GFs = np.array([GDRomega,GODRomega])

T = partial(RE_wormhole_cSYK_STEP, J=J,mu=mu,kappa=kappa,beta=beta,eta=eta)

start_time = time.perf_counter()
sol = fixed_point_egraal(T, GFs, err, ITERMAX = ITERMAX)
stop_time = time.perf_counter()

if DUMP == True:
	np.save(savefile_dump,[GDRomega,GODRomega])
print(f'Exited from golden ration algorithm in {stop_time-start_time} seconds')

values, GFs, step_list = sol

if len(step_list) >= ITERMAX-1:
	print(f"stopped because of ITERMAX = {ITERMAX} reached")
print(f'total steps = {len(step_list)} with final x  = {step_list[-1]}')
print(G.shape)
print(values[-1])

GDRomega, GODRomega = GFs

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
   "J": J,
   "mu": mu,
   "beta": beta,
   "kappa": kappa,
   "M": M, 
   "T": T,
   "omega": omega[comp_omega_slice],
   "rhoGD": -np.imag(GDRomega[comp_omega_slice]),
   "rhoGOD": -np.imag(GODRomega[comp_omega_slice]),
   "compressed": True, 
   "eta": eta
}
	
dict2h5(dictionary, savefile, verbose=True) 

peak_idxlist = find_peaks(-np.imag(GDRomega)[M:],prominence=0.1)[0]
print(omega[M:][peak_idxlist])

c = omega[M:][peak_idxlist][0] / delta
print('predicted peaks = ', c * (np.arange(4) + delta))

if PLOTTING == True:
	fig,ax = plt.subplots(1)
	ax.plot(omega,-np.imag(GDRomega),'.-')
	ax.set_xlabel(r'$\omega$')
	ax.set_ylabel(r'$-Im G^R(\omega)$')

	for peak_idx in peak_idxlist:
		ax.axvline(omega[M:][peak_idx],ls='--')
	plt.show()


print(f"*********Program exited successfully *********")


































