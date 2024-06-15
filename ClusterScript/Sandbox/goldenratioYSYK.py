import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

# if not os.path.exists('../Dump/SupCondYSYKImagDumpfiles'):
#     print("Error - Path to Dump directory not found ")
#     raise Exception("Error - Path to Dump directory not found ")
# else:
#     path_to_dump = '../Dump/SupCondYSYKImagDumpfiles'


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from scipy.linalg import norm
from functools import partial
#import time


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



DUMP = False

Nbig = int(2**14)
err = 1e-6
#err = 1e-2
ITERMAX = 2000

global beta

mu = 0.0
g = 0.5
r = 1.

# target_beta = 500
beta = 200.
kappa = 1.


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)



def stepYSYK(Gtaus,g,beta,Nbig,omega,nu,kappa=1.,mu=0.,r=1.):
	Gtau,Dtau = Gtaus
	Sigmatau = kappa * (g**2) * Gtau * Dtau
	Pitau = 2.0 * (g**2) * Gtau * Gtau[::-1]
	Sigmaomega = Time2FreqF(Sigmatau,Nbig,beta)
	Piomega = Time2FreqB(Pitau,Nbig,beta)
	Gomega = 1./(1j*omega+mu-Sigmaomega)
	Domega = 1./(nu**2 + r - Piomega)
	TGtau = Freq2TimeF(Gomega,Nbig,beta)
	TDtau = Freq2TimeB(Domega,Nbig,beta)
	return np.array((TGtau,TDtau))

x_1 = 0.5 #initialize lambda_{k-1} 
x_2 = 1.*x_1 #initialize lambda_{k-2} 
x = 0.5

# Gtau, Dtau = Gfreetau, Dfreetau
Gtau, Dtau = Gconftau, Dconftau
Gtaus = np.array((Gtau,Dtau))

################## Event Loop ##########################

# while(diff > err and itern < ITERMAX):
# 	if (itern == ITERMAX-1):
# 		print(f'ITERMAX  = {ITERMAX} has been reached')
# 	oldGtau,oldDtau = Gtau.copy(),Dtau.copy() #G_{k-1}
# 	xf = (10./9.) * x_1
# 	num = (norm(Gtau - ))**2


T = partial(stepYSYK, g=g,beta=beta,Nbig=Nbig,omega=omega,nu=nu,kappa=kappa,mu=mu,r=r)

sol = fixed_point_egraal(T, Gtaus, err, ITERMAX = ITERMAX)
values, G, step_list = sol

if len(step_list) >= ITERMAX-1:
	print(f"stopped because of ITERMAX = {ITERMAX} reached")
print(f'total steps = {len(step_list)} with final x  = {step_list[-1]}')
print(G.shape)
print(values[-1])

Gtau, Dtau = G







################## PLOTTING ######################

Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
FreeDtau = DfreeImagtau(tau,r,beta)

fig, ax = plt.subplots(2)

ax[0].plot(tau/beta, np.real(Gtau), 'r', label = 'numerics Gtau')
ax[0].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
ax[0].set_ylim(-1,1)
ax[0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[0].set_ylabel(r'$\Re{G(\tau)}$')
ax[0].legend()

ax[1].plot(tau/beta, np.real(Dtau), 'r', label = 'numerics Dtau')
ax[1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
ax[1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
#ax[1].set_ylim(0,1)
ax[1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
ax[1].set_ylabel(r'$\Re{D(\tau)}$')
ax[1].legend()

#plt.savefig('KoenraadEmails/WithMR_imagtime.pdf',bbox_inches='tight')





fig, ax = plt.subplots(1)
ax.plot(values, '.-', label='values')
ax.set_yscale('log')
ax2 = ax.twinx()
ax2.plot(step_list, '.-', c='r', label = 'step_list')
ax.legend(loc=0)
ax2.legend(loc=0)
plt.show()






















