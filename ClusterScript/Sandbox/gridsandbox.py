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
#import time


DUMP = False

Nbig = int(2**16)
#err = 1e-4
err = 1e-2
ITERMAX = 200

global beta

beta_start = 140
beta = beta_start
mu = 0.0
g = 0.5
r = 1.

target_beta = 200

# g = np.sqrt(10**3)
# r = (10)**2

kappa = 1.
beta_step = 1
betaplus = beta + beta_step


omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')


Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta) 


#print(tau)
print('Nbig = ', Nbig)
# print('tau[0] = ', tau[0])
# print('tau[-1] = ', tau[-1])
# print('dtau = ', tau[2]-tau[1])
# print('beta = ', beta, 'Nbig = ', Nbig, 'beta/Nbig = ', beta/Nbig , 'beta - beta/Nbig = ', beta - beta/Nbig)
# print('beta/2N = ', 0.5*beta/Nbig, 'beta - beta/2N = ', beta - 0.5*beta/Nbig)



tau_beta = ImagGridMaker(Nbig,beta,'tau')
tau_betaplus = ImagGridMaker(Nbig,betaplus,'tau')
#print(list(zip(tau_beta,Gfreetau.real)))
Gfreebetaplus = Freq2TimeF(1./(1j*omega + mu),Nbig,betaplus)

Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
Gconfbetaplus = Freq2TimeF(GconfImag(omega,g,betaplus),Nbig,betaplus)

diff_free = np.sqrt(np.sum(np.abs(Gfreetau - Gfreebetaplus)**2))
diff_conf = np.sqrt(np.sum(np.abs(Gconftau - Gconfbetaplus)**2))

print('beta_step = ', beta_step)
print('diff_free = ', diff_free)
print('diff_conf = ', diff_conf)






