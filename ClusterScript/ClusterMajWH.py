import sys
import os 
if not os.path.exists('./Sources'):
	print("Error - Path to Sources directory not found ")
sys.path.insert(1,'./Sources')
import pickle

import numpy as np
from matplotlib import pyplot as plt
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts


path_to_output = 


savename = 'default_savename'
if len(sys.argv) > 1:
	savename = str(sys.argv[1])


testingscripts.realtimeFFT_validator() # Should return True

##################

def rhotosigma(rhoG,M,dt,t,J,delta=1e-6):
    '''
    returns [Sigma,Pi] given rhos
    '''
    rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
    nLL = (0.5/np.pi)*freq2time((rhoG + rhoGrev) * fermidirac(beta*omega),M,dt)
    nLR = (0.5/np.pi)*freq2time((rhoG - rhoGrev) * fermidirac(beta*omega),M,dt)
    #nLL = (0.25/np.pi**2)*freq2time((rhoG + rhoGrev) * fermidirac(beta*omega),M,dt)
    #nLR = (0.25/np.pi**2)*freq2time((rhoG - rhoGrev) * fermidirac(beta*omega),M,dt)

    
    argSigma = (np.real(nLL**3) - 1j*np.imag(nLR**3)) * np.exp(-np.abs(delta*t)) * np.heaviside(t,1.0)
    Sigma = -2j * (J**2) * time2freq(argSigma,M,dt)

    return Sigma



###################

J = 1.
mu = 0.01
#mu = 0


#beta = 1./(2e-4)
beta = 50.

M = int(2**16) #number of points in the grid
T = int(2**10) #upper cut-off fot the time
dt = (2*T)/((2*M))
t = dt * (np.arange(2*M) - M)

dw = np.pi/(M*dt)
eta = dw*10.
omega = dw * (np.arange(2*M) - M) 
np.testing.assert_almost_equal(dt*dw*M,np.pi,5, "Error in fundamentals")
err = 1e-2

#delta = 0.420374134464041
np.testing.assert_almost_equal(np.max(np.abs(omega)),np.pi*M/T,5,"Error in creating omega grid")

print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
print("dw/temp = ", f'{dw*beta:.4f}')

## State varriables go into .out file
print("######## State Variables ################")
print("J = ", J, '\n')
print("mu = ", mu, '\n')
print("M = ", M, '\n')
print("eta = ", eta, '\n')
print("T = ", T, '\n')
print("err = ", err, '\n')
print("######## End of State variables #########")




#####################

def main():

	GRomega = 1/(omega + 1j*eta - mu)


	itern = 0


	diff = 1. 
	diffG = 1.
	xG = 0.5

	while (diff>err and itern<200): 
	    itern += 1 
	    diffoldG = diffG
	    GRoldomega= 1.0*GRomega
	    
	    rhoG = -1.0*np.imag(GRomega)
	    
	    SigmaOmega = rhotosigma(rhoG,M,dt,t,J,delta=eta)
	    
	    GRomega = 1.0*xG/(omega + 1j*eta - mu - SigmaOmega) + (1-xG)*GRoldomega
	    
	    diffG = (0.5/M) * np.sum((np.abs(GRomega-GRoldomega))**2)
	    diff = diffG
	    
	    if diffG>diffoldG:
	        xG/=2.

	    print("itern = ",itern, " , diff = ", diffG, " , x = ", xG, end = '\r')



	GRt = (0.5/np.pi) * freq2time(GRomega,M,dt)
#Data to be written
dictionary = {
   "J": J,
   "mu": mu,
   "beta": beta,
   "T": T,
   "omega": omega,
   "t": t,
   "dw": dw,
   "Gplusomega": GRomega
}
with open("largedata.pkl", "wb") as outfile:
    pickle.dump(dictionary, outfile)	



print("*********Program exited successfully with itern = %d *********", itern)


if __name__ == "__main__":
	main()




























