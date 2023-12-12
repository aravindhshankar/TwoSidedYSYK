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


savename = 'default_savename'
path_to_output = './Outputs'

if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
	print("Outputs directory created")

if len(sys.argv) > 1:
	savename = str(sys.argv[1])

savefile = os.path.join(path_to_output, savename+'.pkl')

fft_check = testingscripts.realtimeFFT_validator() # Should return True

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
beta = 1./(2e-4)
#beta = 50.

M = int(2**24) #number of points in the grid
T = int(2**16) #upper cut-off fot the time
omega, t = RealGridMaker(M,T)
dw = omega[2]-omega[1]
dt = t[2] - t[1]
grid_flag = testingscripts.RealGridValidator(omega,t, M, T, dt, dw)
err = 1e-2
eta = dw*10.
#delta = 0.420374134464041

print("T = ", T, ", dw =  ", f'{dw:.6f}', ", dt = ", f'{dt:.6f}', ', omega_max = ', f'{omega[-1]:.3f}' ) 
print("dw/temp = ", f'{dw*beta:.4f}')
print("flag fft_check = ", fft_check)
print("grid_flag = ", grid_flag)

## State varriables go into .out file
print("######## State Variables ################")
print("J = ", J)
print("mu = ", mu)
print("M = ", M)
print("eta = ", eta)
print("T = ", T)
print("err = ", err)
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

	    #print("itern = ",itern, " , diff = ", diffG, " , x = ", xG, end = '\r')
	    print("itern = ",itern, " , diff = ", diffG, " , x = ", xG)



	#GRt = (0.5/np.pi) * freq2time(GRomega,M,dt)
	rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
	rhoLL, rhoLR = 0.5 * (rhoG + rhoGrev), 0.5 * (rhoG - rhoGrev)
	GLLomega = -1j*(1.-fermidirac(beta*omega))*rhoLL
	GLRomega = -1j*(1.-fermidirac(beta*omega))*rhoLR
	TLLt = 2 * np.abs((0.5/np.pi)*freq2time(GLLomega,M,dt)) 
	TLRt = 2 * np.abs((0.5/np.pi)*freq2time(GLRomega,M,dt))


	#################Data Compression################
	total_freq_grid_points = 500
	omega_max = 1
	idx = omega_idx(omega_max,dw,M)
	#compress_slice = slice()








	###########Data Writing############ 
	print("\n###########Data Writing############")
	dictionary = {
	   "J": J,
	   "mu": mu,
	   "beta": beta,
	   "M": M, 
	   "T": T,
	   "dw": dw,
	   "rhoLL": rhoLL,
	   "rhoLR": rhoLR, 
	   "TLLt": TLLt,
	   "TLRt": TLRt 
	}
	with open(savefile, "wb") as outfile:
	    pickle.dump(dictionary, outfile)	

	print(f"*********Program exited successfully with itern = {itern} *********")


if __name__ == "__main__":
	main()




#with open('largedata.pkl','rb') as fp:
#   loaded = pickle.load(fp)
#print("success")
#plt.plot(loaded["omega"], loaded["rhoLL])
#loaded['beta'], loaded['T'], loaded['mu']























