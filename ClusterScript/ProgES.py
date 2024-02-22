import sys
import os 
if not os.path.exists('./Sources'):
    print("Error - Path to Sources directory not found ")
    raise(Exception("Error - Path to Sources directory not found "))
sys.path.insert(1,'./Sources')

import numpy as np
from matplotlib import pyplot as plt
from SYK_fft import *
from ConformalAnalytical import *
import testingscripts
from h5_handler import *


savename = 'default_savename'
path_to_output = './Outputs'
if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created")

path_to_subfolder = './Outputs/ProgESOutputs' 
if not os.path.exists(path_to_subfolder):
    os.makedirs(path_to_subfolder)
    print("Subfolder ProgESOutputs created")

path_to_dump = './Dump'
if not os.path.exists(path_to_dump):
    os.makedirs(path_to_dump)
    print("Dump directory created")

if len(sys.argv) > 1: 
    savename = str(sys.argv[1])

docstring = 'NULL'
#docstring = ' rhoLL = -ImG, rhoLR = 1j*ReG '


###################### Initialization Step #########################

Nbig = int(2**23)
#err = 1e-4
err = 1e-5
ITERMAX = 200

global beta

beta_start = 1
beta = beta_start
mu = 0.0
g = 0.5
r = 1.

target_beta = 500

# g = np.sqrt(10**3)
# r = (10)**2

kappa = 1.

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')

Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
delta = 0.420374134464041
omegar2 = ret_omegar2(g,beta)

#Gtau = Gfreetau
#Dtau = Dfreetau

#Gtau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
#Dtau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)

Gtau = -0.5*np.ones(Nbig)
Dtau = 1.0*np.ones(Nbig)

#Gtau,Dtau = np.load('temp.npy')
assert len(Gtau) == Nbig, 'Improperly loaded starting guess'

##################### Calculation Starts ###########################

for beta in range(beta_start, target_beta+1, 1):
    itern = 0
    diff = 1.
    diffG = 1.
    diffD = 1.
    x = 0.5
    xG = 0.5
    xD = 0.5

    print("##### NOW beta = ", beta, "############\n")

    omega = ImagGridMaker(Nbig,beta,'fermion')
    nu = ImagGridMaker(Nbig,beta,'boson')
    tau = ImagGridMaker(Nbig,beta,'tau')

    while(diff>err and itern < ITERMAX):
        itern+=1
        diffold = 1.0*diff
        diffoldG = 1.0*diffG
        diffoldD = 1.0*diffD
        
        oldGtau = 1.0*Gtau
        oldDtau = 1.0*Dtau
        
        if itern == 1:
            oldGomega = Time2FreqF(oldGtau,Nbig,beta)
            oldDomega = Time2FreqB(oldDtau,Nbig,beta)
        else:
            oldGomega = 1.0*Gomega
            oldDomega = 1.0*Domega
        
        Sigmatau = 1.0 * kappa * (g**2) * Dtau * Gtau
        Pitau = 2.0 * g**2 * Gtau * Gtau[::-1] #KMS G(-tau) = -G(beta-tau)
        
        Sigmaomega = Time2FreqF(Sigmatau,Nbig,beta)
        Piomega =  Time2FreqB(Pitau,Nbig,beta)
        # if itern < 15 : 
        #     Piomega[Nbig//2] = 1.0*r - omegar2
        #Piomega[Nbig//2] = 1.0*r - omegar2
        
        
        Gomega = xG*(1./(1j*omega + mu - Sigmaomega)) + (1-xG)*oldGomega
        Domega = xD*(1./(nu**2 + r - Piomega)) + (1-xD)*oldDomega

        Gtau = Freq2TimeF(Gomega,Nbig,beta)
        Dtau = Freq2TimeB(Domega,Nbig,beta)

        
        if itern>0:
            diffG = np.sqrt(np.sum((np.abs(Gtau-oldGtau))**2)) #changed
            diffD = np.sqrt(np.sum((np.abs(Dtau-oldDtau))**2))
            #diff = np.max([diffG,diffD])
            diff = 0.5*(diffG+diffD)
            diffG, diffD = diff, diff
            
            if diffG>diffoldG:
                xG/=2.
            if diffD>diffoldD:
                xD/=2.
            print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD)

    if beta % 100 == 0 :
        dumpfile = savename
        dumpfile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        dumpfile += 'g' + str(g).replace('.','_') + 'r' + str(r) + '.npy'  
        np.save(os.path.join(path_to_dump,dumpfile), np.array([Gtau,Dtau])) 
        print(dumpfile)

        ################      Compression     ######################
        tot_tau_grid_points = int(2**14)
        skip = int(Nbig/tot_tau_grid_points)
        comp_tau_slice = slice(0,-1,skip)
        comp_omega_slice = slice(Nbig//2, Nbig//2 + 10000)

        #################     Data Writing       ############ 
        print("\n###########Data Writing############")
        dictionary = {
           "g": g,
           "mu": mu,
           "beta": beta,
           "kappa": kappa,
           "Nbig": Nbig, 
           "tau": tau[comp_tau_slice], 
           "Gtau": Gtau[comp_tau_slice], 
           "Dtau": Dtau[comp_tau_slice], 
           "omega": omega[comp_omega_slice],
           "nu": nu[comp_omega_slice],  
           "Gomega": Gomega[comp_omega_slice],
           "Domega": Domega[comp_omega_slice],
           "compressed": True, 
           "docstring": docstring
        }
        savedictfile = savename
        savedictfile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
        savedictfile += 'g' + str(g).replace('.','_') + 'r' + str(r).replace('.','_')  + '.h5'  
        dict2h5(dictionary, os.path.join(path_to_subfolder, savedictfile), verbose=True)





print(f"*********Program exited successfully *********")












