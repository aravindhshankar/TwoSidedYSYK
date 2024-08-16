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


savename = 'default_savename'
# path_to_output = './Outputs/RTWH/NFLstart/'
# path_to_dump = './Dump/RTWHDumpfiles/NFLstart'
path_to_output = './Outputs/lowertempRTWH/'
path_to_dump = './Dump/lowertempRTWH/'
# path_to_loadfile = './Dump/ProgRT_YSYK_Dumpfiles/'
# path_to_loadfile = './Dump/RTWHDumpfiles0_01/'
path_to_loadfile = './Dump/lowertempRTWH/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created")

if not os.path.exists(path_to_dump):
    os.makedirs(path_to_dump)
    print("Dump directory created")

if len(sys.argv) > 1:
    savename = str(sys.argv[1])

if not os.path.exists(path_to_loadfile):
    print('Load directory not found!')
    exit(1)
# savefile = os.path.join(path_to_output, savename+'.h5')

DUMP = True
PLOTTING = False

# M = int(2**18) #number of points in the grid
# T = 2**14 #upper cut-off for the time
# M = int(2**16)
# T = 2**12
M = int(2**19)
T = 2**15
# err = 1e-8
err = 1e-3

omega,t  = RealGridMaker(M,T)
dw = omega[2] - omega[1]
dt = t[2] - t[1]

print("dw = ", dw)
print("dt = ", dt)


delta = 0.420374134464041
ITERMAX = 5000
#global beta

mu = 0.0
g = 0.5
r = 1.
kappa = 1.
eta = dw*2.1

# beta_start = 200
beta_start = 500

beta = beta_start
target_beta = 2000
beta_step = 10
betalooplist = np.arange(beta_start,target_beta+beta_step+beta_step, beta_step)
lamb = 0.01
J = 0
print("T Target = ", 1/target_beta)
####### DATA COMPRESSION #######
tot_freq_grid_points = int(2**14)
omega_max = 10
omega_min = -1*omega_max
idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
comp_omega_slice = slice(idx_min,idx_max,skip)
#comp_omega = omega[comp_omega_slice]

#############################

# betasavelist = np.array([20,50,100,150,200,300,400,500,700,800,1000,1200,1500,1800,2000,5000])
betasavelist = np.arange(beta_start,target_beta+beta_step, 10*beta_step)
print(f'betasavelist[-1] = {betasavelist[-1]}')
# betasavelist = np.arange(beta_start,target_beta+1,5) - 1

try:
    # GFs = np.load(os.path.join(path_to_loadfile,'l_01M16T12beta200_0g0_5lamb0_01.npy'))
    GFs = np.load(os.path.join(path_to_loadfile,'lowertempM19T15beta500g0_5lamb0_01.npy'))
except FileNotFoundError:
    print('INPUT FILE NOT FOUND!!!!!!')
    exit(1)

# GDRomega = (omega + 1j*eta + mu)/((omega+1j*eta + mu)**2 - lamb**2)
# DDRomega = (-1.0*(omega + 1j*eta)**2 + r)/((r - (omega+1j*eta)**2)**2 - (J)**2)
# GODRomega = -lamb/((omega+1j*eta + mu)**2 - lamb**2)
# DODRomega = -J / ((r - (omega+1j*eta)**2)**2 - (J)**2)

# GDRomega = 1./ (omega + 1j*eta + mu)
# DDRomega = 1./(-1.0*(omega + 1j*eta)**2 + r)
# GODRomega = np.zeros_like(GDRomega)
# DODRomega = np.zeros_like(DDRomega)

# GFs = [GDRomega,GODRomega,DDRomega,DODRomega]
grid = [M,omega,t]
pars = [g,mu,r]
x=0.1
# while(beta < target_beta):
for beta in betalooplist:
    #beta_step = 0.01 if (beta<1) else 1
    GFs, INFO = RE_WHYSYK_iterator(GFs,grid,pars,beta,lamb,J,err=err,x=x,ITERMAX=ITERMAX,eta = eta,verbose=False,diffcheck=False) 
    # itern, diff, x = INFO
    if beta in betasavelist:
        savefile = savename
        savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
        # savefile += 'beta' + str((round(beta*100))/100.) 
        savefile += 'beta' + str(beta) 
        savefile += 'g' + str(g)
        savefile += 'lamb' + str(lamb)
        savefile = savefile.replace('.','_') 
        savefiledump = savefile + '.npy' 
        savefileoutput = savefile + '.h5'

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
           "rhoGD": -np.imag(GFs[0][comp_omega_slice]),
           "rhoGOD": -np.imag(GFs[1][comp_omega_slice]),
           "rhoDD": -np.imag(GFs[2][comp_omega_slice]),
           "rhoDOD": -np.imag(GFs[3][comp_omega_slice]), 
           "compressed": True, 
           "eta": eta, 
           "INFO": INFO
        }

        if DUMP == True:    
            dict2h5(dictionary, os.path.join(path_to_output,savefileoutput), verbose=True)

        if DUMP == True:
            np.save(os.path.join(path_to_dump,savefiledump), GFs) 
    print("##### Finished beta = ", beta, " INFO = ", INFO, flush = True)
    # beta = beta + beta_step



   
print(f"*********Program exited successfully *********")





