import sys ####
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
path_to_output = './Outputs/RTGapscaling/'
path_to_dump = './Dump/RTGapscaling/'
# path_to_loadfile = './Dump/ProgRT_YSYK_Dumpfiles/'
# path_to_loadfile = './Dump/RTWHDumpfiles0_05/'
path_to_loadfile = './Dump/redoYWH/'

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

# M = int(2**18) #number of points in the grid
# T = 2**14 #upper cut-off for the time
M = int(2**22)
T = 2**18
# err = 1e-8
err = 1e-6

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

# beta_start = 301
# beta = beta_start
# target_beta = 2001.
# beta_step = 1

beta = 1200

lamb_start = 0.05
target_lamb = 0.001
# lamb_start = 0.051
# target_lamb = 0.1
J = 0.
####### DATA COMPRESSION #######
tot_freq_grid_points = int(2**14)
omega_max = 5
omega_min = -1*omega_max
idx_min, idx_max = omega_idx(omega_min,dw,M), omega_idx(omega_max,dw,M)
skip = int(np.ceil((omega_max-omega_min)/(dw*tot_freq_grid_points)))
comp_omega_slice = slice(idx_min,idx_max,skip)
#comp_omega = omega[comp_omega_slice]

#############################

# betasavelist = np.array([20,50,100,150,200,300,400,500,700,800,1000,1200,1500,1800,2000,5000])
# betasavelist = np.arange(beta_start,target_beta+1,5) - 1
lamblooplist = np.arange(lamb_start,target_lamb-1e-10, -0.001)
lambsavelist = lamblooplist

try:
    # GFs = np.load(os.path.join(path_to_loadfile,'l_05M16T12beta199g0_5lamb0_05.npy'))
    GFs = np.load(os.path.join(path_to_loadfile,'YWHM22T18beta1200g0_5lamb0_05.npy'))
    print('Input File successfully loaded')
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
for lamb in lambsavelist:
    #beta_step = 0.01 if (beta<1) else 1
    GFs, INFO = RE_WHYSYK_iterator(GFs,grid,pars,beta,lamb,J,err=err,x=0.01,ITERMAX=ITERMAX,eta = eta,verbose=True,diffcheck=False) 
    # itern, diff, x = INFO
    if np.isclose(lambsavelist,lamb).any():
        savefile = savename
        savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
        # savefile += 'beta' + str((round(beta*100))/100.) 
        savefile += 'beta' + str(beta) 
        savefile += 'g' + str(g)
        savefile += 'lamb' + f'{lamb:.3}'
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
    print("##### Finished lamb = ", lamb, " INFO = ", INFO, flush = True)



   
print(f"*********Program exited successfully *********")





