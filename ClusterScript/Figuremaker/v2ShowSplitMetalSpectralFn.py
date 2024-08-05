import numpy as np 
from matplotlib import pyplot as plt
import sys
import os
from scipy.signal import find_peaks

if not os.path.exists('../Sources'):
	print("Error - Path to sources directory not found")
sys.path.insert(1,'../Sources')
from SYK_fft import *
import h5py
import matplotlib.pyplot as plt
plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.figsize'] = '8,7'


path_to_outfile = '../Dump/RTWHDumpfiles0_01/'
# path_to_outfile = './Dump/RTGapscaling/'
path_to_BH = '../Dump/RTWHDumpfiles0_00/'

# outfile = 'RTgapM16T12beta200g0_5lamb0_06.npy'
BH_outfile = 'l_00M16T12beta200g0_5lamb0_0.npy'


if not os.path.exists('../Dump/'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
    exit(1)
else:
	if not os.path.exists(path_to_outfile):
		print(path_to_outfile)
		raise Exception("Error - Path to Outfile not found ")
		exit(1)
	if not os.path.exists(path_to_BH):
		print(path_to_BH)
		raise Exception("Error - Path to BH not found ")
		exit(1)


M=int(2**16)
T=2**12
beta = 199
# betalist = [40,70]
# betalist = range(109,200,10)
betalist = [80,100,120,140,160,180,200]
temp=1./beta
g = 0.5
# lamb = 0.05
lamb = 0.01
J = 0.
# GDomega,GODomega,DDomega,DODomega = np.load(savepath)
omega,t  = RealGridMaker(M,T)
dt = t[2]-t[1]
dw = omega[2] - omega[1]
fig, ax = plt.subplots(2,2)
titlestring = 'Spectral functions for  ' + r'$\lambda = $ ' + str(lamb) 
# titlestring += ' g = ' + str(g)
fig.suptitle(titlestring)






for i, beta in enumerate(betalist):
	beta = 1.0*beta
	skip = 100
	col = 'C' + str(i)
	lab = r'$\beta = $ ' + str(int(beta))
	# savefile = 'RTgap'
	# savefile = 'l_05'
	savefile = 'l_01'
	savefile += 'M' + str(int(np.log2(M))) + 'T' + str(int(np.log2(T))) 
	# savefile += 'beta' + str((round(beta*100))/100.) 
	savefile += 'beta' + str(beta) 
	savefile += 'g' + str(g)
	savefile += 'lamb' + f'{lamb:.3}'
	savefile = savefile.replace('.','_') 
	savefiledump = savefile + '.npy'
	outfile = savefiledump
	# print(outfile)


	savepath = os.path.join(path_to_outfile, outfile)
	BHsavepath = os.path.join(path_to_BH, BH_outfile)
	if not os.path.exists(savepath):
		raise(Exception(f"WH Output file {outfile} not found"))
	if not os.path.exists(BHsavepath):
		raise(Exception("BH Output file not found"))

	loaded = np.array(np.load(savepath))
	loaded_BH = np.array(np.load(BHsavepath))

	GDomega,GODomega,DDomega,DODomega = loaded
	rhoGD = -np.imag(GDomega)
	rhoGOD = -np.imag(GODomega)
	rhoDD = -np.imag(DDomega)
	rhoDOD = -np.imag(DODomega)


	BHGDomega,BHGODomega,BHDDomega,BHDODomega = loaded_BH
	BHrhoGD = -np.imag(BHGDomega)
	BHrhoGOD = -np.imag(BHGODomega)
	BHrhoDD = -np.imag(BHDDomega)
	BHrhoDOD = -np.imag(BHDODomega)

	# peaks = find_peaks(rhoDOD,prominence=1)[0]
	# print(peaks)
	# peakvals = omega[peaks]
	# print(peakvals)

	# peakmax = omega[M + np.argmax(rhoGD[M:])]
	# print(f'peakmax at {peakmax}')


	ax[0,0].plot(omega, rhoGD, c = col, ls='-', label = lab )
	# ax[0,0].plot(omega, BHrhoGD, c = col, ls = '--')
	# ax[0,0].loglog(omega, rhoGD)
	# ax[0,0].loglog(omega, BHrhoGD, '--', label='BH')
	# ax[0,0].axvline(peakvals[::-1])
	ax[0,0].set_xlim(-0.05,0.05)
	# ax[0,0].set_xlim(-0.1,0.1)
	ax[0,0].set_ylabel(r'$-\Im{G^R_{d}(\omega)}$')
	ax[0,0].set_xlabel(r'$\omega$')
	ax[0,0].legend()



	ax[0,1].plot(omega, rhoGOD, c = col, ls='-', label = lab )
	# ax[0,1].plot(omega, BHrhoGOD,  c = col, ls = '--')
	ax[0,1].set_xlim(-0.05,0.05)
	# ax[0,1].set_xlim(-0.1,0.1)
	ax[0,1].set_ylabel(r'$-\Im{G^R_{od}(\omega)}$')
	ax[0,1].set_xlabel(r'$\omega$')
	ax[0,1].legend()	

	ax[1,0].plot(omega, rhoDD, c = col, ls='-', label = lab)
	ax[1,0].set_xlim(-1,1)
	# ax[1,0].set_ylim(-1.5,1.5)
	# ax[1,0].set_ylim(-0.5,0.5)
	ax[1,0].set_ylabel(r'$-\Im{D^R_{d}(\omega)}$')
	ax[1,0].set_xlabel(r'$\omega$')
	# ax[1,0].plot(omega, BHrhoDD, c = col, ls = '--')
	ax[1,0].legend()	

	ax[1,1].plot(omega, rhoDOD, c = col, ls='-', label = lab)
	ax[1,1].set_xlim(-1,1)
	ax[1,1].set_ylim(-0.3,0.3)
	ax[1,1].set_ylabel(r'$-\Im{D^R_{od}(\omega)}$')
	ax[1,1].set_xlabel(r'$\omega$')
	# ax[1,1].plot(omega, BHrhoDOD, c = col, ls = '--')
	ax[1,1].legend()


# plt.savefig('v2TempEvolnSpectralGapMetalWH.pdf', bbox_inches='tight')
plt.savefig('v3TempEvolnSpectralGapMetalWH.pdf', bbox_inches='tight')
plt.show()