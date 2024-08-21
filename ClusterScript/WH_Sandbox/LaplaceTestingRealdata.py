import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	

if not os.path.exists('../FittingTests'):
	print("Error - Path to FittingTests directory not found ")
	raise Exception("Error - Path to FittingTests directory not found ")
else:	
	sys.path.insert(1,'../FittingTests')	


# if not os.path.exists('../Dump/WHYSYKImagDumpfiles'):
#     print("Error - Path to Dump directory not found ")
#     raise Exception("Error - Path to Dump directory not found ")
# else:
#     path_to_dump = '../Dump/WHYSYKImagDumpfiles'

if not os.path.exists('../Dump/'):
	print("Error - Path to Dump directory not found ")
	raise Exception("Error - Path to Dump directory not found ")
	exit(1)
else:

	path_to_dump_lamb = '../Dump/PushErrDownImagWH/'

	
	if not os.path.exists(path_to_dump_lamb):
		raise Exception('Generate Data first! Path to lamb dump not found')
		exit(1)
	# if not os.path.exists(path_to_dump_temp):
	# 	raise Exception('Generate Data first! Path to temp dump not found')
	# 	exit(1)



from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from scipy.optimize import curve_fit
from prony import prony
from EDSF import fitEDSF
from LaplaceTransform import laplace, PadeLaplacematrixsolver
#import time

# plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# plt.rcParams['legend.fontsize'] = '14'
# print(plt.style.available)
# plt.style.use('seaborn-v0_8')

# Nbig = int(2**14)
Nbig = int(2**16)
err = 1e-5

global beta

betaBH = 50
betaWH = 80
# betalist = [betaBH,betaWH]
# betalist = [80,]
# betalist = [60,61,62,63,64,65,66,67,68,59,70]
# betalist = range(10,90)
beta = 5000
mu = 0.0
g = 0.5
# g = 2.
r = 1.

kappa = 1.
# lamb = 0.05
lamblist = [0.001,]
# lamblist = [0.003,]
J = 0
lamb = lamblist[0]
path_to_dump = path_to_dump_lamb
# path_to_dump = path_to_dump_temp

whichplot = 'GD' #choices GD or DD
# l0,l1 = 0.35,0.4
# l2,l3 = 0.15,0.21
# l4,l5 = 0.04,0.06
# l4,l5 = 0.005,0.008
l0,l1 = 0.05,0.38
l2,l3 = 0.04,0.06
l4,l5 = 0.038,0.04
# l0,l1 = 0.39,0.4
# l2,l3 = 0.15,0.2
# l4,l5 = 0.1,0.11
# l0,l1 = 0.02,0.42
# l2,l3 = 0.04,0.06
# l4,l5 = 0.038,0.04

fig2, ax2 = plt.subplots(1, figsize=(8,7)) 
titlestring = ''
titlestring =  r' $\beta $ = ' + str(beta)
titlestring += r' $\lambda$ = ' + f'{lamb:.3}' 
ax2.set_xlabel(r'$\tau/\beta$')
ax2.set_title(titlestring)

# for i, beta in enumerate(betalist):
for i, lamb in enumerate(lamblist):
	savefile = ''
	# savefile += 'ERR7'
	savefile += 'ERR10'
	savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	savefile += 'lamb' + str(lamb) + 'J' + str(J)
	savefile += 'g' + str(g) + 'r' + str(r)
	savefile = savefile.replace('.','_') 
	savefile += '.npy'

	try :
		#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
		plotfile = os.path.join(path_to_dump, savefile)
	except FileNotFoundError: 
		print('Filename : ', savefile)
		print("INPUT FILE NOT FOUND") 
		exit(1)
	print('savefile = ', savefile)

	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')


	Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
	Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
	delta = 0.420374134464041
	omegar2 = ret_omegar2(g,beta)

	#Gtau = Gfreetau
	#Dtau = Dfreetau

	GDtau, GODtau, DDtau, DODtau = np.load(plotfile)
	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

	GDomega = Time2FreqF(GDtau, Nbig, beta)
	GODomega = Time2FreqF(GODtau, Nbig, beta)
	DDomega = Time2FreqB(DDtau, Nbig, beta)
	DODomega = Time2FreqB(DODtau, Nbig, beta)


	################## PLOTTING ######################
	print(beta), print(tau[-1])
	Gconftau = Freq2TimeF(GconfImag(omega,g,beta),Nbig,beta)
	Dconftau = Freq2TimeB(DconfImag(nu,g,beta),Nbig,beta)
	FreeDtau = DfreeImagtau(tau,r,beta)

	if whichplot == 'GD':
		plottable = np.abs(np.real(GDtau))
		ax2.set_ylabel(r'$|G_d(\tau)|$')
	elif whichplot == 'DD':
		plottable = np.abs(np.real(DDtau))
		ax2.set_ylabel(r'$|D_d(\tau)|$')
	else:
		print('INVALID OPTION FOR PLOTTABLE')
		exit(1)
	startT, stopT = 0, Nbig//2
	# skip = 50
	skip = 1
	xaxis = tau[startT:stopT:skip]/beta
	# yaxis = 
	ax2.semilogy(xaxis, plottable[startT:stopT:skip],'p',label = 'Exact numerics ',markersize=5,c='C2')
	# ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
	ax2.axvline(l0, ls='--')
	ax2.axvline(l1, ls='--')
	# ax2.axvline(1./(beta**2),ls='--', c='gray', label = 'Temperature')
	# ax2.axvline(1./(lambval*beta), ls='--', c='green',label=r'$\lambda^{-1}$')
	#ax2.legend()



	# logder = np.gradient(np.log(plottable),tau)
	start_idx = np.argmin(np.abs(xaxis-l0))
	stop_idx = np.argmin(np.abs(xaxis-l1))
	# print(start_idx,stop_idx)
	fitslice = slice(start_idx,stop_idx)
	# slope = -np.mean(logder[startT:stopT][fitslice])
	# print(slope)
	slope, logA = np.polyfit(xaxis[fitslice],np.log(plottable[startT:stopT:skip][fitslice]),1)
	slope = -1.*slope
	gamma = slope / beta
	A = np.exp(logA)
	ax2.semilogy(xaxis, A*np.exp(-slope * xaxis),ls='-.', label = f' fit with $\\gamma$ calculated to be {gamma:.4}')
	c = gamma/delta
	print(f"position of first peak = {c*delta:.4}")
	print(f'position of second peak = {c*(1+delta):.4}')
	print(c*(np.arange(5)+delta))

	remnant1 = plottable[startT:stopT:skip]-(A*np.exp(-slope * xaxis)) 
	# ax2.semilogy(xaxis, remnant1, ls='-.', label = f' First Remnant')


	# sec_peak = c*(2+delta)
	# ax2.semilogy(xaxis, A*np.exp(-sec_peak *beta * xaxis),ls='-.', label = f' second slope {sec_peak:.4}')
	# # ax2.semilogy(xaxis, A*np.exp(-lamb *beta * xaxis),ls='-.', label = f' second slope {lamb:.4}')
	l0idx = np.argmin(np.abs(l0-xaxis))
	l1idx = np.argmin(np.abs(l1-xaxis))
	s = np.linspace(0,10, 1000)
	# Fs = laplace(tau[startT:stopT:skip],plottable[startT:stopT:skip],s)
	Fs = laplace(tau[startT:stopT:skip],plottable[startT:stopT:skip],s)
	print(len(plottable[l0idx:l1idx]))
	figS, (axS,axgrad) = plt.subplots(2)
	axS.plot(s,np.abs(Fs))
	axS.set_yscale('log')
	axS.set_xlabel('s')
	axS.set_ylabel('F[s]')

	gradlaplace = np.gradient(Fs, s)
	gradlaplace2 = (1./2.) * np.gradient(gradlaplace, s)
	gradlaplace3 = (1./3.) * np.gradient(gradlaplace2, s)
	gradlaplace4 = (1./4.) * np.gradient(gradlaplace3, s)
	axgrad.plot(s,np.abs(gradlaplace),label ='first derivative')
	axgrad.plot(s,np.abs(gradlaplace2),label ='second derivative')
	axgrad.plot(s,np.abs(gradlaplace3),label ='third derivative')
	axgrad.plot(s,np.abs(gradlaplace4),label ='fourth derivative')
	axgrad.set_xlabel('s')
	axgrad.set_ylabel('|grad F(s)|')
	axgrad.set_title('Gradient of the laplace transform')
	axgrad.set_yscale('log')
	axgrad.legend()

	N = 2 # we need at least 2n-1 derivatives
	s0 = 3
	s0idx = np.argmin(np.abs(s-s0))
	assert s0idx < len(s) , "pick a valid s0 ya DOLT!"

	d_arr = np.array((Fs[s0idx], gradlaplace[s0idx], gradlaplace2[s0idx], gradlaplace3[s0idx],gradlaplace4[s0idx]))

	exponents = PadeLaplacematrixsolver(d_arr,N,s0)
	print(f's0 = {s0}')
	print('Found Exponents are ', exponents)
	########## BOTTOM LINE : WE CAN SAFELY PICK s=3 , the first 4 derivatives are converged at that point ##################

	####### Completely hopeless :( ##############


# plt.savefig('../../KoenraadEmails/Fitting Higher exponentials.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
ax2.legend()

plt.show()






















