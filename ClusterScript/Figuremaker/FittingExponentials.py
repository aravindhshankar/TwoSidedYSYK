import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


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
#import time

plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
plt.rcParams['figure.figsize'] = '8,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1.6'
plt.rcParams['axes.labelsize'] = '16'
plt.rcParams['axes.titlesize'] = '16'

plt.rcParams['legend.fontsize'] = '12'
# print(plt.style.available)
# plt.style.use('seaborn-v0_8')

# Nbig = int(2**14)
Nbig = int(2**16)
err = 1e-5

global beta

beta = 5000
mu = 0.0
g = 0.5
# g = 2.
r = 1.

kappa = 1.

lamblist = [0.001,]

J = 0
lamb = lamblist[0]
path_to_dump = path_to_dump_lamb
# path_to_dump = path_to_dump_temp

whichplot = 'GD' #choices GD or DD


l0,l1 = 0.11,0.38
l2,l3 = 0.04,0.09
l4,l5 = 0.038,0.04

# l0,l1 = 0.10,0.38
# l2,l3 = 0.04,0.09
# l4,l5 = 0.038,0.04

l0,l1 = 0.10,0.38
l2,l3 = 0.05,0.09
l4,l5 = 0.038,0.04

# l0,l1 = 0.34,0.4
# l2,l3 = 0.08,0.14
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
	skip = 50
	xaxis = tau[startT:stopT:skip]/beta
	ax2.semilogy(xaxis, plottable[startT:stopT:skip],'p',label = 'Exact numerics ',markersize=1.4,c='C0')
	# ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
	ax2.axvline(l0, ls=':',c='C1')
	ax2.axvline(l1, ls=':',c='C1')


	# logder = np.gradient(np.log(plottable),tau)
	start_idx = np.argmin(np.abs(xaxis-l0))
	stop_idx = np.argmin(np.abs(xaxis-l1))
	
	fitslice = slice(start_idx,stop_idx)
	# slope = -np.mean(logder[startT:stopT][fitslice])
	# print(slope)
	slope, logA = np.polyfit(xaxis[fitslice],np.log(plottable[startT:stopT:skip][fitslice]),1)
	slope = -1.*slope
	gamma = slope / beta
	A = np.exp(logA)
	ax2.semilogy(xaxis, A*np.exp(-slope * xaxis),ls='--', label = f' fit with $\\gamma$ calculated to be {gamma:.4}', c='C1')
	c = gamma/delta

	print('Theoretical En = c(n + deleta) = ', c*(np.arange(5)+delta))

	remnant1 = plottable[startT:stopT:skip]-(A*np.exp(-slope * xaxis)) 


	#############  SECOND FIT STARTS #######################
	secstart_idx = np.argmin(np.abs(xaxis-l2))
	secstop_idx = np.argmin(np.abs(xaxis-l3))

	ax2.semilogy(xaxis, remnant1, ls='-', label = f' First Remnant')

	ax2.axvline(l2,ls=':',c='C2')
	ax2.axvline(l3,ls=':',c='C2')
	secfitslice = slice(secstart_idx,secstop_idx)
	print(f'Number of points in second fit slice = {len(xaxis[secfitslice])}')

	sec_slope, logB= np.polyfit(xaxis[secfitslice],np.log(remnant1[secfitslice]),1)
	B  = np.exp(logB) 
	zeta = - sec_slope/beta
	ax2.semilogy(xaxis, (B)*np.exp(-(zeta)* beta * xaxis),ls='--', c='C2', label = f' fit with $\\zeta$ calculated to be {zeta:.4}')
	print(f'A = {A:.4}, B = {B:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}')

	first_approx = A * np.exp(-gamma*beta*xaxis) + B * np.exp(-zeta*beta*xaxis)
	ax2.semilogy(xaxis, first_approx, label = 'Sum of two exponentials', ls='-.',c='C3')
 
	remnant2 = remnant1 - B * np.exp(-zeta*beta*xaxis)
	ax2.semilogy(xaxis, remnant2, c='C4', ls='-', label = f' Second Remnant')


	print(f'A = {A:.4}, B = {B:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}')
	print(f'zeta - gamma = {zeta-gamma}') 
	print(f'c = {c}')
	print(f'expected spacing = c')



	# ax2.set_xlim(-0.005,0.5)
	if whichplot == 'GD':
		ax2.set_ylim(1e-7,1)
	elif whichplot == 'DD': 
		ax2.set_ylim(1e-5,1)

	textstr = ''
	textstr += f'First slope $\\gamma =  $ {gamma:.4} \n'
	textstr += f'calculated c = $\\gamma/\\Delta = $ {c:.4} \n'
	textstr += f' predicted next slope at $c(1+\\Delta ) = $ {c*(1+delta):.4} \n'
	textstr += f' Fit second slope from remnant = {zeta:.4}'
	# textstr = f'expected spacing {c} \n' + 
	ax2.text(0.29, 0.9, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
	# ax2.text(0.21, 0.9, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')

	percentageerror = 100 * np.abs(zeta - c*(1+delta)) / (c*(1+delta)) 
	print(percentageerror)


# plt.savefig('../../KoenraadEmails/Fitting Higher exponentials.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
ax2.legend(fontsize=8)
if whichplot == 'GD':
	plt.savefig('ExponentsGD.pdf',bbox_inches='tight')
elif whichplot == 'DD':
	plt.savefig('ExponentsDD.pdf',bbox_inches='tight')

plt.show()






















