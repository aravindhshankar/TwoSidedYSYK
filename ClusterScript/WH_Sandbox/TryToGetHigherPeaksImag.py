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
	# path_to_dump = '../Dump/WHYSYKImagDumpfiles/SCWH'
	# path_to_dump = '../Dump/lamb_anneal_dumpfiles/'
	# path_to_dump_lamb = '../Dump/xshift_lamb_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/temp_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/xshift_temp_anneal_dumpfiles/'
	# path_to_dump_temp_fwd = '../Dump/24Aprzoom_x0_01_temp_anneal_dumpfiles/fwd/'
	# path_to_dump_temp_rev = '../Dump/24Aprzoom_x0_01_temp_anneal_dumpfiles/rev/'
	# path_to_dump = '../Dump/gap_powerlawx01_lamb_anneal_dumpfiles/'
	# path_to_dump_lamb = '../Dump/lamb_anneal_dumpfiles/'
	path_to_dump_lamb = '../Dump/LOWTEMP_lamb_anneal_dumpfiles/'

	# path_to_dump_temp = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd'


	
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
#import time

# plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# plt.rcParams['legend.fontsize'] = '14'
# print(plt.style.available)
plt.style.use('seaborn-v0_8')

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
lamblist = [0.003,]
J = 0

path_to_dump = path_to_dump_lamb
# path_to_dump = path_to_dump_temp


# fig, ax = plt.subplots(2,2, figsize=(8,7))
# titlestring = "Imaginary time Green's functions for "
# titlestring =  r' $\beta $ = ' + str(beta)
# # titlestring +=  r', $g = $' + str(g)
# # titlestring += r' $\lambda$ = ' + str(lamb) 
# fig.suptitle(titlestring)
# fig.tight_layout(pad=2)


fig2, ax2 = plt.subplots(1, figsize=(8,7)) 
titlestring = ''
# titlestring =  r' $\beta $ = ' + str(beta)
# titlestring += r' $\lambda$ = ' + f'{lamb:.3}' 
ax2.set_xlabel(r'$\tau/\beta$')
ax2.set_ylabel(r'$|G_d(\tau)|$')
ax2.set_title(titlestring)

# for i, beta in enumerate(betalist):
for i, lamb in enumerate(lamblist):
	savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
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


	plottable = np.abs(np.real(DDtau))
	startT, stopT = 0, Nbig//2
	# skip = 50
	skip = 1
	xaxis = tau[startT:stopT:skip]/beta
	# yaxis = 
	ax2.semilogy(xaxis, plottable[startT:stopT:skip],'p',label = 'numerics DDtau',markersize=2,c='C2')
	# ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
	ax2.axvline(0.1, ls='--')
	ax2.axvline(0.2, ls='--')
	# ax2.axvline(1./(beta**2),ls='--', c='gray', label = 'Temperature')
	# ax2.axvline(1./(lambval*beta), ls='--', c='green',label=r'$\lambda^{-1}$')
	#ax2.legend()



	# logder = np.gradient(np.log(plottable),tau)
	start_idx = np.argmin(np.abs(xaxis-0.1))
	stop_idx = np.argmin(np.abs(xaxis-0.2))
	# print(start_idx,stop_idx)
	fitslice = slice(start_idx,stop_idx)
	# slope = -np.mean(logder[startT:stopT][fitslice])
	# print(slope)
	slope, logA = np.polyfit(xaxis[fitslice],np.log(plottable[startT:stopT:skip][fitslice]),1)
	slope = -1.*slope
	gamma = slope / beta
	A = np.exp(logA)
	ax2.semilogy(xaxis, A*np.exp(-slope * xaxis),ls='-.', label = f' fit with $\gamma$ calculated to be {gamma:.4}')
	c = gamma/delta
	print(f"position of first peak = {c*delta:.4}")
	print(f'position of second peak = {c*(1+delta):.4}')
	print(c*(np.arange(5)+delta))

	remnant1 = plottable[startT:stopT:skip]/(A*np.exp(-slope * xaxis)) - 1 
	ax2.semilogy(xaxis, remnant1, ls='-.', label = f' First Remnant')


	# sec_peak = c*(2+delta)
	# ax2.semilogy(xaxis, A*np.exp(-sec_peak *beta * xaxis),ls='-.', label = f' second slope {sec_peak:.4}')
	# # ax2.semilogy(xaxis, A*np.exp(-lamb *beta * xaxis),ls='-.', label = f' second slope {lamb:.4}')


	##############  SECOND FIT STARTS #######################
	secstart_idx = np.argmin(np.abs(xaxis-0.015))
	secstop_idx = np.argmin(np.abs(xaxis-0.035))
	secfitslice = slice(secstart_idx,secstop_idx)
	print(f'Number of points in second fit slice = {len(xaxis[secfitslice])}')

	sec_slope, logB_A = np.polyfit(xaxis[secfitslice],np.log(remnant1[secfitslice]),1)
	B  = A * np.exp(logB_A) 
	zeta = gamma - sec_slope/beta
	ax2.semilogy(xaxis, (B/A)*np.exp(-(zeta-gamma)* beta * xaxis),ls='-.', label = f' fit with $\zeta$ calculated to be {zeta:.4}')
	print(f'A = {A:.4}, B = {B:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}')

	first_approx = A * np.exp(-gamma*beta*xaxis) + B * np.exp(-zeta*beta*xaxis)
	ax2.semilogy(xaxis, first_approx, label = 'first_approx', ls=':')

	todiv = (B/A) * np.exp(-(zeta-gamma)*beta*xaxis) 
	remnant2 = remnant1/todiv - 1
	ax2.semilogy(xaxis, remnant2, ls='-.', label = f' Second Remnant')

	############# THIRD FIT STARTS #############################
	thistart_idx = np.argmin(np.abs(xaxis-0.004))
	thistop_idx = np.argmin(np.abs(xaxis-0.014))
	thifitslice = slice(thistart_idx,thistop_idx)
	print(f'Number of points in second fit slice = {len(xaxis[thifitslice])}')

	thi_slope, logD_B = np.polyfit(xaxis[thifitslice],np.log(remnant2[thifitslice]),1)
	D  = B * np.exp(logD_B) 
	xi = zeta - thi_slope/beta
	ax2.semilogy(xaxis, (D/B)*np.exp(-(xi-zeta)* beta * xaxis),ls='-.', label = f' fit with $\zeta$ calculated to be {zeta:.4}')
	print(f'A = {A:.4}, B = {B:.4}, D = {D:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}, xi = {xi:.4}')
	print(f'xi - beta  = {xi-beta}')

	second_approx = A * np.exp(-gamma*beta*xaxis) + B * np.exp(-zeta*beta*xaxis) + D * np.exp(-xi * beta * xaxis)
	ax2.semilogy(xaxis, second_approx, label = 'second_approx', ls=':')

	todiv = (D/B) * np.exp(-(xi-zeta)*beta*xaxis) 
	remnant3 = remnant2/todiv - 1
	ax2.semilogy(xaxis, remnant3, ls='-.', label = f' Second Remnant')
















# plt.savefig('GreenFunctionPlotsMetal.pdf', bbox_inches='tight')

#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
ax2.legend()

plt.show()


















