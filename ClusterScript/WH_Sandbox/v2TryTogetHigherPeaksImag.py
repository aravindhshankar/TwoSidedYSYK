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
	# path_to_dump_lamb = '../Dump/LOWTEMP_lamb_anneal_dumpfiles/'
	path_to_dump_lamb = '../Dump/PushErrDownImagWH/'

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
from scipy.optimize import curve_fit
from prony import prony
from EDSF import fitEDSF
#import time

plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
plt.rcParams['figure.figsize'] = '8,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1.6'

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
# l0,l1 = 0.05,0.38 ## These three are the best ones so far
# l2,l3 = 0.04,0.06 ## 
# l4,l5 = 0.038,0.04 ##

l0,l1 = 0.11,0.38
l2,l3 = 0.04,0.09
l4,l5 = 0.038,0.04

l0,l1 = 0.10,0.38
l2,l3 = 0.04,0.09
l4,l5 = 0.038,0.04

l0,l1 = 0.10,0.38
l2,l3 = 0.05,0.09
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
	ax2.semilogy(xaxis, plottable[startT:stopT:skip],'p',label = 'Exact numerics ',markersize=1.6,c='C0')
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
	# print(f"position of first peak = {c*delta:.4}")
	# print(f'position of second peak = {c*(1+delta):.4}')
	print('Theoretical En = c(n + deleta) = ', c*(np.arange(5)+delta))

	remnant1 = plottable[startT:stopT:skip]-(A*np.exp(-slope * xaxis)) 
	# ax2.semilogy(xaxis, remnant1, ls='-.', label = f' First Remnant')


	# sec_peak = c*(2+delta)
	# ax2.semilogy(xaxis, A*np.exp(-sec_peak *beta * xaxis),ls='-.', label = f' second slope {sec_peak:.4}')
	# # ax2.semilogy(xaxis, A*np.exp(-lamb *beta * xaxis),ls='-.', label = f' second slope {lamb:.4}')


	#############  SECOND FIT STARTS #######################
	secstart_idx = np.argmin(np.abs(xaxis-l2))
	secstop_idx = np.argmin(np.abs(xaxis-l3))
	# ax2.semilogy(xaxis[:secstop_idx+5000], remnant1[:secstop_idx+5000], ls='-', label = f' First Remnant')
	ax2.semilogy(xaxis, remnant1, ls='-', label = f' First Remnant')
	# ax2.semilogy(xaxis[:secstop_idx], remnant1[:secstop_idx], ls='-', label = f' First Remnant')
	ax2.axvline(l2,ls='--',c='C3')
	ax2.axvline(l3,ls='--',c='C3')
	secfitslice = slice(secstart_idx,secstop_idx)
	print(f'Number of points in second fit slice = {len(xaxis[secfitslice])}')

	sec_slope, logB= np.polyfit(xaxis[secfitslice],np.log(remnant1[secfitslice]),1)
	B  = np.exp(logB) 
	zeta = - sec_slope/beta
	ax2.semilogy(xaxis, (B)*np.exp(-(zeta)* beta * xaxis),ls=':', c='C3', label = f' fit with $\\zeta$ calculated to be {zeta:.4}')
	print(f'A = {A:.4}, B = {B:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}')

	first_approx = A * np.exp(-gamma*beta*xaxis) + B * np.exp(-zeta*beta*xaxis)
	ax2.semilogy(xaxis, first_approx, label = 'Sum of two exponentials', ls='-.')
 
	remnant2 = remnant1 - B * np.exp(-zeta*beta*xaxis)

	############ THIRD FIT STARTS #############################
	thistart_idx = np.argmin(np.abs(xaxis-l4))
	thistop_idx = np.argmin(np.abs(xaxis-l5))
	# ax2.semilogy(xaxis[:thistop_idx+100], remnant2[:thistop_idx+100], c='C4', ls='-', label = f' Second Remnant')
	# ax2.semilogy(xaxis[:thistop_idx], remnant2[:thistop_idx], c='C4', ls='-', label = f' Second Remnant')
	ax2.semilogy(xaxis, remnant2, c='C4', ls='-', label = f' Second Remnant')
	# ax2.axvline(l4,ls='--',c='C5')
	# ax2.axvline(l5,ls='--',c='C5')
	thifitslice = slice(thistart_idx,thistop_idx)
	print(f'Number of points in third fit slice = {len(xaxis[thifitslice])}')

	thi_slope, logD = np.polyfit(xaxis[thifitslice],np.log(remnant2[thifitslice]),1)
	D  = np.exp(logD) 
	xi =  - thi_slope/beta
	# ax2.semilogy(xaxis, D*np.exp(-xi* beta * xaxis),ls=':', c='C5', label = f' fit with $\\xi$ calculated to be {xi:.4}')
	D,xi = 0., 0.
	print(f'A = {A:.4}, B = {B:.4}, D = {D:.4}') 
	print(f'gamma = {gamma:.4}, zeta = {zeta:.4}, xi = {xi:.4}')
	# print(f'xi - zeta  = {xi-zeta}')
	print(f'zeta - gamma = {zeta-gamma}') 
	print(f'c = {c}')
	print(f'expected spacing = c')

	second_approx = A * np.exp(-gamma*beta*xaxis) + B * np.exp(-zeta*beta*xaxis) + D * np.exp(-xi * beta * xaxis)
	# ax2.semilogy(xaxis, second_approx, label = 'Sum of 3 exponentials', ls='-.')

	remnant3 = remnant2 - D * np.exp(-xi * beta * xaxis)
	# ax2.semilogy(xaxis, remnant3, ls='-.', label = f' Third Remnant')

	# ax2.set_xlim(-0.005,0.5)
	ax2.set_ylim(1e-7,1)

	textstr = ''
	textstr += f'First slope $\\gamma =  $ {gamma:.4} \n'
	textstr += f'calculated c = $\\gamma/\\Delta = $ {c:.4} \n'
	textstr += f' predicted next slope at $c(1+\\Delta ) = $ {c*(1+delta):.4} \n'
	textstr += f' Fit second slope from remnant = {zeta:.4}'
	# textstr = f'expected spacing {c} \n' + 
	ax2.text(0.19, 0.9, textstr, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')

	percentageerror = 100 * np.abs(zeta - c*(1+delta)) / (c*(1+delta)) 
	print(percentageerror)

	########## model fitting #############
	def model(x,a1,a2,a3,e1,e2,e3):
		return a1 * np.exp(-e1*x) + a2 * np.exp(-e2*x) + a3 * np.exp(-e3*x)




	# # initials = [0.02173, 0.002086, 0.03204,  0.00456825, 0.01543535, 0.02630245] #just putting the deltas - seems best
	# initials = [A, B, 0.00001*B,  c*delta, c*(1+delta), c*(2+delta)] 
	# # initials = [0.02173, 0.002086, 0.03204,  0.004568, 0.005827, 0.01793]
	# midslice = slice(np.argmin(np.abs(xaxis-l4)),np.argmin(np.abs(xaxis-l1)))
	# params, covariance = curve_fit(model, xaxis[midslice] * beta, plottable[startT:stopT:skip][midslice],p0=initials)

	# expos = params[-4:-1]
	# sortedexpos = sorted(expos)
	# fitted_gamma = sortedexpos[0]
	# fitted_c = fitted_gamma/delta
	# print(sortedexpos)
	# print(np.diff(sortedexpos))
	# print('fitted spacing = ',fitted_c)

	# ######### PRONY METHOD ###############
	# # midslice = slice(np.argmin(np.abs(xaxis-l4)),np.argmin(np.abs(xaxis-l1)))
	# m = 3
	# a_est, b_est = prony(beta*xaxis[midslice],plottable[startT:stopT:skip][midslice],m)

	# print(f'a_est = {a_est}')
	# print(f'b_est = {b_est}')

	# c_est = b_est[-1] / delta
	# CUT = 0
	# y_fit = [np.sum([a_est[i+CUT] * np.exp(b_est[i+CUT]*beta*xval) for i in range(len(a_est[CUT:]))]) for xval in xaxis]
	# ax2.semilogy(xaxis, y_fit, label = 'Prony', ls='-.')
	# print(f'c_est = {c_est}')

	########### EDSF METHOD #######################
	# a,theta,final_err = fitEDSF(beta*xaxis[midslice],plottable[startT:stopT:skip][midslice])
	# skip = 50
	# nlist = np.arange(0,len(xaxis),skip) 
	# print(len(nlist))
	# Gn = plottable[startT:stopT:skip]
	# astar,thetastar,final_err = fitEDSF(plottable[startT:stopT:skip],nlist , epsilon_1 = 1e-5, epsilon_2 = 1e-5)
	# print(astar,thetastar,final_err)
	# print(f'len(a) = {len(a)}')
	# theta = (2 * Nbig / beta ) * thetastar
	# print('theta = ', theta)

	# startval = np.argmin(np.abs(xaxis-0.005))
	# # startval = 0
	# stopval = np.argmin(np.abs(xaxis-0.4))
	# midslice = slice(startval,stopval)
	# N = 1000
	# rangeys = plottable[startT:stopT:skip][midslice]
	# rangexs = beta*xaxis[midslice] ##units of tau
	# step = len(rangexs)//N
	# selectxs = rangexs[startval:stopval:step] #units of tau
	# selectys = rangeys[startval:stopval:step]
	# print(len(selectxs), len(selectys))
	# nset = np.arange(0,len(selectys))
	# print(len(nset))
	# # fig,ax = plt.subplots(1)
	# # ax.plot(rangexs,rangeys,'.-',markersize=1)
	# # axtw = ax.twiny()
	# # axtw.plot(nset,selectys,'.',markersize=10)
	# astar,thetastar,final_err = fitEDSF(selectys,nset,4)
	# print(np.log(thetastar))


# plt.savefig('../../KoenraadEmails/Fitting Higher exponentials.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
ax2.legend()

plt.show()






















