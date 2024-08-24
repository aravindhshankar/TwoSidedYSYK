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
	path_to_dump_temp = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd'
	
	# if not os.path.exists(path_to_dump_lamb):
	# 	raise Exception('Generate Data first! Path to lamb dump not found')
	# 	exit(1)
	if not os.path.exists(path_to_dump_temp):
		raise Exception('Generate Data first! Path to temp dump not found')
		exit(1)



from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
#import time

plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "120"
plt.rcParams['figure.figsize'] = '8 ,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1.6'
plt.rcParams['axes.labelsize'] = '16'
plt.rcParams['axes.titlesize'] = '16'

plt.rcParams['legend.fontsize'] = '12'

Nbig = int(2**14)
err = 1e-5

global beta

betaBH = 50
betaWH = 80
betalist = [betaBH,betaWH]
# betalist = [60,61,62,63,64,65,66,67,68,59,70]
# betalist = range(10,90)
mu = 0.0
g = 0.5
# g = 2.
r = 1.

kappa = 1.
lamb = 0.05
J = 0

# path_to_dump = path_to_dump_lamb
path_to_dump = path_to_dump_temp


fig, ax = plt.subplots(2,2, figsize=(8,7))
titlestring = "Imaginary time Green's functions for "
# titlestring +=  r', $g = $' + str(g)
titlestring += r' $\lambda$ = ' + str(lamb) 
fig.suptitle(titlestring)
fig.tight_layout(pad=2)

figSL,axSL = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
figSL.suptitle(titlestring)
figSL.tight_layout(pad=2)


figLL,axLL = plt.subplots(2,2)
#fig.set_figwidth(10)
#titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
figLL.suptitle(titlestring)
figLL.tight_layout(pad=2)



for i, beta in enumerate(betalist):
	col = 'C'+str(i)
	lab = r'$\beta = $ ' + str(beta) + ' (2BH)' if beta < 63 else ' (WH)'
	savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	savefile += 'lamb' + str(lamb) + 'J' + str(J)
	savefile += 'g' + str(g) + 'r' + str(r)
	savefile = savefile.replace('.','_') 
	savefile += '.npy'

	try :
		#plotfile = os.path.join(path_to_dump, 'Nbig14beta100_0lamb0_05J0_05g0_5r1_0.npy')
		plotfile = os.path.join(path_to_dump, savefile)
		GDtau, GODtau, DDtau, DODtau = np.load(plotfile)
		GODtau = -GODtau
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




	ax[0,0].plot(tau/beta, np.real(GDtau), c='C'+str(i), ls = '-', label =   r'$\beta$ = ' + str(beta))
	ax[0,0].plot(tau/beta, np.real(Gconftau), c='C'+str(i), ls='--' )
	#ax[0,0].set_ylim(-1,1)
	ax[0,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,0].set_ylabel(r'$\Re{G_{d}(\tau)}$')
	ax[0,0].legend()

	ax[0,1].plot(tau/beta, np.real(GODtau), c='C'+str(i), ls='-',  label =  r'$\beta$ = ' + str(beta))
	# ax[0,1].plot(tau/beta, np.imag(GODtau), label = 'numerics imag GODtau' + r'$\beta$ = ' + str(beta))
	#ax[0,1].plot(tau/beta, np.real(Gconftau), 'b--', label = 'analytical Gtau' )
	#ax[0,1].set_ylim(-1,1)
	ax[0,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[0,1].set_ylabel(r'$\Re{G_{od}(\tau)}$')
	ax[0,1].legend()

	ax[1,0].plot(tau/beta, np.real(DDtau), c='C'+str(i), ls='-', label =  r'$\beta$ = ' + str(beta))
	ax[1,0].plot(tau/beta, np.real(Dconftau), c='C'+str(i), ls= '--')
	# ax[1,0].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
	#ax[1,0].set_ylim(0,1)
	ax[1,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,0].set_ylabel(r'$\Re{D_{d}(\tau)}$')
	ax[1,0].legend()

	ax[1,1].plot(tau/beta, np.real(DODtau), c='C'+str(i), ls='-', label = r'$\beta$ = ' + str(beta))
	# ax[1,1].plot(tau/beta, np.imag(DODtau), label = 'numerics imag DODtau' + r'$\beta$ = ' + str(beta))
	#ax[1,1].plot(tau/beta, np.real(Dconftau), 'b--', label = 'analytical Dtau' )
	#ax[1,1].plot(tau/beta, np.real(FreeDtau), 'g-.', label = 'Free D Dtau' )
	#ax[1,1].set_ylim(0,1)
	ax[1,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
	ax[1,1].set_ylabel(r'$\Re{D_{od}(\tau)}$')
	ax[1,1].legend()

	#fig.suptitle(r'$\beta$ = ', beta)
	#plt.savefig('../../KoenraadEmails/Withx0_01constImagTime.pdf',bbox_inches='tight')
	#plt.show()






	############## LOG-LOG POWER LAW PLOT #####################

	start, stop = Nbig//2, Nbig//2 + 100
	startB, stopB = Nbig//2 + 1 , Nbig//2 + 101
	delta = 0.420374134464041
	alt_delta = 0.116902  

	fitGD_val = -np.imag(GDomega[start+0])*(g**2)
	#fitGD_val = -np.imag(Gconf[start:stop])*(g**2)
	conf_fit_GD = 1 * np.abs(omega/(g**2))**(2*delta - 1)
	conf_fit_GD = conf_fit_GD/conf_fit_GD[start] * fitGD_val

	fitDD_val = np.real(DDomega[startB])*(g**2)
	#fitDD_val = np.real(Dconf[startB:stopB])
	conf_fit_DD = 1 * np.abs(nu[startB:stopB]/(g**2))**(1-4*delta)
	conf_fit_DD = conf_fit_DD/conf_fit_DD[0] * fitDD_val

	fitslice = slice(start+0, start + 15)
	#fitslice = slice(start+25, start + 35)
	functoplot = -np.imag(GDomega)*(g**2)
	m,c = np.polyfit(np.log(np.abs(omega[fitslice])/(g**2)), np.log(functoplot[fitslice]),1)
	print(f'slope of fit = {m:.03f}')
	print('2 Delta - 1 = ', 2*delta-1)

	axLL[0,0].loglog(omega[start:stop]/(g**2), -np.imag(GDomega[start:stop])*(g**2),'p',c=col, label = lab)
	# axLL[0,0].loglog(omega[start:stop]/(g**2), conf_fit_GD[start:stop],'k--',label = 'ES power law')

	# axLL[0,0].loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')
	axLL[0,0].set_xlabel(r'$\omega_n/g^2$')
	axLL[0,0].set_ylabel(r'$-g^2\,\Im{GD(\omega_n)}$')

	axLL[0,0].legend()


	# axLL[0,1].loglog(omega[start:stop]/(g**2), np.imag(GODomega[start:stop])*(g**2),'p',label = 'numerics Im GODomega')
	axLL[0,1].loglog(omega[start:stop]/(g**2), np.abs(GODomega[start:stop])*(g**2),'p',c=col,label = lab)
	axLL[0,1].set_xlabel(r'$\omega_n/g^2$')
	axLL[0,1].set_ylabel(r'$g^2\,GOD(\omega_n)$')
	axLL[0,1].legend()

	axLL[1,0].loglog(nu[startB:stopB]/(g**2), np.real(DDomega[startB:stopB])*(g**2),'p',c=col,label=lab)
	# axLL[1,0].loglog(nu[startB:stopB]/(g**2), conf_fit_DD,'k--',label = 'ES power law')
	axLL[1,0].set_xlabel(r'$\nu_n/g^2$')
	axLL[1,0].set_ylabel(r'$g^2\,\Re{DD(\nu_n)}$',labelpad = None)
	axLL[1,0].legend()


	axLL[1,1].loglog(nu[startB:stopB]/(g**2), -np.real(DODomega[startB:stopB])*(g**2),'p',c=col,label=lab)
	axLL[1,1].set_xlabel(r'$\nu_n/g^2$')
	axLL[1,1].set_ylabel(r'$g^2\,DOD(\nu_n)$',labelpad = None)
	axLL[1,1].legend()



	###################### Log-Linear Plot ###############################



	startT, stopT  = 0, 5000
	startT, stopT  = 0, Nbig//2
	

	fitsliceT = slice(startT+4500, startT + 4600)
	functoplotT = np.abs(np.real(GDtau))
	mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)
	print(f'tau/beta at start of fit = {(tau[fitsliceT][0]/beta):.3f}')
	print(f'slope of fit = {mT:.03f}')

	axSL[0,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GDtau[startT:stopT])),'p',c = col, label = lab)
	# axSL[0,0].semilogy(tau[startT:stopT]/beta, np.exp(mT*tau[startT:stopT] + cT), label=f'Fit with slope {mT:.03f}')
	axSL[0,0].set_xlabel(r'$\tau/\beta$')
	axSL[0,0].set_ylabel(r'$-\Re G_{d}(\tau)$')
	axSL[0,0].legend()
	axSL[0,0].set_yscale('log')


	axSL[0,1].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GODtau[startT:stopT])),'p',c = col, label = lab)
	axSL[0,1].set_xlabel(r'$\tau/\beta$')
	axSL[0,1].set_ylabel(r'$-\Re G_{od}(\tau)$')
	axSL[0,1].legend()
	axSL[0,1].set_yscale('log')

	axSL[1,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(DDtau[startT:stopT])),'p',c=col,label=lab)
	axSL[1,0].set_xlabel(r'$\tau/\beta$')
	axSL[1,0].set_ylabel(r'$g^2\,\Re{D_{d}(\nu_n)}$',labelpad = None)
	axSL[1,0].legend()

	axSL[1,1].semilogy(tau[startT:stopT]/beta, np.abs(np.real(DODtau[startT:stopT])),'p',c=col,label=lab)
	axSL[1,1].set_xlabel(r'$\tau/\beta$')
	axSL[1,1].set_ylabel(r'$g^2\,\Re{D_{od}(\nu_n)}$',labelpad = None)
	axSL[1,1].legend()



# plt.savefig('GreenFunctionPlotsMetal.pdf', bbox_inches='tight')

#plt.savefig('../../KoenraadEmails/lowenergy_powerlaw_ImagTime_SingleYSYK.pdf', bbox_inches = 'tight')
#plt.savefig('../../KoenraadEmails/ImagFreqpowerlaw_withxconst0_01.pdf', bbox_inches = 'tight')
plt.show()


















