import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


# Make 2 directories one for NFL, one for WH, dump GFs there 
if not os.path.exists('../Dump/'):
    print("Error - Path to Dump directory not found ")
    raise Exception("Error - Path to Dump directory not found ")
    exit(1)
else:
	path_to_dump_lamb = '../Dump/kappa10LOWTEMP_lamb_anneal_dumpfiles/'
	# path_to_dump_temp = '../Dump/zoom_xshift_temp_anneal_dumpfiles/rev'
	if not os.path.exists(path_to_dump_lamb):
		# print("Making directory for lamb dump")
		# os.mkdir(path_to_dump_lamb)
		print('Input File not found')
		exit(1)
	# if not os.path.exists(path_to_dump_temp):
	# 	print("Making directory for temp dump")
	# 	os.mkdir(path_to_dump_temp)
	# 	# print('Input File not found')
	# 	# exit(1)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ConformalAnalytical import *
from free_energy import free_energy_YSYKWH 
from annealers import anneal_temp, anneal_lamb
import concurrent.futures


calc = True
# delta = 0.420374134464041
delta = 0.193052

if calc == True:
	### TODO: Implement parallization
	path_to_dump = path_to_dump_lamb
	gaplist = []
	Nbig = int(2**16)
	beta_start = 1 
	target_beta = 5000
	beta = target_beta
	mu = 0.0
	g = 0.5
	r = 1.
	lamb = 0.05
	J = 0
	kappa = 10.
	omegar2 = ret_omegar2(g,beta)
	beta_step = 1
	# betasavelist = [50,100,500,1000,5000,10000]
	betasavelist = [target_beta,]
	lamblooplist = np.arange(1,0.001 - 1e-10,-0.001)
	# lambsavelist = [0.1,0.05,0.01,0.005,0.001]

	lambsavelist = np.arange(0.05,0.01 - 1e-10,-0.001)

	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')
	# lambval = savelist[np.isclose(savelist,lamb)][0]
	lambval = 0.01
	startT, stopT = 0, Nbig//2

	# for lambval in (lambval,):
	for lambval in lambsavelist:
		savefile = 'kappa10_0'
		savefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
		savefile += 'lamb' + f'{lambval:.3}' + 'J' + str(J)
		savefile += 'g' + str(g) + 'r' + str(r)
		savefile = savefile.replace('.','_') 
		savefile += '.npy'
		try:
			GDtau,_,DDtau,_ = np.load(os.path.join(path_to_dump_lamb,savefile))
		except FileNotFoundError: 
			print(f"InputFile not found for lamb = {lambval:.3}")

		plottable = np.abs(np.real(GDtau))
		lambinv = 1./(lambval*beta)
		xaxis = tau[startT:stopT]/beta
		logder = np.gradient(np.log(plottable))
		# start_idx = np.argmin(np.abs(xaxis-lambinv*2))
		# stop_idx = np.argmin(np.abs(xaxis-lambinv*2.5))
		start_idx = np.argmin(np.abs(xaxis-0.008))
		stop_idx = np.argmin(np.abs(xaxis-0.01))

		fitslice = slice(start_idx,stop_idx)
		print(f'lambval = {lambval:.3}, points in fit = {stop_idx-start_idx}, fitscale = {tau[start_idx]/beta:.2}, {tau[stop_idx]/beta :.2}')
		slope = -np.mean(logder[startT:stopT][fitslice])
		gaplist += [slope]

	# titlestring = r'$\beta$ = ' + str(beta) + r', $\log_2{N}$ = ' + str(np.log2(Nbig)) + r', $g = $' + str(g)
	# titlestring += r' $\lambda$ = ' + f'{lambval:.3}' + r' J = ' + str(J)
	# #plottable = np.abs(np.real(GDtau))
	# fig,ax = plt.subplots(2)
	# fig.suptitle(titlestring)
	# startT, stopT = 0, Nbig//2
	# xaxis = tau[startT:stopT]/beta
	# # yaxis = 
	# ax[0].semilogy(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau')
	# ax[0].set_xlabel(r'$\tau/\beta$')
	# ax[0].set_ylabel(r'$-\Re G(\tau)$')
	# ax[0].axvline(1./(beta**2),ls='--', c='gray', label = 'Temperature')
	# ax[0].axvline(1./(lambval*beta), ls='--', c='green',label=r'$\lambda^{-1}$')
	# ax[0].legend()



	# logder = np.gradient(np.log(plottable))
	# ax[1].plot(xaxis, logder[startT:stopT])
	# ax[1].set_xlabel(r'$\tau/\beta$')
	# ax[1].set_ylabel(r'$\frac{d|\Re G(\tau)|}{d\tau}$')

	# start_idx = np.argmin(np.abs(xaxis-0.3))
	# stop_idx = np.argmin(np.abs(xaxis-0.4))
	# print(start_idx,stop_idx)
	# fitslice = slice(start_idx,stop_idx)
	# slope = -np.mean(logder[startT:stopT][fitslice])
	# print(slope)


	slope_expect = 1./(2-2*delta)
	fig,ax = plt.subplots(1)
	ax.loglog(lambsavelist,gaplist,'.')
	m,c = np.polyfit(np.log(lambsavelist),np.log(gaplist),1)
	ax.loglog(lambsavelist, np.exp(c) * lambsavelist**m, label = f'fit with slope {m:.4}')
	print(f'dimensional analysis scaling = {slope_expect:.4}')
	print(f'calculated scaling = {m:.4}')
	ax.set_xlabel(r'$\lambda$')
	ax.set_ylabel(r'mass gap $\gamma\left[\lambda\right]$')
	ax.legend()


	plt.show()








	





else: #if calc == False:
	try:
		lamblooplist, gaplist = np.load('beta100lambgaplist.npy')
	except FileNotFoundError: 
		print('Gaplist not found!')
		exit(1)




















