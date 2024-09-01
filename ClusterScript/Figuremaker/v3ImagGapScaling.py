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
	path_to_dump_lamb = '../Dump/v2LOWTEMP_lamb_anneal_dumpfiles/'
	path_to_dump_lamb = '../Dump/LOWTEMP_lamb_anneal_dumpfiles/'
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
#from annealers import anneal_temp, anneal_lamb
import concurrent.futures


plt.style.use('physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '8'
plt.rcParams['figure.titlesize'] = '10'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams['axes.labelsize'] = '10'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
# plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '0.8'
plt.rcParams['lines.markersize'] = '1.2'
# plt.rcParams['axes.formatter.limits'] = '-2,2'
plt.rcParams['text.usetex'] = 'False'

# plt.rcParams['figure.figsize'] = '8,7'

# fparams =  {'figure.dpi' : 120,
#             'axes.linewidth' : 1,
#             'lines.linewidth' : 2,
#             'axes.labelsize': 22,
#             'axes.titlesize': 20,
#             'font.size': 16,
#             'legend.fontsize': 14,
#             'font.family': 'serif',
#             'font.serif': 'Computer Modern Roman',
#             'xtick.labelsize': 20,
#             'ytick.labelsize': 20,
#             'text.usetex': True,
#             'pdf.fonttype' : 42,
#             'svg.fonttype': 'path',
#             'xtick.major.size': 3.0,
#             'xtick.major.width': 0.5,
#             'legend.numpoints' : 1,
#             'legend.frameon' : False}
# #rcParams.update(fparams)
# #figsize(7.7, 4)
# plt.rcParams.update(fparams)


calc = True
delta = 0.420374134464041

if calc == True:
	### TODO: Implement parallization
	path_to_dump = path_to_dump_lamb
	gaplist = []
	Nbig = int(2**16)
	beta_start = 1 
	target_beta = 2000
	target_beta = 5000
	beta = target_beta
	mu = 0.0
	g = 0.5
	r = 1.
	# lamb = 0.05
	J = 0
	kappa = 1.
	omegar2 = ret_omegar2(g,beta)
	beta_step = 1
	# betasavelist = [50,100,500,1000,5000,10000]
	betasavelist = [target_beta,]
	lamblooplist = np.arange(1,0.01 - 1e-10,-0.001)
	# lambsavelist = [0.1,0.05,0.01,0.005,0.001]

	lambsavelist = np.arange(0.009,0.002 - 1e-10,-0.001)
	lambsavelist = np.arange(0.6,0.5 - 1e-10,-0.001)
	lambsavelist = np.arange(1.,0.5 - 1e-10,-0.005)
	# lambsavelist = np.arange(0.99,0.01 - 1e-10,-0.001)
	# lambsavelist = np.arange(0.006,0.001 - 1e-10,-0.001)
	# lambsavelist = np.arange(0.035,0.005 - 1e-10,-0.001)

	fig,ax = plt.subplots(1)
	fig.set_figwidth(3.25)
	fig.tight_layout()
	ax.set_box_aspect(aspect=1)
	ax.tick_params(axis='both', labelsize=8)
	ax.tick_params(axis='y', pad=1)
	ax.tick_params(axis='x',  pad=1)
	ax.tick_params(axis='x', pad=1)


	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')
	# lambval = savelist[np.isclose(savelist,lamb)][0]
	lambinset = 0.005
	lambinset = 0.9
	startT, stopT = 0, Nbig//2
	startT, stopT = 0, np.argmin(np.abs(tau/beta - 0.01))

	# for lambval in (lambval,):
	for lambval in lambsavelist:
		savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
		savefile += 'lamb' + f'{lambval:.3}' + 'J' + str(J)
		savefile += 'g' + str(g) + 'r' + str(r)
		savefile = savefile.replace('.','_') 
		savefile += '.npy'
		try:
			GDtau,GODtau,DDtau,DODtau = np.load(os.path.join(path_to_dump_lamb,savefile))
		except FileNotFoundError: 
			print(f"InputFile not found for lamb = {lambval:.3}")

		plottable = np.abs(np.real(GDtau))
		lambinv = 1./(lambval*beta)
		xaxis = tau[startT:stopT]/beta
		# logder = np.gradient(np.log(plottable))
		logder = np.gradient(np.log(plottable),tau)
		# start_idx = np.argmin(np.abs(xaxis-lambinv*2))
		# stop_idx = np.argmin(np.abs(xaxis-lambinv*2.5))
		# start_idx = np.argmin(np.abs(xaxis-0.1))
		# stop_idx = np.argmin(np.abs(xaxis-0.13))
		# start_idx = np.argmin(np.abs(xaxis-0.1))
		# stop_idx = np.argmin(np.abs(xaxis-0.2))
		# start_idx = np.argmin(np.abs(xaxis-0.3))
		# stop_idx = np.argmin(np.abs(xaxis-0.35))
		# start_idx = np.argmin(np.abs(xaxis-0.003))
		# stop_idx = np.argmin(np.abs(xaxis-0.004))
		start_idx = np.argmin(np.abs(xaxis-0.001))
		stop_idx = np.argmin(np.abs(xaxis-0.002))
		

		fitslice = slice(start_idx,stop_idx)
		print(f'lambval = {lambval:.3}, points in fit = {stop_idx-start_idx}, fitscale = {tau[start_idx]/beta:.2}, {tau[stop_idx]/beta :.2}')
		slope = -np.mean(logder[startT:stopT][fitslice])
		gaplist += [slope]
		################## INSET #############################
		if np.isclose(lambval,lambinset):
			titlestring =  r' $\beta $ = ' + str(beta)
			titlestring += r', $\lambda$ = ' + f'{lambval:.3}' 
			# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
			left, bottom, width, height = [0.25, 0.52, 0.2, 0.2]
			ax2 = fig.add_axes([left, bottom, width, height])
			#plottable = np.abs(np.real(GDtau))
			# startT, stopT = 0, Nbig//2
			skip = 1
			xaxis = tau[startT:stopT:skip]/beta
			# yaxis = 
			ax2.semilogy(xaxis, plottable[startT:stopT:skip],'.',label = 'numerics GDtau')
			# ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
			ax2.set_xlabel(r'$\tau/\beta$',fontsize=7)
			ax2.set_ylabel(r'$|G_d(\tau)|$',fontsize=7)
			ax2.yaxis.set_label_coords(-0.3,0.5)
			ax2.tick_params(which='major', length=1.5, width=0.4, direction="in", right=True, top=True,labelsize=6,pad=0.1)
			ax2.tick_params(which='minor', length=1, width=0.2, direction="in", right=True, top=True,labelsize=6,pad=0.1)
			ax2.set_title(titlestring,fontsize=7)
			ax2.axvline(xaxis[start_idx], ls='--')
			ax2.axvline(xaxis[stop_idx], ls='--')
			# ax2.axvline(1./(beta**2),ls='--', c='gray', label = 'Temperature')
			# ax2.axvline(1./(lambval*beta), ls='--', c='green',label=r'$\lambda^{-1}$')
			#ax2.legend()




	################## MAIN FIGURE ###################
	slope_expect = 1./(2-2*delta)
	ax.loglog(lambsavelist,gaplist,'.')
	m,c = np.polyfit(np.log(lambsavelist),np.log(gaplist),1)
	# m,c = np.polyfit(np.log(lambsavelist[-10:-1]),np.log(gaplist[-10:-1]),1)
	ax.loglog(lambsavelist, np.exp(c) * lambsavelist**m, label = f'fit slope {m:.4}')
	# ax.loglog([],[],ls='None',label = f'Expected scaling with slope {slope_expect:.4}')
	print(f'dimensional analysis scaling = {slope_expect:.4}')
	print(f'calculated scaling = {m:.4}')
	ax.set_xlabel(r'$\lambda$')
	ax.set_ylabel(r'$\gamma\left[\lambda\right]$',rotation=0)
	ax.set_title(r'$G_d \sim e^{-\lambda \tau}$ at large $\lambda$',fontsize=12)
	ax.legend(fontsize=10,loc='lower right') # add option fontsize = 12 for example
	ax.tick_params(which='major', length=4, width=0.8, direction="in", right=True, top=True,labelsize=8,pad=0.4)
	ax.tick_params(which='minor', length=4, width=0.5, direction="in", right=True, top=True,labelsize=8,pad=0.4)
	ax.yaxis.set_label_coords(-0.06,1)


	# logder = np.gradient(np.log(plottable))
	# ax2[1].plot(xaxis, logder[startT:stopT])
	# ax2[1].set_xlabel(r'$\tau/\beta$')
	# ax2[1].set_ylabel(r'$\frac{d|\Re G(\tau)|}{d\tau}$')

	# start_idx = np.argmin(np.abs(xaxis-0.3))
	# stop_idx = np.argmin(np.abs(xaxis-0.4))
	# print(start_idx,stop_idx)
	# fitslice = slice(start_idx,stop_idx)
	# slope = -np.mean(logder[startT:stopT][fitslice])
	# print(slope)


	fig.savefig('LARGElambdalinearscaling.pdf',bbox_inches='tight')
	# plt.show()








	





else: #if calc == False:
	try:
		lamblooplist, gaplist = np.load('beta100lambgaplist.npy')
	except FileNotFoundError: 
		print('Gaplist not found!')
		exit(1)




















