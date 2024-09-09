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
	# path_to_dump_lamb = '../Dump/v2LOWTEMP_lamb_anneal_dumpfiles/'
	path_to_dump_lamb = '../Dump/kappa10LOWTEMP_lamb_anneal_dumpfiles/'
	# path_to_dump_lamb = '../Dump/LOWTEMP_lamb_anneal_dumpfiles/'
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


plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '10'
plt.rcParams['figure.titlesize'] = '10'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams['axes.labelsize'] = '10'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
# plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1'
# plt.rcParams['axes.formatter.limits'] = '-2,2'
plt.rcParams['text.usetex'] = 'False'




# delta = 0.420374134464041
delta = 0.193052

which = 'DD' 


slope_expect = 1./(2-2*delta)
fig,ax = plt.subplots(1)
fig.tight_layout()
fig.set_figwidth(3.25)
ax.set_box_aspect(aspect=1)

path_to_dump = path_to_dump_lamb
gaplist = []
Nbig = int(2**16)
beta_start = 1 
# target_beta = 2000
target_beta = 5000
beta = target_beta
mu = 0.0
g = 0.5
r = 1.
# lamb = 0.05
J = 0
kappa = 10.
omegar2 = ret_omegar2(g,beta)
beta_step = 1
# betasavelist = [50,100,500,1000,5000,10000]
betasavelist = [target_beta,]
lamblooplist = np.arange(1,0.01 - 1e-10,-0.001)
# lambsavelist = [0.1,0.05,0.01,0.005,0.001]

# lambsavelist = np.arange(0.009,0.002 - 1e-10,-0.001)
lambsavelist = np.arange(0.05,0.01 - 1e-10,-0.001)
# lambsavelist = np.arange(0.006,0.001 - 1e-10,-0.001)
# lambsavelist = np.arange(0.035,0.005 - 1e-10,-0.001)

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')
# lambval = savelist[np.isclose(savelist,lamb)][0]
lambinset = 0.002
lambinset = 0.01
startT, stopT = 0, Nbig//20

# for lambval in (lambval,):
for lambval in lambsavelist:
	savefile = 'kappa10_0' + 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	savefile += 'lamb' + f'{lambval:.3}' + 'J' + str(J)
	savefile += 'g' + str(g) + 'r' + str(r)
	savefile = savefile.replace('.','_') 
	savefile += '.npy'
	try:
		GDtau,GODtau,DDtau,DODtau = np.load(os.path.join(path_to_dump_lamb,savefile))
	except FileNotFoundError: 
		print(f"InputFile not found for lamb = {lambval:.3}")

	if which == 'GD':
		plottable = np.abs(np.real(GDtau))
	elif which =='DD':
		plottable = np.abs(np.real(DDtau))
	else:
		raise(Exception('UNKNOWN VALUE FOR WHICH -- allowed options "GD" and "DD" '))
		exit(1)




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
	startval, stopval = 0.008, 0.01
	# startval, stopval = 0.1, 0.13
	start_idx = np.argmin(np.abs(xaxis-startval))
	stop_idx = np.argmin(np.abs(xaxis-stopval))
	
	if np.isclose(lambval,lambinset):
		################## INSET #############################
		titlestring =  r' $\beta $ = ' + str(beta)
		titlestring += r' $\lambda$ = ' + f'{lambval:.3}' 
		left, bottom, width, height = [0.25, 0.55, 0.2, 0.2]
		# left, bottom, width, height = [0.25, 0.5, 0.2, 0.2]
		ax2 = fig.add_axes([left, bottom, width, height])
		#plottable = np.abs(np.real(GDtau))
		# startT, stopT = 0, Nbig//2
		# skip = 50
		skip = 1
		xaxis = tau[startT:stopT:skip]/beta
		# yaxis = 
		ax2.semilogy(xaxis, plottable[startT:stopT:skip],'p',label = 'numerics DDtau',markersize=2,c='C2')
		# ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
		ax2.set_xlabel(r'$\tau/\beta$')
		insetitle = r'$|G_d(\tau)|$' if which == 'GD' else r'$|D_d(\tau)|$'
		ax2.set_ylabel(insetitle)
		ax2.set_title(titlestring)
		ax2.set_box_aspect(aspect=1)
		# ax2.axvline(0.1, ls='--')
		# ax2.axvline(0.2, ls='--')
		ax2.axvline(startval, ls='--')
		ax2.axvline(stopval, ls='--')
		ax2.tick_params(which='major', length=3, width=0.6, direction="in", right=True, top=True)
		ax2.tick_params(which='minor', length=1, width=0.3, direction="in", right=True, top=True)
		ax2.tick_params(axis='x', labelsize=8)
		ax2.tick_params(axis='y', labelsize=8)


	fitslice = slice(start_idx,stop_idx)
	print(f'lambval = {lambval:.3}, points in fit = {stop_idx-start_idx}, fitscale = {tau[start_idx]/beta:.2}, {tau[stop_idx]/beta :.2}')
	slope = -np.mean(logder[startT:stopT][fitslice])
	gaplist += [slope]




################## MAIN FIGURE ###################





ax.loglog(lambsavelist,gaplist,'.')
m,c = np.polyfit(np.log(lambsavelist),np.log(gaplist),1)
# m,c = np.polyfit(np.log(lambsavelist[-10:-1]),np.log(gaplist[-10:-1]),1)
ax.loglog(lambsavelist, np.exp(c) * lambsavelist**m, label = f'fit with slope {m:.4}')
ax.loglog([],[],ls='None',label = f'Expected slope {slope_expect:.4}')
print(f'dimensional analysis scaling = {slope_expect:.4}')
print(f'calculated scaling = {m:.4}')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'mass gap $\gamma\left[\lambda\right]$')

titleval = r'Gap scaling calculated from $G_d$ for $\kappa = 10$' if which =='GD' else r'Gap scaling calculated from $D_d$ for $\kappa=10$'


ax.set_title(titleval)
ax.legend(loc='lower right') # add option fontsize = 12 for example
ax.tick_params(which='major', length=4, width=0.8, direction="in", right=True, top=True)
ax.tick_params(which='minor', length=2, width=0.5, direction="in", right=True, top=True)
# ax.tick_params(axis='x', labelsize=6)
# ax.tick_params(axis='y', labelsize=6)





if which == 'GD':
	plt.savefig('kappa10GdGapscalingv2.pdf',bbox_inches='tight')
elif which == 'DD':
	plt.savefig('kappa10DdGapscalingv2.pdf',bbox_inches='tight')
else: 
	print("Please try to be less stupid in the future. Kind regards.")
	exit(1)



# plt.show()








	






















