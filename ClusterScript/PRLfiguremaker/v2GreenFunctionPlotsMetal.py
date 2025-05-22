import sys
import os 
if not os.path.exists('../Sources'):
	print("Error - Path to Sources directory not found ")
	raise Exception("Error - Path to Sources directory not found ")
else:	
	sys.path.insert(1,'../Sources')	


if not os.path.exists('../Dump/'):
	print("Error - Path to Dump directory not found ")
	raise Exception("Error - Path to Dump directory not found ")
	exit(1)
else:

	path_to_dump_temp = '../Dump/zoom_xshift_temp_anneal_dumpfiles/fwd'
	

	if not os.path.exists(path_to_dump_temp):
		raise Exception('Generate Data first! Path to temp dump not found')
		exit(1)


path_to_oneside = '../Dump/OnesideMET'
if not os.path.exists(path_to_oneside):
	print('Path to oneside', path_to_oneside, 'does not exist')
	exit(1)


from SYK_fft import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter,NullFormatter
from ConformalAnalytical import *
from Insethelpers import add_subplot_axes
#import time

plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
#plt.rcParams['figure.dpi'] = "120"
# plt.rcParams['figure.figsize'] = '8 ,7'
plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1.2'
plt.rcParams['axes.labelsize'] = '9'
plt.rcParams['axes.titlesize'] = '9'
plt.rcParams['figure.titlesize'] = '9'
plt.rcParams['figure.figsize'] = f'{3.375},{3.375}'
plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

# plt.rcParams['legend.fontsize'] = '12'
# # # plt.rcParams['legend.fontsize'] = '14'
# plt.rcParams['figure.titlesize'] = '10'
# plt.rcParams['axes.titlesize'] = '10'
# plt.rcParams['axes.labelsize'] = '10'
# # plt.rcParams['lines.markersize'] = '6'
# plt.rcParams['lines.linewidth'] = '0.5'
plt.rcParams['axes.formatter.limits'] = '-2,2'

# plt.rcParams['figure.figsize'] = '8,7'
# print(plt.rcParams.keys())

Nbig = int(2**14)
err = 1e-5

global beta

betaBH = 50
betaBH = 42
betaWH = 80
betaWH = 99
betalist = [betaBH,betaWH]

mu = 0.0
g = 0.5
# g = 2.
r = 1.

kappa = 1.
lamb = 0.05
J = 0

path_to_dump = path_to_dump_temp


# fig, ax = plt.subplots(2,2, figsize=(8,7))
titlestring = "Imaginary time Green's functions for "
# # titlestring +=  r', $g = $' + str(g)
titlestring += r' $\lambda$ = ' + str(lamb) 


figSL,axSL = plt.subplots(2,2)
figSL.set_tight_layout(True)
figSL.set_figwidth(3.375)
figSL.suptitle(titlestring,x = 0.55,y=0.92)
# figSL.tight_layout(pad=2)
axSL[0,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSL[0,0].set_ylabel(r'$|\mathrm{Re}\,{G_{d}(\tau)}|$',labelpad = 1)

axSL[0,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSL[0,1].set_ylabel(r'$|\mathrm{Re}\,{G_{od}(\tau)}|$',labelpad = 1)

axSL[1,0].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSL[1,0].set_ylabel(r'$|\mathrm{Re}\,{D_{d}(\tau)}|$',labelpad = 1)

axSL[1,1].set_xlabel(r'$\tau/\beta$',labelpad = 0)
axSL[1,1].set_ylabel(r'$|\mathrm{Re}\,{D_{od}(\tau)}|$',labelpad = 1)


# GDinsetax = add_subplot_axes(axSL[0,0], [-0.08,-0.064,0.4,0.4])
# GODinsetax = add_subplot_axes(axSL[0,1], [0.1,0.0,0.4,0.4])
# DDinsetax = add_subplot_axes(axSL[1,0], [0.5,0.35,0.4,0.4])
# DODinsetax = add_subplot_axes(axSL[1,1], [0.6,0.25,0.4,0.4])
# insetaxes = [GDinsetax,GODinsetax,DDinsetax,DODinsetax]

GDinsetax = add_subplot_axes(axSL[0,0], [0.24,-0.015,0.4,0.4])
GODinsetax = add_subplot_axes(axSL[0,1], [0.4,0.0,0.4,0.4])
DDinsetax = add_subplot_axes(axSL[1,0], [0.47,0.31,0.4,0.4])
DODinsetax = add_subplot_axes(axSL[1,1], [0.6,0.30,0.4,0.4])
insetaxes = [GDinsetax,GODinsetax,DDinsetax,DODinsetax]


# GDinsetax.set_xticklabels([])
# 	insetax.set_xlabel('')
insetfontsize = 8
insetlabelpad = -5
GDinsetax.set_xlabel(r'$\omega_n$',fontsize=insetfontsize,labelpad=-5)
GDinsetax.set_ylabel(r'$-\mathrm{Im}{G_d}(\omega_n)$',fontsize=insetfontsize,labelpad=-4)
GODinsetax.set_xlabel(r'$\omega_n$',fontsize=insetfontsize,labelpad=-4)
GODinsetax.set_ylabel(r'$G_{od}(\omega_n)$',fontsize=insetfontsize,labelpad=-3)
DODinsetax.set_xlabel(r'$\nu_n$',fontsize=insetfontsize,labelpad=-3)
DODinsetax.set_ylabel(r'$D_{od}(\nu_n)$',labelpad = 1,fontsize=insetfontsize)
DDinsetax.set_xlabel(r'$\nu_n$',fontsize=insetfontsize,labelpad=-3)
DDinsetax.set_ylabel(r'$\mathrm{Re}{D_d}(\nu_n)$',labelpad = -1,fontsize=insetfontsize)
# DODinsetax.set_ylabel(r'$|\Re{D_{od}(\tau)}|$')

for axi in axSL:
	for axj in axi:
		axj.tick_params(axis='both', labelsize=8,pad=0.5)
		# axj.set_box_aspect(aspect=1)

# figLL,axLL = plt.subplots(2,2)
# figLL.suptitle(titlestring)
# figLL.tight_layout(pad=2)



for i, beta in enumerate(betalist):
	col = 'C'+str(i+1) #to have same color scheme as superconductor
	lab = r'$\beta = $ ' + str(beta) + (' (2BH)' if beta < 63 else ' (WH)' )
	savefile = 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	savefile += 'lamb' + str(lamb) + 'J' + str(J)
	savefile += 'g' + str(g) + 'r' + str(r)
	savefile = savefile.replace('.','_') 
	savefile += '.npy'

	onesidefile = 'OnesideMET'
	onesidefile += 'Nbig' + str(int(np.log2(Nbig))) + 'beta' + str(beta) 
	onesidefile += 'g' + str(g).replace('.','_') + 'r' + str(r) + '.npy'  

	try :
		plotfile = os.path.join(path_to_dump, savefile)
		GDtau, GODtau, DDtau, DODtau = np.load(plotfile)
		GODtau = -GODtau
	except FileNotFoundError: 
		print('Filename : ', savefile)
		print("INPUT FILE NOT FOUND") 
		exit(1)

	try :
		Gtau,Dtau = np.load(os.path.join(path_to_oneside,onesidefile))
	except FileNotFoundError:
		print('Filename : ', onesidefile)
		print("INPUT FILE NOT FOUND") 
		exit(1)

	omega = ImagGridMaker(Nbig,beta,'fermion')
	nu = ImagGridMaker(Nbig,beta,'boson')
	tau = ImagGridMaker(Nbig,beta,'tau')


	Gfreetau = Freq2TimeF(1./(1j*omega + mu),Nbig,beta)
	Dfreetau = Freq2TimeB(1./(nu**2 + r),Nbig,beta)
	delta = 0.420374134464041
	omegar2 = ret_omegar2(g,beta)



	
	assert len(GDtau) == Nbig, 'Improperly loaded starting guess'

	GDomega = Time2FreqF(GDtau, Nbig, beta)
	GODomega = Time2FreqF(GODtau, Nbig, beta)
	DDomega = Time2FreqB(DDtau, Nbig, beta)
	DODomega = Time2FreqB(DODtau, Nbig, beta)
	Gomega = Time2FreqF(Gtau, Nbig, beta)
	Domega = Time2FreqB(Dtau, Nbig, beta)




	

	###################### Log-Linear Plot ###############################



	startT, stopT  = 0, 5000
	startT, stopT  = 0, Nbig//2
	start, stop = Nbig//2, Nbig//2 + 20
	startB, stopB = Nbig//2 + 1 , Nbig//2 + 21
	delta = 0.420374134464041

	

	fitsliceT = slice(startT+4500, startT + 4600)
	functoplotT = np.abs(np.real(GDtau))
	mT,cT = np.polyfit(np.abs(tau[fitsliceT]), np.log(functoplotT[fitsliceT]),1)


	# llplotslice = slice(np.argmin(np.abs(tau/beta - 0.)),np.argmin(np.abs(tau/beta - 0.5)))
	llplotslice = slice(0,Nbig//2,10)
	axSL[0,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(GDtau[llplotslice])),c = col, label = lab)
	axSL[0,0].semilogy(tau[llplotslice]/beta, np.abs(np.real(Gtau[llplotslice])),c = col, ls='--')
	# axSL[0,0].semilogy(tau[startT:stopT]/beta, np.exp(mT*tau[startT:stopT] + cT), label=f'Fit with slope {mT:.03f}')
	# axSL[0,0].legend(framealpha=0)



	axSL[0,1].semilogy(tau[startT:stopT]/beta, np.abs(np.real(GODtau[startT:stopT])),c = col, label = lab)
	# axSL[0,1].legend(framealpha=0)

	axSL[1,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(DDtau[startT:stopT])),c=col,label=lab)
	axSL[1,0].semilogy(tau[startT:stopT]/beta, np.abs(np.real(Dtau[startT:stopT])),c=col,ls='--')
	# axSL[1,0].legend(framealpha=0)

	axSL[1,1].semilogy(tau[startT:stopT]/beta, np.abs(np.real(DODtau[startT:stopT])),c=col,label=lab)
	# axSL[1,1].legend(framealpha=0)
	# axSL[1,1].tick_params(axis='both', labelsize=4,pad=0)


	# GDinsetax.tick_params(axis='both', labelsize=6,pad=0.5)
	# GODinsetax.tick_params(axis='both', labelsize=6,pad=0.5)
	# DDinsetax.tick_params(axis='both', labelsize=6,pad=0.5)
	# DODinsetax.tick_params(axis='both', labelsize=6,pad=0.5)

	GDinsetax.loglog(omega[start:stop]/(g**2), -np.imag(GDomega[start:stop])*(g**2),'.',c=col, label = lab)
	# GDinsetax.loglog(omega[start:stop]/(g**2), conf_fit_GD[start:stop],'k--',label = 'ES power law')
	GDinsetax.loglog(omega[start:stop]/(g**2), -np.imag(Gomega[start:stop])*(g**2),':',c='b',linewidth=0.8)

	# GDinsetax.loglog(omega[start:stop]/(g**2), np.exp(c)*np.abs(omega[start:stop]/(g**2))**m, label=f'Fit with slope {m:.03f}')


	# axLL[0,1].loglog(omega[start:stop]/(g**2), np.imag(GODomega[start:stop])*(g**2),'p',label = 'numerics Im GODomega')
	GODinsetax.loglog(omega[start:stop]/(g**2), np.abs(GODomega[start:stop])*(g**2),'.',c=col,label = lab)
	# GODinsetax.legend()

	DDinsetax.loglog(nu[startB:stopB]/(g**2), np.real(DDomega[startB:stopB])*(g**2),'.',c=col,label=lab)
	DDinsetax.loglog(nu[startB:stopB]/(g**2), np.real(Domega[startB:stopB])*(g**2),':',c='b',linewidth=0.8)
	# DDinsetax.loglog(nu[startB:stopB]/(g**2), conf_fit_DD,'k--',label = 'ES power law')
	# DDinsetax.legend()


	DODinsetax.loglog(nu[startB:stopB]/(g**2), -np.real(DODomega[startB:stopB])*(g**2),'.',c=col,label=lab)
	# DODinsetax.legend()

	
	# DODinsetax.legend()

# fig.savefig('GreenFunctionPlotsMetal.pdf', bbox_inches='tight')

axSL[0,0].set_ylim(bottom=9e-5)
axSL[1,0].set_ylim(top=2e0)
axSL[1,1].set_ylim(top=2e-1)
axSL[1,1].yaxis.set_minor_formatter(NullFormatter())
# axSL[1,1].ticklabel_format(axis='y', scilimits=(0,0))
# axSL[1,1].tick_params('y',labelsize=2)
handles, labels = axSL[0,0].get_legend_handles_labels()
lgd = figSL.legend(handles, labels, ncol=len(labels)//2+1, loc="lower center", bbox_to_anchor=(1.2,-0.45),frameon=True,fancybox=True,borderaxespad=0, bbox_transform=axSL[1,0].transAxes)
for insetax in insetaxes:
	insetax.set_xticks([],labels=[])
	insetax.set_yticks([],labels=[])

figSL.savefig('LLINSETGreenFunctionPlotsMetal.pdf', bbox_inches='tight')


















