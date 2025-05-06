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
from matplotlib.ticker import StrMethodFormatter, NullFormatter, FixedLocator, FormatStrFormatter, FixedFormatter
from matplotlib.ticker import LogLocator

plt.style.use('../Figuremaker/physrev.mplstyle') # Set full path to if physrev.mplstyle is not in the same in directory as the notebook
# plt.rcParams['figure.dpi'] = "120"
# # plt.rcParams['legend.fontsize'] = '14'
plt.rcParams['legend.fontsize'] = '12'
plt.rcParams['figure.titlesize'] = '12'
plt.rcParams['axes.titlesize'] = '12'
plt.rcParams['axes.labelsize'] = '12'
# plt.rcParams['figure.figsize'] = f'{3.25*2},{3.25*2}'
# plt.rcParams['lines.markersize'] = '6'
plt.rcParams['lines.linewidth'] = '1'
plt.rcParams['axes.formatter.limits'] = '-2,2'
plt.rcParams['text.usetex'] = 'True'




delta = 0.420374134464041

# which = 'GD' 


slope_expect = 1./(2-2*delta)
fig,ax = plt.subplots(1)
fig.tight_layout()
# fig.set_figwidth(3.25)
fig.set_figwidth(3.375 * 2 / 4) #single column width * 2 cols / 4 figs
# ax.set_box_aspect(aspect=1)

path_to_dump = path_to_dump_lamb
gaplist = []
Nbig = int(2**16)
beta_start = 1 
target_beta = 2000
# target_beta = 5000
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
# lambsavelist = np.arange(0.006,0.001 - 1e-10,-0.001)
# lambsavelist = np.arange(0.035,0.005 - 1e-10,-0.001)

omega = ImagGridMaker(Nbig,beta,'fermion')
nu = ImagGridMaker(Nbig,beta,'boson')
tau = ImagGridMaker(Nbig,beta,'tau')
# lambval = savelist[np.isclose(savelist,lamb)][0]
lambinset = 0.002
startT, stopT = 0, Nbig//2

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
    # plottableGD = np.abs(np.real(GDtau))
    # plottableDD = np.abs(np.real(DDtau))
    plottable = [np.abs(np.real(GDtau)),np.abs(np.real(DDtau))]
    lambinv = 1./(lambval*beta)
    xaxis = tau[startT:stopT]/beta
    # logder = np.gradient(np.log(plottable))
    logder = [np.gradient(np.log(plottableval),tau) for plottableval in plottable]
    # start_idx = np.argmin(np.abs(xaxis-lambinv*2))
    # stop_idx = np.argmin(np.abs(xaxis-lambinv*2.5))
    # start_idx = np.argmin(np.abs(xaxis-0.1))
    # stop_idx = np.argmin(np.abs(xaxis-0.13))
    # start_idx = np.argmin(np.abs(xaxis-0.1))
    # stop_idx = np.argmin(np.abs(xaxis-0.2))
    # start_idx = np.argmin(np.abs(xaxis-0.3))
    # stop_idx = np.argmin(np.abs(xaxis-0.35))
    startval, stopval = 0.1, 0.2
    start_idx = np.argmin(np.abs(xaxis-startval))
    stop_idx = np.argmin(np.abs(xaxis-stopval))

    if np.isclose(lambval,lambinset):
        ################## INSET #############################
        titlestring =  r' $\beta $ = ' + str(beta) + '\n'
        titlestring += r' $\lambda$ = ' + f'{lambval:.3}' 
        # left, bottom, width, height = [0.25, 0.55, 0.2, 0.2] #starting default
        left, bottom, width, height = [0.24, 0.70, 0.275, 0.275]
        ax2 = fig.add_axes([left, bottom, width, height])
        #plottable = np.abs(np.real(GDtau))
        startT, stopT = 0, Nbig//2
        skip = 50
        xaxis = tau[startT:stopT:skip]/beta
        # yaxis = 
        for i, plottableval in enumerate(plottable):
            ax2.semilogy(xaxis, plottableval[startT:stopT:skip],'p',label = 'numerics DDtau',markersize=2,c=f'C{i}')
        # ax2.plot(xaxis, plottable[startT:stopT],'p',label = 'numerics GDtau',markersize=2,c='C2')
        ax2.set_xlabel(r'$\tau/\beta$',labelpad=-2)
        insetitle = r'$|G_d(\tau)|$' 
        ax2.set_ylabel(insetitle)
        ax2.set_title(titlestring,pad=-1)
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
    slope = [-np.mean(logderval[startT:stopT][fitslice]) for logderval in logder]
    gaplist += (slope,)


gaplist = np.array(gaplist).T
# temp = [[],[]]
# for i, gaplist_ in enumerate(gaplist):
    # temp[0] += [gaplist_[0]]
    # temp[1] += [gaplist_[1]]
# gaplist = temp
#print(f'dims of gaplist = {gaplist.size}')


################## MAIN FIGURE ###################




for i, gaplistval in enumerate(gaplist):
    ax.loglog(lambsavelist,gaplistval,'.')
print(gaplist)
fitpars = [np.polyfit(np.log(lambsavelist),np.log(gaplistval),1) for gaplistval in gaplist]
# m,c = np.polyfit(np.log(lambsavelist[-10:-1]),np.log(gaplist[-10:-1]),1)
textvals = [r'$E_g^F$', r'$E_g^B$']
for i in [0,1]:
    mval = fitpars[i][0]
    cval = fitpars[i][1]
    ax.loglog(lambsavelist, np.exp(cval) * lambsavelist**mval, label = textvals[i] + f'= {mval:.4}', c=f'C{i}')
    print(f'calculated scaling = {mval:.4}')
ax.loglog([],[],ls='None',label = r'$\frac{1}{2-2\Delta}=$ '+f'{slope_expect:.4}')
print(f'dimensional analysis scaling = {slope_expect:.4}')
ax.set_xlabel(r'$\lambda \times 10^{-3}$')
ax.set_ylabel(r'mass gap $\gamma\left[\lambda\right]$')

titleval = r'Gap scaling calculated from $G_d$' 


ax.set_title(titleval)
ax.legend(loc='lower right') # add option fontsize = 12 for example

ax.set_title('FUCK YOU!')

tick_locs = np.logspace(np.log10(np.min(lambsavelist)), np.log10(np.max(lambsavelist)), num=4)
tick_labels = [f"{val*1000:.0f}" for val in tick_locs]
print('tick_labels', tick_labels)
print('tick_locs = ', tick_locs) 
ax.xaxis.set_minor_locator(FixedLocator(tick_locs))
ax.xaxis.set_minor_formatter(FixedFormatter(tick_labels))
# ax.set_xticks = tick_locs
# ax.tick_params(which='major', length=4, width=0.8, direction="in", right=True, top=True)
# ax.tick_params(which='minor', length=2, width=0.5, direction="in", right=True, top=True)
# ax.tick_params(axis='x', labelsize=6)
# ax.tick_params(axis='y', labelsize=6)




plt.savefig('PRLgapscaling.pdf', bbox_inches='tight')

# if which == 'GD':
        # plt.savefig('GdGapscalingv2.pdf',bbox_inches='tight')
# elif which == 'DD':
        # plt.savefig('DdGapscalingv2.pdf',bbox_inches='tight')
# else: 
        # print("Please try to be less stupid in the future. Kind regards.")
        # exit(1)



# plt.show()































