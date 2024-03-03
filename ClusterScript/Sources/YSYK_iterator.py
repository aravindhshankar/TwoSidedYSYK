import numpy as np 
from SYK_fft import *
from ConformalAnalytical import *
import warnings


def newrhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=1e-6):
    '''
    returns [Sigma,Pi] given rhos
    '''
    #eta = np.pi/(M*dt)*(0.001)
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
    rhoFpp = (1/np.pi)*freq2time(rhoG * fdplus,M,dt)
    rhoFpm = (1/np.pi)*freq2time(rhoG * fdminus,M,dt)
    rhoFmp = (1/np.pi)*freq2time(rhoGrev * fdplus,M,dt)
    rhoFmm = (1/np.pi)*freq2time(rhoGrev * fdminus,M,dt)
    rhoBpp = (1/np.pi)*freq2time(rhoD * beplus,M,dt)
    rhoBpm = (1/np.pi)*freq2time(rhoD * beminus,M,dt)
    
    argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.heaviside(t,1)
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-delta*t) * np.heaviside(t,0)
    Sigma = 1j*(g**2)*kappa * time2freq(argSigma,M,dt)
    
    argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.heaviside(t,1)
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-delta*t) * np.heaviside(t,0)
    Pi = 2*1j*(g**2) * time2freq(argPi,M,dt)
    
    return [Sigma, Pi]




def RE_YSYK_iterator(GRomega,DRomega,omega,t,g,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GRomega,DRomega,omega,t,g,beta,eta=1e-6, verbose=True, diffcheck = False)
    
    '''
    itern = 0
    omegar2 = ret_omegar2(g,beta)

    diff = 1.
    diffG,diffD = (1,0.1.0)
    xG,xD = (0.01,0.01)
    diffseries = []
    flag = True
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]

    while (diff>err and itern<ITERMAX and flag): 
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))
        diffoldG,diffoldD = (diffG,diffD)
        GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)

        rhoG = -1.0*np.imag(GRomega)
        rhoD = -1.0*np.imag(DRomega)

        #SigmaOmega,PiOmega = newcheckrhotosigma(rhoG,rhoD,M,dt,t,g,beta,kappa=1,zeta = 1.,delta=eta)
        SigmaOmega,PiOmega = newrhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=eta)
        if np.imag(SigmaOmega[M] > 0) :
            warnings.warn('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
     
        GRomega = 1.0*xG/(omega + 1j*eta + mu - SigmaOmega) + (1-xG)*GRoldomega
        DRomega = 1.0*xD/(-1.0*(omega+1j*eta)**2 + r - PiOmega) + (1-xD)*DRoldomega
        #DRomega = 1.0*xD/(1.0*(omega+1j*eta)**2 - r - PiOmega) + (1-xD)*DRoldomega #modified


        diffG = np. sqrt(np.sum((np.abs(GRomega-GRoldomega))**2)) #changed
        diffD = np. sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))
        diff = 0.5*(diffG+diffD)
        diffG,diffD = diff,diff
        if diffcheck:
            diffseries += [diff]
            flag = testingscripts.diff_checker(diffseries, tol = 1e-3, periods = 5)
        
        if verbose:
            print("itern = ",itern, " , diff = ", diffG, diffD, " , x = ", xG, xD)

    return (GRomega,DRomega)


