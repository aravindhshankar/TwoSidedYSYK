import numpy as np 
from SYK_fft import *
from ConformalAnalytical import *
import warnings
import testingscripts


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



def newcheckrhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=1e-6):
    '''
    returns [Sigma,Pi] given rhos
    '''
    #eta = np.pi/(M*dt)*(0.001)
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
    rhoFpp = (1/np.pi)*freq2time(rhoG * fdminus,M,dt)
    rhoFpm = (1/np.pi)*freq2time(rhoG * fdplus,M,dt)
    rhoFmp = (1/np.pi)*freq2time(rhoGrev * fdminus,M,dt)
    rhoFmm = (1/np.pi)*freq2time(rhoGrev * fdplus,M,dt)
    rhoBpp = (1/np.pi)*freq2time(rhoD * beminus,M,dt)
    rhoBpm = (1/np.pi)*freq2time(rhoD * beplus,M,dt)
    
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.heaviside(t,1)
    argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-delta*t) * np.heaviside(t,0)
    Sigma = 1j*(g**2)*kappa * time2freq(argSigma,M,dt)
    
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.heaviside(t,1)
    argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-delta*t) * np.heaviside(t,0)
    Pi = 2*1j*(g**2) * time2freq(argPi,M,dt)
    
    return [Sigma, Pi]


def Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=1e-6):
    '''
    Direct implementation of Davide's email
    '''
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    ADt = (1/np.pi) * freq2time(rhoD,M,dt)
    aGt = (1/np.pi) * freq2time(rhoG * fdplus, M,dt)
    AGt = (1/np.pi) * freq2time(rhoG,M,dt)
    aDt = (1/np.pi) * freq2time(rhoD * beplus, M,dt)

    argSigma = (ADt * aGt - AGt * np.conj(aDt)) * np.heaviside(t,0)
    Sigma = -1j*(g**2)*kappa* time2freq(argSigma,M,dt)

    argPi = (AGt * np.conj(aGt) - np.conj(AGt) * (aGt)) * np.heaviside(t,0)
    Pi = 2j*(g**2)*kappa* time2freq(argPi,M,dt)

    return [Sigma,Pi]

    





def RE_YSYK_iterator(GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]
    '''
    M,omega,t = grid
    g,mu,r = pars
    itern = 0
    omegar2 = ret_omegar2(g,beta)

    diff = 1.
    diffG,diffD = (1.0,1.0)
    x = 0.01
    # if beta < 20:
    #     x = 0.5
    # elif beta < 40:
    #     x = 0.2
    # else: 
    #     x = 0.05

    xG, xD = x,x
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

        #SigmaOmega,PiOmega = newcheckrhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=eta)
        SigmaOmega,PiOmega = Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=eta)
        #SigmaOmega,PiOmega = newrhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=eta)
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

    INFO = (itern, diff)
    return (GRomega,DRomega, INFO)






def RE_WHYSYK_iterator(GFs,grid,pars,beta,lamb,J,x = 0.01,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GFs = GDRomega, GODRomega, DDRomega, DODRomega
    GFs,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]
    '''
    GDRomega, GODRomega, DDRomega, DODRomega = GFs
    M,omega,t = grid
    g,mu,r = pars
    itern = 0

    diff = 1.
    diffold = 1.
    #x = 0.01

    diffseries = []
    flag = True
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]

    # x = 0.5 if beta < 10 else 0.01
    # x = 0.5 if beta < 5 else 0.1
    # x = 0.5
    # for x in np.linspace(0.05,1.,10):
    for xval in (x,):
        diff = 1
        while (diff>err and itern<ITERMAX and flag): 
            itern += 1 
            diffold = diff
            if itern == ITERMAX:
                warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))
            #diffoldG,diffoldD = (diffG,diffD)
            GDRoldomega,DDRoldomega = (1.0*GDRomega, 1.0*DDRomega)
            GODRoldomega,DODRoldomega = (1.0*GODRomega, 1.0*DODRomega)

            rhoGD = -1.0*np.imag(GDRomega)
            rhoDD = -1.0*np.imag(DDRomega)
            rhoGOD = -1.0*np.imag(GODRomega)
            rhoDOD = -1.0*np.imag(DODRomega)

            SigmaDomega,PiDomega = Dav_rhotosigma(rhoGD,rhoDD,M,t,g,beta,BMf,kappa=1,delta=eta)
            SigmaODomega,PiODomega = Dav_rhotosigma(rhoGOD,rhoDOD,M,t,g,beta,BMf,kappa=1,delta=eta)
            # if np.imag(SigmaOmega[M] > 0) :
            #     warnings.warn('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
        
            # detGmat = (omega+1j*eta + mu - SigmaDomega)**2 - (lamb - SigmaODomega)**2
            detGmat = (omega+1j*eta + mu - SigmaDomega)**2 - (lamb + SigmaODomega)**2
            detDmat = (r-(omega+1j*eta)**2 - PiDomega)**2 - (J-PiODomega)**2

            GDRomega = xval*((omega+1j*eta + mu - SigmaDomega)/detGmat) + (1-xval)*GDRoldomega
            # GODRomega = xval*(-1.0*(lamb - SigmaODomega)/detGmat) + (1-xval)*GODRoldomega
            GODRomega = xval*((lamb + SigmaODomega)/detGmat) + (1-xval)*GODRoldomega
            DDRomega = xval*((r - (omega+1j*eta)**2 - PiDomega)/detDmat) + (1-xval)*DDRoldomega
            DODRomega = xval*(-1.0*(J - PiODomega)/detDmat) + (1-xval)*DODRoldomega


            diffGD = (1.)*np.sum((np.abs(GDRomega-GDRoldomega))**2) #changed
            diffGOD = (1.)*np.sum((np.abs(GODRomega-GODRoldomega))**2) #changed
            diffDD = (1.)*np.sum((np.abs(DDRomega-DDRoldomega))**2) #changed
            diffDOD = (1.)*np.sum((np.abs(DODRomega-DODRoldomega))**2) #changed
            #diffD = np.sum((np.abs(DRomega-DRoldomega))**2)
            diff = 0.25*(diffGD+diffDOD+diffDD+diffGOD)

            # if diff > diffold and xval*0.9 > 0.01 and itern % 10 == 0:
            #     xval *= 0.9
            # elif diff < diffold and xval*1.1 <= 1. and itern % 10 == 0:
            #     xval *= 1.1
            # if diff < 1e-6 and xval > 0.5:
            #     xval = 1.
            # if xval > 0.9:
            #     xval = 1.
            # if diff < 100.*err: 
            #     xval = 1.
            # if diff > 10 and itern > 100: 
            #     GDRomega = (omega + 1j*eta + mu)/((omega+1j*eta + mu)**2 - lamb**2)
            #     DDRomega = (-1.0*(omega + 1j*eta)**2 + r)/((r - (omega+1j*eta)**2)**2 - (J)**2)
            #     GODRomega = -lamb/((omega+1j*eta + mu)**2 - lamb**2)
            #     DODRomega = -J / ((r - (omega+1j*eta)**2)**2 - (J)**2)
            #     break

            #diffG,diffD = diff,diff
            if diffcheck == True:
                diffseries += [diff]
                if itern >10:
                    flag = testingscripts.diff_checker(diffseries, tol = 1e-6, periods = 7)
            
            if verbose:
                print("itern = ",itern, " , diff = ", diff, " , xval = ", xval,flush=True)

    GFs = [GDRomega, GODRomega, DDRomega, DODRomega]
    INFO = (itern, diff, xval)
    return (GFs, INFO)


























