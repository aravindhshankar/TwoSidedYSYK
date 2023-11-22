import numpy as np
from SYK_fft import *


def rhotosigma(rhoG,rhoD,M,dt,t,omega,g,beta,kappa=1,delta=1e-6):
    '''
    returns [Sigma,Pi] given rhos
    '''
    eta = np.pi/(M*dt)*(0.001)
    rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
    rhoFpp = (1/np.pi)*freq2time(rhoG * fermidirac(beta*omega),M,dt)
    rhoFpm = (1/np.pi)*freq2time(rhoG * fermidirac(-1.*beta*omega),M,dt)
    rhoFmp = (1/np.pi)*freq2time(rhoGrev * fermidirac(beta*(omega)),M,dt)
    rhoFmm = (1/np.pi)*freq2time(rhoGrev * fermidirac(-1.*beta*omega),M,dt)
    rhoBpp = (1/np.pi)*freq2time(rhoD * boseeinstein(beta*(omega+eta)),M,dt)
    rhoBpm = (1/np.pi)*freq2time(rhoD * boseeinstein(-1.*beta*(omega+eta)),M,dt)
    
    argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,1)
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.heaviside(t,1)
    Sigma = 1j*(g**2)*kappa * time2freq(argSigma,M,dt)
    
    argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,1)
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.heaviside(t,1)
    Pi = 2*1j*(g**2) * time2freq(argPi,M,dt)
    
    return [Sigma, Pi]



def GconfImag(omega,g,beta):
    ''' 
    Arguments omega,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    omega is the grid of fermionic matsubara frequencies
    '''
    c1 = 1.154700
    delta = 0.420374134464041
    
    return 1/((1j*omega) * (1 + (c1*np.abs((g**2)/omega)**(2*delta))))

def DconfImag(nu,g,beta):
    ''' 
    Arguments: nu,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    nu is the grid of fermionic matsubara frequencies
    '''
    T = 1.0/beta
    c2 = 0.561228
    c3 = 0.709618
    delta = 0.420374134464041
    omegar2 = c2 * (T/(g**2))**(4*delta - 1)
    
    return 1.0/(nu**2 + omegar2 + c3*(np.abs(nu/(g**2)))**(4*delta - 1))


def DfreeImagtau(tau,r,beta):
    '''
    Arguments: tau,r,beta
    Obtained by contour integration of Dfreeomega
    '''
    m = np.sqrt(r)
    pref = 1./(2*m)
    num = (tau-(beta/2))*m
    den = beta*m/2
    return pref * np.cosh(num)/np.sinh(den)


def ret_omegar2(g,beta):
    ''' Bad programming practice I know, don't judge'''
    T = 1.0/beta
    c2 = 0.561228
    delta = 0.420374134464041
    omegar2 = c2 * (T/(g**2))**(4*delta - 1)
    return omegar2

def DfreeRealt(t,r,eta=1e-6):
    '''
    Arguments t, r(bare boson mass squared)
    Real time retarded boson greens function 
    '''
    omega0 = np.sqrt(r)
    return np.heaviside(t,1.0) * (np.sin(omega0 * t)/omega0)  * np.exp(-eta*np.abs(t))

def DfreeRealomega(omega,r,eta=1e-6):
    '''
    Arguments omega, r(bare boson mass squared) 
    Real frequency retarded boson greens function 
    '''
    Dinv = r - (omega + 1j*eta)**2
    return 1.0/Dinv

def GfreeRealomega(omega,mu,eta=1e-6):
    '''
    Arguments omega, mu 
    Real frequency retarded fermion greens function 
    '''
    return 1. / (omega + 1j*eta - mu)

def GfreeRealt(t,mu,eta=1e-6):
    '''
    Arguments t, mu
    Real time retarded fermion greens function
    '''
    return -1j * np.heaviside(t,1.0) * np.exp(-1j*mu*t) * np.exp(-eta*np.abs(t))


def CrazyGconfReal(omega,g,beta,eta=0):
    ''' 
    Arguments omega,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    omega is the grid of fermionic matsubara frequencies
    '''
    c1 = 1.154700
    delta = 0.420374134464041
    ompluit = omega + 1j*eta
    denom = ompluit + c1*(g**(4*delta)) * (1j**(2*delta)) * ompluit**(1-2*delta)
    print('bar')
    return 1./denom
    #return 1/((omega+1j*eta) * (1 + (c1*np.abs((g**2)/(omega+1j*eta))**(2*delta))))

# def CrazyDconfReal(omega,g,beta,eta=0):
#     ''' 
#     Arguments: nu,g,beta
#     So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
#     nu is the grid of fermionic matsubara frequencies
#     '''
#     T = 1.0/beta
#     c2 = 0.561228
#     c3 = 0.709618
#     delta = 0.420374134464041
#     omegar2 = c2 * (T/(g**2))**(4*delta - 1)
#     print('foo')
#     return 1.0/(-1*(omega+1j*eta)**2 + omegar2 + c3*(np.abs((omega+1j*eta)/(1j* (g**2))))**(4*delta - 1))

def DfreeThermalomega(omega,r,beta,g,eta=1e-6):
    '''
    Arguments omega, r(bare boson mass squared) 
    Real frequency retarded boson greens function 
    '''
    omegar2 = ret_omegar2(g,beta)
    Dinv = omegar2 - (omega + 1j*eta)**2 
    #Dinv = r - (omega + 1j*eta)**2
    #M = len(omega) // 2
    #Dinv[M] = omegar2  
    print('foo barr')
    return -1.0/Dinv

def CrazyDconfReal(omega,g,beta,eta=0):
    ''' 
    Arguments: nu,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    nu is the grid of fermionic matsubara frequencies
    '''
    T = 1.0/beta
    c2 = 0.561228
    c3 = 0.709618
    delta = 0.420374134464041
    omegar2 = c2 * (T/(g**2))**(4*delta - 1)
    print('boo')
    return 1.0/(1*(omega+1j*eta)**2 + omegar2 + c3*(np.abs((omega+1j*eta)/(1j* (g**2))))**(4*delta - 1))
