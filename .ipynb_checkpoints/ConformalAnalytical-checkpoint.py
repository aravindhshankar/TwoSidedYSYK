import numpy as np

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