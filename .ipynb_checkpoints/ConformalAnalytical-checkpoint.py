import numpy as np

def GconfImag(omega,g,beta):
    ''' 
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    omega is the grid of fermionic matsubara frequencies
    '''
    c1 = 1.154700
    delta = 0.420374134464041
    
    return 1/((1j*omega) * (1 + (c1*(np.abs((g**2)/omega)**(2*delta)))))