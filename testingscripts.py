import numpy as np
from SYK_fft import *
import warnings
import inspect

def realtimeFFT_validator():
    '''
    Needs to rerturn True for a successful test
    '''
    M = int(2**18) 
    T = 2000 
    dt = (2*T)/((2*M))
    t = dt * (np.arange(2*M) - M)
    dw = np.pi/(M*dt)
    omega = dw * (np.arange(2*M) - M)  

    sigma = T/20
    x = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(t**2)/(2*(sigma**2)))
    y = time2freq(x,M,dt)
    xprime = 0.5/np.pi * freq2time(y,M,dt)

    return np.allclose(xprime , x) 


def diff_checker(diffseries, tol = 1e-2, periods = 5, verify = False):
    '''
    Used to check if a numerical simulation is converging too slowly. 
    At the moment the implementation requires at least 7 data points. 
    Returns flag False if convering too slowly, True otherwise.
    '''
    flag = True
    data = diffseries[-1-periods:-1]
    if np.var(data)<tol:
        warnings.warn('converging too slowly in function ' + inspect.stack()[1][3])
        flag = False
    return flag
    
    
    