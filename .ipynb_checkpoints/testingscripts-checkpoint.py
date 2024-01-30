import numpy as np
from SYK_fft import *
import warnings
import inspect

def realtimeFFT_validator():
    '''
    Needs to rerturn True for a successful test
    '''
    flag = False
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

    flag = np.allclose(xprime , x)
    if not flag:
        warnings.warn('realtimeFFT_validator FAILED')
    return flag 



def diff_checker(diffseries, tol = 1e-2, periods = 5, verify = False):
    '''
    Used to check if a numerical simulation is converging too slowly. 
    At the moment the implementation requires at least 7 data points. 
    Returns flag False if convering too slowly, True otherwise.
    '''
    flag = True
    if len(diffseries) < periods +2: 
        flag = True
    else:
        data = diffseries[-1-periods:-1]
        if np.var(data)<tol:
            warnings.warn('converging too slowly in function ' + inspect.stack()[1][3])
            flag = False
    return flag
    
    

def RealGridValidator(omega, t, M, T, dt, dw):
    '''
    omega, t are created by RealGridMaker(M,T). upon usage, dt = t[2]-t[1] and 
    likewise dw = omega[2]-omega[1]. 
    This test checks that the diffs are equal to the theoretical value. 
    '''
    flag = False
    theory_dt = (2*T)/((2*M))
    theory_dw = np.pi/(M*dt)
    np.testing.assert_almost_equal(dt*dw*M,np.pi,5, "Error in fundamentals")
    np.testing.assert_almost_equal(np.max(np.abs(omega)),np.pi*M/T,5,"Error in creating omega grid")
    np.testing.assert_almost_equal(theory_dt,dt,5, "Time grid not according to theory")
    np.testing.assert_almost_equal(theory_dw,dw,5, "Frequency grid not according to theory")
    flag = True
    return flag




































    
    