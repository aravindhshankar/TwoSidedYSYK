import numpy as np
from SYK_fft import *

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