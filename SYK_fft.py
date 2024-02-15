from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import numpy as np


def Freq2TimeF(gg, Nbig, beta):
    ''' Fast fourier transform (from fermionic matsubara frequency to time) 
        Input: 
              gg - array of Greens functions depending on the frequency
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the imaginary time    
    '''
    n = np.arange(Nbig)
    return 1./beta *np.exp(1j*np.pi*(Nbig-1)*(2*n+1)/(2*Nbig))*fft(np.exp(-1j*np.pi*n/Nbig)*gg)


def Time2FreqF(gg, Nbig, beta):
    ''' Fast fourier transform (from imaginary time to fermionic frequency) 
        Input: 
              gg - array of Greens functions depending on the imaginary time
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the frequency   
    '''
    n = np.arange(Nbig)
    return (beta)*np.exp(1j*np.pi*n/Nbig)*ifft(np.exp(-1j*np.pi*(Nbig-1)*(n+1./2)/Nbig)*gg) 



def Freq2TimeB(gg, Nbig, beta):
    ''' Fast fourier transform (from Bosonic matsubara frequency to time) 
        Input: 
              gg - array of Greens functions depending on the frequency
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the imaginary time    
    '''
    n = np.arange(Nbig)
    return 1./beta *np.exp(1j*np.pi*(n+1./2))*fft(np.exp(-1j*np.pi*n/Nbig)*gg)


def Time2FreqB(gg, Nbig, beta):
    ''' Fast fourier transform (from imaginary time to bosonic frequency) 
        Input: 
              gg - array of Greens functions depending on the imaginary time
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the frequency   
    '''
    n = np.arange(Nbig)
    return (beta)*np.exp(1j*np.pi*n/Nbig)*ifft(np.exp(-1j*np.pi*(n + 1./2))*gg) 



def time2freq(ftau,M,dt): 
    '''
    Real time/frequency version
    '''
    pref = 2 * M * dt
    OmegaminusM = np.arange(2*M) - M
    tau = np.arange(2*M)
    prefexp = np.exp(-1j * np.pi * OmegaminusM)
    return prefexp * pref * ifft(np.exp(-1j*np.pi*tau)*ftau) 

def freq2time(fomega,M,dt):
    '''
    real frequency/time version
    NOTE: THIS SIMPLY DOES THE INTEGRAL: need to multiply the result by 1/(2pi) for a fourier transforrm
    '''
    dw = np.pi/(M*dt)
    pref = dw
    tauminusM = np.arange(2*M) - M
    omega = np.arange(2*M)
    prefexp = np.exp(1j*np.pi*tauminusM)
    return prefexp * pref * fft(np.exp(1j*np.pi*omega)*fomega)

def fermidirac(arg, default = True):
    '''
    returns 1/(1+ exp(x))
    '''
    if default:
        return (1.0/(1.0 + np.exp(arg)))
    else:
        answer = 1
        if arg < 0:
            answer = 1.0/(1.0 + np.exp(arg))
        else: 
            answer = np.exp(-arg)/(1.0 + np.exp(-arg))
        return answer
    
def boseeinstein(arg, default = True):
    '''
    returns 1/(exp(x)-1)
    Watch out for x=0
    '''
    if default:
        return (1.0/(np.exp(arg)-1))
    else:
        answer = 1
        if arg < 0:
            answer = 1.0/(np.exp(arg)-1)
        elif arg > 0: 
            answer = np.exp(-arg)/(1.0 - np.exp(-arg))
        else :
            answer = 0.
        return answer
    
    
def omega_idx(omegaval,dw,M):
    '''
    returns the index of omegaval on the conventional omega grid
    omega[M] = 0
    '''
    return int(M + np.floor(omegaval/dw))


def RealGridMaker(M,T):
    '''
    returns an omega and t grid used in all the real time (I)FFT
    parameters:
    M : int - Large positive integer, size of the grid is 2M - 1 
    T : float - Upper cutoff on time 
    returns: 
    omega : real frequency grid 
    t : real time grid 
    '''
    dt = (2*T)/((2*M))
    t = dt * (np.arange(2*M) - M)
    dw = np.pi/(M*dt)
    omega = dw * (np.arange(2*M) - M)
    return omega,t


def ImagGridMaker(Nbig,beta,which_grid = 'unknown'):
    '''
    Returns imaginary time grid consistent with FFT in matsubara
    depending on the type asked. 
    parameters: 
    Nbig : int - Large positive integer, size of the grid is Nbig
    beta : inverse temperature
    which_grid: options: 'tau', 'boson', 'fermion'
    returns: 
    depending on type, tau grid from 0 to beta, 
    boson/fermion - corresponding matsubara freq grid nu/omega
    '''
    if which_grid == 'fermion':
        omega = (2 * np.arange(Nbig) - Nbig + 1) * np.pi/beta
        return omega
    elif which_grid == 'boson':
        nu = (2 * np.arange(Nbig) - Nbig ) * np.pi/beta
        return nu 
    elif which_grid == 'tau':
        tau = (np.arange(Nbig) + 1./2) * beta/Nbig
        return tau 
    else :
        raise Exception("which_grid only accepts 'boson', 'fermion', 'tau' ")
        return None