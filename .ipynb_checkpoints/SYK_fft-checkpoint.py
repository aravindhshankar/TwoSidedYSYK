#from scipy.fftpack import fft as fft
#from scipy.fftpack import ifft as ifft
#from scipy.fftpack import fftshift
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

