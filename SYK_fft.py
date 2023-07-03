from scipy.fftpack import fft as fft
from scipy.fftpack import ifft as ifft
import numpy as np


def Freq2Time(gg, Nbig, beta):
    ''' Fast fourier transform (from frequency to time) 
        Input: 
              gg - array of Greens functions depending on the frequency
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the imaginary time    
    '''
    n = np.arange(Nbig)
    return 1./beta *np.exp(1j*np.pi*(Nbig-1)*(2*n+1)/(2*Nbig))*fft(np.exp(-1j*np.pi*n/Nbig)*gg)


def Time2Freq(gg, Nbig, beta):
    ''' Fast fourier transform (from imaginary time to frequency) 
        Input: 
              gg - array of Greens functions depending on the imaginary time
              Nbig - size of a grid
              beta - inverse temperature
        Output: 
               array of Greens functions depending on the frequency   
    '''
    n = np.arange(Nbig)
    return (beta)*np.exp(1j*np.pi*n/Nbig)*ifft(np.exp(-1j*np.pi*(Nbig-1)*(n+1./2)/Nbig)*gg) 