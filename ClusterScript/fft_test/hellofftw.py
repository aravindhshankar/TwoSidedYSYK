import numpy as np 
from scipy.fft import fft, ifft, fftfreq, fftshift
import time 
from matplotlib import pyplot as plt
import sys, os
sys.path.insert(0,'../')
# sys.path.insert(0,'../../')
from Sources.SYK_fft import time2freq, freq2time, RealGridMaker
import pyfftw
from Sources.h5_handler import dict2h5, h52dict

N = int(2**18) #make sure this is an even number 
np.testing.assert_equal(N%2,0)
T = 2**7
t = np.linspace(0,T,N)
dt = t[2] - t[1]
omega = fftfreq(N,dt) 
dw = omega[2] - omega[1]
omega_max = np.max(omega)
omega_min = np.min(omega)
print(f'dw = {dw:.4}, 1/(N dt) = {1./(N*dt)}')
print(f'omega_max = {omega_max}, (N-1)/(2T) = {(N-1)/(2*T)}')
print(f'omega_min = {omega_min}, (N-1)/(2T) = {(N-1)/(2*T)}')
# eta = 1e-2
eta = dw * 20.1
x = np.exp(-eta * t) * np.heaviside(t,0) * (np.exp(2j* np.pi * t) + np.exp(2j * np.pi * 3 * t)) #frequencies 1 and 3  
print(f'dtype x = {x.dtype}')
SYK_omega, SYK_t = RealGridMaker(N//2,T) # N = 2 M , the total number of points in the grid, in this notation
SYK_dw = SYK_omega[2] - SYK_omega[1]
SYK_dt = SYK_t[2] - SYK_t[1]

xSYK = np.exp(-eta * SYK_t) * np.heaviside(SYK_t,0) * (np.exp(2j* np.pi * SYK_t) + np.exp(2j * np.pi * 3 * SYK_t)) #frequencies 1 and 3  
xSYK = np.exp(-eta * SYK_t) * np.heaviside(SYK_t,0) * ( np.exp(2j * np.pi * 1 * SYK_t)) #frequencies 1 and 3  

start = time.perf_counter()
y = dt * fft(x, workers=None) #workers only matters for 2d transforms and beyond by splitting up tasks into 1d transforms
stop = time.perf_counter()
print(f'Scipy fourier transform computed in {stop-start} seconds.')

start = time.perf_counter()
yS = time2freq(xSYK,N//2,dt)
stop = time.perf_counter()
print(f'SYK fourier transform computed in {stop-start} seconds.')

#now for the pyfftw
wisdomflag = False
try : # Warning : the first time for the parameters where there's no wisdom, the computed transform is wrong
    wisdom = h52dict('wisdom_file.h5')
    pyfftw.import_wisdom(wisdom['wisdom'])
except FileNotFoundError : 
    wisdomflag = True
a = pyfftw.empty_aligned(N, dtype='complex128')
b = pyfftw.empty_aligned(N, dtype='complex128')
a[:] = x # the elements should be copied into the created array a 
start = time.perf_counter()
fft_object = pyfftw.FFTW(a, b) #this plan is able to do the fourier transform of the array placed in memeory as a and produce results in place in b
fft_a = fft_object()
stop = time.perf_counter()
# Write newly accumulated wisdom back to file 
wisdom = pyfftw.export_wisdom()
wisdom_dict = {'wisdom':wisdom}
dict2h5(wisdom_dict, 'wisdom_file.h5', verbose=True)

print(fft_object.input_array is a)
print(fft_a is b)
print(f'FFTW fourier transform computed in {stop-start} seconds.')

def defaultRealGridMaker(M,T):
    '''
    T is upper cutoff on time
    '''
    t = np.linspace(0,T,M)
    dt = t[2] - t[1]
    omega = fftfreq(M,dt)
    return omega, t

DEF_omega, DEF_t = defaultRealGridMaker(N//2, T)
DEF_dw, DEF_dt = DEF_omega[2] - DEF_omega[1], DEF_t[2] - DEF_t[1]
print(f'SYK dw = {SYK_dw}, DEF dw = {DEF_dw}')
print(f'SYK dt = {SYK_dt}, DEF dt = {DEF_dt}')

plt.plot(fftshift(omega), fftshift(y.real), '-')
# plt.plot(fftshift(omega), fftshift(y.imag), '--')
# plt.plot(fftshift(omega), fftshift(fft_a.real), '-',c='red')
# plt.plot(fftshift(omega), fftshift(fft_a.imag), '--', c='red')
# plt.plot(SYK_omega,yS.real) 
plt.show()
