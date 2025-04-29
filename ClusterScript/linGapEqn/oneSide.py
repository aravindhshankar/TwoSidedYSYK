import numpy as np
import os, sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
from Sources.ConformalAnalytical import GconfImag, DconfImag
# from scipy.linalg import eigvalsh
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigsh
from scipy.optimize import newton,bisect

#Gconf(omega,g,beta)
#Dconf(nu,g,beta)
cutoff = 100
g = 0.5 
def retOmegan(n:int, beta):
    return (2 * n + 1) * np.pi / beta

def retNun(n:int, beta):
    return (2 * n) * np.pi / beta

def kernelElem(n:int, m:int, g, beta):
    pref = 1.0 * g**2 / beta
    omega = retOmegan(n, beta)
    nu = retNun(n-m, beta)
    return pref * np.abs(GconfImag(omega,g,beta))**2 * DconfImag(nu,g,beta)


nlist = np.arange(-1*cutoff, cutoff, dtype=int)
mlist = np.arange(-1*cutoff, cutoff, dtype=int)

beta = 200
kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
# print(eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0])
print(eigvals(kernelMat)-1)
def objective(beta):
    kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    return max((eigvals(kernelMat))) - 1
# def objective(beta):
    # kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    # return eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0] - 1

# betac = newton(objective, x0=10)
betac = bisect(objective, a=10, b=200)
print(f'betac = {betac}')
print(f'Tc = {1./betac}')


