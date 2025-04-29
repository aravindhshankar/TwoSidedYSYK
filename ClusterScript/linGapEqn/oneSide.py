import numpy as np
import os, sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
from Sources.ConformalAnalytical import GconfImag, DconfImag
# from scipy.linalg import eigvalsh
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigsh
from scipy.optimize import newton,bisect
from scipy.special import gamma as Gamma

#Gconf(omega,g,beta)
#Dconf(nu,g,beta)
cutoff = 100
g = 0.5 
def retOmegan(n:int, beta):
    return (2 * n + 1) * np.pi / beta

def retNun(n:int, beta):
    return (2 * n) * np.pi / beta

# def kernelElem(n:int, m:int, g, beta):
    # pref = 1.0 * g**2 / beta
    # omega = retOmegan(n, beta)
    # nu = retNun(n-m, beta)
    # return pref * np.abs(GconfImag(omega,g,beta))**2 * DconfImag(nu,g,beta)
def kernelElem(n:int, m:int, g, beta): #symmetrized
    pref = 1.0 * g**2 / beta
    omegan = retOmegan(n, beta)
    omegam = retOmegan(m, beta)
    nu = retNun(n-m, beta)
    return pref * np.abs(GconfImag(omegan,g,beta)) * DconfImag(nu,g,beta) * np.abs(GconfImag(omegam,g,beta))
# def kernelElem(n:int, m:int, g, beta):
    # omegan = retOmegan(n, beta)
    # omegam = retOmegan(m, beta)
    # nu = retNun(n-m, beta)
    # deltaf = 0.420374134464041
    # deltab = 1. - 2.*deltaf
    # pref = np.sqrt(np.pi) * 2**(2-deltab) / (g**2)
    # # print(f'pref={pref}') if __debug__ else None 
    # invEuBeta = Gamma(0.5+deltaf+deltab) / (Gamma(0.5+deltaf)*Gamma(deltab))
    # # print(f'invEuBeta={invEuBeta}') if __debug__ else None 
    # term2 = Gamma(1-deltaf) * Gamma(0.5-deltab) / Gamma(1-deltaf-deltab)
    # # print(f'term2={term2}') if __debug__ else None 
    # return pref * invEuBeta * term2 * np.abs(omegan)**(2*deltaf-1) * np.abs(omegam)**(2*deltaf-1) * np.abs(nu)**(2*deltab-1)

    


nlist = np.arange(-1*cutoff, cutoff, dtype=int)
mlist = np.arange(-1*cutoff, cutoff, dtype=int)

beta = 200
kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
# print(np.isnan(kernelMat).any())
print(eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0])
# print(eigvals(kernelMat)-1)
# def objective(beta):
    # kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    # return max((eigvals(kernelMat))) - 1
def objective(beta):
    kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    return eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0] - 1

betac = newton(objective, x0=10)
# betac = bisect(objective, a=10, b=200)
print(f'betac = {betac}')
print(f'Tc = {1./betac}')


