import numpy as np
import os, sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
from Sources.ConformalAnalytical import GconfImag, DconfImag, DImpImag, GImpImag
# from scipy.linalg import eigvalsh
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigsh
from scipy.optimize import newton,bisect
from scipy.special import gamma as Gamma
from functools import partial
from matplotlib import pyplot as plt

#Gconf(omega,g,beta)
#Dconf(nu,g,beta)
cutoff = 20
g = 1
def retOmegan(n:int, beta):
    return (2 * n + 1) * np.pi / beta

def retNun(n:int, beta):
    return (2 * n) * np.pi / beta

# Dimag = DImpImag
# Gimag = GImpImag
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
    if g >= 1.:
        Dimag = DImpImag
        Gimag = GImpImag
    else :
        Dimag = DconfImag
        Gimag = GconfImag
    return pref * np.abs(Gimag(omegan,g,beta)) * Dimag(nu,g,beta) * np.abs(Gimag(omegam,g,beta))
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

# def kernelElem(n:int, m:int, g, beta):
    # return 'M'+f'{n},{m}'


nlist = np.arange(-1*cutoff, cutoff+1, dtype=int)
mlist = np.arange(-1*cutoff, cutoff+1, dtype=int)

# betaval = 200
# kernelMat = np.array([[kernelElem(n,m,g,betaval) for m in mlist] for n in nlist])
# print(np.isnan(kernelMat).any())
# print(eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0])
# print(eigvals(kernelMat)-1)
# def objective(beta):
    # kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    # return max((eigvals(kernelMat))) - 1
def objective(beta,g):
    kernelMat = np.array([[kernelElem(n,m,g,beta) for m in mlist] for n in nlist])
    return eigsh(kernelMat,1,which='LA',return_eigenvectors=False)[0] - 1

glist = np.concatenate([np.linspace(0.1,1,10), np.arange(1,11,1)])
betac = np.array([newton(partial(objective,g=gval), x0=10) for gval in glist])
Tc = 1./betac
# betac = bisect(objective, a=10, b=200)
print(f'betac = {betac}')
print(f'Tc = {1./betac}')


plt.plot(glist,Tc,ls='-',marker='x')
plt.ylabel(r'$\frac{T_c}{\omega_0}$')
plt.xlabel(r'$g$')
plt.savefig('myTcvsg.pdf')
plt.show()
