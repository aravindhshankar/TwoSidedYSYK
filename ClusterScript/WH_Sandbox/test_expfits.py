from prony import prony
from EDSF import fitEDSF
import numpy as np
import matplotlib.pyplot as plt 

def test_prony():
	t = np.linspace(0,50,10000)
	y = np.exp(-t) + 2.4 * np.exp(-2.3 * t )  + 10.3 * np.exp(-1.4 * t)
	y = np.exp(-0.75*t) + 0.1 * np.exp(-0.5 * t )  + 0.001 * np.exp(-0.25 * t)
	m = 3
	a,b = prony(y,t,m)
	print(a,b)


def test_EDSF():
	n = np.arange(0,100)
	# y = np.exp(-n) + 0.1 * np.exp(-2.3 * n )  + 0.01 * np.exp(-1.4 * n)
	y = np.exp(-0.0075*n) + 0.1 * np.exp(-0.005 * n )  + 0.001 * np.exp(-0.0025 * n)
	a,theta,final_err = fitEDSF(y,n)
	print(a,theta,final_err)
	print('actual exponents:', np.log(theta))




	# rates = (0.1,0.5,0.8)
	# N = 100
	# n = np.arange(0,N,1)
	# y = np.zeros_like(n,dtype = np.float64)
	# for r in rates:
	# 	y += r**n

	# a,theta,err = fitEDSF(y,n) 
	# print(a,theta,err)
	# a,theta,err = fitEDSF(y,n,3) 
	# print(a,theta,err)



def main(): 
	# test_prony() ## Utter shit at fitting 
	print('Now EDSF')
	test_EDSF() 


if __name__ == '__main__':
	main()