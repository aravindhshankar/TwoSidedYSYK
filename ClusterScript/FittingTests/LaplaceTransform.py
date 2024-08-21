import numpy as np 
from scipy.integrate import simpson, quad
from matplotlib import pyplot as plt
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
import time 
from scipy.optimize import curve_fit



NO_OF_SAMPLES = 2**14
t = np.linspace(10,5000,NO_OF_SAMPLES)
alist = np.array([0.1,0.002,0.00012, 0.001,0.0003 ])
# blist = np.array([1,2])
blist = np.array([0.00461,0.01559,0.022,0.033,0.044])
ft = np.array([np.sum(alist * np.exp(-1.0*blist*tval)) for tval in t])


def laplace(t,f,s, INTEGRATOR = 'quad'):
	'''
	We're assuming the t grid runs all the way to infinity 
	For simpson make sure that t is an np array
	'''
	if INTEGRATOR == 'quad': 
		# interpol = Akima1DInterpolator(t, f, method = 'makima')
		interpol = PchipInterpolator(x=t, y=f)
		# callintegrand = lambda tval : np.exp(-sval*tval) * interpol(tval)
		Fs = np.array([quad(lambda tval : np.exp(-sval*tval) * interpol(tval), t[0],t[-1])[0] for sval in s])
	elif INTEGRATOR == 'simpson':
		Fs = np.array([simpson(np.exp(-sval*t) * f , x=t) for sval in s])
	else:
		raise(Exception('Valid choices of integrator are \'quad\' and \'simpson\' '))

	return Fs


def manual():
	linfitslice = slice(0,-1)
	m,logc = np.polyfit(t[linfitslice],np.log(ft[linfitslice]),1)
	c = np.exp(logc)
	loglinfit = c * np.exp(m * t)

	remnant1 = ft - loglinfit
	linfitslice2 = slice(5,20)
	m2,logc2 = np.polyfit(t[linfitslice2],np.log(remnant1[linfitslice2]),1)
	c2 = np.exp(logc2)
	loglinfit2 = c2 * np.exp(m2 * t)


	fig,ax = plt.subplots(1)
	ax.plot(t,ft,'.')
	ax.plot(t, loglinfit,'--', label = f'log lin fit with exponent {m:.2}')
	ax.plot(t, remnant1,':', label = 'first remnant' )
	ax.plot(t,loglinfit2, '--', label = f'second fit with exponent {m2:.2}')
	ax.set_yscale('log')
	ax.set_xlabel('t')
	ax.set_ylabel(r'$f(t)$')
	ax.legend()


	# def model(x, *alist,*blist):
	# 	return np.array([np.sum(alist * np.exp(-1.0*blist*tval)) for tval in t])

	def model(x, a1,a2,a3,b1,b2,b3):
		return a1*np.exp(-b1*x)+a2*np.exp(-b2*x)+a3*np.exp(-b3*x)

	print('Actual data exponents', blist)
	popt,pcov = curve_fit(model,t,ft,bounds=(0,[np.inf,np.inf,np.inf,1,1,1]),p0=(1,1,1,0.01,0.02,0.02)
							, jac = '3-point',ftol=1e-14)
	print('popt = ', popt)
	# print('pcov = ', pcov)
	perr = np.sqrt(np.diag(pcov))
	print('One std dev error', perr)

	# plt.show()


def testing_laplace():
	# s = np.linspace(0.01,5,100)

	# manual()
	s = np.linspace(0,10,1000)
	start = time.perf_counter()
	FsQUAD = laplace(t,ft,s, INTEGRATOR = 'quad')
	print(f'TypeFsQUAD = {type(FsQUAD)}')
	stop = time.perf_counter()
	print(f'Finished quad integrator in {stop - start} seconds')

	start = time.perf_counter()
	FsSIMPS = laplace(t,ft,s, INTEGRATOR = 'simpson')
	print(f'TypeFsSimps = {type(FsSIMPS)}')
	stop = time.perf_counter()
	print(f'Finished Simpson integrator in {stop-start} seconds')
	figL, (axL,axgrad) = plt.subplots(2)
	axL.set_xlabel('s')
	axL.set_ylabel(r'$F(s)$')
	# axL.set_ylim(-5,5)
	axL.plot(s,FsQUAD, '.-', label = 'quad')
	axL.plot(s,FsSIMPS,':', label = 'simpson')
	axL.set_yscale('log')
	axL.legend()

	gradlaplace = np.gradient(FsQUAD, s)
	gradlaplace2 = np.gradient(gradlaplace, s)
	axgrad.plot(s,np.abs(gradlaplace),label ='first derivative')
	axgrad.plot(s,np.abs(gradlaplace2),label ='second derivative')
	axgrad.set_xlabel('s')
	axgrad.set_ylabel('|grad F(s)|')
	axgrad.set_title('Gradient of the laplace transform')
	axgrad.set_yscale('log')
	axgrad.legend()

def PadeLaplacematrixsolver(d,n,s0):
	'''
	d is the list of derivatives, appropriately normalized by factorials
	n is the order 
	'''
	assert len(d) >= 2 * n, "Not enough derivatives" 
	dmat = np.zeros((n,n))
	dvec = np.zeros(n)

	for i in np.arange(n):
		dvec[i] = -1.0 * d[n+i]
		for j in np.arange(n):
			dmat[i,j] = d[n+i-j-1]

	#now solve for the b list 
	blist = np.linalg.solve(dmat,dvec) #### solves a x = b

	#blist contains now b1, b2 ......
	polycoeffs = np.append(blist[::-1],[1.,])

	#numpy roots needs the polynomial coeffs in descending order
	np.testing.assert_equal(len(polycoeffs), n+1)

	pminuss0 = np.roots(polycoeffs)

	exponents = pminuss0 + s0
	return exponents

def testingExponentsPade():
	# s = np.linspace(0.01,5,100)

	manual()
	s = np.linspace(0,10,1000)
	start = time.perf_counter()
	FsQUAD = laplace(t,ft,s, INTEGRATOR = 'quad')
	print(f'TypeFsQUAD = {type(FsQUAD)}')
	stop = time.perf_counter()
	print(f'Finished quad integrator in {stop - start} seconds')

	start = time.perf_counter()
	FsSIMPS = laplace(t,ft,s, INTEGRATOR = 'simpson')
	print(f'TypeFsSimps = {type(FsSIMPS)}')
	stop = time.perf_counter()
	print(f'Finished Simpson integrator in {stop-start} seconds')
	figL, (axL,axgrad) = plt.subplots(2)
	axL.set_xlabel('s')
	axL.set_ylabel(r'$F(s)$')
	# axL.set_ylim(-5,5)
	axL.plot(s,FsQUAD, '.-', label = 'quad')
	axL.plot(s,FsSIMPS,':', label = 'simpson')
	axL.set_yscale('log')
	axL.legend()

	gradlaplace = np.gradient(FsQUAD, s)
	gradlaplace2 = (1./2.) * np.gradient(gradlaplace, s)
	gradlaplace3 = (1./3.) * np.gradient(gradlaplace2, s)
	gradlaplace4 = (1./4.) * np.gradient(gradlaplace3, s)
	axgrad.plot(s,np.abs(gradlaplace),label ='first derivative')
	axgrad.plot(s,np.abs(gradlaplace2),label ='second derivative')
	axgrad.plot(s,np.abs(gradlaplace3),label ='third derivative')
	axgrad.plot(s,np.abs(gradlaplace4),label ='fourth derivative')
	axgrad.set_xlabel('s')
	axgrad.set_ylabel('|grad F(s)|')
	axgrad.set_title('Gradient of the laplace transform')
	axgrad.set_yscale('log')
	axgrad.legend()

	N = 2 # we need at least 2n-1 derivatives
	s0 = 1.6
	s0idx = np.argmin(np.abs(s-s0))
	assert s0idx < len(s) , "pick a valid s0 ya DOLT!"

	d_arr = np.array((FsQUAD[s0idx], gradlaplace[s0idx], gradlaplace2[s0idx], gradlaplace3[s0idx],gradlaplace4[s0idx]))

	exponents = PadeLaplacematrixsolver(d_arr,N,s0)
	print(f's0 = {s0}')
	print('Found Exponents are ', exponents)


def main():
	# testing_laplace()
	# testingExponentsPade()
	manual()
	plt.show()






if __name__ == '__main__':
	main()