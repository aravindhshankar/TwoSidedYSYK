import numpy as np 
from scipy.integrate import simpson, quad
from matplotlib import pyplot as plt
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
import time 


NO_OF_SAMPLES = 100
t = np.linspace(0,100,NO_OF_SAMPLES)
alist = np.array([0.1,0.02])
blist = np.array([1,2])
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
	linfitslice = slice(-10,-1)
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
	# plt.show()


def testing_laplace():
	# s = np.linspace(0.01,5,100)

	# manual()
	s = np.linspace(-2,10,1000)
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

def main():
	testing_laplace()
	plt.show()






if __name__ == '__main__':
	main()