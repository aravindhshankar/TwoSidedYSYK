def prony(t, F, m):
	"""Input  : real arrays t, F of the same size (ti, Fi)
			: integer m - the number of modes in the exponential fit
		Output : arrays a and b such that F(t) ~ sum ai exp(bi*t)"""

	import numpy as np
	import numpy.polynomial.polynomial as poly

	# Solve LLS problem in step 1
	# Amat is (N-m)*m and bmat is N-m*1
	N    = len(t)
	Amat = np.zeros((N-m, m))
	bmat = F[m:N]

	for jcol in range(m):
		Amat[:, jcol] = F[m-jcol-1:N-1-jcol]
		
	sol = np.linalg.lstsq(Amat, bmat,rcond=None)
	d = sol[0]

	# Solve the roots of the polynomial in step 2
	# first, form the polynomial coefficients
	c = np.zeros(m+1)
	c[m] = 1.
	for i in range(1,m+1):
		c[m-i] = -d[i-1]

	u = poly.polyroots(c)
	b_est = np.log(u)/(t[1] - t[0])

	# Set up LLS problem to find the "a"s in step 3
	Amat = np.zeros((N, m))
	bmat = F

	for irow in range(N):
		Amat[irow, :] = u**irow
		
	sol = np.linalg.lstsq(Amat, bmat,rcond=None)
	a_est = sol[0]

	return a_est, b_est