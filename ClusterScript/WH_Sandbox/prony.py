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


import numpy as np
from scipy.linalg import lstsq, toeplitz

def prony_fit(x, y, n):
    """
    Fit a sum of exponentials to the data using the Prony method.

    Parameters:
        x : numpy array
            Independent variable (e.g., time).
        y : numpy array
            Dependent variable (e.g., signal).
        n : int
            The number of exponentials to fit.

    Returns:
        A : numpy array
            Amplitudes of the exponentials.
        alpha : numpy array
            Decay rates of the exponentials.
    """
    # Step 1: Create the Hankel matrix
    m = len(x)
    Y = np.array([y[i:m-n+i] for i in range(n)]).T
    
    # Step 2: Compute the characteristic polynomial coefficients
    b = -y[n:m]
    coefficients = lstsq(Y, b)[0]
    coefficients = np.concatenate(([1], coefficients))

    # Step 3: Find the roots of the characteristic polynomial to get decay rates
    roots = np.roots(coefficients)
    alpha = np.log(roots) / (x[1] - x[0])

    # Step 4: Set up and solve the Vandermonde system for amplitudes
    V = np.exp(np.outer(x[:n], alpha))
    A = lstsq(V, y[:n])[0]

    return A, alpha

# Example usage
x = np.linspace(0, 10, 100)  # Example x data (time)
y = 3 * np.exp(-2 * x) + 2 * np.exp(-0.5 * x)  # Example y data (signal)

# Fit 2 exponentials to the data
A, alpha = prony_fit(x, y, 2)

print("Amplitudes:", A)
print("Decay rates:", alpha)
