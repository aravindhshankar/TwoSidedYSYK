import numpy as np 

def E(n,a,theta):
	# Exponential sum function for n and the model (a,theta)
	y = np.zeros_like(n,dtype = np.float64) # np.float64
	for ai,theta_i in zip(a,theta):
		y += ai*theta_i**n
	return np.abs(y) # return y

def R0(X,n,a,theta):
	# Empirical risk associated with the current model (a,theta)
	return np.sum((X - E(n,a,theta))**2)

def P(grid_thetas,X,n,a,theta):
	# Residual polynomial (from the paper)
	y = np.zeros_like(grid_thetas)
	for ni in n:
		y += 2*(E(ni,a,theta) - X[ni])*grid_thetas**ni
	return y

def linear_stage(A,X): 
	# Simple linear regression to find the amplitudes
	sol = np.linalg.lstsq(A, X, rcond=None)
	a_new = sol[0]
	return a_new

def drop_negatives(a_old, a_new): 
	# Begin minor iteration to remove negative terms from the fit
	if (np.min(a_new) < 0): 
		min_idx = -1 # index k, in the paper 
		min_beta = float('inf') # beta_k, in the paper
		for idx, (aoi, ani) in enumerate(zip(a_old, a_new)):
			# aoi = a_old idx, ani = a_new idx
			if ani < 0: 
				# guaranteed to be true for one 
				# of the amplitudes, see outer "if" statement
				beta = aoi / (aoi - ani)
				if beta < min_beta: 
					min_beta = beta
					min_idx = idx
		# Now update a_old
		a_old = (1 - min_beta) * a_old + min_beta*a_new
		a_old[min_idx] = 0
		return (min_idx,a_old)
	else: 
		return (None,None)



def coalesceTerms(a, theta, M):
	# Coalesces terms until M terms remain
	N = len(theta)
	if N <= M: # quit if we have M (or fewer) terms
		return (a,theta)
	while N > M:
		# find closest pair
		mind = float('inf')
		minidx = (0,0)
		for i in range(N):
			for j in range(N):
				d = np.abs(theta[i] - theta[j])
				if d < mind: 
					mind = d
					minidx = (i,j)

		a_new = a[i] + a[j]
		t_new = 0.5*(theta[i] + theta[j])
		a[i] = a_new
		theta[i] = t_new
		a = np.delete(a,j)
		theta = np.delete(theta,j)
		N = len(theta)
	return (a,theta)


def fitEDSF(y, n, M = None, epsilon_1 = 10e-20, epsilon_2 = 10e-20): 
	# fit exponential decay sum function
	# y: the exponential sum fit values to be fitted against
	# n: equally-spaced and contiguous points of n
	#    for instance, n = np.array([1,2,3,4,5,6,7,8])
	# M: model order. If None, then the model runs to convergence.
	#    if given, then the model returns exactly M components
	# epsilon_1: convergence criteria 1 from paper, default works well
	# epsilon_2: convergence criteria 2 from paper, default works well

	# Convergence flags
	converged = False
	R0_old = None
	R0_new = None

	# Parameters
	a_old = np.array([]) # set of amplitude values from previous iter
	a = np.array([]) # set of amplitude values 
	theta = [] # theta is a list, need to append often 


	iters = 0 
	while not converged:
		a_old = a # update previous iteration values

		# 1. Nonlinear part - find a good theta (rate)
		# add theta to minimize the residual polynomial P
		grid_thetas = np.linspace(0,1,100000)
		P_theta = P(grid_thetas,y,n,a,theta)
		idx = np.argmin(P_theta)
		theta_new = grid_thetas[idx]
		theta.append(theta_new)

		# 2. Now set up linear regression problem to find the 
		# amplitudes from the system y = A*a
		A = np.zeros((len(n),len(theta)),dtype = np.float64)
		for i,theta_i in enumerate(theta):
			A[:,i] = theta_i**n

		a_new = linear_stage(A,y)
		a_old = np.append(a_old,0)

		# 3. Eliminate terms with negative a_i
		drop_index, a_old = drop_negatives(a_old, a_new)
		while drop_index is not None:
			# delete the negative term
			a_old = np.delete(a_old,drop_index) 
			A = np.delete(A,drop_index,axis = 1)
			del theta[drop_index]
			a_new = linear_stage(A,y) # find new a_i
			# ensure new a_i > 0
			drop_index, a_old = drop_negatives(a_old, a_new) 


		# 4. Have we converged?
		a = a_new
		R0_new = R0(y,n,a,theta)

		if R0_old is not None: 
			E1 = (R0_old - R0_new)/(R0_old) 
			if E1 < epsilon_1: 
				converged = True

		E2 = P_theta[idx]
		if E2 >= epsilon_2: 
			converged = True

		# If we don't have M components
		# keep iterating
		if M is not None:
			if len(theta) < M:
				converged = False
		iters += 1 
		if iters > 200: 
			converged = True

		R0_old = R0_new

	# Found a good fit, return coefficients
	theta = np.array(theta)
	a = np.array(a)
	if M is not None:
		a,theta = coalesceTerms(a, theta, M)

	final_err = R0(y,n,a,theta)
	return (a,theta,final_err)