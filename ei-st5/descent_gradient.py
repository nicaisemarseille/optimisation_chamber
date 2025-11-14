import _env
import numpy
import projected_chi
import preprocessing

def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		print("Robin")
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi



def int_gamma(chi, domain_omega):
	nb = 0.0
	M,N = numpy.shape(domain_omega)
	int_nn = 0.0
	for i in range(0, M):
		for j in range(0, N):
			if domain_omega[i, j] == _env.NODE_ROBIN:
				nb += 1
				int_nn += chi[i, j]
	return int_nn / nb  

def dist_L_inf(chi1,chi2):
    return numpy.max(numpy.abs(chi1-chi2))   # Norme infinie de chi1 - chi2
    


def project(chi, prop, domain_omega):
    """
    data type : 
    chi = array((M,N), dtype=complex)
    prop = float
    """
    (M, N) = numpy.shape(domain_omega)
    def find_l(prop, chi, domain_omega, tol=0.0001):
        # Initial limits for the dichotomy
        l_min, l_max = -2, 2  # Adjust according to your application
        chi1 = numpy.maximum(0, numpy.minimum(chi + l_min, 1))
        chi1 = preprocessing.set2zero(chi1, domain_omega)
        # While the difference is greater than tolerance
        while l_max - l_min > tol:
            l_mid = (l_min + l_max) / 2
            chi1 = preprocessing.set2zero(numpy.maximum(0, numpy.minimum(chi + l_mid, 1)),domain_omega)
            int_val = int_gamma(chi1, domain_omega)
            
            # Compare with the target
            if int_val < prop:
                l_min = l_mid
            else:
                l_max = l_mid

        return (l_min + l_max) / 2

    l = find_l(prop, chi, domain_omega)
    return preprocessing.set2zero(numpy.maximum(0, numpy.minimum(chi + l, 1)),domain_omega)


def descent_and_project(chi, prop, domain_omega,grad , mu):
	descendu = compute_gradient_descent(chi, grad, domain_omega, mu)
	return project(descendu,prop,domain_omega)