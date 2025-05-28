import numpy as np
from numpy.polynomial.legendre import leggauss
from collections               import namedtuple


Ds_2d      = namedtuple('Ds_2d',    ['D11','D22','D12','D1','D2'])
Ds_3d      = namedtuple('Ds_3d', ['D11','D22','D33','D12','D13','D23','D1','D2','D3'])

def cheb(p):
	"""
	Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
	
	Parameters:
	  p: The polynomial degree
	  
	Returns:
	  D: The Chebyshev differentiation matrix
	  x: The Chebyshev points (flipped)
	"""
	# Compute Chebyshev nodes on [-1,1]
	x = np.cos(np.pi * np.arange(p) / (p - 1))
	# Compute weights: endpoints are 2 and interior are 1, with alternating signs.
	c = np.concatenate((np.array([2]), np.ones(p - 2), np.array([2])))
	c = c * (-1)**np.arange(p)
	# Create difference matrix
	X = x.repeat(p).reshape((-1, p))
	dX = X - X.T
	# Off-diagonal entries
	D = np.outer(c, 1 / c) / (dX + np.eye(p))
	# Diagonal: force row sum to zero
	D = D - np.diag(np.sum(D, axis=1))
	return np.flip(x), np.flip(np.flip(D, axis=0), axis=1)

def lagrange_basis(x, k, nodes):
	"""
	Evaluate the k-th Lagrange basis polynomial at x given the interpolation nodes.
	
	Parameters:
		x : float or numpy.ndarray
			The point(s) at which to evaluate the basis polynomial.
		k : int
			The index of the basis polynomial.
		nodes : array-like
			The interpolation nodes.
	
	Returns:
		float or numpy.ndarray:
			The value of the k-th Lagrange basis polynomial at x.
	"""
	factors = [(x - nodes[j]) / (nodes[k] - nodes[j]) 
			   for j in range(len(nodes)) if j != k]
	return np.prod(factors)

def legfcheb_matrix(p, q):
	"""
	Constructs a transformation matrix to convert a vector tabulated
	on Chebyshev nodes to one tabulated on Legendre nodes.

	Parameters:
		p : int
			The number of nodes for Legendre.
		q : int
			The number of nodes for Chebyshev.
	
	Returns:
		numpy.ndarray: Transformation matrix of shape (p, q).
	"""
	cheb_nodes     = cheb(q)[0]
	legendre_nodes = leggauss(p)[0]
	
	transformation_matrix = np.zeros((p, q))
	for i, x_leg in enumerate(legendre_nodes):
		for j in range(q):
			transformation_matrix[i, j] = lagrange_basis(x_leg, j, cheb_nodes)
	
	return transformation_matrix

def chebfleg_matrix(p, q):
	"""
	Constructs a transformation matrix that converts a vector tabulated
	on p Legendre nodes to one tabulated on q Chebyshev nodes.
	
	This inverse mapping is defined as the left pseudoinverse of the 
	transformation matrix from Chebyshev to Legendre nodes.
	
	Parameters:
		p : int
			The number of nodes for Legendre.
		q : int
			The number of nodes for Chebyshev.
	
	Returns:
		numpy.ndarray: Transformation matrix of shape (q, p).
			When applied to data at Legendre nodes, it yields an approximation
			of the function values at Chebyshev nodes.
	"""
	leg_nodes = leggauss(p)[0]
	cheb_nodes = cheb(q)[0]
	
	T = np.zeros((q, p))
	for i, x_val in enumerate(cheb_nodes):
		for j in range(p):
			T[i, j] = lagrange_basis(x_val, j, leg_nodes)
	
	return T

def legfcheb_matrix_2d(p, q):
	"""
	Constructs a 2D transformation matrix to convert a vector tabulated
	on a Chebyshev grid (q x q) to one tabulated on a Legendre grid (p x p).

	The resulting matrix has shape (p*p, q*q) and is built as the Kronecker 
	product of the 1D transformation matrices.

	Parameters:
		p : int
			The number of Legendre nodes in each dimension.
		q : int
			The number of Chebyshev nodes in each dimension.

	Returns:
		numpy.ndarray: The 2D transformation matrix.
	"""
	T1 = legfcheb_matrix(p, q)
	T2 = legfcheb_matrix(p, q)
	return np.kron(T2, T1)

def chebfleg_matrix_2d(p, q):
	"""
	Constructs a 2D transformation matrix to convert a vector tabulated
	on a Legendre grid (p x p) to one tabulated on a Chebyshev grid (q x q).

	The resulting matrix has shape (q*q, p*p) and is built as the Kronecker 
	product of the 1D transformation matrices.

	Parameters:
		p : int
			The number of Legendre nodes in each dimension.
		q : int
			The number of Chebyshev nodes in each dimension.

	Returns:
		numpy.ndarray: The 2D transformation matrix.
	"""
	T1 = chebfleg_matrix(p, q)
	T2 = chebfleg_matrix(p, q)
	return np.kron(T2, T1)

#################################### Cheb utils for 2d and 3d ##########################################

def cheb_2d(a, p):
	xvec,D = cheb(p)
	I      = np.eye(p)
	ainv   = 1/a

	D1     = np.kron(ainv[0] * D, I)
	D2     = np.kron(I, ainv[1] * D)

	Dsq    = D @ D
	D11    = np.kron(ainv[0]**2 * Dsq, I)
	D22    = np.kron(I, ainv[1]**2 * Dsq)
	D12    = np.kron(ainv[0] * D, ainv[1] * D)
	
	# Create grid using meshgrid for consistency:
	X, Y = np.meshgrid(a[0] * xvec, a[1] * xvec,indexing='ij')
	zz = np.vstack((X.flatten(), Y.flatten()))
	
	Ds = Ds_2d(D1=D1, D2=D2, D11=D11, D22=D22, D12=D12)
	return zz, Ds


def cheb_3d(a,p):
	xvec,D = cheb(p)
	I      = np.eye(p)
	ainv   = 1/a

	D1  = np.kron(ainv[0] * D, np.kron(I,I))
	D2  = np.kron(I,np.kron(ainv[1] * D,I))
	D3  = np.kron(I,np.kron(I,ainv[2] * D))
	Dsq = D @ D
	D11 = np.kron(ainv[0]**2 * Dsq,np.kron(I,I))
	D22 = np.kron(I,np.kron(ainv[1]**2 * Dsq,I))
	D33 = np.kron(I,np.kron(I,ainv[2]**2 * Dsq))
	D12 = np.kron(ainv[0] * D,np.kron(ainv[1] * D,I))
	D13 = np.kron(ainv[0] * D,np.kron(I,ainv[2] * D))
	D23 = np.kron(I,np.kron(ainv[1] * D,ainv[2] * D))

	X, Y, Z = np.meshgrid(a[0] * xvec, a[1] * xvec, a[2] * xvec, indexing='ij')
	zz      = np.vstack((X.flatten(),Y.flatten(),Z.flatten()))
	Ds      = Ds_3d(D1= D1, D2= D2, D3= D3, D11= D11, D22= D22, D33= D33,
		 D12= D12, D13= D13, D23= D23)
	return zz, Ds