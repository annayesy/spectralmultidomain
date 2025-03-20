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

def legfcheb_matrix(a,p):
	"""
	Constructs a transformation matrix to convert a vector tabulated
	on Chebyshev nodes to one tabulated on Legendre nodes.
	
	Parameters:
	p : int
		The number of nodes for Chebyshev and Legendre.
	
	Returns:
	numpy.ndarray
		Transformation matrix of shape (p, p+1).
	"""
	cheb_nodes     = cheb(p+1)[0]
	legendre_nodes = leggauss(p)[0]

	# Construct Lagrange basis function
	def lagrange_basis(x, k, nodes):
		l_k = np.prod([(x - nodes[j]) / (nodes[k] - nodes[j]) for j in range(len(nodes)) if j != k], axis=0)
		return l_k

	# Populate the transformation matrix
	transformation_matrix = np.zeros((p, p+1))
	for i, x_leg in enumerate(legendre_nodes):  # Loop over Legendre nodes
		for j in range(p+1):  # Loop over Chebyshev nodes
			transformation_matrix[i, j] = lagrange_basis(x_leg, j, cheb_nodes)
	
	return transformation_matrix

def chebfleg_matrix(a, p):
    """
    Constructs a transformation matrix that converts a vector tabulated
    on p Legendre nodes to one tabulated on p+1 Chebyshev nodes.
    
    This inverse mapping is defined as the left pseudoinverse of the 
    transformation matrix from Chebyshev to Legendre nodes.
    
    Parameters:
        a : float
            Scaling factor for the nodes (nodes lie in [-a, a]).
        p : int
            Number of Legendre nodes. The Chebyshev nodes will be p+1.
    
    Returns:
        numpy.ndarray
            Transformation matrix of shape (p+1, p).
            When applied to data at Legendre nodes, it yields an approximation
            of the function values at Chebyshev nodes.
    """

    # Get p Legendre nodes on [-1,1] and scale by a
    leg_nodes = leggauss(p)[0]
    cheb_nodes = cheb(p+1)[0]
    
    # Initialize the transformation matrix
    T = np.zeros((p+1, p))
    
    # Populate the transformation matrix using Lagrange basis polynomials.
    # For each Chebyshev node x_val, evaluate the j-th Lagrange basis polynomial constructed
    # with the Legendre nodes.
    for i_idx, x_val in enumerate(cheb_nodes):
        for j in range(p):
            # Compute the j-th Lagrange basis polynomial L_j(x_val)
            L_j = 1.0
            for k in range(p):
                if k == j:
                    continue
                L_j *= (x_val - leg_nodes[k]) / (leg_nodes[j] - leg_nodes[k])
            T[i_idx, j] = L_j
    return T

#################################### Cheb utils for 2d and 3d ##########################################

def cheb_2d(a, p):
	xvec,D = cheb(p)
	xvec = a * xvec  # scale the nodes
	D = (1 / a) * D  # adjust the differentiation matrix accordingly
	I = np.eye(p)
	D1 = np.kron(D, I)
	D2 = np.kron(I, D)
	Dsq = D @ D
	D11 = np.kron(Dsq, I)
	D22 = np.kron(I, Dsq)
	D12 = np.kron(D, D)
	
	# Create grid using meshgrid for consistency:
	X, Y = np.meshgrid(xvec, xvec,indexing='ij')
	zz = np.vstack((X.flatten(), Y.flatten()))
	
	Ds = Ds_2d(D1=D1, D2=D2, D11=D11, D22=D22, D12=D12)
	return zz, Ds


def cheb_3d(a,p):
	xvec,D = cheb(p)
	xvec = a * xvec
	D = (1/a) * D
	I = np.eye(p)
	D1  = np.kron(D,np.kron(I,I))
	D2  = np.kron(I,np.kron(D,I))
	D3  = np.kron(I,np.kron(I,D))
	Dsq = D @ D
	D11 = np.kron(Dsq,np.kron(I,I))
	D22 = np.kron(I,np.kron(Dsq,I))
	D33 = np.kron(I,np.kron(I,Dsq))
	D12 = np.kron(D,np.kron(D,I))
	D13 = np.kron(D,np.kron(I,D))
	D23 = np.kron(I,np.kron(D,D))

	X, Y, Z = np.meshgrid(xvec, xvec, xvec, indexing='ij')
	zz      = np.vstack((X.flatten(),Y.flatten(),Z.flatten()))
	Ds      = Ds_3d(D1= D1, D2= D2, D3= D3, D11= D11, D22= D22, D33= D33,
		 D12= D12, D13= D13, D23= D23)
	return zz, Ds