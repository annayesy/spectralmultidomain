import numpy as np

import numpy as np
from collections import namedtuple
import scipy
from time import time
from numpy.polynomial.legendre import leggauss
import scipy.linalg

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Ds_2d      = namedtuple('Ds_2d',    ['D11','D22','D12','D1','D2'])
JJ_2d      = namedtuple('JJ_2d',    ['Jl','Jr','Jd','Ju','Ji'])
JJext_2d   = namedtuple('JJext_2d', ['Jl','Jr','Jd','Ju'])

Ds_3d    = namedtuple('Ds_3d', ['D11','D22','D33','D12','D13','D23','D1','D2','D3'])
JJ_3d    = namedtuple('JJ_3d', ['Jl','Jr','Jd','Ju','Jb','Jf','Ji'])

def cheb(p):
	"""
	Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
	
	Parameters:
	- p: The polynomial degree
	
	Returns:
	- D: The Chebyshev differentiation matrix
	- x: The Chebyshev points
	"""
	x = np.cos(np.pi * np.arange(p) / (p-1))
	c = np.concatenate((np.array([2]), np.ones(p-2), np.array([2])))
	c = np.multiply(c,np.power(np.ones(p) * -1, np.arange(p)))
	X = x.repeat(p).reshape((-1,p))
	dX = X - X.T
	# create the off diagonal entries of D
	D = np.divide(np.outer(c,np.divide(np.ones(p),c)), dX + np.eye(p))
	D = D - np.diag(np.sum(D,axis=1))
	return D,np.flip(x)

def chebyshev_to_legendre_matrix(a,p):
    """
    Constructs a transformation matrix to convert a vector tabulated
    on Chebyshev nodes to one tabulated on Legendre nodes.
    
    Parameters:
    p : int
        The number of nodes for Chebyshev and Legendre.
    
    Returns:
    numpy.ndarray
        Transformation matrix of shape (p, p).
    """
    cheb_nodes     = a * cheb(p)[1]
    legendre_nodes = a * leggauss(p)[0]

    # Construct Lagrange basis function
    def lagrange_basis(x, k, nodes):
        l_k = np.prod([(x - nodes[j]) / (nodes[k] - nodes[j]) for j in range(len(nodes)) if j != k], axis=0)
        return l_k

    # Populate the transformation matrix
    transformation_matrix = np.zeros((p, p))
    for i, x_leg in enumerate(legendre_nodes):  # Loop over Legendre nodes
        for j in range(p):  # Loop over Chebyshev nodes
            transformation_matrix[i, j] = lagrange_basis(x_leg, j, cheb_nodes)
    
    return transformation_matrix


#################################### Cheb utils for 2d and 3d ##########################################

def cheb_2d(a,p):
	D,xvec = cheb(p)

	xvec = a * xvec
	D = (1/a) * D
	I = np.eye(p)
	D1 = -np.kron(D,I)
	D2 = -np.kron(I,D)
	Dsq = D @ D
	D11 = np.kron(Dsq,I)
	D22 = np.kron(I,Dsq)
	D12 = np.kron(D,D)

	zz1 = np.repeat(xvec,p)
	zz2 = np.repeat(xvec,p).reshape(-1,p).T.flatten()
	zz = np.vstack((zz1,zz2))
	Ds = Ds_2d(D1= D1, D2= D2, D11= D11, D22= D22, D12= D12)
	return zz, Ds 


def cheb_3d(a,p):
	D,xvec = cheb(p)
	xvec = a * xvec
	D = (1/a) * D
	I = np.eye(p)
	D1 = -np.kron(D,np.kron(I,I))
	D2 = -np.kron(I,np.kron(D,I))
	D3 = -np.kron(I,np.kron(I,D))
	Dsq = D @ D
	D11 = np.kron(Dsq,np.kron(I,I))
	D22 = np.kron(I,np.kron(Dsq,I))
	D33 = np.kron(I,np.kron(I,Dsq))
	D12 = np.kron(D,np.kron(D,I))
	D13 = np.kron(D,np.kron(I,D))
	D23 = np.kron(I,np.kron(D,D))

	zz1 = np.repeat(xvec,p*p)
	zz2 = np.repeat(xvec,p*p).reshape(p*p,p).T.flatten()
	zz3 = np.repeat(xvec,p*p).reshape(p,p*p).T.flatten()
	zz = np.vstack((zz1,zz2,zz3))
	Ds = Ds_3d(D1= D1, D2= D2, D3= D3, D11= D11, D22= D22, D33= D33,
		 D12= D12, D13= D13, D23= D23)
	return zz, Ds

#################################### Discretization utils for 2d and 3d ##########################################

def leaf_discretization_2d(a,p):
	zz,Ds = cheb_2d(a,p)
	hmin  = zz[1,1] - zz[0,1]

	Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
	Jl    = np.argwhere(zz[0,:] < - a + 0.5 * hmin).reshape(p,)
	Jr    = np.argwhere(zz[0,:] > + a - 0.5 * hmin).reshape(p,)
	Jd    = np.argwhere(zz[1,:] < - a + 0.5 * hmin).reshape(p,)
	Ju    = np.argwhere(zz[1,:] > + a - 0.5 * hmin).reshape(p,)

	Ji    = np.argwhere(np.logical_and(Jc0,Jc1)).reshape((p-2)**2,)
	
	JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Ji = Ji)
	return zz,Ds,JJ,hmin

def ext_discretization_2d(a,p):

	leg_nodes,_ = leggauss(p);

	zz             = np.zeros((2,4*p))

	zz [0,0*p:1*p] = -a                # left
	zz [1,0*p:1*p] = a * leg_nodes     # left

	zz [0,1*p:2*p] = +a                # right
	zz [1,1*p:2*p] = a * leg_nodes     # right

	zz [0,2*p:3*p] = a * leg_nodes     # down
	zz [1,2*p:3*p] = -a                # down

	zz [0,3*p:4*p] = a * leg_nodes     # up
	zz [1,3*p:4*p] = +a                # up

	JJ = JJext_2d(Jl = np.arange(p), Jr = np.arange(p,2*p), \
		Jd = np.arange(2*p,3*p), Ju = np.arange(3*p,4*p))

	return zz,JJ

def leaf_discretization_3d(a,p):
	zz,Ds = cheb_3d(a,p)
	hmin  = zz[2,1] - zz[2,0]

	Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
	Jc2   = np.abs(zz[2,:]) < a - 0.5*hmin
	Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin)).reshape(p**2,)
	Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin)).reshape(p**2,)
	Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin)).reshape(p**2,)
	Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin)).reshape(p**2,)
	Jb    = np.argwhere(np.logical_and(zz[2,:] < - a + 0.5 * hmin)).reshape(p**2,)
	Jf    = np.argwhere(np.logical_and(zz[2,:] > + a - 0.5 * hmin)).reshape(p**2,)

	Ji    = np.argwhere(np.logical_and(Jc0,np.logical_and(Jc1,Jc2))).reshape((p-2)**3,)

	JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb, Jf=Jf, Ji=Ji)
	return zz,Ds,JJ,hmin


def get_diff_ops(Ds,JJ,d):
	if (d == 2):
		Nl = Ds.D1[JJ.Jl]
		Nr = Ds.D1[JJ.Jr]
		Nd = Ds.D2[JJ.Jd]
		Nu = Ds.D2[JJ.Ju]

		Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu))
	else:
		Nl = Ds.D1[JJ.Jl]
		Nr = Ds.D1[JJ.Jr]
		Nd = Ds.D2[JJ.Jd]
		Nu = Ds.D2[JJ.Ju]
		Nb = Ds.D3[JJ.Jb]
		Nf = Ds.D3[JJ.Jf]

		Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu,-Nb,+Nf))
	return Nx

class PatchUtils:
	def __init__(self,a,p,ndim=2):
		"""
		Initializes utilities associated with an HPS patch.
		
		Parameters:
		- a: Half the size of the computational domain
		- p: The polynomial degree for Chebyshev discretization
		"""

		self._discretize(a,p,d=ndim)
		self.a = a; self.p = p; self.ndim = ndim
		
	def _discretize(self,a,p,d):
		if (d == 2):
			zz_int,self.Ds,self.JJ_int,self.hmin = leaf_discretization_2d(a,p)
			zz_ext,self.JJ_ext                   = ext_discretization_2d (a,p)

		elif (d == 3):
			zz_tmp,self.Ds,self.JJ,self.hmin = leaf_discretization_3d(a,p)

		else:
			raise ValueError

		self.zz_int = zz_int.T
		self.zz_ext = zz_ext.T
		self.Nx     = get_diff_ops(self.Ds,self.JJ_int,d)