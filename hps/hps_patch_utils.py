import numpy as np

import numpy as np
from collections import namedtuple
import scipy
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx','Jc'])

Ds_3d    = namedtuple('Ds_3d', ['D11','D22','D33','D12','D13','D23','D1','D2','D3'])
JJ_3d    = namedtuple('JJ_3d', ['Jl','Jr','Jd','Ju','Jb','Jf','Jx','Jc'])

def cheb(p):
	"""
	Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
	
	Parameters:
	- p: The polynomial degree
	
	Returns:
	- D: The Chebyshev differentiation matrix
	- x: The Chebyshev points
	"""
	x = np.cos(np.pi * np.arange(p+1) / p)
	c = np.concatenate((np.array([2]), np.ones(p-1), np.array([2])))
	c = np.multiply(c,np.power(np.ones(p+1) * -1, np.arange(p+1)))
	X = x.repeat(p+1).reshape((-1,p+1))
	dX = X - X.T
	# create the off diagonal entries of D
	D = np.divide(np.outer(c,np.divide(np.ones(p+1),c)), dX + np.eye(p+1))
	D = D - np.diag(np.sum(D,axis=1))
	return D,x


#################################### Cheb utils for 2d and 3d ##########################################

def cheb_2d(a,p):
	D,xvec = cheb(p-1)
	xvec = a * np.flip(xvec)
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
	D,xvec = cheb(p-1)
	xvec = a * np.flip(xvec)
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
	Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,Jc1))
	Jl    = Jl.copy().reshape(p-2,)
	Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,Jc1))
	Jr    = Jr.copy().reshape(p-2,)
	Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,Jc0))
	Jd    = Jd.copy().reshape(p-2,)
	Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,Jc0))
	Ju    = Ju.copy().reshape(p-2,)
	Jc    = np.argwhere(np.logical_and(Jc0,Jc1))
	Jc    = Jc.copy().reshape((p-2)**2,)
	Jx    = np.concatenate((Jl,Jr,Jd,Ju))
	
	JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, 
			 Jx= Jx, Jc= Jc)
	return zz,Ds,JJ,hmin


def leaf_discretization_3d(a,p):
	zz,Ds = cheb_3d(a,p)
	hmin  = zz[2,1] - zz[2,0]

	Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
	Jc2   = np.abs(zz[2,:]) < a - 0.5*hmin
	Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,
									   np.logical_and(Jc1,Jc2)))
	Jl    = Jl.copy().reshape((p-2)**2,)
	Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,
									   np.logical_and(Jc1,Jc2)))
	Jr    = Jr.copy().reshape((p-2)**2,)
	Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,
									   np.logical_and(Jc0,Jc2)))
	Jd    = Jd.copy().reshape((p-2)**2,)
	Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,
									   np.logical_and(Jc0,Jc2)))
	Ju    = Ju.copy().reshape((p-2)**2,)
	Jb    = np.argwhere(np.logical_and(zz[2,:] < - a + 0.5 * hmin,
									   np.logical_and(Jc0,Jc1)))
	Jb    = Jb.copy().reshape((p-2)**2,)
	Jf    = np.argwhere(np.logical_and(zz[2,:] > + a - 0.5 * hmin,
									   np.logical_and(Jc0,Jc1)))
	Jf    = Jf.copy().reshape((p-2)**2,)

	Jc    = np.argwhere(np.logical_and(Jc0,
									   np.logical_and(Jc1,Jc2)))
	Jc    = Jc.copy().reshape((p-2)**3,)
	Jx    = np.concatenate((Jl,Jr,Jd,Ju,Jb,Jf))

	JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb,
			 Jf=Jf, Jx=Jx, Jc=Jc)
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
			zz_tmp,self.Ds,self.JJ,self.hmin = leaf_discretization_2d(a,p)
		elif (d == 3):
			zz_tmp,self.Ds,self.JJ,self.hmin = leaf_discretization_3d(a,p)

		else:
			raise ValueError

		self.zz = zz_tmp.T
		self.Nx = get_diff_ops(self.Ds,self.JJ,d)