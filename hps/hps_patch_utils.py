import numpy as np
from collections            import namedtuple
from scipy.linalg           import block_diag,lu_factor,lu_solve
from hps.cheb_utils         import *
from scipy.linalg           import null_space
from scipy.spatial.distance import cdist
import scipy

import sys

from matplotlib import pyplot as plt

# Define named tuples for storing partial differential operators (PDOs) 
# and differential schemes (Ds) for both 2D and 3D problems, 
# along with indices (JJ) for domain decomposition.

JJ_2d    = namedtuple('JJ_2d',    ['Jl','Jr','Jd','Ju','Ji'])
JJext_2d = namedtuple('JJext_2d', ['Jl','Jr','Jd','Ju'])
JJ_3d    = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf','Ji'])
JJext_3d = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf'])

######################## Discretization utils for 2d and 3d ########################

def min_axis_aligned_gap(pts):
    """
    pts: an (N, d) array of point coordinates
    Returns:
      mins_per_dim: length-d array, where mins_per_dim[k]
                    is the smallest |x_i[k] - x_j[k]| over all i<j
      overall_min: the single smallest axis-aligned gap min_k mins_per_dim[k]
    """
    pts = np.asarray(pts)
    # sort each column (axis)
    sorted_by_dim = np.sort(pts, axis=0)        # shape (N,d)
    # compute differences between neighbors along each column
    diffs = np.diff(sorted_by_dim, axis=0)      # shape (N-1, d)
    # the minimum gap in each dimension
    mins_per_dim = diffs.min(axis=0)            # shape (d,)
    # the overall smallest axis-aligned gap
    overall_min = mins_per_dim.min()
    return overall_min

def leaf_discretization_2d(a,q):
	zz,Ds = cheb_2d(a,q)

	hmin  = min( np.max(np.abs(zz[:,1]-zz[:,0])), \
		np.max(np.abs(zz[:,q]-zz[:,0])) ) 

	Jc0   = np.abs(zz[0,:]) < a[0] - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a[1] - 0.5*hmin
	Jl    = np.argwhere(zz[0,:] < - a[0] + 0.5 * hmin).reshape(q,)
	Jr    = np.argwhere(zz[0,:] > + a[0] - 0.5 * hmin).reshape(q,)
	Jd    = np.argwhere(zz[1,:] < - a[1] + 0.5 * hmin).reshape(q,)
	Ju    = np.argwhere(zz[1,:] > + a[1] - 0.5 * hmin).reshape(q,)

	Ji    = np.argwhere(np.logical_and(Jc0,Jc1)).reshape((q-2)**2,)
	
	JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Ji = Ji)

	return zz,Ds,JJ,hmin

def ext_discretization_2d(a,p):

	leg_nodes,_    = leggauss(p);
	zz             = np.zeros((2,4*p))

	zz [0,0*p:1*p] = -a[0]                # left
	zz [1,0*p:1*p] = a[1] * leg_nodes     # left

	zz [0,1*p:2*p] = +a[0]                # right
	zz [1,1*p:2*p] = a[1] * leg_nodes     # right

	zz [0,2*p:3*p] = a[0] * leg_nodes     # down
	zz [1,2*p:3*p] = -a[1]                # down

	zz [0,3*p:4*p] = a[0] * leg_nodes     # up
	zz [1,3*p:4*p] = +a[1]                # up

	JJ = JJext_2d(Jl = np.arange(p), Jr = np.arange(p,2*p), \
		Jd = np.arange(2*p,3*p), Ju = np.arange(3*p,4*p))
	return zz,JJ

def leaf_discretization_3d(a,q):
	zz,Ds = cheb_3d(a,q)
	hmin  = zz[2,1] - zz[2,0]

	Jc0   = np.abs(zz[0,:]) < a[0] - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a[1] - 0.5*hmin
	Jc2   = np.abs(zz[2,:]) < a[2] - 0.5*hmin
	Jl    = np.argwhere(zz[0,:] < - a[0] + 0.5 * hmin).reshape(q**2,)
	Jr    = np.argwhere(zz[0,:] > + a[0] - 0.5 * hmin).reshape(q**2,)
	Jd    = np.argwhere(zz[1,:] < - a[1] + 0.5 * hmin).reshape(q**2,)
	Ju    = np.argwhere(zz[1,:] > + a[1] - 0.5 * hmin).reshape(q**2,)
	Jb    = np.argwhere(zz[2,:] < - a[2] + 0.5 * hmin).reshape(q**2,)
	Jf    = np.argwhere(zz[2,:] > + a[2] - 0.5 * hmin).reshape(q**2,)

	Ji    = np.argwhere(np.logical_and(Jc0,np.logical_and(Jc1,Jc2))).reshape((q-2)**3,)

	JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb, Jf=Jf, Ji=Ji)
	return zz,Ds,JJ,hmin

def ext_discretization_3d(a,p):

	leg_nodes,_    = leggauss(p);
	face_size      = p**2
	zz             = np.zeros((3,6*face_size))

	JJ = JJext_3d(Jl = np.arange(face_size), Jr = np.arange(face_size,2*face_size), \
		Jd = np.arange(2*face_size,3*face_size), Ju = np.arange(3*face_size,4*face_size),\
		Jb = np.arange(4*face_size,5*face_size), Jf = np.arange(5*face_size,6*face_size))

	Xtmp,Ytmp      = np.meshgrid(a[1] * leg_nodes, a[2] * leg_nodes, indexing='ij')
	zz_grid        = np.vstack((Xtmp.flatten(),Ytmp.flatten()))

	zz [0,JJ.Jl] = -a[0]
	zz [1,JJ.Jl] = zz_grid[0]
	zz [2,JJ.Jl] = zz_grid[1]

	zz [0,JJ.Jr] = +a[0]
	zz [1,JJ.Jr] = zz_grid[0]
	zz [2,JJ.Jr] = zz_grid[1]

	Xtmp,Ytmp      = np.meshgrid(a[0] * leg_nodes, a[2] * leg_nodes, indexing='ij')
	zz_grid        = np.vstack((Xtmp.flatten(),Ytmp.flatten()))

	zz [0,JJ.Jd] = zz_grid[0]
	zz [1,JJ.Jd] = -a[1]
	zz [2,JJ.Jd] = zz_grid[1]

	zz [0,JJ.Ju] = zz_grid[0]
	zz [1,JJ.Ju] = +a[1]
	zz [2,JJ.Ju] = zz_grid[1]

	Xtmp,Ytmp      = np.meshgrid(a[0] * leg_nodes, a[1] * leg_nodes, indexing='ij')
	zz_grid        = np.vstack((Xtmp.flatten(),Ytmp.flatten()))

	zz [0,JJ.Jb] = zz_grid[0]
	zz [1,JJ.Jb] = zz_grid[1]
	zz [2,JJ.Jb] = -a[2]

	zz [0,JJ.Jf] = zz_grid[0]
	zz [1,JJ.Jf] = zz_grid[1]
	zz [2,JJ.Jf] = +a[2]

	return zz,JJ

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
	def __init__(self,a,p,ndim=2, q=-1):
		"""
		Initializes utilities associated with an HPS patch.
		
		Parameters:
		- a: Half the size of the computational domain
		- p: The polynomial degree for Chebyshev discretization
		"""

		self.a = a; self.p = p; self.ndim = ndim
		self.q = p + 2 if (q <= 0) else q
		assert self.q > self.p
		self.ndim = ndim
		self._discretize()
		
	def _discretize(self):
		if (self.ndim == 2):
			zz_int,self.Ds,self.JJ_int,self.hmin = leaf_discretization_2d(self.a,self.q)
			zz_ext,self.JJ_ext                   = ext_discretization_2d (self.a,self.p)

		elif (self.ndim == 3):
			zz_int,self.Ds,self.JJ_int,self.hmin = leaf_discretization_3d(self.a,self.q)
			zz_ext,self.JJ_ext                   = ext_discretization_3d (self.a,self.p)

		else:
			raise ValueError

		self.zz_int   = zz_int.T
		self.zz_ext   = zz_ext.T
		self.Nx_stack = get_diff_ops(self.Ds,self.JJ_int,self.ndim)


	# Input:  vector of values collocated on exterior chebyshev points
	# Output: vector of values collocated on exterior legendre  points
	# The points are in order [l,r,d,u] in 2D and [l,r,d,u,b,f] in 3D
	@property
	def legfcheb_exterior_mat(self):

		if (self.ndim == 2):
			T = legfcheb_matrix(self.p,self.q)
			return block_diag(T,T,T,T)
		else:
			T = legfcheb_matrix_2d(self.p,self.q)
			return block_diag(T,T,T,T,T,T)

	def find_equality_constraints(self,ind_list):

		zz_int = self.zz_int; q = self.q

		if (self.ndim == 2):
			face_size   = q
			constraints = np.zeros((4,4*face_size))
		else:
			face_size   = q**2
			constraints = np.zeros((12,6*face_size))

		offset = 0
		for j in range (len(ind_list)):
			for k in range (j+1,len(ind_list)):

				D = cdist(zz_int[ind_list[j]],\
					zz_int[ind_list[k]])

				inds_zero = np.where(D == 0)

				if (inds_zero[0].shape[0] == 0):
					continue
				else:
					constraints[offset,j*face_size+inds_zero[0]] = +1
					constraints[offset,k*face_size+inds_zero[1]] = -1
					offset += 1
		assert offset == constraints.shape[0]

		return constraints


	# Input:  vector of values collocated on exterior legendre  points
	# Output: vector of values collocated on exterior chebyshev points
	# The points are in order [l,r,d,u] in 2D and [l,r,d,u,b,f] in 3D
	@property
	def chebfleg_exterior_mat(self):

		if (self.ndim == 2):
			T                = chebfleg_matrix(self.p,self.q)
			tmp_exterior_mat = block_diag(T,T,T,T)

			constraints = self.find_equality_constraints(
				[self.JJ_int.Jl, self.JJ_int.Jr,\
				self.JJ_int.Jd, self.JJ_int.Ju])
		else:
			T                = chebfleg_matrix_2d(self.p,self.q)
			tmp_exterior_mat = block_diag(T,T,T,T,T,T)

			constraints = self.find_equality_constraints(
				[self.JJ_int.Jl, self.JJ_int.Jr,\
				self.JJ_int.Jd, self.JJ_int.Ju,\
				self.JJ_int.Jb, self.JJ_int.Jf])

		N = null_space(constraints)
		return N @ N.T @ tmp_exterior_mat