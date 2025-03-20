import numpy as np
from collections         import namedtuple
from scipy.linalg        import block_diag,lu_factor,lu_solve
from hps.cheb_utils      import *
from hps.compatible_proj import project_chebyshev_square
from scipy.linalg        import null_space

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.

JJ_2d    = namedtuple('JJ_2d',    ['Jl','Jr','Jd','Ju','Ji'])
JJext_2d = namedtuple('JJext_2d', ['Jl','Jr','Jd','Ju'])
JJ_3d    = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf','Ji'])
JJext_3d = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf'])

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

	leg_nodes,_    = leggauss(p);
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
	Jl    = np.argwhere(zz[0,:] < - a + 0.5 * hmin).reshape(p**2,)
	Jr    = np.argwhere(zz[0,:] > + a - 0.5 * hmin).reshape(p**2,)
	Jd    = np.argwhere(zz[1,:] < - a + 0.5 * hmin).reshape(p**2,)
	Ju    = np.argwhere(zz[1,:] > + a - 0.5 * hmin).reshape(p**2,)
	Jb    = np.argwhere(zz[2,:] < - a + 0.5 * hmin).reshape(p**2,)
	Jf    = np.argwhere(zz[2,:] > + a - 0.5 * hmin).reshape(p**2,)

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
			zz_tmp,self.Ds,self.JJ,self.hmin     = leaf_discretization_3d(a,p)
			raise ValueError("param map in progress for 3d")

		else:
			raise ValueError

		self.zz_int   = zz_int.T
		self.zz_ext   = zz_ext.T
		self.Nx_stack = get_diff_ops(self.Ds,self.JJ_int,d)


	# Input:  vector of values collocated on exterior chebyshev points
	# Output: vector of values collocated on exterior legendre  points
	# The points are in order [l,r,d,u] in 2D and [l,r,d,u,b,f] in 3D
	@property
	def legfcheb_exterior_mat(self):

		assert self.ndim == 2
		T = legfcheb_matrix(self.a,self.p)
		return block_diag(T,T,T,T)

	# Input:  vector of values collocated on exterior legendre  points
	# Output: vector of values collocated on exterior chebyshev points
	# The points are in order [l,r,d,u] in 2D and [l,r,d,u,b,f] in 3D
	@property
	def chebfleg_exterior_mat(self):

		assert self.ndim == 2
		p = self.p
		Jx_stack = np.hstack((self.JJ_int.Jl, self.JJ_int.Jr,\
			self.JJ_int.Jd, self.JJ_int.Ju))
		zz_tmp   = self.zz_int[Jx_stack]

		LU_mat = lu_factor(self.legfcheb_exterior_mat)
		chebfleg_exterior_mat = lu_solve(LU_mat,np.eye(self.zz_ext.shape[0]))

		constraints = np.zeros((4,self.zz_ext.shape[0]))
		constraints[0,0]      = 1  
		constraints[0,2*p]    = -1 # bottom left corner
		assert np.linalg.norm(zz_tmp[0] - zz_tmp[2*p]) < 1e-15

		constraints[1,3*p-1]  = 1  
		constraints[1,p]      = -1 # bottom right corner
		assert np.linalg.norm(zz_tmp[3*p-1] - zz_tmp[p]) < 1e-15

		constraints[2,p-1]    = 1  
		constraints[2,3*p]    = -1 # upper left corner
		assert np.linalg.norm(zz_tmp[p-1] - zz_tmp[3*p]) < 1e-15

		constraints[3,2*p-1]  = 1  
		constraints[3,4*p-1]  = -1 # upper right corner
		assert np.linalg.norm(zz_tmp[2*p-1] - zz_tmp[4*p-1]) < 1e-15

		N = null_space(constraints)
		return N @ N.T @ chebfleg_exterior_mat