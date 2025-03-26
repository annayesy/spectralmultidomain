import numpy as np
from collections         import namedtuple
from scipy.linalg        import block_diag,lu_factor,lu_solve
from hps.cheb_utils      import *
from scipy.linalg        import null_space

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.

JJ_2d    = namedtuple('JJ_2d',    ['Jl','Jr','Jd','Ju','Ji'])
JJext_2d = namedtuple('JJext_2d', ['Jl','Jr','Jd','Ju'])
JJ_3d    = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf','Ji'])
JJext_3d = namedtuple('JJ_3d',    ['Jl','Jr','Jd','Ju','Jb','Jf'])

#################################### Discretization utils for 2d and 3d ##########################################

def leaf_discretization_2d(a,q):
	zz,Ds = cheb_2d(a,q)
	hmin  = zz[1,1] - zz[0,1]

	Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
	Jl    = np.argwhere(zz[0,:] < - a + 0.5 * hmin).reshape(q,)
	Jr    = np.argwhere(zz[0,:] > + a - 0.5 * hmin).reshape(q,)
	Jd    = np.argwhere(zz[1,:] < - a + 0.5 * hmin).reshape(q,)
	Ju    = np.argwhere(zz[1,:] > + a - 0.5 * hmin).reshape(q,)

	Ji    = np.argwhere(np.logical_and(Jc0,Jc1)).reshape((q-2)**2,)
	
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

def leaf_discretization_3d(a,q):
	zz,Ds = cheb_3d(a,q)
	hmin  = zz[2,1] - zz[2,0]

	Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
	Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
	Jc2   = np.abs(zz[2,:]) < a - 0.5*hmin
	Jl    = np.argwhere(zz[0,:] < - a + 0.5 * hmin).reshape(q**2,)
	Jr    = np.argwhere(zz[0,:] > + a - 0.5 * hmin).reshape(q**2,)
	Jd    = np.argwhere(zz[1,:] < - a + 0.5 * hmin).reshape(q**2,)
	Ju    = np.argwhere(zz[1,:] > + a - 0.5 * hmin).reshape(q**2,)
	Jb    = np.argwhere(zz[2,:] < - a + 0.5 * hmin).reshape(q**2,)
	Jf    = np.argwhere(zz[2,:] > + a - 0.5 * hmin).reshape(q**2,)

	Ji    = np.argwhere(np.logical_and(Jc0,np.logical_and(Jc1,Jc2))).reshape((q-2)**3,)

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
	def __init__(self,a,p,ndim=2, q=-1):
		"""
		Initializes utilities associated with an HPS patch.
		
		Parameters:
		- a: Half the size of the computational domain
		- p: The polynomial degree for Chebyshev discretization
		"""

		self.a = a; self.p = p; self.ndim = ndim
		self.q = p + 1 if (q <= 0) else q
		assert self.q > self.p
		self.ndim = ndim
		self._discretize()
		
	def _discretize(self):
		if (self.ndim == 2):
			zz_int,self.Ds,self.JJ_int,self.hmin = leaf_discretization_2d(self.a,self.q)
			zz_ext,self.JJ_ext                   = ext_discretization_2d (self.a,self.p)

		elif (self.ndim == 3):
			zz_tmp,self.Ds,self.JJ,self.hmin     = leaf_discretization_3d(self.a,self.q)
			raise ValueError("param map in progress for 3d")

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

		assert self.ndim == 2
		T = legfcheb_matrix(self.p,self.q)
		return block_diag(T,T,T,T)

	# Input:  vector of values collocated on exterior legendre  points
	# Output: vector of values collocated on exterior chebyshev points
	# The points are in order [l,r,d,u] in 2D and [l,r,d,u,b,f] in 3D
	@property
	def chebfleg_exterior_mat(self):

		assert self.ndim == 2
		p = self.p; q = self.q
		Jx_stack = np.hstack((self.JJ_int.Jl, self.JJ_int.Jr,\
			self.JJ_int.Jd, self.JJ_int.Ju))
		zz_tmp   = self.zz_int[Jx_stack]

		T = chebfleg_matrix(self.p,self.q)
		chebfleg_exterior_mat = block_diag(T,T,T,T)

		constraints = np.zeros((4,4*q))
		constraints[0,0]          = 1  
		constraints[0,2*q]        = -1 # bottom left corner
		assert np.linalg.norm(zz_tmp[0] - zz_tmp[2*q]) < 1e-15

		constraints[1,3*q-1]  = 1  
		constraints[1,q]      = -1 # bottom right corner
		assert np.linalg.norm(zz_tmp[3*q-1] - zz_tmp[q]) < 1e-15

		constraints[2,q-1]    = 1  
		constraints[2,3*q]    = -1 # upper left corner
		assert np.linalg.norm(zz_tmp[q-1] - zz_tmp[3*q]) < 1e-15

		constraints[3,2*q-1]  = 1  
		constraints[3,4*q-1]  = -1 # upper right corner
		assert np.linalg.norm(zz_tmp[2*q-1] - zz_tmp[4*q-1]) < 1e-15

		N = null_space(constraints)
		return N @ N.T @ chebfleg_exterior_mat