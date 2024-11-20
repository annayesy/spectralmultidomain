import numpy as np
from   abc              import ABCMeta, abstractmethod, abstractproperty
from   hps.pdo          import get_known_greens
from   hps.sparse_utils import SparseSolver

class AbstractPDESolver(metaclass=ABCMeta):

	#################################################

	@abstractproperty
	def geom(self):
		pass

	@property
	def ndim(self):
		return self.geom.bounds.shape[-1]

	#################################################

	@abstractproperty
	def XX(self):
		pass

	@abstractproperty
	def p(self):
		pass

	@abstractproperty
	def I_C(self):
		pass

	@abstractproperty
	def I_X(self):
		pass

	@abstractproperty
	def npoints_dim(self):
		pass

	#################################################

	@abstractproperty
	def A_CC(self):
		pass

	@abstractproperty
	def A_CX(self):
		pass

	def setup_solver_CC(self,solve_op=None):
		if (solve_op is None):
			self.solve_op = SparseSolver(self.A_CC).solve_op
		else:
			self.solve_op = solve_op

	@property
	def solver_CC(self):

		if (not hasattr(self,'solve_op')):
			self.setup_solver_CC()
		return self.solve_op

	#################################################

	def get_nwaves_dim(self,kh):
		nwaves_unit = kh / (2*np.pi)
		nunits_dim  = self.geom.bounds[1] - self.geom.bounds[0]
		nwaves_dim  = nwaves_unit * nunits_dim

		return nwaves_dim

	def get_ppw_dim(self,kh):
		ppw_ndim = self.npoints_dim / self.get_nwaves_dim(kh)
		return ppw_ndim

	def solve_dir(self,uu_dir):

		# body load non-zero functionality
		# will be added later

		if (uu_dir.ndim == 1):
			uu_tmp = uu_dir[:,np.newaxis]
		else:
			uu_tmp = uu_dir

		result = - self.solver_CC ( self.A_CX @ uu_tmp)

		if (uu_dir.ndim == 1):
			result = result.flatten()
		return result

	def verify_discretization(self,kh):

		uu      = get_known_greens(self.XX,kh,center = self.geom.bounds[1]+10)
		uu_sol  = self.solve_dir(uu[self.I_X])
		uu_true = uu[self.I_C]

		err     = np.linalg.norm(uu_sol - uu_true,ord=2)
		relerr  = err / np.linalg.norm(uu_true,ord=2)
		return relerr
