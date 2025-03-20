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
	def Ji(self):
		# index vector for interior points
		pass

	@abstractproperty
	def Jx(self):
		# index vector for exterior points
		pass

	@abstractproperty
	def npoints_dim(self):
		pass

	#################################################

	@abstractproperty
	def Aii(self):
		pass

	@abstractproperty
	def Aix(self):
		pass

	@abstractproperty
	def Axx(self):
		pass

	@abstractproperty
	def Axi(self):
		pass

	def setup_solver_Aii(self,solve_op=None,use_approx=False):
		if (solve_op is None):
			self.solve_op = SparseSolver(self.Aii,use_approx=use_approx).solve_op
		else:
			self.solve_op = solve_op

	@property
	def solver_Aii(self):

		if (not hasattr(self,'solve_op')):
			self.setup_solver_Aii()
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

		result = - self.solver_Aii ( self.Aix @ uu_tmp)

		if (uu_dir.ndim == 1):
			result = result.flatten()
		return result

	def verify_discretization(self,kh):

		XX      = self.geom.parameter_map(self.XX) if hasattr(self.geom,'parameter_map') else self.XX

		uu      = get_known_greens(XX,kh,center = self.geom.bounds[1]+10)
		uu_sol  = self.solve_dir(uu[self.Jx])
		uu_true = uu[self.Ji]

		err     = np.linalg.norm(uu_sol - uu_true,ord=2)
		relerr  = err / np.linalg.norm(uu_true,ord=2)
		return relerr
