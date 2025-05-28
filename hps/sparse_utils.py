import numpy as np
from   scipy.sparse.linalg import LinearOperator, splu
from   scipy.sparse        import csr_matrix, coo_matrix
from   time import time
try:
	from petsc4py import PETSc
	petsc_imported = True
	print("PETSC IMPORTED")
except:
	print("PETSC NOT IMPORTED")	
	petsc_imported = False

def petscdense_to_nparray(pM):
	M = pM.getValues(range(0,pM.getSize()[0]), \
		range(0,pM.getSize()[1]))
	return M

def setup_ksp(A,use_approx=False):

	A = A.tocsr()

	ksp = PETSc.KSP().create()
	pA  = PETSc.Mat().createAIJ(A.shape, csr=(A.indptr,A.indices,A.data))
	ksp.setOperators(pA)

	if (not use_approx):

		ksp.setType('preonly')
		ksp.getPC().setType('lu')
		ksp.getPC().setFactorSolverType('mumps')
	else:
		ksp.setType('gmres')
		ksp.getPC().setType('hypre')
		ksp.setTolerances(rtol=5e-14)
		ksp.setConvergenceHistory()
	ksp.setUp()
	return ksp

def get_vecsolve(ksp):

	def vecsolve(b):

		pb = PETSc.Vec().createWithArray(b)
		px = PETSc.Vec().createWithArray(np.zeros(b.shape))
		ksp.solve(pb,px)
		
		result = px.getArray()
		px.destroy(); pb.destroy()
		return result

	return vecsolve

def get_matsolve(ksp,use_approx):

	def matsolve(B):
		pB = PETSc.Mat().createDense([B.shape[0],B.shape[1]],None,B)
		pX = PETSc.Mat().createDense([B.shape[0],B.shape[1]])
		pX.zeroEntries()

		ksp.matSolve(pB,pX)
		result = petscdense_to_nparray(pX)
		pX.destroy(); pB.destroy()
		return result

	def seq_vec_solve(B):
		vec_solve = get_vecsolve(ksp)
		res = np.zeros(B.shape)
		for j in range(B.shape[-1]):
			res[:,j] = vec_solve(B[:,j])
		return res

	return matsolve if not use_approx else seq_vec_solve

######################################################################################################

class SparseSolver:

	def __init__(self,A,use_approx=False):

		v                 = np.random.rand(A.shape[0],)
		self.is_symmetric = np.linalg.norm(A @ v - A.T @ v) < 1e-12
		self.N            = A.shape[0]

		self.use_petsc    = petsc_imported
		self.use_approx   = use_approx

		if (self.use_petsc):

			self.ksp = setup_ksp(A,use_approx)
			if (not self.is_symmetric):
				# on some installations of petsc
				# there are issues with transpose matsolve
				self.ksp_adj = setup_ksp(A.T,use_approx)
		else:
			self.ksp = splu(A.tocsc())

		rhs  = A @ v
		tic  = time()
		res  = self.solve_op.matvec(rhs)
		toc  = time() - tic

		if (self.use_petsc):
			niter= self.ksp.getIterationNumber()

			#print("\t use_approx = %s, time to solve %5.2e with relerr %5.2e in niter=%d" % \
			#	(use_approx,toc, np.linalg.norm(res-v),niter))

	@property
	def solve_op(self):

		if (self.use_petsc and self.is_symmetric):

			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp),\
				matmat =get_matsolve(self.ksp,self.use_approx),\
				rmatmat=get_matsolve(self.ksp,self.use_approx))
			
		elif (self.use_petsc and not self.is_symmetric):

			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp_adj),\
				matmat =get_matsolve(self.ksp,self.use_approx),\
				rmatmat=get_matsolve(self.ksp_adj,self.use_approx))

		elif (not self.use_petsc and self.is_symmetric):
			return LinearOperator(shape=(self.N,self.N),\
				matvec =lambda x: self.ksp.solve(x),\
				rmatvec=lambda x: self.ksp.solve(x),\
				matmat =lambda x: self.ksp.solve(x),\
				rmatmat=lambda x: self.ksp.solve(x))

		else:
			return LinearOperator(shape=(self.N,self.N),\
				matvec =lambda x: self.ksp.solve(x),\
				rmatvec=lambda x: self.ksp.solve(x,trans='T'),\
				matmat =lambda x: self.ksp.solve(x),\
				rmatmat=lambda x: self.ksp.solve(x,trans='T'))

class CSRBuilder:

	def __init__(self,M,N,nnz):

		self.M    = M
		self.N    = N

		self.row  = np.zeros(nnz,dtype=int)
		self.col  = np.zeros(nnz,dtype=int)
		self.data = np.zeros(nnz)
		self.acc  = 0

	def add_data(self,mat):

		mat   = mat.tocoo()
		ndata = mat.row.shape[0]
		assert self.acc + ndata < self.data.shape[0]

		self.row [self.acc : self.acc + ndata] = mat.row
		self.col [self.acc : self.acc + ndata] = mat.col
		self.data[self.acc : self.acc + ndata] = mat.data

		self.acc += ndata

	def tocsr(self):
		return coo_matrix((self.data[:self.acc],\
			(self.row[:self.acc],self.col[:self.acc]))).tocsr()
