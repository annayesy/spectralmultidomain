import jax
# enable 64-bit (double) precision globally
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functools
import numpy as np
import os

##########################################################################################################################
#####################################      NUMPY HELPERS              ####################################################

def query_total_memory():
	pages = os.sysconf('SC_PHYS_PAGES')
	page_size = os.sysconf('SC_PAGE_SIZE')
	return pages * page_size

def diag_mult(diag, M):
	return diag[..., None] * M

def build_pdo_terms(pdo, Ds, ndim, nt_cheb, device):
	"""
	Assemble everything having to do with pdo.*  into:
	  1) a single JAX array Ds_stack  of shape (K, m, m),
	  2) a JAX array consts       of shape (K,),
	  3) a Python list func_list  of length K, where each element is a function handle.

	Arguments:
	  pdo      : some object with attributes c11, c22, c12, c1, c2, c, c33, c13, c23, c3, ...
	  Ds       : an object whose attributes D11, D22, D12, D1, D2, I (eye), D33, D13, D23, D3, ... are already JAX arrays
				 of shape (m, m), sitting on 'device'.
	  ndim     : either 2 or 3
	  nt_cheb  : the size m of the Chebyshev‐local points
	  device   : a JAX device (so that we can .astype(..., device=device) if needed)

	Returns:
	  Ds_stack : jnp.ndarray of shape (K, m, m), dtype=float64, device=device
	  consts   : jnp.ndarray of shape (K,),   dtype=float64, device=device
	  func_list: Python list of length K, each entry is a callable that maps xxloc → a (batch, m, 1)-shaped array
	"""

	Ds_list    = []
	consts_list= []
	func_list  = []

	def helper_append(Dmat,func,const):
		Ds_list.append(jnp.array(Dmat,dtype=jnp.float64,device=device))
		func_list.append(func)
		consts_list.append(const)

	helper_append(Ds.D11,pdo.c11,-1.0)
	helper_append(Ds.D22,pdo.c22,-1.0)

	if pdo.c12 is not None:
		helper_append(Ds.D12,pdo.c12,-2.0)

	if pdo.c1 is not None:
		helper_append(Ds.D1,pdo.c1,+1.0)

	if pdo.c2 is not None:
		helper_append(Ds.D2,pdo.c2,+1.0)

	# c (scalar), I, +1
	if getattr(pdo, "c", None) is not None:
		I = jnp.eye(nt_cheb, dtype=jnp.float64, device=device)
		helper_append(I,pdo.c,+1.0)

	if ndim == 3:

		helper_append(Ds.D33,pdo.c33,-1.0)

		if pdo.c13 is not None:
			helper_append(Ds.D13,pdo.c13,-2.0)
		if pdo.c23 is not None:
			helper_append(Ds.D13,pdo.c23,-2.0)

		if pdo.c3 is not None:
			helper_append(Ds.D2,pdo.c3,+1.0)

	# Stack everything along axis=0 so we get shape (K, m, m).
	Ds_stack = jnp.stack(Ds_list, axis=0)
	consts   = jnp.array(consts_list, dtype=jnp.float64, device=device)

	return Ds_stack, consts, tuple(func_list)

##########################################################################################################################
#####################################      JIT-COMPILED FUNCTIONS     ####################################################

@functools.partial(jax.jit,
				   static_argnums=(-1,))
def _get_Aloc(xxloc, Ds_stack, consts, funcs):
	m = xxloc.shape[-2]

	if xxloc.ndim == 2:
		out_shape = (m, m)
	else:
		out_shape = (xxloc.shape[0], m, m)

	Aloc = jnp.zeros(out_shape, dtype=jnp.float64)
	K    = Ds_stack.shape[0]

	for i in range(K):
		c_fun = funcs[i]           # Python callable (static: not traced)
		D_mat = Ds_stack[i, :, :]  # jnp.ndarray (m, m)
		coeff = consts[i]          # jnp.scalar

		cvals = c_fun(xxloc)       # → (m,1)  or  (batch, m,1)
		Aloc = Aloc + coeff * diag_mult(cvals, D_mat)

	return Aloc

@functools.partial(jax.jit,
				   # argument indices 4 (c_fns), 5 (consts), 8 (ni_cheb) are static
				   static_argnums=(-2,-1))
def solve_dir_helper_with_tile(
	xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
	uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
	ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
	chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
	Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
	D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
	consts:       jnp.ndarray,  # (N_terms,)
	c_fns:        tuple,        # length N_terms tuple of Python callables
	ni_cheb:      int           # number of interior Chebyshev nodes
) -> jnp.ndarray:

	B = xxloc_bnd.shape[0]
	nrhs = uu_dir.shape[-1]

	Aloc_bnd = _get_Aloc(xxloc_bnd, D_stack, consts, c_fns)

	# 4) Partition into interior‐interior and interior‐boundary blocks:
	Aii = Aloc_bnd[:, :ni_cheb, :ni_cheb]   # (batch, ni_cheb, ni_cheb)
	Aib = Aloc_bnd[:, :ni_cheb, ni_cheb:]   # (batch, ni_cheb, nb_cheb)

	# 5) Compute the “equivalent Dirichlet data” on the boundary:
	equiv_dir = chebfleg_mat @ uu_dir       # (batch, nb_cheb, nrhs)

	# 6) Build RHS for interior solve: ff_body_interior − Aib @ equiv_dir
	rhs = ff_body[:, :ni_cheb, :] - (Aib @ equiv_dir)   # (batch, ni_cheb, nrhs)

	# 7) Solve interior block:
	sol_i = jnp.linalg.solve(Aii, rhs)   # (batch, ni_cheb, nrhs)

	# 8) Concatenate interior solution + boundary “equiv_dir” → (batch, nt_cheb, nrhs)
	return jnp.concatenate([sol_i, jnp.tile(equiv_dir,(B,1,1))], axis=1)

@functools.partial(jax.jit,
				   # argument indices 4 (c_fns), 5 (consts), 8 (ni_cheb) are static
				   static_argnums=(-2,-1))
def solve_dir_helper(
	xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
	uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
	ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
	chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
	Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
	D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
	consts:       jnp.ndarray,  # (N_terms,)
	c_fns:        tuple,        # length N_terms tuple of Python callables
	ni_cheb:      int           # number of interior Chebyshev nodes
) -> jnp.ndarray:

	B = xxloc_bnd.shape[0]
	nrhs = uu_dir.shape[-1]

	Aloc_bnd = _get_Aloc(xxloc_bnd, D_stack, consts, c_fns)

	# 4) Partition into interior‐interior and interior‐boundary blocks:
	Aii = Aloc_bnd[:, :ni_cheb, :ni_cheb]   # (batch, ni_cheb, ni_cheb)
	Aib = Aloc_bnd[:, :ni_cheb, ni_cheb:]   # (batch, ni_cheb, nb_cheb)

	# 5) Compute the “equivalent Dirichlet data” on the boundary:
	equiv_dir = chebfleg_mat @ uu_dir       # (batch, nb_cheb, nrhs)

	# 6) Build RHS for interior solve: ff_body_interior − Aib @ equiv_dir
	rhs = ff_body[:, :ni_cheb, :] - (Aib @ equiv_dir)   # (batch, ni_cheb, nrhs)

	# 7) Solve interior block:
	sol_i = jnp.linalg.solve(Aii, rhs)   # (batch, ni_cheb, nrhs)

	# 8) Concatenate interior solution + boundary “equiv_dir” → (batch, nt_cheb, nrhs)
	return jnp.concatenate([sol_i, equiv_dir],axis=1)

@functools.partial(jax.jit,
				   static_argnums=(-2,-1))
def compute_chunk_DtN(
	xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
	uu_dir:       jnp.ndarray,  # (1, nx_leg, nx_leg)
	ff_body:      jnp.ndarray,  # (1, nt_cheb, 1)
	legfcheb_mat: jnp.ndarray,  # (nx_leg, nb_cheb)
	chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
	Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
	D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
	consts:       jnp.ndarray,  # (N_terms,)	
	c_fns:        tuple,        # length N_terms
	ni_cheb:      int           # number of interior Chebyshev nodes
) -> jnp.ndarray:
	"""
	Compute a DtN‐block for each leaf in xx_chunk:
	  1) Solve with Dirichlet = eye_block, body = 0
	  2) interior→Nx_cheb→ boundary via chebfleg_mat
	Returns: (batch, nx_leg, nx_leg).
	"""
	# 1) Solve with tile:
	loc = solve_dir_helper_with_tile(
		xxloc_bnd,
		uu_dir,
		ff_body,
		chebfleg_mat,
		Nx_cheb,
		D_stack,
		consts,
		c_fns,
		ni_cheb
	)  # (batch, nt_cheb, nx_leg)

	tmp  = Nx_cheb @ loc
	return legfcheb_mat @ tmp

@functools.partial(jax.jit,static_argnums=(-1,-2))
def reduce_chunk_body_load(
	xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
	uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
	ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
	legfcheb_mat: jnp.ndarray,  # (nx_leg, nb_cheb)
	chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
	Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
	D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
	consts:       jnp.ndarray,  # (N_terms,)
	c_fns:        tuple,        # length N_terms tuple of Python callables
	ni_cheb:      int           # number of interior Chebyshev nodes):
	):

	loc_sol = solve_dir_helper_with_tile(
		xxloc_bnd,uu_dir,-ff_body,chebfleg_mat,Nx_cheb,\
		D_stack,consts,c_fns,ni_cheb
	)

	tmp     = Nx_cheb[:, :ni_cheb] @ loc_sol[:, :ni_cheb]		
	return legfcheb_mat @ tmp

##########################################################################################################################

class LeafSubdomain:
	def __init__(self, box_centers, pdo, patch_utils, verbose):

		# box_centers: (batch, ndim)
		box_centers     = box_centers if box_centers.ndim > 1 else box_centers[None, :]

		self.p          = patch_utils.p
		self.ndim       = box_centers.shape[-1]
		self.utils      = patch_utils

		self.nbatch     = box_centers.shape[0]

		try:
			device  = jax.devices('gpu')[0]
			max_mem = device.memory_stats()['bytes_limit']
		except (IndexError,RuntimeError):
			device = jax.devices('cpu')[0]
			max_mem = max( 5e9, int(query_total_memory()/5))

		const_overhead  = 4; const_nbytes = 8
		chunk_calc      = int(max_mem // ((self.p**self.ndim)**2  * const_overhead * const_nbytes))
		self.chunk_size = max( min(self.nbatch, chunk_calc), 1)
		self.nbatch_ext = ((self.nbatch+self.chunk_size-1)//self.chunk_size) * self.chunk_size

		if (verbose):
			print("Using device:",device,"for HPS leaf computations (",\
			self.nbatch, "leaves with parallel chunk size",self.chunk_size,")")

		# indexing for the Chebyshev discretization
		JJc   = patch_utils.JJ_int
		if self.ndim == 2:
			Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju])
		else:
			Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju, JJc.Jb, JJc.Jf])

		Jx_cheb_uni, inds_uni = jnp.unique(Jx_cheb,return_index=True)
		Ji_cheb               = JJc.Ji

		Jorder                = jnp.hstack([Ji_cheb,Jx_cheb_uni])
		Ds                    = jax.tree.map(lambda x: jnp.array(x[Jorder][:,Jorder], dtype=jnp.float64,device=device),patch_utils.Ds)

		# differentiate on total patch and restrict to Jx_cheb
		self.Nx_cheb       = jnp.array(patch_utils.Nx_stack[:,Jorder],dtype=jnp.float64,device=device)

		# interpolate from Legendre exterior grid to Jx_cheb_uni
		self.chebfleg_mat  = jnp.array(patch_utils.chebfleg_exterior_mat[inds_uni],dtype=jnp.float64,device=device)
		# interpolate from Jx_cheb to Legendre exterior grid
		self.legfcheb_mat  = jnp.array(patch_utils.legfcheb_exterior_mat,dtype=jnp.float64,device=device)

		zz_patch_cheb = patch_utils.zz_int[Jorder]
		zz_patch_leg  = patch_utils.zz_ext

		self.nt_cheb  = zz_patch_cheb.shape[0]; self.nx_leg = zz_patch_leg.shape[0]
		self.ni_cheb  = Ji_cheb.shape[0]
		self.inv_perm_ni_cheb = np.argsort(Jorder)
		self.perm_ni_cheb     = Jorder

		# interior/exterior local points: (batch, nt, ndim)
		box_centers      = jnp.vstack(( box_centers, jnp.zeros((self.nbatch_ext-self.nbatch,box_centers.shape[-1])) ))
		self._xxloc_int  = jnp.array(zz_patch_cheb[None, :, :] + box_centers[:, None, :],dtype=jnp.float64,device=device)
		self._xxloc_ext  = jnp.array(zz_patch_leg [None, :, :] + box_centers[:, None, :],dtype=jnp.float64,device=device)

		self._eye_block = jnp.eye(self.nx_leg, dtype=jnp.float64,device=device)[None, :, :]
		self._zero_ff   = jnp.zeros((1, self.nt_cheb, 1), dtype=jnp.float64,device=device) 
		self._zero_dir  = jnp.zeros((1, self.nx_leg, 1), dtype=jnp.float64,device=device) 

		self.D_stack, self.consts, self.c_fns = \
			build_pdo_terms(pdo, Ds, self.ndim, self.nt_cheb, device)

	def Aloc(self):
		return _get_Aloc(self._xxloc_int,self.D_stack, self.c_fns, self.consts)[:,self.nbatch]

	@property
	def xxloc_int(self):
		return self._xxloc_int[:self.nbatch,self.inv_perm_ni_cheb]

	@property
	def xxloc_ext(self):
		return self._xxloc_ext[:self.nbatch]

	def solve_dir(self, uu_dir, ff_body=None):
		"""
		solve_dir: solves for each patch in chunk_size‐sized pieces.
		uu_dir:  (batch, nx_leg, nrhs)  or  (nx_leg, nrhs)
		ff_body: (batch, nt_cheb, nrhs) or  (nt_cheb, nrhs) or None
		Returns: (batch, nt_cheb, nrhs)
		"""
		device = self._xxloc_int.device
		nrhs   = uu_dir.shape[-1]

		if (self.nbatch == 1 and uu_dir.ndim == 2):
			uu_dir = uu_dir[None,:,:]

		if ff_body is None:
			ff_body = np.zeros((self.nbatch, self.nt_cheb, nrhs))
		elif (self.nbatch == 1 and ff_body.ndim == 2):
			ff_body == ff_body[None,:,:]
		assert ff_body.shape[0] == self.nbatch
		assert uu_dir.shape[0]  == self.nbatch

		uu_ext   = np.zeros((self.nbatch_ext,self.nx_leg,nrhs))
		uu_ext[:self.nbatch] = uu_dir

		ff_ext   = np.zeros((self.nbatch_ext,self.nt_cheb,nrhs))
		ff_ext[:self.nbatch] = ff_body[:,self.perm_ni_cheb]

		out_host = np.zeros((self.nbatch,self.nt_cheb,nrhs))
		nchunks  = self.nbatch_ext // self.chunk_size

		for i in range(nchunks):
			start = i * self.chunk_size
			end   = start + self.chunk_size

			xx_chunk = self._xxloc_int[start:end]
			uu_chunk = jnp.array(uu_ext[start:end],device=device)
			ff_chunk = jnp.array(ff_ext[start:end],device=device)
			frag_dev = solve_dir_helper(xx_chunk,uu_chunk,ff_chunk,\
				self.chebfleg_mat,       # (nb_cheb, nx_leg)
				self.Nx_cheb,            # (nx_leg, nt_cheb)
				self.D_stack,            # (N_terms, nt_cheb, nt_cheb)
				self.consts,             # jnp
				self.c_fns,              # tuple of callables (static)
				self.ni_cheb             # int (static))
				)

			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start 

			if length > 0:
				frag_chunk_host = np.array(frag_dev)[:length,self.inv_perm_ni_cheb]  
				out_host[real_start:real_end, :, :] = frag_chunk_host
		return out_host

	def reduce_body_load(self, ff_body):
		if (self.nbatch == 1 and ff_body.ndim == 2):
			ff_body == ff_body[None,:,:]
		assert ff_body.shape[0] == self.nbatch

		nrhs  = ff_body.shape[-1]
		device= self._xxloc_int.device

		ff_ext   = np.zeros((self.nbatch_ext,self.nt_cheb,nrhs))
		ff_ext[:self.nbatch] = ff_body[:,self.perm_ni_cheb]

		out_host = np.zeros((self.nbatch,self.nx_leg,nrhs))
		nchunks  = self.nbatch_ext // self.chunk_size

		for i in range(nchunks):
			start = i * self.chunk_size
			end   = start + self.chunk_size

			xx_chunk = self._xxloc_int[start:end]
			ff_chunk = jnp.array(ff_ext[start:end],device=device)
			frag_dev = reduce_chunk_body_load(xx_chunk,\
				self._zero_dir, ff_chunk, 
				self.legfcheb_mat,
				self.chebfleg_mat,       # (nb_cheb, nx_leg)
				self.Nx_cheb,            # (nx_leg, nt_cheb)
				self.D_stack,            # (N_terms, nt_cheb, nt_cheb)
				self.consts,             # jnp
				self.c_fns,              # tuple of callables (static)
				self.ni_cheb             # int (static))
				)

			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start 

			if length > 0:
				frag_chunk_host = np.array(frag_dev[:length]) 
				out_host[real_start:real_end, :, :] = frag_chunk_host
		return out_host

	# ------------------------------------------------------------------------
	def DtN(self) -> np.ndarray:
		"""
		Compute the DtN “matrix” for each of the nbatch leaves, by slicing
		through self._xxloc_int in chunks of size self.chunk_size. We never
		allocate the full (nbatch_ext × nx_leg × nx_leg) on the device at once.
		Instead, we fill a host‐side NumPy buffer chunk by chunk.
		"""
		device = self._xxloc_int.device  # still a JAX array on some device

		# 1) Pre‐allocate a host‐side buffer for “real” leaves only:
		out_host = np.zeros((self.nbatch, self.nx_leg, self.nx_leg),
							dtype=np.float64)

		nchunks = self.nbatch_ext // self.chunk_size

		for i in range(nchunks):
			start = i * self.chunk_size
			end   = start + self.chunk_size

			# 2) Slice the device‐side xxloc_int for this chunk:
			xx_chunk = self._xxloc_int[start:end]   # JAX array (chunk_size, nt_cheb, ndim)

			# 3) Call the standalone JIT’d DtN‐helper:
			frag_dev = compute_chunk_DtN(
				xx_chunk,
				self._eye_block,         # (1, nx_leg, nx_leg)
				self._zero_ff,           # (1, nt_cheb, 1)
				self.legfcheb_mat,
				self.chebfleg_mat,       # (nb_cheb, nx_leg)
				self.Nx_cheb,            # (nx_leg, nt_cheb)
				self.D_stack,            # (N_terms, nt_cheb, nt_cheb)
				self.consts,             # jnp
				self.c_fns,              # tuple of callables (static)
				self.ni_cheb             # int (static)
			)  # → JAX array (chunk_size, nx_leg, nx_leg)

			# 4) Copy only the “real” leaves (the first length rows) back to host:
			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start

			if length > 0:
				sub_np = np.array(frag_dev[:length])   # blocks, copies (length, nx_leg, nx_leg)
				out_host[real_start:real_end, :, :] = sub_np

		return out_host