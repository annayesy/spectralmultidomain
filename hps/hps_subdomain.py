import jax
# enable 64-bit (double) precision globally
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functools

import os
import numpy as np

def query_total_memory():
	pages = os.sysconf('SC_PHYS_PAGES')
	page_size = os.sysconf('SC_PAGE_SIZE')
	return pages * page_size

def diag_mult(diag, M):
	return diag[..., None] * M

class LeafSubdomain:
	def __init__(self, box_centers, pdo, patch_utils):

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
		print("Using device:",device,"for HPS leaf computations (", self.nbatch, "leaves with parallel chunk size",self.chunk_size,")")

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

		terms_pdo = [(pdo.c11, Ds.D11, -1), (pdo.c22, Ds.D22, -1)]
		if pdo.c12 is not None:
			terms_pdo.append((pdo.c12, Ds.D12, -2))
		if pdo.c1  is not None:
			terms_pdo.append((pdo.c1,  Ds.D1,  +1))
		if pdo.c2  is not None:
			terms_pdo.append((pdo.c2,  Ds.D2,  +1))
		if pdo.c   is not None:
			terms_pdo.append((pdo.c, jnp.eye(self.nt_cheb, dtype=jnp.float64, device=device), +1))

		if self.ndim == 3:
			terms_pdo.append((pdo.c33, Ds.D33, -1))
			if pdo.c13 is not None:
				terms_pdo.append((pdo.c13, Ds.D13, -2))
			if pdo.c23 is not None:
				terms_pdo.append((pdo.c23, Ds.D23, -2))
			if pdo.c3 is not None:
				terms_pdo.append((pdo.c3, Ds.D3, +1))

		@jax.jit
		def _get_Aloc(xxloc):
			"""
			JAX version of get_Aloc.
			xxloc: either (m, ndim) or (batch, m, ndim)
			returns: (m, m) or (batch, m, m)
			"""
			m = xxloc.shape[-2]
			ndim = xxloc.shape[-1]
			# Determine output shape
			out_shape = (m, m) if xxloc.ndim == 2 else (xxloc.shape[0], m, m)
			Aloc = jnp.zeros(out_shape)

			for (c_func,Dterm,const) in terms_pdo:
				Aloc += const * diag_mult(c_func(xxloc),Dterm)

			return Aloc

		self._get_Aloc = _get_Aloc

	def Aloc(self):
		return self._get_Aloc(self._xxloc_int)[:,self.nbatch]

	@property
	def xxloc_int(self):
		return self._xxloc_int[:self.nbatch,self.inv_perm_ni_cheb]

	@property
	def xxloc_ext(self):
		return self._xxloc_ext[:self.nbatch]

	@functools.partial(jax.jit,static_argnums=0)
	def solve_dir_helper_with_tile(self, xxloc_bnd, uu_dir, ff_body):
		B       = xxloc_bnd.shape[0]
		nrhs    = uu_dir.shape[-1] if uu_dir is not None else self.nx_leg

		Aloc_bnd = self._get_Aloc(xxloc_bnd)
		Aib = Aloc_bnd[:, :self.ni_cheb, self.ni_cheb:]
		Aii = Aloc_bnd[:, :self.ni_cheb, :self.ni_cheb]

		equiv_dir = self.chebfleg_mat @ uu_dir

		rhs   = ff_body[:,:self.ni_cheb] - (Aib @ equiv_dir)
		sol_i = jnp.linalg.solve(Aii, rhs)  # solves each batch & RHS

		return jnp.concatenate([sol_i,jnp.tile(equiv_dir,(B,1,1))],axis=1)

	@functools.partial(jax.jit,static_argnums=0)
	def solve_dir_helper(self, xxloc_bnd, uu_dir, ff_body):
		B       = xxloc_bnd.shape[0]
		nrhs    = uu_dir.shape[-1] if uu_dir is not None else self.nx_leg

		Aloc_bnd = self._get_Aloc(xxloc_bnd)
		Aib = Aloc_bnd[:, :self.ni_cheb, self.ni_cheb:]
		Aii = Aloc_bnd[:, :self.ni_cheb, :self.ni_cheb]

		equiv_dir = self.chebfleg_mat @ uu_dir

		rhs   = ff_body[:,:self.ni_cheb] - (Aib @ equiv_dir)
		sol_i = jnp.linalg.solve(Aii, rhs)  # solves each batch & RHS
		return jnp.concatenate([sol_i,equiv_dir],axis=1)

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
			frag_dev = self.solve_dir_helper(xx_chunk,uu_chunk,ff_chunk)

			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start 

			if length > 0:
				frag_chunk_host = np.array(frag_dev)[:length,self.inv_perm_ni_cheb]   # shape (length, nx_leg, nx_leg)
				out_host[real_start:real_end, :, :] = frag_chunk_host
		return out_host

	@functools.partial(jax.jit,static_argnums=0)
	def _reduce_chunk_body_load(self, xx_chunk, ff_chunk):

		loc_sol = self.solve_dir_helper_with_tile(xx_chunk, self._zero_dir, -ff_chunk)
		tmp     = self.Nx_cheb[:, :self.ni_cheb] @ loc_sol[:, :self.ni_cheb]		
		return self.legfcheb_mat @ tmp

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
			frag_dev = self._reduce_chunk_body_load(xx_chunk,ff_chunk)

			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start 

			if length > 0:
				frag_chunk_host = np.array(frag_dev[:length])   # shape (length, nx_leg, nx_leg)
				out_host[real_start:real_end, :, :] = frag_chunk_host
		return out_host


	@functools.partial(jax.jit, static_argnums=0)
	def _compute_chunk_DtN(self, xx_chunk):
			"""
			xx_chunk: a JAX array of shape (chunk_size, nt_cheb, ndim)
			Returns a DeviceArray frag of shape (chunk_size, nx_leg, nx_leg),
			which is exactly the DtN‐block corresponding to these chunk_size leaves.
			"""
			loc = self.solve_dir_helper_with_tile(
				xx_chunk,
				uu_dir  = self._eye_block,
				ff_body = self._zero_ff
			)

			tmp  = self.Nx_cheb @ loc
			frag = self.legfcheb_mat @ tmp

			return frag

	def DtN(self):
		"""
		Compute the DtN “matrix” for each of the `nbatch` leaves, by slicing
		through self._xxloc_int in chunks of size self.chunk_size.  We never
		allocate the full (nbatch_ext × nx_leg × nx_leg) array on the device.
		Instead, we build a host‐side NumPy array of shape (nbatch × nx_leg × nx_leg),
		fill it chunk by chunk, and return it as a NumPy array.
		"""
		# Pre‐allocate a host‐side numpy buffer for “real” leaves:
		out_host = np.zeros((self.nbatch, self.nx_leg, self.nx_leg), dtype=np.float64)

		nchunks = self.nbatch_ext // self.chunk_size

		for i in range(nchunks):
			start = i * self.chunk_size
			end   = start + self.chunk_size

			xx_chunk = self._xxloc_int[start:end]   # JAX array of shape (chunk_size, nt_cheb, ndim)
			frag_dev = self._compute_chunk_DtN(xx_chunk)

			real_start = start
			real_end   = min(end, self.nbatch)
			length     = real_end - real_start

			if length > 0:
				frag_chunk_host = np.array(frag_dev[:length])   # shape (length, nx_leg, nx_leg)
				out_host[real_start:real_end, :, :] = frag_chunk_host
		return out_host