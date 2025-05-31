import jax
# enable 64-bit (double) precision globally
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functools
from jax import lax

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
			max_mem = 5e9

		const_overhead  = 8; const_nbytes = 8
		self.chunk_size = min(self.nbatch, max_mem // ((self.p**self.ndim)**2  * const_overhead * const_nbytes))

		self.nbatch_ext = ((self.nbatch+self.chunk_size-1)//self.chunk_size) * self.chunk_size
		print("Using device:",device,"for HPS leaf computations (", self.nbatch, "leaves with parallel chunk size",self.chunk_size,")")

		Ds              = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float64,device=device),patch_utils.Ds)
		box_centers     = jnp.vstack(( box_centers, jnp.zeros((self.nbatch_ext-self.nbatch,box_centers.shape[-1])) ))

		# interior/exterior local points: (batch, nt, ndim)
		self._xxloc_int  = jnp.array(patch_utils.zz_int[None, :, :] + box_centers[:, None, :],dtype=jnp.float64,device=device)
		self._xxloc_ext  = jnp.array(patch_utils.zz_ext[None, :, :] + box_centers[:, None, :],dtype=jnp.float64,device=device)

		self.nx_leg     = self._xxloc_ext.shape[1]
		self.nt_cheb    = self._xxloc_int.shape[1]

		JJc   = patch_utils.JJ_int
		if self.ndim == 2:
			self.Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju])
		else:
			self.Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju, JJc.Jb, JJc.Jf])
		self.Ji_cheb     = jnp.hstack([JJc.Ji])
		self.Jx_cheb_uni = jnp.setdiff1d(jnp.arange(self.nt_cheb), self.Ji_cheb)

		self.Jx_cheb     = jax.device_put(self.Jx_cheb,device=device)
		self.Ji_cheb     = jax.device_put(self.Ji_cheb,device=device)
		self.Jx_cheb_uni = jax.device_put(self.Jx_cheb_uni,device=device)

		self.inds_uni    = jnp.unique(self.Jx_cheb,return_index=True)[1]

		self.Nx_cheb       = jnp.array(patch_utils.Nx_stack,dtype=jnp.float64,device=device)
		self.chebfleg_mat  = jnp.array(patch_utils.chebfleg_exterior_mat,dtype=jnp.float64,device=device)
		self.legfcheb_mat  = jnp.array(patch_utils.legfcheb_exterior_mat,dtype=jnp.float64,device=device)

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
		return self._xxloc_int[:self.nbatch]

	@property
	def xxloc_ext(self):
		return self._xxloc_ext[:self.nbatch]

	def solve_dir_helper(self, xxloc_bnd, uu_dir, ff_body):
		B       = xxloc_bnd.shape[0]
		nrhs    = uu_dir.shape[-1] if uu_dir is not None else self.nx_leg

		Aloc_bnd = self._get_Aloc(xxloc_bnd)
		Aib = Aloc_bnd[:, self.Ji_cheb][:, :, self.Jx_cheb_uni]
		Aii = Aloc_bnd[:, self.Ji_cheb][:, :, self.Ji_cheb]

		res = jnp.zeros((B, self.nt_cheb, nrhs))
		tmp = self.chebfleg_mat @ uu_dir
		res = res.at[:, self.Jx_cheb_uni].set(tmp[...,self.inds_uni,:])

		rhs   = ff_body[:,self.Ji_cheb] - (Aib @ res[:, self.Jx_cheb_uni])
		sol_i = jnp.linalg.solve(Aii, rhs)  # solves each batch & RHS
		res   = res.at[:, self.Ji_cheb].set(sol_i)
		return res

	def solve_dir(self, uu_dir, ff_body=None):
		"""
		solve_dir: solves for each patch in chunk_size‐sized pieces.
		uu_dir:  (batch, nx_leg, nrhs)  or  (nx_leg, nrhs)
		ff_body: (batch, nt_cheb, nrhs) or  (nt_cheb, nrhs) or None
		Returns: (batch, nt_cheb, nrhs)
		"""
		if uu_dir.ndim == 2:
			uu_dir = uu_dir[None, :, :]
		assert uu_dir.shape[0] == self.nbatch

		device = self._xxloc_int.device
		nrhs   = uu_dir.shape[-1]

		if ff_body is None:
			ff_body = jnp.zeros((self.nbatch, self.nt_cheb, nrhs),
								dtype=jnp.float64,
								device=device)
		assert ff_body.shape[0] == self.nbatch

		return self._solve_dir(uu_dir,ff_body)

	@functools.partial(jax.jit, static_argnums=0)
	def _solve_dir(self, uu_dir,ff_body):
		nrhs        = uu_dir.shape[-1]

		uu_dir_ext  = jnp.zeros((self.nbatch_ext, self.nx_leg, nrhs),
								dtype=jnp.float64)
		uu_dir_ext  = uu_dir_ext.at[: self.nbatch].set(uu_dir)

		ff_body_ext = jnp.zeros((self.nbatch_ext, self.nt_cheb, nrhs),
								dtype=jnp.float64)
		ff_body_ext = ff_body_ext.at[: self.nbatch].set(ff_body)
		res_ext = jnp.zeros((self.nbatch_ext, self.nt_cheb, nrhs),
							dtype=jnp.float64)

		nchunks = self.nbatch_ext // self.chunk_size

		def body_fn(i, res_carry):

			start = i * self.chunk_size

			xxloc_chunk  = lax.dynamic_slice(
				self._xxloc_int,
				(start, 0, 0),
				(self.chunk_size, self.nt_cheb, self.ndim)
			)
			uu_chunk     = lax.dynamic_slice(
				uu_dir_ext,
				(start, 0, 0),
				(self.chunk_size, self.nx_leg, nrhs)
			)
			ff_chunk     = lax.dynamic_slice(
				ff_body_ext,
				(start, 0, 0),
				(self.chunk_size, self.nt_cheb, nrhs)
			)

			local_result = self.solve_dir_helper(xxloc_chunk, uu_chunk, ff_chunk)

			res_carry = lax.dynamic_update_slice(
				res_carry,
				local_result,         # shape (chunk_size, nt_cheb, nrhs)
				(start, 0, 0)         # starting indices
			)
			return res_carry

		res_final = lax.fori_loop(0, nchunks, body_fn, res_ext)
		return res_final[: self.nbatch]

	def reduce_body_load(self, ff_body):
		if ff_body.ndim == 2:
			ff_body = ff_body[None, :, :]
		nrhs  = ff_body.shape[-1]
		device= self._xxloc_int.device

		loc_sol = self._solve_dir(jnp.zeros((self.nbatch, self.nx_leg, 1),device=device), -ff_body)
		tmp     = self.Nx_cheb[:, self.Ji_cheb] @ loc_sol[:, self.Ji_cheb]
		return self.legfcheb_mat @ tmp

	@functools.partial(jax.jit, static_argnums=0)
	def DtN(self):
		"""
		Compute a (nbatch_ext × nx_leg × nx_leg) DtN in chunks of size=self.chunk_size
		using lax.fori_loop, then return only the first self.nbatch “real” rows.
		"""
		DtNs = jnp.zeros(
			(self.nbatch_ext, self.nx_leg, self.nx_leg),
			dtype=jnp.float64,
			device=self._xxloc_int.device
		)

		nchunks = self.nbatch_ext // self.chunk_size

		def body_fn(i, DtNs_carry):
			start = i * self.chunk_size

			xx_chunk = lax.dynamic_slice(
				self._xxloc_int,
				(start, 0, 0),
				(self.chunk_size, self.nt_cheb, self.ndim)
			)

			loc  = self.solve_dir_helper(xx_chunk, uu_dir = jnp.eye(self.nx_leg)[None, :, :],\
				ff_body=jnp.zeros((1, self.nt_cheb, 1)))
			tmp  = self.Nx_cheb @ loc        
			frag = self.legfcheb_mat @ tmp  

			DtNs_carry = lax.dynamic_update_slice(
				DtNs_carry,      # big array: (nbatch_ext, nx_leg, nx_leg)
				frag,            # small array: (chunk_size, nx_leg, nx_leg)
				(start, 0, 0)    # starting indices along each axis
			)
			return DtNs_carry

		DtNs_full = lax.fori_loop(0, nchunks, body_fn, DtNs)
		return DtNs_full[: self.nbatch]