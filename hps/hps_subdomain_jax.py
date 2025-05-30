import jax
# enable 64-bit (double) precision globally
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functools

@jax.jit
def diag_mult(diag, M):
    """
    JAX version of diag_mult.
    diag: shape (m,) or (batch, m)
    M:    shape (m, n) or (batch, m, n)
    """
    if diag.ndim == 1:
        # (m,) * (m,n) -> (m,n)
        return (diag[:, None] * M)
    elif diag.ndim == 2:
        # (batch, m, 1) * (batch, m, n)
        return diag[..., :, None] * M
    else:
        raise ValueError(f"Unsupported diag.ndim={diag.ndim}")


class LeafSubdomainJAX:
    def __init__(self, box_centers, pdo, patch_utils):
        # centers: (batch, ndim)
        centers = box_centers if box_centers.ndim > 1 else box_centers[None, :]

        device          = jax.devices('gpu')[0] if len(jax.devices('gpu')) > 0 else jax.devices('cpu')[0]
        Ds              = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float64,device=device),patch_utils.Ds)

        self.utils      = patch_utils
        self.ndim       = patch_utils.ndim

        # interior/exterior local points: (batch, nt, ndim)
        self.xxloc_int  = jnp.array(patch_utils.zz_int[None, :, :] + centers[:, None, :],dtype=jnp.float64,device=device)
        self.xxloc_ext  = jnp.array(patch_utils.zz_ext[None, :, :] + centers[:, None, :],dtype=jnp.float64,device=device)

        self.p          = patch_utils.p
        self.nx_leg     = self.xxloc_ext.shape[1]
        self.nt_cheb    = self.xxloc_int.shape[1]
        self.nbatch     = centers.shape[0]

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

            # helper to accumulate terms
            def accum(A, coeff_fn, D_op, factor=1.0):
                c = coeff_fn(xxloc)  # shape (m,) or (batch,m)
                return A - factor * diag_mult(c, D_op)

            # second‐order terms
            Aloc = accum(Aloc, pdo.c11, Ds.D11)
            Aloc = accum(Aloc, pdo.c22, Ds.D22)
            if pdo.c12 is not None:
                Aloc = accum(Aloc, pdo.c12, Ds.D12, factor=2.0)

            # lower‐order terms
            if pdo.c1  is not None: Aloc = Aloc + diag_mult(pdo.c1(xxloc), Ds.D1)
            if pdo.c2  is not None: Aloc = Aloc + diag_mult(pdo.c2(xxloc), Ds.D2)
            if pdo.c   is not None: Aloc = Aloc + diag_mult(pdo.c(xxloc),  jnp.eye(m))

            if ndim == 3:
                Aloc = accum(Aloc, pdo.c33, Ds.D33)
                if pdo.c13 is not None: Aloc = accum(Aloc, pdo.c13, Ds.D13, factor=2.0)
                if pdo.c23 is not None: Aloc = accum(Aloc, pdo.c23, Ds.D23, factor=2.0)
                if pdo.c3  is not None: Aloc = Aloc + diag_mult(pdo.c3(xxloc), Ds.D3)

            return Aloc

        self._get_Aloc = _get_Aloc

    @property
    @functools.partial(jax.jit, static_argnums=0)
    def Aloc(self):
        return self._get_Aloc(self.xxloc_int)

    @functools.partial(jax.jit, static_argnums=0)
    def solve_dir_helper(self, xxloc_bnd, uu_dir=None, ff_body=None):
        B       = xxloc_bnd.shape[0]
        nrhs    = uu_dir.shape[-1] if uu_dir is not None else self.nx_leg

        if ff_body is None:
            ff_tmp = jnp.zeros((1, self.Ji_cheb.shape[0], nrhs))
        else:
            ff_tmp = ff_body[:, self.Ji_cheb]

        if uu_dir is None:
            uu_tmp = jnp.eye(self.nx_leg)[None, :, :]
        else:
            uu_tmp = uu_dir

        Aloc_bnd = self._get_Aloc(xxloc_bnd)
        Aib = Aloc_bnd[:, self.Ji_cheb][:, :, self.Jx_cheb_uni]
        Aii = Aloc_bnd[:, self.Ji_cheb][:, :, self.Ji_cheb]

        res = jnp.zeros((B, self.nt_cheb, nrhs))

        tmp        = self.chebfleg_mat @ uu_tmp

        res = res.at[:, self.Jx_cheb_uni].set(tmp[...,self.inds_uni,:])

        rhs = ff_tmp - (Aib @ res[:, self.Jx_cheb_uni])
        sol_i = jnp.linalg.solve(Aii, rhs)  # solves each batch & RHS
        res = res.at[:, self.Ji_cheb].set(sol_i)
        return res

    @functools.partial(jax.jit, static_argnums=0)
    def solve_dir(self, uu_dir, ff_body=None):
        if uu_dir.ndim == 2:
            uu_dir = uu_dir[None, :, :]
        assert uu_dir.shape[0] == self.nbatch

        res = jnp.zeros((self.nbatch, self.nt_cheb, uu_dir.shape[-1]),device=self.xxloc_int.device)

        for start in range(0, self.nbatch, 10):
            end = min(start + 10, self.nbatch)
            res = res.at[start:end].set(
                self.solve_dir_helper(self.xxloc_int[start:end], uu_dir[start:end], ff_body[start:end])
            )
        return res

    @functools.partial(jax.jit, static_argnums=0)
    def reduce_body_load(self, ff_body):
        if ff_body.ndim == 2:
            ff_body = ff_body[None, :, :]

        loc_sol = self.solve_dir(jnp.zeros((self.nbatch, self.nx_leg, 1),device=self.xxloc_int.device), -ff_body)
        tmp     = self.Nx_cheb[:, self.Ji_cheb] @ loc_sol[:, self.Ji_cheb]
        return self.legfcheb_mat @ tmp

    @property
    @functools.partial(jax.jit, static_argnums=0)
    def DtN(self):
        DtNs = jnp.zeros((self.nbatch, self.nx_leg, self.nx_leg),device=self.xxloc_int.device)
        for start in range(0, self.nbatch, 10):
            end = min(start + 10, self.nbatch)
            loc = self.solve_dir_helper(self.xxloc_int[start:end])
            DtNs = DtNs.at[start:end].set(self.legfcheb_mat @ (self.Nx_cheb @ loc))
        return DtNs
