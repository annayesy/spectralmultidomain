import jax
# Enable 64-bit (double) precision globally
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import functools
import numpy as np
import os

# ------------------------------------------------------------------------------
# NUMPY HELPERS
# ------------------------------------------------------------------------------

def query_total_memory():
    """
    Return the total physical memory (in bytes) available on the system.
    """
    pages = os.sysconf('SC_PHYS_PAGES')
    page_size = os.sysconf('SC_PAGE_SIZE')
    return pages * page_size

def diag_mult(diag, M):
    """
    Multiply each row of matrix M by the corresponding element of the vector diag.
    
    Arguments:
      diag:   Array of shape (..., m, 1) or (..., m) to use as diagonal entries.
      M:      Matrix of shape (..., m, m).
    
    Returns:
      An array of shape (..., m, m) where each row i of M is multiplied by diag[i].
    """
    return diag[..., None] * M

def build_pdo_terms(pdo, Ds, ndim, nt_cheb, device):
    """
    Assemble the PDE operator (pdo) terms into coefficient matrices, constants, and functions.

    Returns:
      Ds_stack:  jnp.ndarray of shape (K, m, m), containing each D matrix.
      consts:    jnp.ndarray of shape (K,), containing scalar multipliers.
      func_list: Tuple of K Python callables, each taking xxloc -> (batch, m, 1) array.
    """
    Ds_list = []
    consts_list = []
    func_list = []

    def helper_append(Dmat, func, const):
        """
        Append a term: 
          - Dmat (m, m) as a JAX array
          - func: callable that maps xxloc -> (m, 1) or (batch, m, 1)
          - const: scalar multiplier
        """
        Ds_list.append(jnp.array(Dmat, dtype=jnp.float64, device=device))
        func_list.append(func)
        consts_list.append(const)

    # Second-derivative terms (negative sign convention for Laplacian-like terms)
    helper_append(Ds.D11, pdo.c11, -1.0)
    helper_append(Ds.D22, pdo.c22, -1.0)

    if pdo.c12 is not None:
        helper_append(Ds.D12, pdo.c12, -2.0)

    # First-derivative terms (positive sign)
    if pdo.c1 is not None:
        helper_append(Ds.D1, pdo.c1, +1.0)
    if pdo.c2 is not None:
        helper_append(Ds.D2, pdo.c2, +1.0)

    # Zero-th order term (scalar coefficient times identity)
    if getattr(pdo, "c", None) is not None:
        I = jnp.eye(nt_cheb, dtype=jnp.float64, device=device)
        helper_append(I, pdo.c, +1.0)

    if ndim == 3:
        # Third-dimension second-derivative
        helper_append(Ds.D33, pdo.c33, -1.0)

        if pdo.c13 is not None:
            helper_append(Ds.D13, pdo.c13, -2.0)
        if pdo.c23 is not None:
            helper_append(Ds.D23, pdo.c23, -2.0)

        # Third-dimension first-derivative
        if pdo.c3 is not None:
            helper_append(Ds.D3, pdo.c3, +1.0)

    # Stack into a single array of shape (K, m, m)
    Ds_stack = jnp.stack(Ds_list, axis=0)
    consts   = jnp.array(consts_list, dtype=jnp.float64, device=device)

    return Ds_stack, consts, tuple(func_list)

# ------------------------------------------------------------------------------
# JAX HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def _get_Aloc(xxloc, Ds_stack, consts, funcs):
    """
    Evaluate the local PDE operator A at points xxloc in each patch.

    Arguments:
      xxloc:    Array of shape (batch, m, ndim) or (m, ndim).
      Ds_stack: Array of shape (K, m, m), stacked coefficient matrices.
      consts:   Array of shape (K,), scalar multipliers for each term.
      funcs:    Tuple of K callables, each mapping xxloc -> (batch, m, 1) or (m, 1).

    Returns:
      Aloc:     Array of shape (batch, m, m) or (m, m), the assembled operator.
    """
    m = xxloc.shape[-2]
    
    # Determine output shape
    if xxloc.ndim == 2:
        out_shape = (m, m)
    else:
        out_shape = (xxloc.shape[0], m, m)

    Aloc = jnp.zeros(out_shape, dtype=jnp.float64)
    K = Ds_stack.shape[0]

    for i in range(K):
        c_fun = funcs[i]            # Python callable (static)
        D_mat = Ds_stack[i]         # (m, m)
        coeff = consts[i]           # scalar multiplier

        # Evaluate coefficient function at each local point
        cvals = c_fun(xxloc)        # (m,1) or (batch, m,1)
        Aloc = Aloc + coeff * diag_mult(cvals, D_mat)

    return Aloc

def solve_dir_helper_with_tile(
    xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
    uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
    ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
    chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
    Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
    D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
    consts:       jnp.ndarray,  # (N_terms,)
    c_fns:        tuple,        # length N_terms tuple of callables
    ni_cheb:      int           # number of interior Chebyshev nodes (static)
) -> jnp.ndarray:
    """
    Solve Dirichlet problem in one patch with tiled boundary conditions.

    Steps:
      1. Compute Aloc on boundary.
      2. Partition Aloc into interior-interior (Aii) and interior-boundary (Aib) blocks.
      3. Compute equivalent Dirichlet data on boundary via Chebyshev-to-Legendre.
      4. Build RHS: ff_body_interior - Aib @ equiv_dir.
      5. Solve interior block Aii * sol_i = RHS.
      6. Concatenate interior solution with boundary values and return.

    Returns:
      (batch, nt_cheb, nrhs) array of local solutions.
    """
    B = xxloc_bnd.shape[0]
    nrhs = uu_dir.shape[-1]

    # 1) Evaluate local operator on boundary nodes
    Aloc_bnd = _get_Aloc(xxloc_bnd, D_stack, consts, c_fns)

    # 2) Partition Aii and Aib
    Aii = Aloc_bnd[:, :ni_cheb, :ni_cheb]   # (batch, ni_cheb, ni_cheb)
    Aib = Aloc_bnd[:, :ni_cheb, ni_cheb:]   # (batch, ni_cheb, nb_cheb)

    # 3) Compute equivalent Dirichlet data on boundary
    equiv_dir = chebfleg_mat @ uu_dir       # (batch, nb_cheb, nrhs)

    # 4) Build the RHS for interior solve
    rhs = ff_body[:, :ni_cheb, :] - (Aib @ equiv_dir)  # (batch, ni_cheb, nrhs)

    # 5) Solve the interior block
    sol_i = jnp.linalg.solve(Aii, rhs)     # (batch, ni_cheb, nrhs)

    # 6) Concatenate interior solution and boundary data
    return jnp.concatenate([sol_i, jnp.tile(equiv_dir, (B, 1, 1))], axis=1)

def solve_dir_helper(
    xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
    uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
    ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
    chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
    Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
    D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
    consts:       jnp.ndarray,  # (N_terms,)
    c_fns:        tuple,        # length N_terms tuple of callables
    ni_cheb:      int           # number of interior Chebyshev nodes (static)
) -> jnp.ndarray:
    """
    Solve Dirichlet problem in one patch without tiling (assumes boundary data directly).
    """
    B = xxloc_bnd.shape[0]
    nrhs = uu_dir.shape[-1]

    # 1) Evaluate local operator on boundary nodes
    Aloc_bnd = _get_Aloc(xxloc_bnd, D_stack, consts, c_fns)

    # 2) Partition Aii and Aib
    Aii = Aloc_bnd[:, :ni_cheb, :ni_cheb]   # (batch, ni_cheb, ni_cheb)
    Aib = Aloc_bnd[:, :ni_cheb, ni_cheb:]   # (batch, ni_cheb, nb_cheb)

    # 3) Compute equivalent Dirichlet data on boundary
    equiv_dir = chebfleg_mat @ uu_dir       # (batch, nb_cheb, nrhs)

    # 4) Build the RHS for interior solve
    rhs = ff_body[:, :ni_cheb, :] - (Aib @ equiv_dir)  # (batch, ni_cheb, nrhs)

    # 5) Solve the interior block
    sol_i = jnp.linalg.solve(Aii, rhs)     # (batch, ni_cheb, nrhs)

    # 6) Concatenate interior solution and boundary data
    return jnp.concatenate([sol_i, equiv_dir], axis=1)

def compute_chunk_DtN(
    xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
    uu_dir:       jnp.ndarray,  # (1, nx_leg, nx_leg) — identity block for Dirichlet BC
    ff_body:      jnp.ndarray,  # (1, nt_cheb, 1) — zero body force
    legfcheb_mat: jnp.ndarray,  # (nx_leg, nb_cheb)
    chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
    Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
    D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
    consts:       jnp.ndarray,  # (N_terms,)
    c_fns:        tuple,        # length N_terms tuple of callables
    ni_cheb:      int           # number of interior Chebyshev nodes (static)
) -> jnp.ndarray:
    """
    Compute the Dirichlet-to-Neumann (DtN) map for each patch in the chunk.

    Steps:
      1. Solve with Dirichlet = identity on boundary, zero body force.
      2. Convert interior solution to boundary via legfcheb_mat and chebfleg_mat.
    
    Returns:
      Array of shape (batch, nx_leg, nx_leg) representing local DtN operator.
    """
    # Solve with tiled 'identity' boundary data
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

    # Map interior solution to Legendre grid and then back to boundary Chebyshev nodes
    tmp = Nx_cheb @ loc
    return legfcheb_mat @ tmp

def reduce_chunk_body_load(
    xxloc_bnd:    jnp.ndarray,  # (batch, nt_cheb, ndim)
    uu_dir:       jnp.ndarray,  # (batch, nx_leg, nrhs) or (1, nx_leg, nrhs)
    ff_body:      jnp.ndarray,  # (batch, nt_cheb, nrhs) or (1, nt_cheb, nrhs)
    legfcheb_mat: jnp.ndarray,  # (nx_leg, nb_cheb)
    chebfleg_mat: jnp.ndarray,  # (nb_cheb, nx_leg)
    Nx_cheb:      jnp.ndarray,  # (nx_leg, nt_cheb)
    D_stack:      jnp.ndarray,  # (N_terms, nt_cheb, nt_cheb)
    consts:       jnp.ndarray,  # (N_terms,)
    c_fns:        tuple,        # length N_terms tuple of callables
    ni_cheb:      int           # number of interior Chebyshev nodes (static)
) -> jnp.ndarray:
    """
    Compute boundary load contributions for interior body force.

    Steps:
      1. Solve with Dirichlet = zero, body = -ff_body.
      2. Project interior solution to boundary via legfcheb_mat and chebfleg_mat.
    
    Returns:
      Array of shape (batch, nx_leg, nrhs).
    """
    # 1) Solve for interior with zero Dirichlet boundary data
    loc_sol = solve_dir_helper_with_tile(
        xxloc_bnd,
        uu_dir,
        -ff_body,
        chebfleg_mat,
        Nx_cheb,
        D_stack,
        consts,
        c_fns,
        ni_cheb
    )

    # 2) Take interior part and map to boundary grid
    tmp = Nx_cheb[:, :ni_cheb] @ loc_sol[:, :ni_cheb]
    return legfcheb_mat @ tmp

# ------------------------------------------------------------------------------
# LEAF SUBDOMAIN CLASS
# ------------------------------------------------------------------------------

class LeafSubdomain:
    def __init__(self, box_centers, pdo, patch_utils, verbose=False):
        """
        Parameters:
          box_centers:  Array of shape (batch, ndim) or (ndim,), center coordinates for each leaf.
          pdo:          PDE operator object with attributes for coefficients (c11, c22, etc.).
          patch_utils:  Utility object providing differentiation matrices, interpolation,
                        and local grid points.
          verbose:      If True, print device and chunk-size information.
        """
        # Ensure box_centers has shape (batch, ndim)
        box_centers = box_centers if box_centers.ndim > 1 else box_centers[None, :]

        self.p = patch_utils.p
        self.ndim = box_centers.shape[-1]
        self.utils = patch_utils
        self.nbatch = box_centers.shape[0]

        # Choose GPU if available, otherwise CPU, and estimate chunk size by available memory
        try:
            device = jax.devices('gpu')[0]
            max_mem = device.memory_stats()['bytes_limit']
        except (IndexError, RuntimeError):
            device = jax.devices('cpu')[0]
            max_mem = max(int(query_total_memory() / 5), int(5e9))

        # Estimate how many patches to process in parallel
        const_overhead = 12
        const_nbytes   = 8
        chunk_calc = int(max_mem // ((self.p ** self.ndim) ** 2 * const_overhead * const_nbytes))
        self.chunk_size = max(min(self.nbatch, chunk_calc), 1)
        self.nbatch_ext = ((self.nbatch + self.chunk_size - 1) // self.chunk_size) * self.chunk_size

        if verbose:
            print(
                "Using device:", device,
                "for HPS leaf computations (",
                self.nbatch, "leaves with chunk size", self.chunk_size, ")"
            )

        # Build index sets for Chebyshev discretization:
        # Ji_cheb: interior indices, Jx_cheb: boundary indices
        JJc = patch_utils.JJ_int
        if self.ndim == 2:
            Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju])
        else:
            Jx_cheb = jnp.hstack([JJc.Jl, JJc.Jr, JJc.Jd, JJc.Ju, JJc.Jb, JJc.Jf])

        Jx_cheb_uni, inds_uni = jnp.unique(Jx_cheb, return_index=True)
        Ji_cheb = JJc.Ji
        Jorder = jnp.hstack([Ji_cheb, Jx_cheb_uni])

        # Reorder differentiation matrices according to Jorder
        Ds = jax.tree.map(
            lambda x: jnp.array(x[Jorder][:, Jorder], dtype=jnp.float64, device=device),
            patch_utils.Ds
        )

        # Differentiation matrix from Legendre exterior to Chebyshev interior+boundary
        self.Nx_cheb = jnp.array(patch_utils.Nx_stack[:, Jorder],
                                 dtype=jnp.float64, device=device)
        # Interpolation from Legendre exterior grid to unique Chebyshev boundary nodes
        self.chebfleg_mat = jnp.array(
            patch_utils.chebfleg_exterior_mat[inds_uni],
            dtype=jnp.float64, device=device
        )
        # Interpolation from Chebyshev boundary nodes to Legendre exterior grid
        self.legfcheb_mat = jnp.array(
            patch_utils.legfcheb_exterior_mat,
            dtype=jnp.float64, device=device
        )

        zz_patch_cheb = patch_utils.zz_int[Jorder]
        zz_patch_leg = patch_utils.zz_ext

        self.nt_cheb = zz_patch_cheb.shape[0]
        self.nx_leg = zz_patch_leg.shape[0]
        self.ni_cheb = Ji_cheb.shape[0]

        # Permutation arrays for interior nodes
        self.inv_perm_ni_cheb = np.argsort(Jorder)
        self.perm_ni_cheb = Jorder

        # Build local coordinate arrays for interior and exterior nodes
        box_centers_padded = jnp.vstack((
            box_centers,
            jnp.zeros((self.nbatch_ext - self.nbatch, self.ndim))
        ))
        self._xxloc_int = jnp.array(
            zz_patch_cheb[None, :, :] + box_centers_padded[:, None, :],
            dtype=jnp.float64, device=device
        )
        self._xxloc_ext = jnp.array(
            zz_patch_leg[None, :, :] + box_centers_padded[:, None, :],
            dtype=jnp.float64, device=device
        )

        # Identity and zero arrays for boundary solves
        self._eye_block = jnp.eye(self.nx_leg, dtype=jnp.float64, device=device)[None, :, :]
        self._zero_ff = jnp.zeros((1, self.nt_cheb, 1), dtype=jnp.float64, device=device)
        self._zero_dir = jnp.zeros((1, self.nx_leg, 1), dtype=jnp.float64, device=device)

        # Build the PDE operator terms once for all leaves
        self.D_stack, self.consts, self.c_fns = build_pdo_terms(
            pdo, Ds, self.ndim, self.nt_cheb, device
        )

    def Aloc(self):
        """
        Return the local PDE operator evaluated at interior nodes
        for the first batch entry (used for diagnostics).
        """
        # Note: slicing with [:, self.nbatch] yields the first (only) interior block
        return _get_Aloc(
            self._xxloc_int,
            self.D_stack,
            self.consts,
            self.c_fns
        )[:, self.nbatch]

    @property
    def xxloc_int(self):
        """
        Return the interior node coordinates (batch, ni_cheb, ndim).
        """
        return self._xxloc_int[:self.nbatch, self.inv_perm_ni_cheb]

    @property
    def xxloc_ext(self):
        """
        Return the exterior (boundary) node coordinates (batch, nx_leg, ndim).
        """
        return self._xxloc_ext[:self.nbatch]

    def solve_dir(self, uu_dir, ff_body=None):
        """
        Solve the interior Dirichlet problem for each leaf in chunks.

        Arguments:
          uu_dir:   Array with Dirichlet data of shape (batch, nx_leg, nrhs)
                    or (nx_leg, nrhs) for a single batch.
          ff_body:  Optional body load of shape (batch, nt_cheb, nrhs)
                    or (nt_cheb, nrhs). If None, assumed zero.

        Returns:
          Array of shape (batch, nt_cheb, nrhs) containing the solution at Chebyshev nodes.
        """
        device = self._xxloc_int.device
        nrhs = uu_dir.shape[-1]

        # Ensure batch dimension
        if self.nbatch == 1 and uu_dir.ndim == 2:
            uu_dir = uu_dir[None, :, :]

        # Default ff_body to zero array if not provided
        if ff_body is None:
            ff_body = np.zeros((self.nbatch, self.nt_cheb, nrhs))
        elif self.nbatch == 1 and ff_body.ndim == 2:
            ff_body = ff_body[None, :, :]

        assert ff_body.shape[0] == self.nbatch
        assert uu_dir.shape[0] == self.nbatch

        # Pad uu_dir and ff_body to match nbatch_ext
        uu_ext = np.zeros((self.nbatch_ext, self.nx_leg, nrhs))
        uu_ext[:self.nbatch] = uu_dir

        ff_ext = np.zeros((self.nbatch_ext, self.nt_cheb, nrhs))
        ff_ext[:self.nbatch] = ff_body[:, self.perm_ni_cheb]

        out_host = np.zeros((self.nbatch, self.nt_cheb, nrhs))
        nchunks = self.nbatch_ext // self.chunk_size

        # Process leaves in chunks to fit into device memory
        for i in range(nchunks):
            start = i * self.chunk_size
            end = start + self.chunk_size

            xx_chunk = self._xxloc_int[start:end]
            uu_chunk = jnp.array(uu_ext[start:end], device=device)
            ff_chunk = jnp.array(ff_ext[start:end], device=device)

            frag_dev = solve_dir_helper(
                xx_chunk,
                uu_chunk,
                ff_chunk,
                self.chebfleg_mat,  # (nb_cheb, nx_leg)
                self.Nx_cheb,       # (nx_leg, nt_cheb)
                self.D_stack,       # (N_terms, nt_cheb, nt_cheb)
                self.consts,        # (N_terms,)
                self.c_fns,         # tuple of callables
                self.ni_cheb        # int
            )  # -> (chunk_size, nt_cheb, nrhs)

            real_end = min(end, self.nbatch)
            length = real_end - start

            if length > 0:
                # Reorder interior nodes back to original ordering
                frag_chunk_host = np.array(frag_dev)[:length, self.inv_perm_ni_cheb]
                out_host[start:real_end, :, :] = frag_chunk_host

        return out_host

    def reduce_body_load(self, ff_body):
        """
        Compute the Neumann boundary load on each leaf due to an interior body force.

        Arguments:
          ff_body:  Array of body load values, shape (batch, nt_cheb, nrhs)
                    or (nt_cheb, nrhs) for a single batch.

        Returns:
          Array of shape (batch, nx_leg, nrhs) containing boundary load contributions.
        """
        if self.nbatch == 1 and ff_body.ndim == 2:
            ff_body = ff_body[None, :, :]

        assert ff_body.shape[0] == self.nbatch
        nrhs = ff_body.shape[-1]
        device = self._xxloc_int.device

        # Pad ff_body to match nbatch_ext and reorder interior nodes
        ff_ext = np.zeros((self.nbatch_ext, self.nt_cheb, nrhs))
        ff_ext[:self.nbatch] = ff_body[:, self.perm_ni_cheb]

        out_host = np.zeros((self.nbatch, self.nx_leg, nrhs))
        nchunks = self.nbatch_ext // self.chunk_size

        for i in range(nchunks):
            start = i * self.chunk_size
            end = start + self.chunk_size

            xx_chunk = self._xxloc_int[start:end]
            ff_chunk = jnp.array(ff_ext[start:end], device=device)

            frag_dev = reduce_chunk_body_load(
                xx_chunk,
                self._zero_dir,
                ff_chunk,
                self.legfcheb_mat,
                self.chebfleg_mat,   # (nb_cheb, nx_leg)
                self.Nx_cheb,       # (nx_leg, nt_cheb)
                self.D_stack,       # (N_terms, nt_cheb, nt_cheb)
                self.consts,        # (N_terms,)
                self.c_fns,         # tuple of callables
                self.ni_cheb        # int
            )  # -> (chunk_size, nx_leg, nrhs)

            real_end = min(end, self.nbatch)
            length = real_end - start

            if length > 0:
                out_host[start:real_end, :, :] = np.array(frag_dev[:length])

        return out_host

    def DtN(self) -> np.ndarray:
        """
        Compute the Dirichlet-to-Neumann (DtN) matrix for each leaf.

        Returns:
          Array of shape (batch, nx_leg, nx_leg), where each slice is the DtN map for one leaf.
        """
        device = self._xxloc_int.device

        # Pre-allocate host buffer
        out_host = np.zeros((self.nbatch, self.nx_leg, self.nx_leg), dtype=np.float64)
        nchunks = self.nbatch_ext // self.chunk_size

        for i in range(nchunks):
            start = i * self.chunk_size
            end = start + self.chunk_size

            # Slice local coordinates for this chunk
            xx_chunk = self._xxloc_int[start:end]

            # Compute the DtN for this chunk with identity Dirichlet boundary and zero body force
            frag_dev = compute_chunk_DtN(
                xx_chunk,
                self._eye_block,   # (1, nx_leg, nx_leg)
                self._zero_ff,     # (1, nt_cheb, 1)
                self.legfcheb_mat,
                self.chebfleg_mat, # (nb_cheb, nx_leg)
                self.Nx_cheb,      # (nx_leg, nt_cheb)
                self.D_stack,      # (N_terms, nt_cheb, nt_cheb)
                self.consts,       # (N_terms,)
                self.c_fns,        # tuple of callables
                self.ni_cheb       # int
            )  # -> (chunk_size, nx_leg, nx_leg)

            real_end = min(end, self.nbatch)
            length = real_end - start

            if length > 0:
                out_host[start:real_end, :, :] = np.array(frag_dev[:length])

        return out_host
