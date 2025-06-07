import numpy as np

from hps.hps_subdomain import LeafSubdomain
from hps.hps_patch_utils import PatchUtils
from scipy.sparse import block_diag
from hps.sparse_utils import CSRBuilder
from hps.pde_solver import AbstractPDESolver
from time import time

def get_leaf_DtNs(pdo, box_geom, a, p, verbose):
    """
    Partition the computational domain into leaf subdomains, compute their centers,
    and instantiate a LeafSubdomain object that can compute per‐leaf DtN maps.

    Parameters:
    - pdo:        PDE operator object containing coefficient functions.
    - box_geom:   Array of shape (2, ndim) giving lower‐ and upper‐bounds of the domain.
    - a:          Array of length ndim or scalar giving half‐width of each patch in each dimension.
    - p:          Polynomial degree (number of Chebyshev nodes per direction) for each leaf.
    - verbose:    If True, print debug information about device and chunk size.

    Returns:
    - npan_dim:           ndarray of ints giving number of panels/patches per dimension.
    - leaf_subdomains:    LeafSubdomain instance containing all leaf information.
    """
    # Compute the total length in each dimension
    len_dim = box_geom[1] - box_geom[0]
    # Calculate how many patches fit along each dimension (must be integer)
    npan_dim = np.round(len_dim / (2 * a)).astype(int)
    # Verify that npan_dim * (2*a) exactly matches len_dim (up to numerical tolerance)
    assert np.linalg.norm(npan_dim * (2 * a) - len_dim, ord=2) < 1e-14

    ndim = npan_dim.shape[0]
    # Initialize patch utilities (handles Chebyshev grids, differentiation matrices, etc.)
    patch_utils = PatchUtils(a, p, ndim=ndim)
    # Prepare an array to hold the center of each leaf patch (flattened ordering)
    box_centers = np.zeros((np.prod(npan_dim), ndim))

    if ndim == 2:
        # Loop over 2D grid of patches
        for j in range(npan_dim[1]):
            for i in range(npan_dim[0]):
                # Compute lower‐corner (root) location of current patch
                root_loc = np.zeros(ndim)
                root_loc[0] = 2 * a[0] * i + box_geom[0, 0]
                root_loc[1] = 2 * a[1] * j + box_geom[0, 1]

                # Compute flattened index of this patch
                box_ind = i + j * npan_dim[0]
                # Center of patch = lower corner + a (half‐width)
                box_centers[box_ind] = root_loc + a

    else:
        # 3D case: loop over k, j, i
        for k in range(npan_dim[2]):
            for j in range(npan_dim[1]):
                for i in range(npan_dim[0]):
                    root_loc = np.zeros(ndim)
                    root_loc[0] = 2 * a[0] * i + box_geom[0, 0]
                    root_loc[1] = 2 * a[1] * j + box_geom[0, 1]
                    root_loc[2] = 2 * a[2] * k + box_geom[0, 2]

                    box_ind = i + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]
                    box_centers[box_ind] = root_loc + a

    # Return number of patches per dimension and the LeafSubdomain object
    return npan_dim, box_centers, patch_utils

def get_duplicated_interior_points_2d(p, npan_dim):
    """
    Identify pairs of indices corresponding to duplicated interior Chebyshev points
    between adjacent leaf patches in 2D. These points must be enforced to have
    identical values across patch interfaces.

    Parameters:
    - p:         Number of Chebyshev nodes along one edge of a patch.
    - npan_dim:  Array of length 2 giving number of patches in x and y.

    Returns:
    - Icopy1:    1D array of length N_dup giving source indices (flattened).
    - Icopy2:    1D array of length N_dup giving target indices (flattened).
    """
    # Each patch has p nodes along each boundary side
    size_bnd = p
    # Each patch's boundary is subdivided into 4 edges, each with p points
    size_ext = 4 * size_bnd

    # Preallocate storage for maximum possible number of duplicated points
    total_possible = np.prod(npan_dim) * size_ext
    Icopy1 = np.zeros(total_possible, dtype=int)
    Icopy2 = np.zeros(total_possible, dtype=int)
    offset = 0

    # Loop over all patches in row‐major order
    for j in range(npan_dim[1]):
        for i in range(npan_dim[0]):
            curr_box = i + j * npan_dim[0]

            # If there is a patch to the left (i > 0), match its right edge to our left edge
            if i > 0:
                prev_box = (i - 1) + j * npan_dim[0]
                # Right boundary of previous box: indices [1*p : 2*p) in that patch's flattened boundary
                Icopy1[offset: offset + size_bnd] = np.arange(size_bnd) + prev_box * size_ext + 1 * size_bnd
                # Left boundary of current box: indices [0*p : 1*p)
                Icopy2[offset: offset + size_bnd] = np.arange(size_bnd) + curr_box * size_ext + 0 * size_bnd
                offset += size_bnd

            # If there is a patch below (j > 0), match its top edge to our bottom edge
            if j > 0:
                prev_box = i + (j - 1) * npan_dim[0]
                # Top boundary of previous box: indices [3*p : 4*p)
                Icopy1[offset: offset + size_bnd] = np.arange(size_bnd) + prev_box * size_ext + 3 * size_bnd
                # Bottom boundary of current box: indices [2*p : 3*p)
                Icopy2[offset: offset + size_bnd] = np.arange(size_bnd) + curr_box * size_ext + 2 * size_bnd
                offset += size_bnd

    # Only return the filled portion of the arrays
    return Icopy1[:offset], Icopy2[:offset]


def get_duplicated_interior_points_3d(p, npan_dim):
    """
    Identify pairs of indices corresponding to duplicated interior Chebyshev points
    between adjacent leaf patches in 3D. These points must be enforced to have
    identical values across patch interfaces.

    Parameters:
    - p:         Number of Chebyshev nodes along one edge of a patch (so p^2 per face).
    - npan_dim:  Array of length 3 giving number of patches in x, y, z.

    Returns:
    - Icopy1:    1D array of length N_dup giving source indices (flattened).
    - Icopy2:    1D array of length N_dup giving target indices (flattened).
    """
    size_bnd = p**2
    # Each patch has 6 faces, each with p^2 interior points
    size_ext = 6 * size_bnd

    total_possible = np.prod(npan_dim) * size_ext
    Icopy1 = np.zeros(total_possible, dtype=int)
    Icopy2 = np.zeros(total_possible, dtype=int)
    offset = 0

    # Loop over 3D grid of patches
    for k in range(npan_dim[2]):
        for j in range(npan_dim[1]):
            for i in range(npan_dim[0]):
                curr_box = i + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]

                # If there is a patch to the left (i > 0), match its right face to our left face
                if i > 0:
                    prev_box = (i - 1) + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]
                    # Right face of previous box: indices [1*p^2 : 2*p^2)
                    Icopy1[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + prev_box * size_ext + 1 * size_bnd
                    )
                    # Left face of current box: indices [0*p^2 : 1*p^2)
                    Icopy2[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + curr_box * size_ext + 0 * size_bnd
                    )
                    offset += size_bnd

                # If there is a patch behind (j > 0), match its front face to our back face
                if j > 0:
                    prev_box = i + (j - 1) * npan_dim[0] + k * npan_dim[0] * npan_dim[1]
                    # Front face of previous box: indices [3*p^2 : 4*p^2)
                    Icopy1[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + prev_box * size_ext + 3 * size_bnd
                    )
                    # Back face of current box: indices [2*p^2 : 3*p^2)
                    Icopy2[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + curr_box * size_ext + 2 * size_bnd
                    )
                    offset += size_bnd

                # If there is a patch below (k > 0), match its top face to our bottom face
                if k > 0:
                    prev_box = i + j * npan_dim[0] + (k - 1) * npan_dim[0] * npan_dim[1]
                    # Top face of previous box: indices [5*p^2 : 6*p^2)
                    Icopy1[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + prev_box * size_ext + 5 * size_bnd
                    )
                    # Bottom face of current box: indices [4*p^2 : 5*p^2)
                    Icopy2[offset: offset + size_bnd] = (
                        np.arange(size_bnd) + curr_box * size_ext + 4 * size_bnd
                    )
                    offset += size_bnd

    # Return only the filled entries
    return Icopy1[:offset], Icopy2[:offset]


# ------------------------------------------------------------------------------
# HPS Multidomain Solver
# ------------------------------------------------------------------------------
class HPSMultidomain(AbstractPDESolver):
    def __init__(self, pdo, geom, a, p, verbose=False):
        """
        Initialize the HPS multidomain solver by:
          1. Determining patch layout and centers.
          2. Building leaf subdomains and computing their DtN maps.
          3. Assembling global sparse Schur complements for interior‐interior, interior‐boundary, etc.

        Parameters:
        - pdo:     PDE operator object with coefficient functions.
        - geom:    Object representing the global domain; must have attribute `bounds` of shape (2, ndim).
        - a:       Scalar or length‐ndim array specifying half‐width of each patch.
        - p:       Polynomial degree for Chebyshev discretization on each patch.
        - verbose: If True, print timing and device information.
        """
        # Determine problem dimension from geometry bounds
        ndim = geom.bounds.shape[-1]

        # Ensure `a` is an array of length `ndim`
        a = np.array(a, dtype=float)
        if a.ndim == 0:
            # If scalar, replicate to all dimensions
            a = np.full(ndim, a)
        elif a.ndim == 1 and a.shape[0] == ndim:
            a = a.copy()
        else:
            raise ValueError(f"'a' must be a scalar or length-{ndim} array; got shape {a.shape}")

        # Store basic properties
        self._box_geom = geom.bounds
        self._geom     = geom
        self._p        = p
        self.pdo       = pdo

        # Partition domain into leaves and build LeafSubdomain
        self.npan_dim, self.box_centers, self.patch_utils = get_leaf_DtNs(pdo, self._box_geom, a, p, verbose)

        leaf_subdomains = LeafSubdomain(self.box_centers, self.pdo, self.patch_utils,verbose)
        # Retrieve exterior (boundary) and interior coordinates for all leaves
        xxext_list = leaf_subdomains.xxloc_ext  # shape (nbatch, nx_leg, ndim)
        xxint_list = leaf_subdomains.xxloc_int  # shape (nbatch, ni_cheb, ndim)
        nbatch     = leaf_subdomains.nbatch

        # Flatten coordinates for global indexing
        self._XX = xxext_list.reshape(xxext_list.shape[0] * xxext_list.shape[1], ndim)
        self._XXfull = xxint_list.reshape(xxint_list.shape[0] * xxint_list.shape[1], ndim)

        # Build index arrays for matching duplicated interface points
        if ndim == 2:
            self._Jcopy1, self._Jcopy2 = get_duplicated_interior_points_2d(p, self.npan_dim)
        elif ndim == 3:
            self._Jcopy1, self._Jcopy2 = get_duplicated_interior_points_3d(p, self.npan_dim)
        else:
            raise ValueError("Unsupported dimension; must be 2 or 3.")

        # Jx: indices of boundary DOFs that are not duplicated
        self._Jx = np.setdiff1d(np.arange(self._XX.shape[0]), np.union1d(self._Jcopy1, self._Jcopy2))
        # Ji: indices of one copy of interior interface nodes
        self._Ji = self._Jcopy1

        # Time DtN computation
        tic = time()
        DtN_list = leaf_subdomains.DtN()  # Compute DtN for each leaf
        toc_dtn = time() - tic

        # Build a block‐diagonal sparse matrix whose blocks are per‐leaf DtN maps
        A = block_diag(tuple(DtN_list), format='csr')
        # Discard DtN_list to free memory
        del DtN_list

        # Time sparse Schur complement assembly
        tic = time()

        # Build Aii: interactions among duplicated interior nodes
        Aii = CSRBuilder(self._Jcopy1.shape[0], self._Jcopy1.shape[0], A.nnz)
        Aii.add_data(A[self._Jcopy1][:, self._Jcopy1])  # block from copy1→copy1
        Aii.add_data(A[self._Jcopy1][:, self._Jcopy2])  # block from copy1→copy2
        Aii.add_data(A[self._Jcopy2][:, self._Jcopy1])  # block from copy2→copy1
        Aii.add_data(A[self._Jcopy2][:, self._Jcopy2])  # block from copy2→copy2
        Aii = Aii.tocsr()

        # Build Aix: interactions from duplicated interior to unique boundary
        Aix = CSRBuilder(self._Jcopy1.shape[0], self._Jx.shape[0], A.nnz)
        Aix.add_data(A[self._Jcopy1][:, self._Jx])
        Aix.add_data(A[self._Jcopy2][:, self._Jx])
        Aix = Aix.tocsr()

        # Build Axi: interactions from unique boundary to duplicated interior
        Axi = CSRBuilder(self._Jx.shape[0], self._Jcopy1.shape[0], A.nnz)
        Axi.add_data(A[self._Jx][:, self._Jcopy1])
        Axi.add_data(A[self._Jx][:, self._Jcopy2])
        Axi = Axi.tocsr()

        toc_sparse = time() - tic

        # Store assembled sub‐blocks and timing stats
        self._Aii = Aii
        self._Aix = Aix
        self._Axi = Axi
        self._Axx = A[self._Jx][:, self._Jx]  # boundary→boundary interactions
        self.stats = {'toc_sparse': toc_sparse, 'toc_dtn': toc_dtn}

    def solve_dir_full(self, uu_dir, ff_body=None):
        """
        Solve the PDE on the full multidomain:
          1. Reduce body loads to interface boundaries.
          2. Solve global Schur system for boundary unknowns.
          3. Solve each leaf's Dirichlet problem given boundary values and interior body load.

        Parameters:
        - uu_dir:   Array of Dirichlet data on all global boundary DOFs, shape (n_bnd, nrhs).
        - ff_body:  Optional interior body load given at all Chebyshev nodes (global), shape (n_int, nrhs).

        Returns:
        - uu_sol_int:  Solution values at all interior Chebyshev nodes, shape (n_int, nrhs).
        """
        assert uu_dir.ndim == 2
        nrhs = uu_dir.shape[-1]

        # If no interior body load given, assume zero
        if ff_body is None:
            ff_body = np.zeros((self._XXfull.shape[0], nrhs))

        subdomains = LeafSubdomain(self.box_centers, self.pdo, self.patch_utils)

        # Reshape global ff_body into per‐leaf (batch, nt_cheb, nrhs)
        ff_body = ff_body.reshape(subdomains.nbatch, subdomains.nt_cheb, nrhs)
        # Compute reduced boundary load from interior body on each leaf
        ff_red = subdomains.reduce_body_load(ff_body)
        # Flatten reduced values to global ordering
        ff_red = ff_red.reshape((self._XX.shape[0], nrhs))

        # For duplicated interface DOFs, sum contributions from both sides
        ff_dup = ff_red[self._Jcopy1] + ff_red[self._Jcopy2]

        # Build global boundary RHS (including body‐load contributions)
        uu_sol_bnd = np.zeros((self._XX.shape[0], nrhs))
        # Solve the Schur system implicitly via the method solve_dir (assumes Axx, Axi, etc. used inside)
        uu_sol_bnd[self._Jcopy1] = self.solve_dir(uu_dir, ff_dup)
        # Enforce uniqueness of duplicated DOFs
        uu_sol_bnd[self._Jcopy2] = uu_sol_bnd[self._Jcopy1]
        # Boundary DOFs that are not interfaces equal the given Dirichlet data
        uu_sol_bnd[self._Jx] = uu_dir

        # Reshape to per‐leaf for final interior solve
        uu_sol_bnd = uu_sol_bnd.reshape(subdomains.nbatch, subdomains.nx_leg, nrhs)
        # Solve interior Dirichlet problems on each leaf
        uu_sol_int = subdomains.solve_dir(uu_sol_bnd, ff_body)

        # Return flattened interior solution
        return uu_sol_int.reshape(self._XXfull.shape[0], nrhs)

    @property
    def npoints_dim(self):
        """Return the total number of Chebyshev points per dimension (npan_dim * p)."""
        return self.npan_dim * self.p

    @property
    def geom(self):
        """Access the geometry object."""
        return self._geom

    @property
    def XX(self):
        """Return the array of all boundary (exterior) node coordinates, shape (n_bnd, ndim)."""
        return self._XX

    @property
    def Ji(self):
        """Return indices of one copy of interior interface DOFs."""
        return self._Ji

    @property
    def Jx(self):
        """Return indices of unique boundary DOFs (not duplicated)."""
        return self._Jx

    @property
    def Aii(self):
        """Return the sparse block corresponding to interior‐interior interactions (duplicated interfaces)."""
        return self._Aii

    @property
    def Aix(self):
        """Return the sparse block corresponding to interior‐boundary interactions (interface to unique boundary)."""
        return self._Aix

    @property
    def Axi(self):
        """Return the sparse block corresponding to boundary‐interior interactions (unique boundary to interface)."""
        return self._Axi

    @property
    def Axx(self):
        """Return the sparse block corresponding to boundary‐boundary interactions (unique boundary only)."""
        return self._Axx

    @property
    def p(self):
        """Return the polynomial degree used for each patch."""
        return self._p
