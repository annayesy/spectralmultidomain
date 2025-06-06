import numpy as np

from scipy.sparse import kron, diags, block_diag
from scipy.sparse import eye as speye, linalg as spla
from hps.pde_solver import AbstractPDESolver

#######          GRID FOR 2D and 3D      #########
# Given box geometry and mesh spacing h, generate grid points in a rectangle (or box).
def grid(box_geom, h):
    """
    Create a regular Cartesian grid over the domain defined by box_geom with spacing h.

    Parameters:
    - box_geom: array of shape (2, d), where box_geom[0] is the lower corner and
                box_geom[1] is the upper corner in each dimension.
    - h:        mesh spacing (scalar)

    Returns:
    - ns:   array of length d with the number of grid points in each dimension
    - XX:   array of shape (N, d) containing all grid coordinates, flattened in row-major order
    - Ji:   indices of interior points (strictly away from each boundary by at least ~0.25*h)
    - Jx:   indices of boundary points (complement of Ji)
    """
    # Determine spatial dimension from box_geom
    d = box_geom.shape[-1]

    # Build 1D arrays of coordinates in each dimension with spacing h
    xx0 = np.arange(box_geom[0, 0], box_geom[1, 0] + 0.5 * h, h)
    xx1 = np.arange(box_geom[0, 1], box_geom[1, 1] + 0.5 * h, h)
    if d == 3:
        xx2 = np.arange(box_geom[0, 2], box_geom[1, 2] + 0.5 * h, h)

    # Count the number of points in each 1D grid
    if d == 2:
        ns = np.array([xx0.shape[0], xx1.shape[0]], dtype=int)
    else:
        ns = np.array([xx0.shape[0], xx1.shape[0], xx2.shape[0]], dtype=int)

    # Form a meshgrid and flatten to list all coordinates
    if d == 2:
        # 2D: meshgrid of shape (n0, n1)
        XX0, XX1 = np.meshgrid(xx0, xx1, indexing='ij')
        # Stack and flatten to shape (2, n0*n1), then transpose to (n0*n1, 2)
        XX = np.vstack((XX0.flatten(), XX1.flatten()))
    else:
        # 3D: meshgrid of shape (n0, n1, n2)
        XX0, XX1, XX2 = np.meshgrid(xx0, xx1, xx2, indexing='ij')
        XX = np.vstack((XX0.flatten(), XX1.flatten(), XX2.flatten()))

    XX = XX.T  # Now shape is (N, d) with N = product of ns
    # Compute approximate grid spacing using the first two points
    hmin = np.max(XX[1] - XX[0])

    # Build Boolean masks to identify interior points at least ~0.25*h from boundary
    cond0 = np.logical_and(
        XX[:, 0] > box_geom[0, 0] + 0.25 * hmin,
        XX[:, 0] < box_geom[1, 0] - 0.25 * hmin
    )
    cond1 = np.logical_and(
        XX[:, 1] > box_geom[0, 1] + 0.25 * hmin,
        XX[:, 1] < box_geom[1, 1] - 0.25 * hmin
    )
    if d == 3:
        cond2 = np.logical_and(
            XX[:, 2] > box_geom[0, 2] + 0.25 * hmin,
            XX[:, 2] < box_geom[1, 2] - 0.25 * hmin
        )
    else:
        cond2 = True  # In 2D, ignore third coordinate

    # Interior indices are those satisfying all cond0, cond1, cond2
    Ji = np.where(np.logical_and(np.logical_and(cond0, cond1), cond2))[0]
    # Boundary indices Jx are the complement of Ji
    Jx = np.setdiff1d(np.arange(XX.shape[0]), Ji)

    return ns, XX, Ji, Jx


def assemble_sparse(pdo_op, npoints_dim, XX):
    """
    Assemble a sparse finite-difference approximation of the PDO operator over the grid.

    This constructs the global sparse matrix A = discretization of -div(C grad u) + lower-order terms.

    Parameters:
    - pdo_op:       an object with callable attributes c11, c22 (and optionally c12, c1, c2, c)
                    which evaluate coefficient functions at any set of points XX.
    - npoints_dim:  array of length d giving number of grid points in each dimension.
    - XX:           array of shape (N, d) listing coordinates of all grid points.

    Returns:
    - A: sparse matrix of shape (N, N) representing the discretized PDO on the grid.
    """
    d = XX.shape[-1]                # spatial dimension
    # Recompute h from the first two points (assumes uniform spacing)
    h = np.max(XX[1] - XX[0])

    if d == 2:
        # 2D case: build 1D second-derivative and first-derivative matrices in each direction
        n0, n1 = npoints_dim

        # Second-derivative in x (d0sq) and y (d1sq) with Dirichlet stencil [1, -2, 1] / h^2
        d0sq = (1 / (h * h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0), format='csc')
        d1sq = (1 / (h * h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1), format='csc')

        # First-derivative in x (d0) and y (d1) with centered stencil [-1, 0, +1] / (2h)
        d0 = (1 / (2 * h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n0, n0), format='csc')
        d1 = (1 / (2 * h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n1, n1), format='csc')

        # Kronecker products to lift 1D to 2D: 
        #    D00 approximates ∂²/∂x² over 2D grid, D11 approximates ∂²/∂y²
        D00 = kron(d0sq, speye(n1))
        D11 = kron(speye(n0), d1sq)

        # Diagonal coefficient matrices C00 = diag(c11(X)), C11 = diag(c22(X))
        c00_diag = np.array(pdo_op.c11(XX)).reshape(n0 * n1,)
        C00 = diags(c00_diag, 0, shape=(n0 * n1, n0 * n1))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(n0 * n1,)
        C11 = diags(c11_diag, 0, shape=(n0 * n1, n0 * n1))

        # Assemble the main second-derivative part: -C00 * D00 - C11 * D11
        A = -C00 @ D00 - C11 @ D11

        # If mixed second-derivative term c12 exists, add -2 * diag(c12) * (d0 ⊗ d1)
        if pdo_op.c12 is not None:
            c_diag = np.array(pdo_op.c12(XX)).reshape(n0 * n1,)
            S = diags(c_diag, 0, shape=(n0 * n1, n0 * n1))
            D01 = kron(d0, d1)  # approximates ∂²/∂x∂y
            A -= 2 * S @ D01

        # If first-order term c1 exists, add diag(c1) * (d0 ⊗ I)
        if pdo_op.c1 is not None:
            c_diag = np.array(pdo_op.c1(XX)).reshape(n0 * n1,)
            S = diags(c_diag, 0, shape=(n0 * n1, n0 * n1))
            D0 = kron(d0, speye(n1))  # approximates ∂/∂x
            A += S @ D0

        # If first-order term c2 exists, add diag(c2) * (I ⊗ d1)
        if pdo_op.c2 is not None:
            c_diag = np.array(pdo_op.c2(XX)).reshape(n0 * n1,)
            S = diags(c_diag, 0, shape=(n0 * n1, n0 * n1))
            D_ = kron(speye(n0), d1)  # approximates ∂/∂y
            A += S @ D_

        # If zeroth-order term c exists, add diag(c)
        if pdo_op.c is not None:
            c_diag = np.array(pdo_op.c(XX)).reshape(n0 * n1,)
            S = diags(c_diag, 0, shape=(n0 * n1, n0 * n1))
            A += S

    elif d == 3:
        # 3D case: similar but with three spatial directions
        n0, n1, n2 = npoints_dim

        # 1D second-derivative matrices in x, y, z
        d0sq = (1 / (h * h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0), format='csc')
        d1sq = (1 / (h * h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1), format='csc')
        d2sq = (1 / (h * h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2), format='csc')

        # Lift 1D to 3D via Kron: 
        # D00 = ∂²/∂x² ⊗ I_y ⊗ I_z, D11 = I_x ⊗ ∂²/∂y² ⊗ I_z, D22 = I_x ⊗ I_y ⊗ ∂²/∂z²
        D00 = kron(d0sq, kron(speye(n1), speye(n2)))
        D11 = kron(speye(n0), kron(d1sq, speye(n2)))
        D22 = kron(speye(n0), kron(speye(n1), d2sq))

        # Build diagonal coefficient matrices for second derivatives:
        N = n0 * n1 * n2
        c00_diag = np.array(pdo_op.c11(XX)).reshape(N,)
        C00 = diags(c00_diag, 0, shape=(N, N))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(N,)
        C11 = diags(c11_diag, 0, shape=(N, N))
        c22_diag = np.array(pdo_op.c33(XX)).reshape(N,)
        C22 = diags(c22_diag, 0, shape=(N, N))

        # Assemble main second-derivative part
        A = -C00 @ D00 - C11 @ D11 - C22 @ D22

        # In this simplified version, we assume no mixed or first-order terms exist in 3D
        if (
            (pdo_op.c1 is not None)
            or (pdo_op.c2 is not None)
            or (pdo_op.c3 is not None)
            or (pdo_op.c12 is not None)
            or (pdo_op.c13 is not None)
            or (pdo_op.c23 is not None)
        ):
            raise ValueError("Mixed or first-order terms not handled in 3D FD assembly")

        # If zeroth-order term c exists, add diag(c)
        if pdo_op.c is not None:
            c_diag = np.array(pdo_op.c(XX)).reshape(N,)
            S = diags(c_diag, 0, shape=(N, N))
            A += S

    return A


#######          FINITE-DIFFERENCE DISCRETIZATION CLASS      #########
# Inherits AbstractPDESolver, providing the required interfaces for PDE solve routines.
class FDDiscretization(AbstractPDESolver):
    """
    A finite-difference-based PDE solver in 2D or 3D over a regular grid.

    This class implements the AbstractPDESolver interface by providing:
      - XX:    coordinates of all grid points
      - p:     stencil degree (here fixed to 2 for second-order FD)
      - geom:  geometry object with domain bounds
      - npoints_dim: number of points in each dimension
      - Ji, Jx: indices of interior vs boundary points
      - Aii, Aix, Axx, Axi: subblocks of the global matrix A
    """

    def __init__(self, pdo, geom, h, kh=0):
        """
        Parameters:
        - pdo:   PDO operator object with coefficient functions (c11, c22, etc.)
        - geom:  geometry object providing bounds
        - h:     mesh spacing
        - kh:    wavenumber*scale (optional; not used here aside from pdo)
        """
        self._geom = geom
        # Build grid: get number of points per dim, all coordinates, interior indices, boundary indices
        self._npoints_dim, self._XX, self._Ji, self._Jx = grid(self.geom.bounds, h)
        self.pdo = pdo

        # Assemble the global sparse operator matrix A over all grid points
        self.A = assemble_sparse(self.pdo, self.npoints_dim, self._XX)

    @property
    def XX(self):
        """
        Return all grid coordinates, shape (N, d).
        """
        return self._XX

    @property
    def p(self):
        """
        Return FD stencil order; here fixed to 2 (second-order FD).
        """
        return 2

    @property
    def geom(self):
        """
        Return the geometry object (with bounds).
        """
        return self._geom

    @property
    def npoints_dim(self):
        """
        Return the number of grid points in each dimension.
        """
        return self._npoints_dim

    @property
    def Ji(self):
        """
        Return indices of interior grid points (used for Schur complement).
        """
        return self._Ji

    @property
    def Jx(self):
        """
        Return indices of boundary grid points.
        """
        return self._Jx

    @property
    def Aii(self):
        """
        Return the submatrix A[Ji, Ji], coupling interior points among themselves.
        """
        return self.A[self.Ji][:, self.Ji]

    @property
    def Aix(self):
        """
        Return the submatrix A[Ji, Jx], coupling interior points to boundary points.
        """
        return self.A[self.Ji][:, self.Jx]

    @property
    def Axx(self):
        """
        Return the submatrix A[Jx, Jx], coupling boundary points among themselves.
        """
        return self.A[self.Jx][:, self.Jx]

    @property
    def Axi(self):
        """
        Return the submatrix A[Jx, Ji], coupling boundary points to interior points.
        """
        return self.A[self.Jx][:, self.Ji]
