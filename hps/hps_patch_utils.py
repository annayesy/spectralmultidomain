import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag, lu_factor, lu_solve
from hps.cheb_utils import *
from scipy.linalg import null_space
from scipy.spatial.distance import cdist
import scipy

# ------------------------------------------------------------------------------------
# Named tuples to organize index sets (JJ) for interior and boundary nodes in 2D/3D
# ------------------------------------------------------------------------------------
# JJ_2d stores indices for Left, Right, Down, Up, and Interior points on a 2D Chebyshev leaf.
JJ_2d = namedtuple('JJ_2d', ['Jl', 'Jr', 'Jd', 'Ju', 'Ji'])
# JJext_2d stores indices for exterior points (Left, Right, Down, Up) on a 2D patch.
JJext_2d = namedtuple('JJext_2d', ['Jl', 'Jr', 'Jd', 'Ju'])

# JJ_3d stores indices for Left, Right, Down, Up, Back, Front, and Interior points on a 3D leaf.
JJ_3d = namedtuple('JJ_3d', ['Jl', 'Jr', 'Jd', 'Ju', 'Jb', 'Jf', 'Ji'])
# JJext_3d stores indices for all six faces (Left, Right, Down, Up, Back, Front) of a 3D patch.
JJext_3d = namedtuple('JJext_3d', ['Jl', 'Jr', 'Jd', 'Ju', 'Jb', 'Jf'])


######################## Discretization utils for 2D and 3D ########################

def leaf_discretization_2d(a, q):
    """
    Compute Chebyshev grid points, differentiation operators, and index sets (JJ) for a 2D leaf.

    Parameters:
    - a: length-2 array [a_x, a_y], half-length of the rectangular patch in x and y.
    - q: int, number of Chebyshev nodes per direction (including endpoints).

    Returns:
    - zz: 2×(q^2) array of 2D coordinates of Chebyshev nodes (flattened columns).
    - Ds: Ds_2d namedtuple containing 2D differentiation matrices (D1, D2, D11, D22, D12).
    - JJ: JJ_2d namedtuple with index arrays for Left, Right, Down, Up, Interior nodes (all length q or (q-2)^2).
    - hmin: smallest spacing between Chebyshev nodes in this patch (to identify interior).
    """
    # 1) Get Chebyshev grid and differentiation operators for 2D: zz has shape 2×(q^2)
    zz, Ds = cheb_2d(a, q)

    # 2) Estimate minimum grid spacing hmin among nearest neighbors in x- or y-direction.
    #    We compare differences along first column vs first row for simplicity.
    hmin = min(
        np.max(np.abs(zz[:, 1] - zz[:, 0])),
        np.max(np.abs(zz[:, q] - zz[:, 0]))
    )

    # 3) Identify which nodes lie strictly inside (not on boundary) by checking |x| < a_x - hmin/2, same for y.
    Jc0 = np.abs(zz[0, :]) < a[0] - 0.5 * hmin
    Jc1 = np.abs(zz[1, :]) < a[1] - 0.5 * hmin

    # 4) Identify indices of boundary points on each side:
    #    - Left boundary: x < -a_x + hmin/2
    Jl = np.argwhere(zz[0, :] < -a[0] + 0.5 * hmin).reshape(q,)
    #    - Right boundary: x > +a_x - hmin/2
    Jr = np.argwhere(zz[0, :] > +a[0] - 0.5 * hmin).reshape(q,)
    #    - Down boundary: y < -a_y + hmin/2
    Jd = np.argwhere(zz[1, :] < -a[1] + 0.5 * hmin).reshape(q,)
    #    - Up boundary: y > +a_y - hmin/2
    Ju = np.argwhere(zz[1, :] > +a[1] - 0.5 * hmin).reshape(q,)

    # 5) Interior points: nodes satisfying both Jc0 and Jc1; there are (q-2)^2 of them
    Ji = np.argwhere(np.logical_and(Jc0, Jc1)).reshape((q - 2) ** 2,)

    # 6) Bundle indices into namedtuple
    JJ = JJ_2d(Jl=Jl, Jr=Jr, Jd=Jd, Ju=Ju, Ji=Ji)
    return zz, Ds, JJ, hmin


def ext_discretization_2d(a, p):
    """
    Compute exterior (boundary) nodes for a 2D patch, using p Legendre nodes per edge.

    These points lie on the perimeter of a rectangle of half-lengths a_x, a_y.

    Parameters:
    - a: length-2 array [a_x, a_y], half-length of the patch.
    - p: int, number of Legendre points per edge.

    Returns:
    - zz: 2×(4p) array of 2D coordinates of boundary (Legendre) nodes, ordered [Left, Right, Down, Up].
    - JJ: JJext_2d namedtuple with index arrays for each side: Jl, Jr, Jd, Ju (each length p).
    """
    # 1) Get 1D Legendre nodes on [-1,1] (p points)
    leg_nodes, _ = leggauss(p)

    # 2) Initialize an array to hold 4p boundary points (2 coordinates each)
    zz = np.zeros((2, 4 * p))

    # 3) Fill Left edge: x = -a_x; y = a_y * legendre nodes
    zz[0, 0 * p:1 * p] = -a[0]
    zz[1, 0 * p:1 * p] = a[1] * leg_nodes

    # 4) Fill Right edge: x = +a_x; y = a_y * legendre nodes
    zz[0, 1 * p:2 * p] = +a[0]
    zz[1, 1 * p:2 * p] = a[1] * leg_nodes

    # 5) Fill Down edge: x = a_x * legendre nodes; y = -a_y
    zz[0, 2 * p:3 * p] = a[0] * leg_nodes
    zz[1, 2 * p:3 * p] = -a[1]

    # 6) Fill Up edge: x = a_x * legendre nodes; y = +a_y
    zz[0, 3 * p:4 * p] = a[0] * leg_nodes
    zz[1, 3 * p:4 * p] = +a[1]

    # 7) Index sets for each contiguous block of p points on each side
    JJ = JJext_2d(
        Jl=np.arange(p),
        Jr=np.arange(p, 2 * p),
        Jd=np.arange(2 * p, 3 * p),
        Ju=np.arange(3 * p, 4 * p)
    )
    return zz, JJ


def leaf_discretization_3d(a, q):
    """
    Compute Chebyshev grid points, differentiation operators, and index sets (JJ) for a 3D leaf.

    Parameters:
    - a: length-3 array [a_x, a_y, a_z], half-lengths of the rectangular box.
    - q: int, number of Chebyshev nodes per direction.

    Returns:
    - zz: 3×(q^3) array of 3D coordinates of Chebyshev nodes (flattened).
    - Ds: Ds_3d namedtuple containing 3D differentiation matrices.
    - JJ: JJ_3d namedtuple with indices for Left, Right, Down, Up, Back, Front, and Interior nodes.
    - hmin: smallest spacing among nearest neighbors in x, y, or z direction.
    """
    # 1) Compute 3D Chebyshev grid and differentiation operators
    zz, Ds = cheb_3d(a, q)

    # 2) Estimate minimum spacing by comparing distances between first few points
    hmin = min(
        np.max(np.abs(zz[:, 1] - zz[:, 0])),
        np.max(np.abs(zz[:, q] - zz[:, 0])),
        np.max(np.abs(zz[:, q**2] - zz[:, 0]))
    )

    # 3) Build Boolean masks for interior check: |x|<a_x - hmin/2, |y|<a_y - hmin/2, |z|<a_z - hmin/2
    Jc0 = np.abs(zz[0, :]) < a[0] - 0.5 * hmin
    Jc1 = np.abs(zz[1, :]) < a[1] - 0.5 * hmin
    Jc2 = np.abs(zz[2, :]) < a[2] - 0.5 * hmin

    # 4) Identify boundary indices on each face (each face has q^2 nodes):
    #    - Left (x = -a_x), Right (x = +a_x)
    Jl = np.argwhere(zz[0, :] < -a[0] + 0.5 * hmin).reshape(q**2,)
    Jr = np.argwhere(zz[0, :] > +a[0] - 0.5 * hmin).reshape(q**2,)
    #    - Down (y = -a_y), Up (y = +a_y)
    Jd = np.argwhere(zz[1, :] < -a[1] + 0.5 * hmin).reshape(q**2,)
    Ju = np.argwhere(zz[1, :] > +a[1] - 0.5 * hmin).reshape(q**2,)
    #    - Back (z = -a_z), Front (z = +a_z)
    Jb = np.argwhere(zz[2, :] < -a[2] + 0.5 * hmin).reshape(q**2,)
    Jf = np.argwhere(zz[2, :] > +a[2] - 0.5 * hmin).reshape(q**2,)

    # 5) Interior indices: nodes satisfying all three interior conditions
    Ji = np.argwhere(np.logical_and(Jc0, np.logical_and(Jc1, Jc2))).reshape((q - 2) ** 3,)

    # 6) Bundle indices into namedtuple
    JJ = JJ_3d(Jl=Jl, Jr=Jr, Jd=Jd, Ju=Ju, Jb=Jb, Jf=Jf, Ji=Ji)
    return zz, Ds, JJ, hmin


def ext_discretization_3d(a, p):
    """
    Compute exterior (boundary) nodes for a 3D patch, using p×p Legendre nodes per face.

    These points lie on the six faces of a rectangular box of half-lengths a_x, a_y, a_z.

    Parameters:
    - a: length-3 array [a_x, a_y, a_z], half-lengths of the patch.
    - p: int, number of Legendre points per direction on each face.

    Returns:
    - zz: 3×(6*p^2) array of 3D coordinates of boundary (Legendre) nodes in order [L, R, D, U, B, F].
    - JJ: JJext_3d namedtuple containing index arrays for each face: Jl, Jr, Jd, Ju, Jb, Jf.
    """
    # 1) Get 1D Legendre nodes for p points
    leg_nodes, _ = leggauss(p)
    face_size = p ** 2
    # Allocate array for 6 faces × p^2 nodes each
    zz = np.zeros((3, 6 * face_size))

    # Build index sets for each of the six faces
    JJ = JJext_3d(
        Jl=np.arange(face_size),
        Jr=np.arange(face_size, 2 * face_size),
        Jd=np.arange(2 * face_size, 3 * face_size),
        Ju=np.arange(3 * face_size, 4 * face_size),
        Jb=np.arange(4 * face_size, 5 * face_size),
        Jf=np.arange(5 * face_size, 6 * face_size)
    )

    # 2) Build grids for the Right/Left faces: y-z plane coordinates
    Xtmp, Ytmp = np.meshgrid(a[1] * leg_nodes, a[2] * leg_nodes, indexing='ij')
    zz_grid = np.vstack((Xtmp.flatten(), Ytmp.flatten()))  # shape 2×(p^2)

    # Left face: x = -a_x, y and z vary over legendre grid
    zz[0, JJ.Jl] = -a[0]
    zz[1, JJ.Jl] = zz_grid[0]
    zz[2, JJ.Jl] = zz_grid[1]

    # Right face: x = +a_x, y and z vary over legendre grid
    zz[0, JJ.Jr] = +a[0]
    zz[1, JJ.Jr] = zz_grid[0]
    zz[2, JJ.Jr] = zz_grid[1]

    # 3) Build grids for the Down/Up faces: x-z plane coordinates
    Xtmp, Ytmp = np.meshgrid(a[0] * leg_nodes, a[2] * leg_nodes, indexing='ij')
    zz_grid = np.vstack((Xtmp.flatten(), Ytmp.flatten()))

    # Down face: y = -a_y, x and z vary
    zz[0, JJ.Jd] = zz_grid[0]
    zz[1, JJ.Jd] = -a[1]
    zz[2, JJ.Jd] = zz_grid[1]

    # Up face: y = +a_y, x and z vary
    zz[0, JJ.Ju] = zz_grid[0]
    zz[1, JJ.Ju] = +a[1]
    zz[2, JJ.Ju] = zz_grid[1]

    # 4) Build grids for the Back/Front faces: x-y plane coordinates
    Xtmp, Ytmp = np.meshgrid(a[0] * leg_nodes, a[1] * leg_nodes, indexing='ij')
    zz_grid = np.vstack((Xtmp.flatten(), Ytmp.flatten()))

    # Back face: z = -a_z, x and y vary
    zz[0, JJ.Jb] = zz_grid[0]
    zz[1, JJ.Jb] = zz_grid[1]
    zz[2, JJ.Jb] = -a[2]

    # Front face: z = +a_z, x and y vary
    zz[0, JJ.Jf] = zz_grid[0]
    zz[1, JJ.Jf] = zz_grid[1]
    zz[2, JJ.Jf] = +a[2]

    return zz, JJ


def get_diff_ops(Ds, JJ, d):
    """
    Extract normal derivative operators (D ∂/∂n) at boundary faces for 2D or 3D.

    The returned Nx stacks the outward normal derivatives in the order:
     - 2D: [ -D1 @ left, +D1 @ right, -D2 @ down, +D2 @ up ]
     - 3D: [ -D1@left, +D1@right, -D2@down, +D2@up, -D3@back, +D3@front ]

    Parameters:
    - Ds: Ds_2d or Ds_3d namedtuple containing 1st-derivative matrices D1, D2, (D3).
    - JJ: JJ_2d or JJ_3d namedtuple with indices of each face.
    - d:  int, spatial dimension (2 or 3).

    Returns:
    - Nx: 2D array of shape ((number_of_faces * face_size), q^d) where each block is the normal derivative
          on that face, concatenated in the order described above.
    """
    if d == 2:
        # Extract D1 row-blocks at left (Jl) and right (Jr), D2 at down (Jd) and up (Ju)
        Nl = Ds.D1[JJ.Jl]   # ∂/∂x on left face
        Nr = Ds.D1[JJ.Jr]   # ∂/∂x on right face
        Nd = Ds.D2[JJ.Jd]   # ∂/∂y on down face
        Nu = Ds.D2[JJ.Ju]   # ∂/∂y on up face

        # Apply outward normal: left = -∂/∂x, right = +∂/∂x, down = -∂/∂y, up = +∂/∂y
        Nx = np.concatenate((-Nl, +Nr, -Nd, +Nu))

    else:
        # 3D case: extract D1 for left/right, D2 for down/up, D3 for back/front
        Nl = Ds.D1[JJ.Jl]
        Nr = Ds.D1[JJ.Jr]
        Nd = Ds.D2[JJ.Jd]
        Nu = Ds.D2[JJ.Ju]
        Nb = Ds.D3[JJ.Jb]
        Nf = Ds.D3[JJ.Jf]

        # Sign according to outward normal on each face
        Nx = np.concatenate((-Nl, +Nr, -Nd, +Nu, -Nb, +Nf))
    return Nx


class PatchUtils:
    """
    Utilities for an HPS patch, including interior/exterior node locations,
    differentiation operators on boundaries, and transformation matrices between
    Chebyshev and Legendre collocation.

    Methods and properties:
    - zz_int: internal coordinates (num_points × ndim) of interior Chebyshev nodes.
    - zz_ext: external coordinates (num_points × ndim) of boundary Chebyshev nodes.
    - Nx_stack: stacked normal derivative operators on all boundary faces.
    - legfcheb_exterior_mat: maps Chebyshev boundary data → Legendre boundary data.
    - chebfleg_exterior_mat: maps Legendre boundary data → Chebyshev boundary data with constraints.
    """

    def __init__(self, a, p, ndim=2, q=-1):
        """
        Initialize patch utilities, including discretization of interior and exterior.

        Parameters:
        - a: length-ndim array, half side-lengths of the patch in each dimension.
        - p: polynomial degree (number of Chebyshev nodes minus 2 for interior).
        - ndim: int, 2 or 3 for problem dimension.
        - q: int, number of Chebyshev nodes per direction for interior. If q <= 0,
             defaults to p+2 to ensure two extra layers of boundary nodes.
        """
        self.a = a
        self.p = p
        self.ndim = ndim
        # If q not provided or invalid, take q = p + 2
        self.q = p + 2 if (q <= 0) else q
        assert self.q > self.p, "Interior Chebyshev degree q must exceed boundary degree p."
        # Perform the discretization immediately
        self._discretize()

    def _discretize(self):
        """
        Perform interior and exterior discretization according to ndim.

        - Computes zz_int:  interior Chebyshev coordinates (N_int × ndim).
        - Computes zz_ext:  exterior Chebyshev coordinates (N_ext × ndim).
        - Computes Nx_stack: stacked boundary normal derivative operators on interior nodes.
        """
        if self.ndim == 2:
            # 2D: leaf_discretization_2d → (zz_int, Ds, JJ_int, hmin)
            zz_int, self.Ds, self.JJ_int, self.hmin = leaf_discretization_2d(self.a, self.q)
            # 2D: ext_discretization_2d → (zz_ext, JJ_ext)
            zz_ext, self.JJ_ext = ext_discretization_2d(self.a, self.p)

        elif self.ndim == 3:
            # 3D: leaf_discretization_3d → (zz_int, Ds, JJ_int, hmin)
            zz_int, self.Ds, self.JJ_int, self.hmin = leaf_discretization_3d(self.a, self.q)
            # 3D: ext_discretization_3d → (zz_ext, JJ_ext)
            zz_ext, self.JJ_ext = ext_discretization_3d(self.a, self.p)

        else:
            raise ValueError("Unsupported dimension: must be 2 or 3.")

        # Transpose to get coordinates in row-major form (num_points × ndim)
        self.zz_int = zz_int.T
        self.zz_ext = zz_ext.T

        # Build stacked normal derivative operator for all interior boundary faces
        self.Nx_stack = get_diff_ops(self.Ds, self.JJ_int, self.ndim)

    @property
    def legfcheb_exterior_mat(self):
        """
        Build the matrix that maps Chebyshev boundary data → Legendre boundary data.

        For each exterior face:
          - In 2D: four edges → block diagonal of 4 copies of the 1D legfcheb_matrix(p, q).
          - In 3D: six faces → block diagonal of 6 copies of the 2D legfcheb_matrix_2d(p, q).

        Returns:
        - Mat: scipy.linalg.block_diag(...) of appropriate shape.
        """
        if self.ndim == 2:
            T = legfcheb_matrix(self.p, self.q)  # shape (p, q)
            # Four faces: Left, Right, Down, Up → block_diag to size (4p × 4q)
            return block_diag(T, T, T, T)
        else:
            # 3D: each face is p×p → use 2D transform of shape (p^2, q^2)
            T = legfcheb_matrix_2d(self.p, self.q)
            # Six faces: L, R, D, U, B, F → block_diag to size (6p^2 × 6q^2)
            return block_diag(T, T, T, T, T, T)

    def find_equality_constraints(self, ind_list):
        """
        Find linear constraints enforcing equality of values at duplicated interior nodes
        across adjacent faces, for use in projecting from Legendre to Chebyshev boundary.

        ind_list: list of index arrays (each of length face_size) corresponding
                  to one set of boundary nodes for each face in pairwise comparison.
                  For 2D: [Jl, Jr, Jd, Ju]; for 3D: [Jl, Jr, Jd, Ju, Jb, Jf].

        Returns:
        - constraints: 2D array of shape (num_eqs, total_exterior_points)
                       Each row represents an equation of the form value(face_i, idx) - value(face_j, idx) = 0.
        """
        zz_int = self.zz_int  # interior coordinates (N_int × ndim)
        q = self.q

        if self.ndim == 2:
            face_size = q
            # We expect 4 edges, so (4 choose 2)=6 pairs, but only interior-adjacent give nonzero matches.
            constraints = np.zeros((4, 4 * face_size))
        else:
            face_size = q**2
            # In 3D, there are 6 faces → many pairs; we allocate for 12 matching equations
            constraints = np.zeros((12, 6 * face_size))

        offset = 0
        # Compare each pair of face index lists (j, k) from ind_list
        for j in range(len(ind_list)):
            for k in range(j + 1, len(ind_list)):
                # Compute distances between all points on face j vs face k
                D = cdist(zz_int[ind_list[j]], zz_int[ind_list[k]])
                # Find pairs where distance is zero (exactly matching interior nodes)
                inds_zero = np.where(D == 0)

                if inds_zero[0].shape[0] == 0:
                    # No overlapping nodes between these two faces
                    continue
                else:
                    # For each matching pair (i, j), add +1 at face j index, -1 at face k index
                    constraints[offset, j * face_size + inds_zero[0]] = +1
                    constraints[offset, k * face_size + inds_zero[1]] = -1
                    offset += 1

        # Ensure we filled all rows
        assert offset == constraints.shape[0]
        return constraints

    @property
    def chebfleg_exterior_mat(self):
        """
        Build the matrix that maps Legendre boundary data → Chebyshev boundary data,
        while enforcing equality constraints at duplicated interior nodes.

        Procedure:
         1. Build temporary block-diagonal matrix T: either four 1D chebfleg matrices (2D)
            or six 2D chebfleg matrices (3D).
         2. Build equality constraints between overlapping boundary nodes (using find_equality_constraints).
         3. Compute null space of constraint matrix C, call it N (so C N = 0).
         4. Return projection N N^T @ T, which forces T to respect equality of duplicated nodes.

        Returns:
        - final_mat: matrix of shape (total_chebyshev_exterior, total_legendre_exterior).
        """
        if self.ndim == 2:
            # 2D: temporary block diagonal of 4 copies of chebfleg_matrix(p, q)
            T = chebfleg_matrix(self.p, self.q)  # shape (q, p)
            tmp_exterior_mat = block_diag(T, T, T, T)  # shape (4q, 4p)

            # Build equality constraints among the four edges (Jl, Jr, Jd, Ju)
            constraints = self.find_equality_constraints([
                self.JJ_int.Jl, self.JJ_int.Jr,
                self.JJ_int.Jd, self.JJ_int.Ju
            ])
        else:
            # 3D: temporary block diagonal of 6 copies of chebfleg_matrix_2d(p, q)
            T = chebfleg_matrix_2d(self.p, self.q)  # shape (q^2, p^2)
            tmp_exterior_mat = block_diag(T, T, T, T, T, T)  # shape (6q^2, 6p^2)

            # Build equality constraints among the six faces
            constraints = self.find_equality_constraints([
                self.JJ_int.Jl, self.JJ_int.Jr,
                self.JJ_int.Jd, self.JJ_int.Ju,
                self.JJ_int.Jb, self.JJ_int.Jf
            ])

        # Compute an orthonormal basis for the null space of C
        N = null_space(constraints)
        # Project T onto the null space to enforce equality: N N^T tmp_exterior_mat
        return N @ N.T @ tmp_exterior_mat
