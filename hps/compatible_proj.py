import numpy as np
from scipy.linalg               import null_space
from numpy.polynomial.chebyshev import chebfit, chebval
from hps.cheb_utils             import *

def compatible_projection(p):
    """
    Constructs a projection matrix for compatible boundary conditions.
    Handles polynomial coefficients for four edges.
    """
    A = np.zeros((4, 4 * p))
    
    # Top-left corner
    A[0, 0:p] = 1
    A[0, 3*p:4*p] = -(-1)**np.arange(p)
    
    # Bottom-left corner
    A[1, 0:p] = (-1)**np.arange(p)
    A[1, 2*p:3*p] = -(-1)**np.arange(p)
    
    # Top-right corner
    A[2, p:2*p] = 1
    A[2, 3*p:4*p] = -np.ones(p)
    
    # Bottom-right corner
    A[3, p:2*p] = (-1)**np.arange(p)
    A[3, 2*p:3*p] = -np.ones(p)
    
    VV = null_space(A)  # Null space of A
    P = VV @ VV.T       # Projection matrix
    return P

def project_chebyshev_square(values_edges, p):
    """
    Projects values collocated on a Chebyshev square to ensure corner continuity.
    
    Parameters:
      values_edges : list of ndarray
          List of 4 arrays, each containing function values collocated on Chebyshev nodes for the edges.
      p : int
          Number of polynomial coefficients (degree + 1).
    
    Returns:
      list of ndarray
          List of 4 arrays, each containing modified Chebyshev collocated values with corner continuity.
    """
    n_cheb = values_edges.shape[1]  # Number of Chebyshev nodes
    nodes = cheb(n_cheb)[0]

    coeffs_edges    = np.polynomial.chebyshev.chebfit(nodes, \
        values_edges.T, deg=p-1).T
    coeffs_flat     = coeffs_edges.reshape(4*p,)
    
    # Step 3: Apply compatible projection (4p x 4p)
    P = compatible_projection(p)
    projected_coeffs_flat = P @ coeffs_flat
    
    # Step 4: Split the projected coefficients back into edges
    projected_coeffs_edges  = projected_coeffs_flat.reshape(4,p)
    projected_values_edges  = np.polynomial.chebyshev.chebval(nodes, projected_coeffs_edges.T)

    return projected_values_edges


if __name__ == '__main__':

    box_geom = np.array([[0.5, 0.5], [1.0, 1.0]])
    a = 0.25; p = 8; kh = 2
    # The following hps_subdomain, patch_utils, pdo, etc. remain unchanged.
    from hps_subdomain import LeafSubdomain
    from hps_patch_utils import PatchUtils
    from pdo import PDO2d, PDO3d, const, get_known_greens

    ndim = box_geom.shape[-1]
    patch_utils = PatchUtils(a, p, ndim=ndim)
    pdo = PDO2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

    Jx_stack = np.hstack((leaf_subdomain.JJ_int.Jl, leaf_subdomain.JJ_int.Jr,
                           leaf_subdomain.JJ_int.Jd, leaf_subdomain.JJ_int.Ju))

    uu_exact_cheb = get_known_greens(leaf_subdomain.xxloc_int[Jx_stack], kh).reshape(4*p,)
    values_edges  = uu_exact_cheb.reshape(4,p)
    projected_values_edges = project_chebyshev_square(values_edges, p)

    for i in range(4):
        print(values_edges[i] - projected_values_edges[i])
