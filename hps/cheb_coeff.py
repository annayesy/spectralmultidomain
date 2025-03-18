
from hps_subdomain   import LeafSubdomain
from hps_patch_utils import *
from pdo             import PDO2d,PDO3d,const,get_known_greens
from scipy.linalg import null_space

import matplotlib.pyplot as plt

def chebyshev_to_coefficients(values, deg):
    """
    Converts Chebyshev collocated values to coefficients in Chebyshev basis.
    """
    n = len(values)
    # Fit to specified degree
    return np.polynomial.chebyshev.chebfit(np.linspace(-1, 1, n), values, deg=deg)

def coefficients_to_chebyshev(coefficients, nodes):
    """
    Converts Chebyshev coefficients back to values collocated on Chebyshev nodes.
    """
    return np.polynomial.chebyshev.chebval(nodes, coefficients)

def compatible_projection(p):
    """
    Constructs a projection matrix for compatible boundary conditions.
    Handles polynomial coefficients for four edges.
    """
    A = np.zeros((4, 4 * p))
    
    # Top-left corner
    A[0, 0:p] = 1
    A[0, 3 * p:] = -(-1) ** np.arange(p)
    
    # Bottom-left corner
    A[1, 0:p] = (-1) ** np.arange(p)
    A[1, 2 * p:3 * p] = -(-1) ** np.arange(p)
    
    # Top-right corner
    A[2, p:2 * p] = 1
    A[2, 3 * p:] = -1
    
    # Bottom-right corner
    A[3, p:2 * p] = (-1) ** np.arange(p)
    A[3, 2 * p:3 * p] = -1
    
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
    n_cheb = len(values_edges[0])  # Number of Chebyshev nodes
    cheb_nodes = np.linspace(-1, 1, n_cheb)
    
    # Step 1: Map Chebyshev collocated values to coefficients, truncate/pad to size p
    coeffs_edges = []
    for values in values_edges:
        coeffs = chebyshev_to_coefficients(values, deg=p - 1)
        if len(coeffs) < p:
            coeffs = np.pad(coeffs, (0, p - len(coeffs)), mode='constant')
        else:
            coeffs = coeffs[:p]
        coeffs_edges.append(coeffs)
    
    # Step 2: Flatten coefficients into a single vector
    coeffs_flat = np.concatenate(coeffs_edges)
    
    # Step 3: Apply compatible projection (4p x 4p)
    P = compatible_projection(p)
    projected_coeffs_flat = P @ coeffs_flat
    
    # Step 4: Split the projected coefficients back into edges
    projected_coeffs_edges = [
        projected_coeffs_flat[i * p:(i + 1) * p] for i in range(4)
    ]
    
    # Step 5: Map coefficients back to Chebyshev collocated values
    projected_values_edges = [
        coefficients_to_chebyshev(coeffs, cheb_nodes) for coeffs in projected_coeffs_edges
    ]
    
    return projected_values_edges

box_geom = np.array([[0.5,0.5],[1.0,1.0]]); a = 0.25; p = 20; kh = 2
ndim           = box_geom.shape[-1]
patch_utils    = PatchUtils(a,p,ndim=ndim)

pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

Jx_stack     = np.hstack((leaf_subdomain.JJ_int.Jl, leaf_subdomain.JJ_int.Jr,\
    leaf_subdomain.JJ_int.Jd, leaf_subdomain.JJ_int.Ju))

uu_exact_cheb = get_known_greens(leaf_subdomain.xxloc_int[Jx_stack],kh).reshape(4*p,)

values_edges = [uu_exact_cheb[:p], uu_exact_cheb[p:2*p], uu_exact_cheb[2*p:3*p], uu_exact_cheb[3*p:]]

projected_values_edges = project_chebyshev_square(values_edges, p)

for i in range(4):
    print(values_edges[i] - projected_values_edges[i])