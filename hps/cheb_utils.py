import numpy as np
from numpy.polynomial.legendre import leggauss
from collections import namedtuple

# ------------------------------------------------------------------------------------
# Named tuples to hold differentiation matrices for 2D and 3D Chebyshev grids
# ------------------------------------------------------------------------------------
Ds_2d = namedtuple('Ds_2d', ['D11', 'D22', 'D12', 'D1', 'D2'])
Ds_3d = namedtuple('Ds_3d', ['D11', 'D22', 'D33', 'D12', 'D13', 'D23', 'D1', 'D2', 'D3'])


def cheb(p):
    """
    Compute the Chebyshev differentiation matrix and Chebyshev points of degree p.

    Parameters:
    - p: int
        Number of Chebyshev nodes (polynomial degree).

    Returns:
    - x_flipped: 1D numpy.ndarray of length p
        Chebyshev points in ascending order on [-1, 1].
    - D_flipped: 2D numpy.ndarray of shape (p, p)
        Chebyshev differentiation matrix corresponding to those points.
    """
    # 1) Compute Chebyshev nodes x on [-1, 1] in descending order
    x = np.cos(np.pi * np.arange(p) / (p - 1))  # x[0] = cos(0)=1, x[-1]=cos(pi)= -1

    # 2) Build the vector of weights c (for differentiation formula):
    #    c[0] = 2, c[-1] = 2, interior = 1, with alternating signs
    c = np.concatenate((np.array([2]), np.ones(p - 2), np.array([2])))
    c = c * (-1) ** np.arange(p)  # multiply by (-1)^i to alternate signs

    # 3) Form the matrix of node differences: X_i - X_j
    X = x.repeat(p).reshape((p, p))       # repeat each x-value p times to build a p×p matrix
    dX = X - X.T                          # difference matrix dX[i,j] = x[i] - x[j]

    # 4) Off-diagonal entries of D: D[i,j] = (c[i] / c[j]) / (x[i] - x[j])
    D = np.outer(c, 1 / c) / (dX + np.eye(p))  # add identity to avoid division by zero on diagonal

    # 5) Subtract row sums from diagonal to ensure sum_j D[i,j] = 0
    D = D - np.diag(np.sum(D, axis=1))

    # 6) Flip x and D so that they are in ascending order (from -1 to 1)
    x_flipped = np.flip(x)
    D_flipped = np.flip(np.flip(D, axis=0), axis=1)
    return x_flipped, D_flipped


def lagrange_basis(x, k, nodes):
    """
    Evaluate the k-th Lagrange basis polynomial at point(s) x given interpolation nodes.

    L_k(x) = prod_{j != k} (x - nodes[j]) / (nodes[k] - nodes[j])

    Parameters:
    - x: float or numpy.ndarray
        Point(s) at which to evaluate the basis polynomial.
    - k: int
        Index of the basis polynomial (0 <= k < len(nodes)).
    - nodes: 1D array-like of length n
        Interpolation nodes.

    Returns:
    - float or numpy.ndarray
        Value(s) of the k-th Lagrange basis polynomial at x.
    """
    # Multiply factors for all j != k:
    factors = [
        (x - nodes[j]) / (nodes[k] - nodes[j])
        for j in range(len(nodes))
        if j != k
    ]
    # Take product over all factors
    return np.prod(factors, axis=0)


def legfcheb_matrix(p, q):
    """
    Build the 1D transformation matrix that maps values on Chebyshev nodes (q of them)
    to values on p Legendre (Gauss) nodes.

    In other words, if f_C is a vector of function values at q Chebyshev nodes,
    then legfcheb_matrix(p, q) @ f_C ≈ f_L, function values at p Legendre nodes.

    Parameters:
    - p: int
        Number of Legendre (Gauss) nodes.
    - q: int
        Number of Chebyshev nodes.

    Returns:
    - transformation_matrix: 2D numpy.ndarray of shape (p, q)
        Each row i corresponds to evaluating all q Lagrange basis polynomials
        (based on Chebyshev nodes) at the i-th Legendre node.
    """
    # 1) Get Chebyshev nodes (ascending order) of length q
    cheb_nodes = cheb(q)[0]
    # 2) Get Legendre (Gauss) nodes of length p (range [-1,1])
    legendre_nodes = leggauss(p)[0]

    # 3) Initialize transformation matrix of zeros (p × q)
    transformation_matrix = np.zeros((p, q))

    # 4) For each Legendre node, evaluate the q Chebyshev-based Lagrange basis polynomials
    for i, x_leg in enumerate(legendre_nodes):
        for j in range(q):
            # Lagrange basis L_j(x_leg) using Chebyshev nodes as interpolation nodes
            transformation_matrix[i, j] = lagrange_basis(x_leg, j, cheb_nodes)

    return transformation_matrix


def chebfleg_matrix(p, q):
    """
    Build the 1D transformation matrix that maps values on p Legendre nodes to
    approximated values on q Chebyshev nodes.

    This is effectively the (left) pseudoinverse of the Chebyshev-to-Legendre matrix,
    mapping f_L -> f_C approximately.

    Parameters:
    - p: int
        Number of Legendre nodes.
    - q: int
        Number of Chebyshev nodes.

    Returns:
    - T: 2D numpy.ndarray of shape (q, p)
        Each row i corresponds to evaluating the p Legendre-based Lagrange basis
        polynomials at the i-th Chebyshev node.
    """
    # 1) Legendre nodes (p of them) and Chebyshev nodes (q of them)
    leg_nodes = leggauss(p)[0]
    cheb_nodes = cheb(q)[0]

    # 2) Initialize the transformation matrix (q × p)
    T = np.zeros((q, p))

    # 3) For each Chebyshev node, evaluate the p Lagrange basis polynomials (Legendre nodes)
    for i, x_val in enumerate(cheb_nodes):
        for j in range(p):
            T[i, j] = lagrange_basis(x_val, j, leg_nodes)

    return T


def legfcheb_matrix_2d(p, q):
    """
    Build the 2D transformation matrix mapping values defined on a q×q Chebyshev grid
    to values on a p×p Legendre grid.

    Exploits the tensor-product structure: 2D transform = Kron(1D_transform, 1D_transform).

    Parameters:
    - p: int
        Number of Legendre nodes along each dimension.
    - q: int
        Number of Chebyshev nodes along each dimension.

    Returns:
    - 2D transformation matrix of shape (p*p, q*q)
    """
    T1 = legfcheb_matrix(p, q)  # shape (p, q)
    T2 = legfcheb_matrix(p, q)  # same as T1
    # Kron(T2, T1) yields shape (p*p, q*q)
    return np.kron(T2, T1)


def chebfleg_matrix_2d(p, q):
    """
    Build the 2D transformation matrix mapping values defined on a p×p Legendre grid
    to values on a q×q Chebyshev grid.

    Uses the tensor-product of the 1D chebfleg_matrix.

    Parameters:
    - p: int
        Number of Legendre nodes per dimension.
    - q: int
        Number of Chebyshev nodes per dimension.

    Returns:
    - 2D transformation matrix of shape (q*q, p*p)
    """
    T1 = chebfleg_matrix(p, q)  # shape (q, p)
    T2 = chebfleg_matrix(p, q)  # same as T1
    return np.kron(T2, T1)


# ------------------------------------------------------------------------------------
# Chebyshev utilities for 2D and 3D, returning grid points zz and differentiation matrices Ds
# ------------------------------------------------------------------------------------

def cheb_2d(a, p):
    """
    Build 2D Chebyshev grid points and differentiation matrices on a rectangular domain
    scaled by vector a = [a_x, a_y], using p Chebyshev nodes per dimension.

    Points zz are on the tensor-product grid (p×p), flattened as 2×(p^2).
    Ds contains the 2D differentiation matrices for once and twice derivatives.

    Parameters:
    - a: length-2 array-like [a_x, a_y], half-lengths of domain in x and y directions.
    - p: int, number of Chebyshev nodes per direction.

    Returns:
    - zz: 2D numpy.ndarray of shape (2, p^2)
        Coordinates of Chebyshev points in 2D, flattened.
    - Ds: Ds_2d namedtuple with fields:
        D1:  First-derivative matrix in x (size p^2 × p^2)
        D2:  First-derivative matrix in y (size p^2 × p^2)
        D11: Second-derivative matrix in x (size p^2 × p^2)
        D22: Second-derivative matrix in y (size p^2 × p^2)
        D12: Mixed second-derivative matrix (∂^2/∂x∂y)
    """
    # 1) Compute 1D Chebyshev nodes and differentiation matrix on [-1,1]
    xvec, D = cheb(p)          # xvec: length-p nodes, D: p×p diff matrix
    I = np.eye(p)

    # 2) Scale differentiation matrices by 1/a to map [-1,1] → [-a,a]
    ainv = 1.0 / np.array(a)   # [1/a_x, 1/a_y]

    # 3) Build 2D first-derivative matrices via Kronecker products:
    #    D1 acts in x-direction: (ainv_x * D) ⊗ I
    #    D2 acts in y-direction: I ⊗ (ainv_y * D)
    D1 = np.kron(ainv[0] * D, I)
    D2 = np.kron(I, ainv[1] * D)

    # 4) Build 2D second-derivatives:
    Dsq = D @ D  # 1D second-derivative on [-1,1]
    #    D11 = (ainv_x^2 * Dsq) ⊗ I
    D11 = np.kron(ainv[0] ** 2 * Dsq, I)
    #    D22 = I ⊗ (ainv_y^2 * Dsq)
    D22 = np.kron(I, ainv[1] ** 2 * Dsq)
    #    D12 = (ainv_x * D) ⊗ (ainv_y * D)
    D12 = np.kron(ainv[0] * D, ainv[1] * D)

    # 5) Build 2D coordinate grid (flattened)
    #    X_ij = a_x * xvec[i], Y_ij = a_y * xvec[j], for i,j = 0..p-1
    X, Y = np.meshgrid(a[0] * xvec, a[1] * xvec, indexing='ij')
    #    zz is 2×(p^2), each column is (x_i, y_j)
    zz = np.vstack((X.flatten(), Y.flatten()))

    # 6) Collect differentiation matrices into a namedtuple
    Ds = Ds_2d(D1=D1, D2=D2, D11=D11, D22=D22, D12=D12)
    return zz, Ds


def cheb_3d(a, p):
    """
    Build 3D Chebyshev grid points and differentiation matrices on a rectangular domain
    scaled by a = [a_x, a_y, a_z], using p Chebyshev nodes per dimension.

    Points zz are on the tensor-product grid (p×p×p), flattened as 3×(p^3).
    Ds contains the 3D differentiation matrices for first and second derivatives.

    Parameters:
    - a: length-3 array-like [a_x, a_y, a_z], half-lengths of domain in x, y, z directions.
    - p: int, number of Chebyshev nodes per direction.

    Returns:
    - zz: 2D numpy.ndarray of shape (3, p^3)
        Coordinates of Chebyshev points in 3D, flattened.
    - Ds: Ds_3d namedtuple with fields:
        D1, D2, D3:        First-derivative matrices in x, y, z (size p^3 × p^3)
        D11, D22, D33:     Second-derivative matrices in x, y, z
        D12, D13, D23:     Mixed second-derivative matrices
    """
    # 1) Compute 1D Chebyshev nodes and differentiation matrix on [-1,1]
    xvec, D = cheb(p)
    I = np.eye(p)

    # 2) Scale differentiation matrices by 1/a to map [-1,1] → [-a,a]
    ainv = 1.0 / np.array(a)  # [1/a_x, 1/a_y, 1/a_z]

    # 3) Build 3D first-derivative matrices via Kronecker products:
    #    D1 acts in x: (ainv_x * D) ⊗ I ⊗ I
    D1 = np.kron(ainv[0] * D, np.kron(I, I))
    #    D2 acts in y: I ⊗ (ainv_y * D) ⊗ I
    D2 = np.kron(I, np.kron(ainv[1] * D, I))
    #    D3 acts in z: I ⊗ I ⊗ (ainv_z * D)
    D3 = np.kron(I, np.kron(I, ainv[2] * D))

    # 4) Build 3D second-derivative matrices:
    Dsq = D @ D  # 1D second-derivative
    #    D11 = (ainv_x^2 * Dsq) ⊗ I ⊗ I
    D11 = np.kron(ainv[0] ** 2 * Dsq, np.kron(I, I))
    #    D22 = I ⊗ (ainv_y^2 * Dsq) ⊗ I
    D22 = np.kron(I, np.kron(ainv[1] ** 2 * Dsq, I))
    #    D33 = I ⊗ I ⊗ (ainv_z^2 * Dsq)
    D33 = np.kron(I, np.kron(I, ainv[2] ** 2 * Dsq))

    #    Mixed derivatives:
    #    D12 = (ainv_x * D) ⊗ (ainv_y * D) ⊗ I
    D12 = np.kron(ainv[0] * D, np.kron(ainv[1] * D, I))
    #    D13 = (ainv_x * D) ⊗ I ⊗ (ainv_z * D)
    D13 = np.kron(ainv[0] * D, np.kron(I, ainv[2] * D))
    #    D23 = I ⊗ (ainv_y * D) ⊗ (ainv_z * D)
    D23 = np.kron(I, np.kron(ainv[1] * D, ainv[2] * D))

    # 5) Build 3D coordinate grid (flattened):
    #    X[i,j,k] = a_x * xvec[i], Y[i,j,k] = a_y * xvec[j], Z[i,j,k] = a_z * xvec[k]
    X, Y, Z = np.meshgrid(a[0] * xvec, a[1] * xvec, a[2] * xvec, indexing='ij')
    zz = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))  # shape (3, p^3)

    # 6) Collect differentiation matrices into namedtuple
    Ds = Ds_3d(
        D1=D1, D2=D2, D3=D3,
        D11=D11, D22=D22, D33=D33,
        D12=D12, D13=D13, D23=D23
    )
    return zz, Ds
