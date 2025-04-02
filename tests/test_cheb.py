import numpy as np
from   hps.cheb_utils import *
import numpy.testing as npt

def test_cheb_1d():
    """
    Test the 1D Chebyshev differentiation matrix.
    We approximate the derivative of f(x)=cos(x) on [-1,1].
    The analytical derivative is -sin(x).
    """
    p = 20
    a = 1.0
    x,D = cheb(p)
    # Test function f(x)=cos(x)
    f = np.cos(x)
    df_exact = -np.sin(x)
    df_num = D @ f

    relerr = np.linalg.norm(df_exact - df_num) / np.linalg.norm(df_exact)
    assert relerr < 1e-10

def test_cheb_2d():
    """
    Test the 2D Chebyshev differentiation matrices.
    We use f(x,y) = cos(x)*cos(y) on a square domain.
    Its partial derivatives are:
      f_x = -sin(x)*cos(y)
      f_y = -cos(x)*sin(y)
    """
    p = 20
    a = 1.0
    zz, Ds = cheb_2d(a, p)
    # Reshape grid for evaluation.
    xvec = zz[0, :].reshape(p, p)
    yvec = zz[1, :].reshape(p, p)
    # Evaluate the function
    f = np.cos(xvec) * np.cos(yvec)
    f_flat = f.flatten()
    # Compute numerical partial derivatives
    f_x_num_flat = Ds.D1 @ f_flat
    f_y_num_flat = Ds.D2 @ f_flat
    f_x_num = f_x_num_flat.reshape(p, p)
    f_y_num = f_y_num_flat.reshape(p, p)
    # Exact derivatives:
    f_x_exact = -np.sin(xvec) * np.cos(yvec)
    f_y_exact = -np.cos(xvec) * np.sin(yvec)
    # Compute maximum error
    err_x = np.max(np.abs(f_x_exact - f_x_num))
    err_y = np.max(np.abs(f_y_exact - f_y_num))
    assert err_x < 1e-10
    assert err_y < 1e-10

def test_legfcheb_matrix():
    """
    Test the transformation matrix that maps function values
    from Chebyshev nodes to Legendre nodes.
    
    We use the polynomial function f(x) = x^2. Since this is a 
    degree-2 polynomial, it should be exactly reproduced by 
    the interpolation provided the number of nodes p >= 3.
    
    The test computes:
      - f_cheb: f evaluated at Chebyshev nodes.
      - f_leg_exact: f evaluated at Legendre nodes.
      - f_leg_approx: f_cheb mapped to Legendre nodes via the transformation matrix.
    
    The test asserts that the maximum error is below a tight tolerance.
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss

    p = 10; q = 15
    a = 1.0
    
    # Compute the transformation matrix
    T = legfcheb_matrix(p,q)
    
    cheb_nodes     = a * cheb(q)[0]
    legendre_nodes = a * leggauss(p)[0]
    
    # Define the test function f(x)=x^2 (a polynomial of degree 2)
    f = lambda x: x**2
    
    # Evaluate f on the Chebyshev and Legendre nodes
    f_cheb = f(cheb_nodes)
    f_leg_exact = f(legendre_nodes)
    
    # Compute the interpolated values at Legendre nodes from Chebyshev values
    f_leg_approx = T @ f_cheb
    
    # Compute the maximum error
    err = np.max(np.abs(f_leg_exact - f_leg_approx))
    print("Maximum error in transformation:", err)
    
    # Assert that the error is very small
    assert err < 1e-10, f"Transformation error {err} exceeds tolerance"


def test_chebfleg_matrix():
    """
    Test the transformation matrix that maps function values
    from Chebyshev nodes to Legendre nodes.
    
    We use the polynomial function f(x) = x^2. Since this is a 
    degree-2 polynomial, it should be exactly reproduced by 
    the interpolation provided the number of nodes p >= 3.
    
    The test computes:
      - f_cheb: f evaluated at Chebyshev nodes.
      - f_leg_exact: f evaluated at Legendre nodes.
      - f_leg_approx: f_cheb mapped to Legendre nodes via the transformation matrix.
    
    The test asserts that the maximum error is below a tight tolerance.
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss

    p = 10; q = 12
    a = 1.0
    
    # Compute the transformation matrix
    T = chebfleg_matrix(p,q)
    
    # Get Chebyshev nodes (using the second output from cheb for nodes)
    cheb_nodes     = a * cheb(q)[0]
    legendre_nodes = a * leggauss(p)[0]
    
    # Define the test function f(x)=x^2 (a polynomial of degree 2)
    f = lambda x: x**2
    
    # Evaluate f on the Chebyshev and Legendre nodes
    f_cheb_exact = f(cheb_nodes)       # function values at Chebyshev nodes (length p+1)
    f_leg_exact  = f(legendre_nodes)     # function values at Legendre nodes (length p)
    
    # Compute the interpolated values at Legendre nodes from Chebyshev values
    f_cheb_approx = T @ f_leg_exact      # should yield approximations at Legendre nodes
    
    # Compute the maximum error
    err = np.max(np.abs(f_cheb_approx - f_cheb_exact))
    print("Maximum error in transformation:", err)
    
    # Assert that the error is very small
    assert err < 1e-10, f"Transformation error {err} exceeds tolerance"

def test_cheb_3d():
    """
    Test the 3D Chebyshev differentiation matrices.
    We use f(x,y,z) = cos(x)*cos(y)*cos(z) on a cubic domain.
    Its partial derivatives are:
      f_x = -sin(x)*cos(y)*cos(z)
      f_y = -cos(x)*sin(y)*cos(z)
      f_z = -cos(x)*cos(y)*sin(z)
    """
    p = 20
    a = 1.0
    grid, Ds = cheb_3d(a, p)
    
    # Reshape grid for evaluation.
    X = grid[0, :].reshape(p, p, p)
    Y = grid[1, :].reshape(p, p, p)
    Z = grid[2, :].reshape(p, p, p)
    
    # Evaluate the function.
    f = np.cos(X) * np.cos(Y) * np.cos(Z)
    f_flat = f.flatten()
    
    # Compute numerical partial derivatives.
    f_x_num_flat = Ds.D1 @ f_flat
    f_y_num_flat = Ds.D2 @ f_flat
    f_z_num_flat = Ds.D3 @ f_flat
    
    f_x_num = f_x_num_flat.reshape(p, p, p)
    f_y_num = f_y_num_flat.reshape(p, p, p)
    f_z_num = f_z_num_flat.reshape(p, p, p)
    
    # Exact derivatives.
    f_x_exact = -np.sin(X) * np.cos(Y) * np.cos(Z)
    f_y_exact = -np.cos(X) * np.sin(Y) * np.cos(Z)
    f_z_exact = -np.cos(X) * np.cos(Y) * np.sin(Z)
    
    # Compute maximum errors.
    err_x = np.max(np.abs(f_x_exact - f_x_num))
    err_y = np.max(np.abs(f_y_exact - f_y_num))
    err_z = np.max(np.abs(f_z_exact - f_z_num))
    
    assert err_x < 1e-10, f"Max error in x-derivative: {err_x}"
    assert err_y < 1e-10, f"Max error in y-derivative: {err_y}"
    assert err_z < 1e-10, f"Max error in z-derivative: {err_z}"

def test_transformation_matrices_2d_poly():
    # Set the number of nodes in each dimension (assumed equal)
    p = 7  # Legendre nodes per dimension
    q = 7  # Chebyshev nodes per dimension

    # Get Chebyshev nodes (assumed to be in [-1, 1])
    cheb_nodes = cheb(q)[0]  # e.g., cheb(q) returns (nodes, weights)
    
    # Construct a Chebyshev grid in 2D.
    X_cheb, Y_cheb = np.meshgrid(cheb_nodes, cheb_nodes,indexing='ij')
    X_cheb_flat    = X_cheb.flatten()
    Y_cheb_flat    = Y_cheb.flatten()
    
    # Define a simple polynomial function that is exactly representable.
    def f(x, y):
        return 1 + x + y

    # Evaluate the function at Chebyshev nodes.
    f_cheb = f(X_cheb_flat, Y_cheb_flat)

    # Build 2D transformation matrices.
    # T_legfcheb: maps data from Chebyshev grid to Legendre grid.
    # T_chebfleg: maps data from Legendre grid back to Chebyshev grid.
    T_legfcheb = legfcheb_matrix_2d(p, q)
    T_chebfleg = chebfleg_matrix_2d(p, q)

    # Transform from Chebyshev to Legendre.
    f_leg = T_legfcheb @ f_cheb

    # Transform back from Legendre to Chebyshev.
    f_cheb_reconstructed = T_chebfleg @ f_leg

    # The reconstructed values should match the original function values.
    npt.assert_allclose(f_cheb_reconstructed, f_cheb, atol=1e-8)
