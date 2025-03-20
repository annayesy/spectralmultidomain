import numpy as np
from   hps.cheb_utils import *

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

    p = 10
    a = 1.0
    
    # Compute the transformation matrix
    T = legfcheb_matrix(a, p)
    
    # Get Chebyshev nodes (using the second output from cheb for nodes)
    cheb_nodes = a * cheb(p)[0]
    
    # Get Legendre nodes
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
