import numpy as np
from numpy.polynomial.legendre import leggauss

# Generate Chebyshev nodes in [-1, 1]
def chebyshev_nodes(n_nodes):
    return np.cos(np.pi * (2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes))

# Generate Legendre nodes in [-1, 1]
def leg_nodes(n_nodes):
    legendre_nodes, _ = leggauss(n_nodes)
    return legendre_nodes

# Create transformation matrix from Chebyshev to Legendre nodes
def chebyshev_to_legendre_matrix(n_nodes):
    """
    Constructs a transformation matrix to convert a vector tabulated
    on Chebyshev nodes to one tabulated on Legendre nodes.
    
    Parameters:
    n_nodes : int
        The number of nodes for Chebyshev and Legendre.
    
    Returns:
    numpy.ndarray
        Transformation matrix of shape (n_nodes, n_nodes).
    """
    cheb_nodes = chebyshev_nodes(n_nodes)
    legendre_nodes = leg_nodes(n_nodes)

    # Construct Lagrange basis function
    def lagrange_basis(x, k, nodes):
        l_k = np.prod([(x - nodes[j]) / (nodes[k] - nodes[j]) for j in range(len(nodes)) if j != k], axis=0)
        return l_k

    # Populate the transformation matrix
    transformation_matrix = np.zeros((n_nodes, n_nodes))
    for i, x_leg in enumerate(legendre_nodes):  # Loop over Legendre nodes
        for j in range(n_nodes):  # Loop over Chebyshev nodes
            transformation_matrix[i, j] = lagrange_basis(x_leg, j, cheb_nodes)
    
    return transformation_matrix

# Create 2D transformation matrices for tensor product grids
def chebyshev_to_legendre_2d_matrix(n_x, n_y):
    """
    Constructs transformation matrices for 2D tensor product grids.
    
    Parameters:
    n_x : int
        Number of nodes along the x-axis.
    n_y : int
        Number of nodes along the y-axis.
    
    Returns:
    tuple
        A tuple (T_x, T_y) where T_x and T_y are the 1D transformation matrices
        for the x-axis and y-axis respectively.
    """
    T_x = chebyshev_to_legendre_matrix(n_x)
    T_y = chebyshev_to_legendre_matrix(n_y)
    return T_x, T_y

# Apply the 2D transformation to a tensor product grid
def transform_2d_tensor_product_grid(values, T_x, T_y):
    """
    Applies the transformation to a 2D tensor product grid.
    
    Parameters:
    values : ndarray
        2D array of function values tabulated on Chebyshev nodes.
    T_x : ndarray
        Transformation matrix along the x-axis.
    T_y : ndarray
        Transformation matrix along the y-axis.
    
    Returns:
    ndarray
        2D array of function values tabulated on Legendre nodes.
    """
    # Transform along both axes
    transformed_values = T_x @ values @ T_y.T
    return transformed_values

# Main script for testing
if __name__ == "__main__":

    # 1D Example: Chebyshev to Legendre transformation
    n = 20  # Number of nodes
    matrix = chebyshev_to_legendre_matrix(n)

    # Example function: f(x) = sin(pi * x)
    cheb_nodes = chebyshev_nodes(n)
    f_values_on_cheb = np.sin(np.pi * cheb_nodes)  # Values on Chebyshev nodes
    f_values_on_leg = matrix @ f_values_on_cheb  # Values on Legendre nodes

    legendre_nodes = leg_nodes(n)
    exact_values = np.sin(np.pi * legendre_nodes)  # Exact function values on Legendre nodes

    # Compute error
    error = np.linalg.norm(f_values_on_leg - exact_values, ord=np.inf)
    print(f"1D Transformation - Max error: {error:.5e}")

    # 2D Example: Chebyshev to Legendre transformation
    n_x, n_y = 12, 12  # Nodes along x and y axes
    T_x, T_y = chebyshev_to_legendre_2d_matrix(n_x, n_y)

    # Define function: f(x, y) = sin(pi * x) * cos(pi * y)
    cheb_x = chebyshev_nodes(n_x)
    cheb_y = chebyshev_nodes(n_y)
    X, Y = np.meshgrid(cheb_x, cheb_y, indexing='ij')
    f_values = np.sin(np.pi * X) * np.cos(np.pi * Y)

    # Transform to Legendre nodes
    leg_values = transform_2d_tensor_product_grid(f_values, T_x, T_y)

    # Verify against exact values
    leg_x = leg_nodes(n_x)
    leg_y = leg_nodes(n_y)
    X_leg, Y_leg = np.meshgrid(leg_x, leg_y, indexing='ij')
    exact_values = np.sin(np.pi * X_leg) * np.cos(np.pi * Y_leg)

    # Compute error
    error = np.linalg.norm(leg_values - exact_values, ord=np.inf)
    print(f"2D Transformation - Max error: {error:.5e}")
