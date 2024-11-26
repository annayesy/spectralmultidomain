import numpy as np
from scipy.linalg import null_space
from numpy.polynomial.chebyshev import chebfit, chebval
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

# Function to compute and display accuracy
def test_accuracy(values_edges, projected_values_edges):
    n_edges = len(values_edges)
    
    # Check corner continuity
    print("Corner continuity checks:")
    corners = [
        (projected_values_edges[0][-1], projected_values_edges[1][0], "Edge 1 end", "Edge 2 start"),
        (projected_values_edges[1][-1], projected_values_edges[2][0], "Edge 2 end", "Edge 3 start"),
        (projected_values_edges[2][-1], projected_values_edges[3][0], "Edge 3 end", "Edge 4 start"),
        (projected_values_edges[3][-1], projected_values_edges[0][0], "Edge 4 end", "Edge 1 start"),
    ]
    for val1, val2, desc1, desc2 in corners:
        diff = np.abs(val1 - val2)
        print(f"{desc1} vs {desc2}: Difference = {diff:.5e}")
    
    # Compute error norms for each edge
    print("\nApproximation errors:")
    for i in range(n_edges):
        original = values_edges[i]
        projected = projected_values_edges[i]
        diff = original - projected
        max_error = np.max(np.abs(diff))
        l2_error = np.sqrt(np.sum(diff ** 2))
        print(f"Edge {i + 1}: Max Error = {max_error:.5e}, L2 Error = {l2_error:.5e}")
    
    # Optional: Plot original and projected functions
    import matplotlib.pyplot as plt
    cheb_nodes = np.linspace(-1, 1, len(values_edges[0]))
    for i in range(n_edges):
        plt.figure()
        plt.plot(cheb_nodes, values_edges[i], 'b-', label='Original')
        plt.plot(cheb_nodes, projected_values_edges[i], 'r--', label='Projected')
        plt.title(f"Edge {i + 1}")
        plt.legend()
        plt.xlabel('Chebyshev Nodes')
        plt.ylabel('Function Value')
        plt.show()

# Example Usage
if __name__ == "__main__":
    n_cheb = 16  # Number of Chebyshev nodes
    p = 16        # Polynomial degree + 1 (number of coefficients)
    
    # Example functions on Chebyshev nodes for 4 edges
    cheb_nodes = np.linspace(-1, 1, n_cheb)
    edge_1 = np.sin(np.pi * cheb_nodes)
    edge_2 = np.sin(2*np.pi * cheb_nodes)
    edge_3 = np.sin(np.pi * cheb_nodes)
    edge_4 = np.sin(2*np.pi * cheb_nodes)
    
    values_edges = [edge_1, edge_2, edge_3, edge_4]
    
    # Project to ensure corner continuity
    projected_values_edges = project_chebyshev_square(values_edges, p)
    
    # Test accuracy
    test_accuracy(values_edges, projected_values_edges)
