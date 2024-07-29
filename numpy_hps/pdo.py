import numpy as np

def const(c=1):
    """
    Returns a function that generates a tensor of constants for a given set of locations.
    
    Parameters:
    c: The constant value to use. Defaults to 1.
    
    Returns:
    A function that takes `xxloc`, an n x d matrix of locations, and returns a tensor of constants c.
    """
    def const_func(xxloc):
        assert xxloc.shape[1] == 2
        return c * np.ones(xxloc.shape[0],)
    return const_func

class PDO2d:
    """
    Represents a 2-dimensional Partial Differential Operator (PDO) with coefficients for the PDE terms.
    
    Parameters:
    c11, c22, c12: Coefficients for the second-order partial derivatives.
    c1, c2: Coefficients for the first-order derivatives.
    c: Coefficient for the zeroth-order term.

    The partial differential operator is given by:
    A [u](x) = - c11(x) [D11 u](x) - c22(x) [D22 u](x) 
    - 2 c12(x) [D12 u](x) + c1(x) [D1 u](x) + c2(x) [D2 u(x)] + c(x) [u](x)

    We assume the coefficients are smooth. 
    We also assume that the matrix [c11,c12; c12, c22] is positive definite in the domain to ensure ellipticity.
    These assumptions are from the paper by Hao, Martinsson on 3D HPS.
    See: https://users.oden.utexas.edu/~pgm/Pubs/2016_HPS_3D_final.pdf
    """
    def __init__(self, c11, c22, c12=None, c1= None, c2 = None, c = None):
        self.c11, self.c22 = c11, c22
        self.c12 = c12
        self.c1, self.c2 = c1, c2
        self.c = c