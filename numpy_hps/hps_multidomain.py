import numpy as np

# HPS Multidomain class for handling multidomain discretizations
class Multidomain:
    
    def __init__(self, pdo, domain, a, p):
        """
        Initializes the HPS multidomain solver with domain information and discretization parameters.
        
        Parameters:
        - pdo: An object representing the partial differential operator.
        - domain: The computational domain represented as an array.
        - a (float): Characteristic length scale for the domain.
        - p (int): Polynomial degree for spectral methods or discretization parameter.
        """
        self.pdo    = pdo
        self.domain = domain
        self.p      = p
        self.a      = a