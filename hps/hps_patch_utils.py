import numpy as np

import numpy as np
from collections import namedtuple
import scipy
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx','Jc'])

def cheb(p):
    """
    Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
    
    Parameters:
    - p: The polynomial degree
    
    Returns:
    - D: The Chebyshev differentiation matrix
    - x: The Chebyshev points
    """
    x = np.cos(np.pi * np.arange(p+1) / p)
    c = np.concatenate((np.array([2]), np.ones(p-1), np.array([2])))
    c = np.multiply(c,np.power(np.ones(p+1) * -1, np.arange(p+1)))
    X = x.repeat(p+1).reshape((-1,p+1))
    dX = X - X.T
    # create the off diagonal entries of D
    D = np.divide(np.outer(c,np.divide(np.ones(p+1),c)), dX + np.eye(p+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D,x


#################################### 2D discretization ##########################################

def cheb_2d(a,p):
    D,xvec = cheb(p-1)
    xvec = a * np.flip(xvec)
    D = (1/a) * D
    I = np.eye(p)
    D1 = -np.kron(D,I)
    D2 = -np.kron(I,D)
    Dsq = D @ D
    D11 = np.kron(Dsq,I)
    D22 = np.kron(I,Dsq)
    D12 = np.kron(D,D)

    zz1 = np.repeat(xvec,p)
    zz2 = np.repeat(xvec,p).reshape(-1,p).T.flatten()
    zz = np.vstack((zz1,zz2))
    Ds = Ds_2d(D1= D1, D2= D2, D11= D11, D22= D22, D12= D12)
    return zz, Ds 

def leaf_discretization_2d(a,p):
    zz,Ds = cheb_2d(a,p)
    hmin  = zz[1,1] - zz[0,1]

    Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
    Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
    Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,Jc1))
    Jl    = Jl.copy().reshape(p-2,)
    Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,Jc1))
    Jr    = Jr.copy().reshape(p-2,)
    Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,Jc0))
    Jd    = Jd.copy().reshape(p-2,)
    Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,Jc0))
    Ju    = Ju.copy().reshape(p-2,)
    Jc    = np.argwhere(np.logical_and(Jc0,Jc1))
    Jc    = Jc.copy().reshape((p-2)**2,)
    Jx    = np.concatenate((Jl,Jr,Jd,Ju))
    
    JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, 
             Jx= Jx, Jc= Jc)
    return zz,Ds,JJ,hmin

def get_diff_ops(Ds,JJ,d):
    assert (d == 2)
    Nl = Ds.D1[JJ.Jl]
    Nr = Ds.D1[JJ.Jr]
    Nd = Ds.D2[JJ.Jd]
    Nu = Ds.D2[JJ.Ju]

    Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu))

    return Nx

class PatchUtils:
    def __init__(self,a,p):
        """
        Initializes utilities associated with an HPS patch.
        
        Parameters:
        - a: Half the size of the computational domain
        - p: The polynomial degree for Chebyshev discretization
        """
        self._discretize(a,p,d=2)
        self.a = a; self.p = p; self.d = 2
        
    def _discretize(self,a,p,d):
        assert d == 2
        zz_tmp,self.Ds,self.JJ,self.hmin = leaf_discretization_2d(a,p)
        self.zz = zz_tmp.T
        self.Nx = get_diff_ops(self.Ds,self.JJ,d)