import numpy as np

from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils
from scipy.sparse        import block_diag

def get_leaf_DtNs(pdo, box_geom, a, p):

    len_dim  = box_geom[1] - box_geom[0]
    npan_dim = np.round( len_dim / (2*a)).astype(int)
    assert np.linalg.norm(npan_dim * (2*a) - len_dim,ord=2) < 1e-14

    ndim        = npan_dim.shape[0]
    patch_utils = PatchUtils(a,p,ndim=npan_dim.shape[0])
    Jx_shape    = patch_utils.JJ.Jx.shape[0]

    xx_list     = np.zeros((np.prod(npan_dim),Jx_shape, ndim))
    DtN_list    = np.zeros((np.prod(npan_dim),Jx_shape, Jx_shape))

    if (ndim == 2):

        for j in range(npan_dim[1]):
            for i in range(npan_dim[0]):

                root_loc    = np.zeros(ndim,)

                root_loc[0] = 2 * a * i + box_geom[0,0]
                root_loc[1] = 2 * a * j + box_geom[0,1]

                box_loc  = np.vstack((root_loc,root_loc+2*a))
                leaf_loc = LeafSubdomain(box_loc,pdo,patch_utils)

                xx_list[i+j*npan_dim[0]]  = leaf_loc.xxloc[leaf_loc.Jx]
                DtN_list[i+j*npan_dim[0]] = leaf_loc.DtN

    else:
        raise ValueError

    return npan_dim,xx_list,DtN_list

def get_duplicated_interior_points_2d(p,npan_dim):

    size_bnd = p-2
    size_ext = 4*size_bnd

    Icopy1 = np.zeros(np.prod(npan_dim) * size_ext, dtype=int)
    Icopy2 = np.zeros(np.prod(npan_dim) * size_ext, dtype=int)
    offset = 0

    for j in range(npan_dim[1]):
        for i in range(npan_dim[0]):

            curr_box = i + j * npan_dim[0]
            if (i > 0):
                prev_box = (i-1) + j * npan_dim[0]

                # right boundary of previous box
                Icopy1[offset: offset+size_bnd] = np.arange(size_bnd) + prev_box*size_ext + 1*size_bnd
                # left boundary of current box
                Icopy2[offset: offset+size_bnd] = np.arange(size_bnd) + curr_box*size_ext + 0*size_bnd

                offset += size_bnd

            if (j > 0):
                prev_box = i + (j-1) * npan_dim[0]

                # up   boundary of previous box
                Icopy1[offset: offset+size_bnd] = np.arange(size_bnd) + prev_box*size_ext + 3*size_bnd
                # down boundary of current box
                Icopy2[offset: offset+size_bnd] = np.arange(size_bnd) + curr_box*size_ext + 2*size_bnd

                offset += size_bnd

    return Icopy1[:offset],Icopy2[:offset]


# HPS Multidomain class for handling multidomain discretizations
class Multidomain:
    
    def __init__(self, pdo, box_geom, a, p):
        """
        Initializes the HPS multidomain solver with domain 
        information and discretization parameters.
        
        Parameters:
        - pdo: An object representing the partial differential operator.
        - domain: The computational domain represented as an array.
        - a (float): Characteristic length scale for the domain.
        - p (int): Polynomial degree for spectral methods or discretization parameter.
        """
        self.pdo      = pdo
        self.box_geom = box_geom
        self.p        = p
        self.a        = a
        self.ndim     = box_geom.shape[-1]

        self.npan_dim,xx_list,DtN_list = \
        get_leaf_DtNs(pdo,box_geom,a,p)

        self.XX   = xx_list.reshape(xx_list.shape[0] * xx_list.shape[1],self.ndim)
        self.A    = block_diag(tuple(DtN_list),format='csr')

        if  (self.ndim == 2):
            self.Icopy1,self.Icopy2 = \
            get_duplicated_interior_points_2d(self.p,self.npan_dim)
        else:
            raise ValueError

        self.I_X = np.setdiff1d(np.arange(self.XX.shape[0]), \
            np.union1d(self.Icopy1,self.Icopy2))



