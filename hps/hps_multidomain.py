import numpy as np

from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils
from scipy.sparse        import block_diag
from hps.sparse_utils    import CSRBuilder
from hps.pde_solver      import AbstractPDESolver

def get_leaf_DtNs(pdo, box_geom, a, p):

    len_dim  = box_geom[1] - box_geom[0]
    npan_dim = np.round( len_dim / (2*a)).astype(int)
    assert np.linalg.norm(npan_dim * (2*a) - len_dim,ord=2) < 1e-14

    ndim        = npan_dim.shape[0]
    patch_utils = PatchUtils(a,p,ndim=npan_dim.shape[0])
    Jx_shape    = patch_utils.zz_ext.shape[0]
    Ji_shape    = patch_utils.zz_int.shape[0]

    xxext_list  = np.zeros((np.prod(npan_dim),Jx_shape, ndim))
    xxint_list  = np.zeros((np.prod(npan_dim),Ji_shape, ndim))
    DtN_list    = np.zeros((np.prod(npan_dim),Jx_shape, Jx_shape))

    sub_list    = [np.array([]) for j in range(np.prod(npan_dim))]

    if (ndim == 2):

        for j in range(npan_dim[1]):
            for i in range(npan_dim[0]):

                root_loc    = np.zeros(ndim,)

                root_loc[0] = 2 * a * i + box_geom[0,0]
                root_loc[1] = 2 * a * j + box_geom[0,1]

                box_loc  = np.vstack((root_loc,root_loc+2*a))
                leaf_loc = LeafSubdomain(box_loc,pdo,patch_utils)

                box_ind  = i+j*npan_dim[0]

                xxext_list[box_ind] = leaf_loc.xxloc_ext
                xxint_list[box_ind] = leaf_loc.xxloc_int
                DtN_list  [box_ind] = leaf_loc.DtN
                sub_list  [box_ind] = leaf_loc

    else:

        for k in range(npan_dim[2]):
            for j in range(npan_dim[1]):
                for i in range(npan_dim[0]):

                    root_loc    = np.zeros(ndim,)

                    root_loc[0] = 2 * a * i + box_geom[0,0]
                    root_loc[1] = 2 * a * j + box_geom[0,1]
                    root_loc[2] = 2 * a * k + box_geom[0,2]

                    box_loc  = np.vstack((root_loc,root_loc+2*a))
                    leaf_loc = LeafSubdomain(box_loc,pdo,patch_utils)

                    box_ind  = i + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]

                    xxext_list [box_ind] = leaf_loc.xxloc_ext
                    xxint_list [box_ind] = leaf_loc.xxloc_int
                    DtN_list   [box_ind] = leaf_loc.DtN
                    sub_list   [box_ind] = leaf_loc

    return patch_utils,npan_dim,xxext_list,xxint_list,sub_list,DtN_list

def get_duplicated_interior_points_2d(p,npan_dim):

    size_bnd = p
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

def get_duplicated_interior_points_3d(p,npan_dim):

    size_bnd = p**2
    size_ext = 6*size_bnd

    Icopy1 = np.zeros(np.prod(npan_dim) * size_ext, dtype=int)
    Icopy2 = np.zeros(np.prod(npan_dim) * size_ext, dtype=int)
    offset = 0

    for k in range(npan_dim[2]):
        for j in range(npan_dim[1]):
            for i in range(npan_dim[0]):

                curr_box = i + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]
                if (i > 0):
                    prev_box = (i-1) + j * npan_dim[0] + k * npan_dim[0] * npan_dim[1]

                    # right boundary of previous box
                    Icopy1[offset: offset+size_bnd] = np.arange(size_bnd) + prev_box*size_ext + 1*size_bnd
                    # left boundary of current box
                    Icopy2[offset: offset+size_bnd] = np.arange(size_bnd) + curr_box*size_ext + 0*size_bnd

                    offset += size_bnd

                if (j > 0):
                    prev_box = i + (j-1) * npan_dim[0] + k * npan_dim[0] * npan_dim[1]

                    # up   boundary of previous box
                    Icopy1[offset: offset+size_bnd] = np.arange(size_bnd) + prev_box*size_ext + 3*size_bnd
                    # down boundary of current box
                    Icopy2[offset: offset+size_bnd] = np.arange(size_bnd) + curr_box*size_ext + 2*size_bnd

                    offset += size_bnd

                if (k > 0):
                    prev_box = i + j * npan_dim[0] + (k-1) * npan_dim[0] * npan_dim[1]

                    # fron boundary of previous box
                    Icopy1[offset: offset+size_bnd] = np.arange(size_bnd) + prev_box*size_ext + 5*size_bnd
                    # back boundary of current box
                    Icopy2[offset: offset+size_bnd] = np.arange(size_bnd) + curr_box*size_ext + 4*size_bnd

                    offset += size_bnd

    return Icopy1[:offset],Icopy2[:offset]

# HPS Multidomain class for handling multidomain discretizations
class HPSMultidomain(AbstractPDESolver):
    
    def __init__(self, pdo, geom, a, p):
        """
        Initializes the HPS multidomain solver with domain 
        information and discretization parameters.
        
        Parameters:
        - pdo: An object representing the partial differential operator.
        - domain: The computational domain represented as an array.
        - a (float): Characteristic length scale for the domain.
        - p (int): Polynomial degree for spectral methods or discretization parameter.
        """

        self._box_geom = geom.bounds
        self._geom     = geom
        self._p        = p

        self.patch_utils,self.npan_dim,xxext_list,xxint_list,\
        self.sub_list,DtN_list = get_leaf_DtNs(pdo,self._box_geom, a, self.p)
        
        self._XX     = xxext_list.reshape(xxext_list.shape[0] * xxext_list.shape[1],self.ndim)
        self._XXfull = xxint_list.reshape(xxint_list.shape[0] * xxint_list.shape[1],self.ndim)

        if  (self.ndim == 2):
            self._Jcopy1,self._Jcopy2 = \
            get_duplicated_interior_points_2d(self.p, self.npan_dim)
        elif (self.ndim == 3):
            self._Jcopy1,self._Jcopy2 = \
            get_duplicated_interior_points_3d(self.p, self.npan_dim)
        else:
            raise ValueError

        self._Jx = np.setdiff1d(np.arange(self.XX.shape[0]), \
            np.union1d(self._Jcopy1,self._Jcopy2))
        self._Ji = self._Jcopy1

        A    = block_diag(tuple(DtN_list),format='csr')
        del DtN_list

        #### accumulate coo matrix
        Aii = CSRBuilder(self._Jcopy1.shape[0],\
            self._Jcopy1.shape[0],A.nnz)
        Aii.add_data(A[self._Jcopy1][:,self._Jcopy1])
        Aii.add_data(A[self._Jcopy1][:,self._Jcopy2])
        Aii.add_data(A[self._Jcopy2][:,self._Jcopy1])
        Aii.add_data(A[self._Jcopy2][:,self._Jcopy2])
        Aii = Aii.tocsr()

        Aix = CSRBuilder(self._Jcopy1.shape[0],\
            self.Jx.shape[0],A.nnz)
        Aix.add_data(A[self._Jcopy1][:,self.Jx])
        Aix.add_data(A[self._Jcopy2][:,self.Jx])
        Aix = Aix.tocsr()

        Axi = CSRBuilder(self.Jx.shape[0],
            self._Jcopy1.shape[0],A.nnz)
        Axi.add_data(A[self._Jx][:,self._Jcopy1])
        Axi.add_data(A[self._Jx][:,self._Jcopy2])
        Axi = Axi.tocsr()

        self._Aii      = Aii
        self._Aix      = Aix
        self._Axi      = Axi
        self._Axx      = A[self.Jx][:,self.Jx]

    def solve_dir_full(self,uu_dir):

        assert uu_dir.ndim == 2
        nrhs       = uu_dir.shape[-1]

        uu_sol_bnd = np.zeros((self._XX.shape[0],nrhs))
        uu_sol_bnd[self._Jcopy1] = self.solve_dir(uu_dir)
        uu_sol_bnd[self._Jcopy2] = uu_sol_bnd[self._Jcopy1]
        uu_sol_bnd[self._Jx]     = uu_dir

        uu_sol_bnd = uu_sol_bnd.reshape(np.prod(self.npan_dim),self.patch_utils.zz_ext.shape[0],nrhs)
        uu_sol_int = np.zeros((np.prod(self.npan_dim),self.patch_utils.zz_int.shape[0],nrhs))

        for j in range(np.prod(self.npan_dim)):
            uu_sol_int[j] = self.sub_list[j].solve_dir(uu_sol_bnd[j])
        return uu_sol_int.reshape(self._XXfull.shape[0],nrhs)
        
    @property
    def npoints_dim(self):
        return self.npan_dim * self.p

    @property
    def geom(self):
        return self._geom

    @property
    def XX(self):
        return self._XX

    @property
    def Ji(self):
        return self._Ji

    @property
    def Jx(self):
        return self._Jx

    @property
    def Aii(self):
        return self._Aii
    
    @property
    def Aix(self):
        return self._Aix

    @property
    def Axi(self):
        return self._Axi
    
    @property
    def Axx(self):
        return self._Axx

    @property
    def p(self):
        return self._p
    
