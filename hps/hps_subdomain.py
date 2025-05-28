import numpy as np

def diag_mult(diag,M):
    """
    Performs multiplication of a diagonal matrix (represented by a vector) with a matrix.
    
    Parameters:
    - diag: A vector representing the diagonal of a diagonal matrix
    - M: A matrix to be multiplied
    
    Returns:
    - The result of the diagonal matrix multiplied by M
    """
    return (diag * M.T).T

def get_Aloc(pdo,xxloc,Ds):
    """
    Returns Aloc according to the definition of the PDO at local points xxloc
    using differential operators Ds.
    """

    m = xxloc.shape[0];
    assert Ds.D11.shape[0] == m; assert Ds.D11.shape[1] == m 
    ndim = xxloc.shape[1]; assert ndim == 2 or ndim == 3

    Aloc = np.zeros((m,m))
    Aloc -= diag_mult( pdo.c11(xxloc), Ds.D11)
    Aloc -= diag_mult( pdo.c22(xxloc), Ds.D22)

    if (pdo.c12 is not None):
        Aloc -= 2*diag_mult (pdo.c12(xxloc), Ds.D12)

    if (pdo.c1 is not None):
        Aloc +=   diag_mult (pdo.c1(xxloc),Ds.D1)
    if (pdo.c2 is not None):
        Aloc +=   diag_mult (pdo.c2(xxloc),Ds.D2)
    if (pdo.c is not None):
        Aloc +=   diag_mult (pdo.c(xxloc), np.eye(m))

    if (ndim == 3):

        Aloc -= diag_mult( pdo.c33(xxloc), Ds.D33)

        if (pdo.c13 is not None):
            Aloc -= 2*diag_mult (pdo.c13(xxloc), Ds.D13)        
        if (pdo.c23 is not None):
            Aloc -= 2*diag_mult (pdo.c23(xxloc), Ds.D23)
        if (pdo.c3 is not None):
            Aloc +=   diag_mult (pdo.c3(xxloc),Ds.D3)

    return Aloc

class LeafSubdomain:

    def __init__(self,box_geom,pdo,patch_utils):

        lim_max = box_geom[1]
        lim_min = box_geom[0]

        box_len = lim_max - lim_min
        c       = box_len * 0.5 + lim_min

        assert np.abs( box_len[0] - 2 * patch_utils.a[0]) < 1e-14
        assert np.abs( box_len[1] - 2 * patch_utils.a[1]) < 1e-14
        assert box_geom.shape[-1] == patch_utils.ndim

        self.pdo       = pdo
        self.xxloc_int = patch_utils.zz_int + c
        self.xxloc_ext = patch_utils.zz_ext + c
        self.diff_ops  = patch_utils.Ds
        self.p         = patch_utils.p
        self.utils     = patch_utils

        # Jx is ordered as Jl, Jr, Jd, Ju
        self.JJ_int  = patch_utils.JJ_int
        self.JJ_ext  = patch_utils.JJ_ext

        self.chebfleg_mat = self.utils.chebfleg_exterior_mat
        self.legfcheb_mat = self.utils.legfcheb_exterior_mat

    @property
    def Aloc(self):
        return get_Aloc(self.pdo,self.xxloc_int,self.diff_ops)

    def solve_dir(self,uu_dir,ff_body=None):

        if (self.utils.ndim == 2):

            Jx_stack = np.hstack((self.JJ_int.Jl, self.JJ_int.Jr,\
                self.JJ_int.Jd, self.JJ_int.Ju))
        else:
            Jx_stack = np.hstack((self.JJ_int.Jl, self.JJ_int.Jr,\
                self.JJ_int.Jd, self.JJ_int.Ju, self.JJ_int.Jb, self.JJ_int.Jf ))
        Ji       = self.JJ_int.Ji
        Jx       = np.setdiff1d(np.arange(self.xxloc_int.shape[0]),Ji)

        assert uu_dir.ndim == 2
        assert uu_dir.shape[0] == self.xxloc_ext.shape[0]

        nrhs = uu_dir.shape[-1]
        res  = np.zeros((self.xxloc_int.shape[0],nrhs))

        Aloc = self.Aloc
        if (ff_body is None):
            ff_body = np.zeros((self.xxloc_int.shape[0],nrhs))

        res[Jx_stack] = self.chebfleg_mat @ uu_dir
        res[Ji]       = np.linalg.solve(Aloc[Ji][:,Ji], \
            ff_body[Ji] - Aloc[Ji][:,Jx] @ res[Jx])

        return res

    def reduce_body_load(self,ff_body):

        Nx_stack = self.utils.Nx_stack

        loc_sol  = self.solve_dir(np.zeros((self.xxloc_ext.shape[0],1)),-ff_body)
        return self.legfcheb_mat @ (Nx_stack[:,self.JJ_int.Ji] @ loc_sol[self.JJ_int.Ji])

    @property
    def DtN(self):

        Nx_stack = self.utils.Nx_stack
        mat      = np.eye(self.xxloc_ext.shape[0])

        loc_sol  = self.solve_dir(np.eye(self.xxloc_ext.shape[0]))
        neu_sol  = Nx_stack @ loc_sol

        return self.legfcheb_mat @ neu_sol
