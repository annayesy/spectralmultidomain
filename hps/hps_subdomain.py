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

        assert np.abs( np.min(box_len) - 2 * patch_utils.a) < 1e-14
        assert np.abs( np.max(box_len) - 2 * patch_utils.a) < 1e-14
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
        assert ff_body is None

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
            ff_body = np.zeros((Ji.shape[0],nrhs))

        res[Jx_stack] = self.chebfleg_mat @ uu_dir
        res[Ji]       = np.linalg.solve(Aloc[Ji][:,Ji], \
            ff_body - Aloc[Ji][:,Jx] @ res[Jx])

        return res

    @property
    def DtN(self):

        Nx_stack = self.utils.Nx_stack
        mat      = np.eye(self.xxloc_ext.shape[0])

        loc_sol  = self.solve_dir(np.eye(self.xxloc_ext.shape[0]))
        neu_sol  = Nx_stack @ loc_sol

        return self.legfcheb_mat @ neu_sol

if __name__ == '__main__':

    
    from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
    from hps.hps_patch_utils import *
    from hps.cheb_utils      import *

    import matplotlib.pyplot as plt

    ndim = 3

    if (ndim == 2):
        box_geom = np.array([[0.5,0.25],[1.0,0.75]]); a = 0.25; p = 20; kh = 0
        patch_utils  = PatchUtils(a,p,ndim=2)

        pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
    else:

        box_geom = np.array([[0.5,0.25,0.25],[1.0,0.75,0.75]]); a = 0.25; p = 10; kh = 0
        patch_utils  = PatchUtils(a,p,ndim=3)

        pdo = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))

    leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

    xx_int = leaf_subdomain.xxloc_int
    xx_ext = leaf_subdomain.xxloc_ext

    uu_exact_ext  = get_known_greens(leaf_subdomain.xxloc_ext,kh)
    uu_exact_cheb = get_known_greens(leaf_subdomain.xxloc_int,kh)

    ####### legendre f cheb

    Ji_cheb      = leaf_subdomain.JJ_int.Ji
    if (ndim == 2):
        Jx_stack     = np.hstack((leaf_subdomain.JJ_int.Jl, leaf_subdomain.JJ_int.Jr,\
            leaf_subdomain.JJ_int.Jd, leaf_subdomain.JJ_int.Ju))

    else:
        Jx_stack     = np.hstack((leaf_subdomain.JJ_int.Jl, leaf_subdomain.JJ_int.Jr,\
            leaf_subdomain.JJ_int.Jd, leaf_subdomain.JJ_int.Ju,\
            leaf_subdomain.JJ_int.Jb,leaf_subdomain.JJ_int.Jf))

    uu_loc_ext   = patch_utils.legfcheb_exterior_mat @ uu_exact_cheb[Jx_stack]

    assert np.linalg.norm(uu_loc_ext - uu_exact_ext) < 1e-10

    ####### cheb f legendre
    Aloc         = leaf_subdomain.Aloc

    uu_calc      = np.zeros(uu_exact_cheb.shape)
    Jx_cheb      = np.setdiff1d(np.arange(xx_int.shape[0]),Ji_cheb)

    uu_calc[Jx_stack] = patch_utils.chebfleg_exterior_mat @ uu_exact_ext
    uu_calc[Ji_cheb]  = - np.linalg.solve(Aloc[Ji_cheb][:,Ji_cheb], \
        Aloc[Ji_cheb][:,Jx_cheb] @ uu_calc[Jx_cheb])
    uu_diff = uu_calc - uu_exact_cheb

    assert np.linalg.norm(uu_diff[Jx_stack]) < 1e-10
    assert np.linalg.norm(uu_diff[Ji_cheb])  < 1e-10

    ###############################################

    center = np.zeros(ndim,)

    uu_exact_ext  = get_known_greens(leaf_subdomain.xxloc_ext,kh=0,center=center)
    uu_exact_cheb = get_known_greens(leaf_subdomain.xxloc_int,kh=0,center=center)

    uu_sol = leaf_subdomain.solve_dir(uu_exact_ext)
    assert np.linalg.norm(uu_sol - uu_exact_cheb) < 1e-7

    neumann_from_dtn = leaf_subdomain.DtN @ uu_exact_ext

    if (ndim == 2):

        def r_sq(xx):
            return (xx[:,0]**2 + xx[:,1]**2).reshape(xx.shape[0],1)

        xx_ext    = leaf_subdomain.xxloc_ext
        neu_true  = np.zeros(neumann_from_dtn.shape)

        neu_true[:p]     = - box_geom[0,0] / r_sq(xx_ext[leaf_subdomain.JJ_ext.Jl])
        neu_true[p:2*p]  = + box_geom[1,0] / r_sq(xx_ext[leaf_subdomain.JJ_ext.Jr])
        neu_true[2*p:3*p]= - box_geom[0,1] / r_sq(xx_ext[leaf_subdomain.JJ_ext.Jd])
        neu_true[3*p:]   = + box_geom[1,1] / r_sq(xx_ext[leaf_subdomain.JJ_ext.Ju])
    else:
        raise ValueError

    assert np.linalg.norm(neumann_from_dtn - neu_true) < 1e-12

    print("Checks passed")