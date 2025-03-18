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

        # Jx is ordered as Jl, Jr, Jd, Ju
        self.JJ_int  = patch_utils.JJ_int
        self.JJ_ext  = patch_utils.JJ_ext
        #self.Nx  = patch_utils.Nx

    @property
    def Aloc(self):
        return get_Aloc(self.pdo,self.xxloc_int,self.diff_ops)

    def solve_dir(self,uu_dir,ff_body=None):

        assert uu_dir.ndim == 2
        assert uu_dir.shape[0] == self.Jx.shape[0]

        nrhs = uu_dir.shape[-1]

        Aloc = self.Aloc
        if (ff_body is None):
            ff_body = np.zeros((self.Jc.shape[0],nrhs))

        Acc = Aloc[self.Jc][:,self.Jc]
        Acx = Aloc[self.Jc][:,self.Jx]

        return np.linalg.solve(Aloc[self.Jc][:,self.Jc], ff_body - Acx @ uu_dir)

    @property
    def DtN(self):

        Aloc = self.Aloc

        Acc = Aloc[self.Jc][:,self.Jc]
        Acx = Aloc[self.Jc][:,self.Jx]

        Axc = Aloc[self.Jx][:,self.Jc]

        DtN = self.Nx[:,self.Jx] - self.Nx[:,self.Jc] @ np.linalg.solve(Acc,Acx)
        return DtN

if __name__ == '__main__':

    from hps_patch_utils import PatchUtils,chebyshev_to_legendre_matrix
    from pdo             import PDO2d,const,get_known_greens
    from scipy.linalg    import block_diag
    import matplotlib.pyplot as plt
    from compatible_proj import project_chebyshev_square

    box_geom     = np.array([[-0.25,-0.25],[+0.25,+0.25]]); a = 0.25; p = 20; kh = 10
    patch_utils  = PatchUtils(a,p,ndim=2)

    pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

    leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

    xx_tot = leaf_subdomain.xxloc_int
    xx_int = leaf_subdomain.xxloc_int[leaf_subdomain.JJ_int.Ji]
    xx_tmp = leaf_subdomain.xxloc_int[leaf_subdomain.JJ_int.Jl]
    xx_ext = leaf_subdomain.xxloc_ext

    fig = plt.figure()
    ax  = fig.add_subplot()

    ax.scatter(xx_tot[:,0], xx_tot[:,1],color='black')
    ax.scatter(xx_int[:,0], xx_int[:,1],color='tab:blue')
    ax.scatter(xx_ext[:,0], xx_ext[:,1],color='tab:red')

    ax.set_aspect('equal','box')

    plt.savefig("xxloc.png",transparent=True,dpi=300)

    uu_exact_ext  = get_known_greens(leaf_subdomain.xxloc_ext,kh)
    uu_exact_cheb = get_known_greens(leaf_subdomain.xxloc_int,kh)

    ####### cheb to legendre

    Ji_cheb      = leaf_subdomain.JJ_int.Ji
    Jx_stack     = np.hstack((leaf_subdomain.JJ_int.Jl, leaf_subdomain.JJ_int.Jr,\
        leaf_subdomain.JJ_int.Jd, leaf_subdomain.JJ_int.Ju))

    T            = chebyshev_to_legendre_matrix(a,p)
    uu_loc_ext   = block_diag(T,T,T,T) @ uu_exact_cheb[Jx_stack]

    print("Cheb to legendre %5.2e" % np.linalg.norm(uu_loc_ext - uu_exact_ext) / np.linalg.norm(uu_exact_ext))

    ####### legendre to cheb
    Aloc         = leaf_subdomain.Aloc

    uu_calc      = np.zeros(uu_exact_cheb.shape)
    Jx_cheb      = np.setdiff1d(np.arange(p**2),Ji_cheb)

    tmp_cheb          = np.linalg.solve(block_diag(T,T,T,T),uu_exact_ext )
    uu_calc[Jx_stack] = project_chebyshev_square(tmp_cheb.reshape(4,p),p).reshape(4*p,1)
    uu_calc[Ji_cheb]  = - np.linalg.solve(Aloc[Ji_cheb][:,Ji_cheb], \
        Aloc[Ji_cheb][:,Jx_cheb] @ uu_calc[Jx_cheb])

    print("Legendre to cheb %5.2e"% np.linalg.norm(uu_calc - uu_exact_cheb) / np.linalg.norm(uu_exact_cheb))