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
    assert xxloc.shape[1] == 2

    m = xxloc.shape[0];
    assert Ds.D11.shape[0] == m; assert Ds.D11.shape[1] == m 

    Aloc = np.zeros((m,m))
    # accumulate Aloc according to the PDO

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
    return Aloc


class LeafSubdomain:

    def __init__(self,box_geom,pdo,patch_utils):

        lim_max = box_geom[1]
        lim_min = box_geom[0]

        box_len = lim_max - lim_min
        c       = box_len * 0.5 + lim_min

        assert np.abs( np.min(box_len) - 2 * patch_utils.a) < 1e-14
        assert np.abs( np.max(box_len) - 2 * patch_utils.a) < 1e-14

        self.pdo     = pdo
        self.xxloc   = patch_utils.zz + c
        self.diff_ops= patch_utils.Ds

        # Jx is ordered as Jl, Jr, Jd, Ju
        self.Jx  = patch_utils.JJ.Jx
        self.Jc  = patch_utils.JJ.Jc
        self.Nx  = patch_utils.Nx

    @property
    def Aloc(self):
        return get_Aloc(self.pdo,self.xxloc,self.diff_ops)

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

        return self.Nx[:,self.Jx] - self.Nx[:,self.Jc] @ np.linalg.solve(Acc,Acx)