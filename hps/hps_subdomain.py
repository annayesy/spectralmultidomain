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

    def __init__(self,box_centers,pdo,patch_utils):

        assert box_centers.shape[-1] == patch_utils.ndim
        if (box_centers.ndim == 1):
            self.centers   = box_centers[None,:]
        else:
            self.centers   = box_centers

        self.pdo       = pdo
        self.xxloc_int = patch_utils.zz_int[None,:,:] + self.centers[:,None,:]
        self.xxloc_ext = patch_utils.zz_ext[None,:,:] + self.centers[:,None,:]
        self.diff_ops  = patch_utils.Ds
        self.p         = patch_utils.p
        self.utils     = patch_utils
        self.ndim      = patch_utils.ndim

        self.nx_leg       = self.xxloc_ext.shape[1]
        self.nt_cheb      = self.xxloc_int.shape[1]
        self.nbatch       = self.centers.shape[0]

        JJ_cheb  = patch_utils.JJ_int
        JJext_leg= patch_utils.JJ_ext

        if (self.utils.ndim == 2):

            self.Jx_cheb = np.hstack((JJ_cheb.Jl, JJ_cheb.Jr,\
                JJ_cheb.Jd, JJ_cheb.Ju))
        else:
            self.Jx_cheb = np.hstack((JJ_cheb.Jl, JJ_cheb.Jr,\
                JJ_cheb.Jd, JJ_cheb.Ju, JJ_cheb.Jb, JJ_cheb.Jf ))
        self.Ji_cheb     = JJ_cheb.Ji
        self.Jx_cheb_uni = np.setdiff1d(np.arange(self.nt_cheb),self.Ji_cheb)

        self.Nx_cheb     = self.utils.Nx_stack

        self.chebfleg_mat = self.utils.chebfleg_exterior_mat
        self.legfcheb_mat = self.utils.legfcheb_exterior_mat

    @property
    def Aloc(self):

        Aloc = np.zeros((self.nbatch,self.nt_cheb,self.nt_cheb))

        for j in range(self.nbatch):
            Aloc[j,:,:] = get_Aloc(self.pdo,self.xxloc_int[j],self.diff_ops)

        return Aloc

    def solve_dir(self,uu_dir,ff_body=None):

        if (uu_dir.ndim == 2):
            uu_dir = uu_dir[None,:,:]
        assert uu_dir.shape[0] == self.nbatch
        assert uu_dir.shape[1] == self.nx_leg

        nrhs = uu_dir.shape[-1]
        res  = np.zeros((self.nbatch,self.nt_cheb,nrhs))

        Aloc = self.Aloc
        if (ff_body is None):
            ff_body = np.zeros((self.nbatch,self.nt_cheb,nrhs))

        for j in range(self.nbatch):

            res[j,self.Jx_cheb] = self.chebfleg_mat @ uu_dir[j]
            res[j,self.Ji_cheb] = np.linalg.solve(Aloc[j,self.Ji_cheb][:,self.Ji_cheb], \
                ff_body[j,self.Ji_cheb] - Aloc[j,self.Ji_cheb][:,self.Jx_cheb_uni] @ res[j,self.Jx_cheb_uni])

        return res

    def reduce_body_load(self,ff_body):

        if (ff_body.ndim == 2):
            ff_body = ff_body[None,:,:]
        assert ff_body.shape[0] == self.nbatch
        assert ff_body.shape[1] == self.nt_cheb

        loc_sol  = self.solve_dir(np.zeros((self.nbatch,self.nx_leg,1)),-ff_body)
        red_f    = self.legfcheb_mat @ (self.Nx_cheb[:,self.Ji_cheb] @ loc_sol[:,self.Ji_cheb])
        return red_f

    @property
    def DtN(self):

        mat      = mat = np.eye(self.nx_leg)[None, :, :].repeat(self.nbatch, axis=0)
        loc_sol  = self.solve_dir(mat)
        neu_sol  = self.Nx_cheb @ loc_sol

        DtNs     = self.legfcheb_mat @ neu_sol
        return DtNs
