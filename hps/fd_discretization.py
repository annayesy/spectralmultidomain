import numpy as np

from scipy.sparse        import kron, diags, block_diag
from scipy.sparse        import eye as speye, linalg as spla
from hps.pde_solver      import AbstractPDESolver

#######          GRID FOR 2D and 3D      #########
# Given box geometry, generates the grid points in a rectangle.
def grid(box_geom,h):

    # discretize each box dimension
    d = box_geom.shape[-1]
    xx0 = np.arange(box_geom[0,0],box_geom[1,0]+0.5*h,h)
    xx1 = np.arange(box_geom[0,1],box_geom[1,1]+0.5*h,h)
    if (d == 3):
        xx2 = np.arange(box_geom[0,2],box_geom[1,2]+0.5*h,h)

    if (d == 2):
        ns = np.array([xx0.shape[0],xx1.shape[0]],dtype=int)
    else:
        ns = np.array([xx0.shape[0],xx1.shape[0],xx2.shape[0]],dtype=int)

    # get meshgrid
    if (d == 2):
        XX0,XX1 = np.meshgrid(xx0,xx1,indexing='ij')
        XX  = np.vstack((XX0.flatten(),XX1.flatten()))

    elif (d == 3):
        XX0,XX1, XX2 = np.meshgrid(xx0,xx1,xx2,indexing='ij')
        XX  = np.vstack((XX0.flatten(),\
            XX1.flatten(),XX2.flatten()))

    XX   = XX.T
    hmin = np.max(XX[1] - XX[0])

    cond0 = np.logical_and(XX[:,0] > box_geom[0,0] + 0.25*hmin,\
        XX[:,0] < box_geom[1,0] - 0.25*hmin) 
    cond1 = np.logical_and(XX[:,1] > box_geom[0,1] + 0.25*hmin,\
        XX[:,1] < box_geom[1,1] - 0.25*hmin)

    if (d == 3): 
        cond2 = np.logical_and(XX[:,2] > box_geom[0,2] + 0.25*hmin,\
            XX[:,2] < box_geom[1,2] - 0.25*hmin) 
    else:
        cond2 = True

    I_C = np.where(np.logical_and(np.logical_and(cond0,cond1),cond2))[0]
    I_X = np.setdiff1d(np.arange(XX.shape[0]),I_C)

    return ns,XX,I_C,I_X

def assemble_sparse(pdo_op,npoints_dim,XX):
    d = XX.shape[-1]
    h = np.max(XX[1] - XX[0])

    if (d == 2):

        n0,n1 = npoints_dim
        d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
        d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')

        d0   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n0,n0),format='csc')
        d1   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n1,n1),format='csc')

        D00 = kron(d0sq,speye(n1))
        D11 = kron(speye(n0),d1sq)

        c00_diag = np.array(pdo_op.c11(XX)).reshape(n0*n1,)
        C00 = diags(c00_diag, 0, shape=(n0*n1,n0*n1))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(n0*n1,)
        C11 = diags(c11_diag, 0, shape=(n0*n1,n0*n1))

        A = - C00 @ D00 - C11 @ D11

        if (pdo_op.c12 is not None):
            c_diag = np.array(pdo_op.c12(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D01 = kron(d0,d1)
            A  -= 2 * S @ D01

        if (pdo_op.c1 is not None):
            c_diag = np.array(pdo_op.c1(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D0 = kron(d0,speye(n1))
            A  += S @ D0

        if (pdo_op.c2 is not None):
            c_diag = np.array(pdo_op.c1(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D0 = kron(speye(n0),d1)
            A  += S @ D1

        if (pdo_op.c is not None):
            c_diag = np.array(pdo_op.c(XX)).reshape(n0*n1,)
            S = diags(c_diag, 0, shape=(n0*n1,n0*n1))
            A += S

    elif (d == 3):

        n0,n1,n2 = npoints_dim
        d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
        d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
        d2sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2),format='csc')

        D00 = kron(d0sq,kron(speye(n1),speye(n2)))
        D11 = kron(speye(n0),kron(d1sq,speye(n2)))
        D22 = kron(speye(n0),kron(speye(n1),d2sq))

        N = n0*n1*n2
        c00_diag = np.array(pdo_op.c11(XX)).reshape(N,)
        C00 = diags(c00_diag, 0, shape=(N,N))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(N,)
        C11 = diags(c11_diag, 0, shape=(N,N))
        c22_diag = np.array(pdo_op.c33(XX)).reshape(N,)
        C22 = diags(c22_diag, 0, shape=(N,N))

        A = - C00 @ D00 - C11 @ D11 - C22 @ D22

        if ((pdo_op.c1 is not None) or \
            (pdo_op.c2 is not None) or \
            (pdo_op.c3 is not None) or \
            (pdo_op.c12 is not None) or \
            (pdo_op.c13 is not None) or \
            (pdo_op.c23 is not None)):
            raise ValueError

        if (pdo_op.c is not None):
            c_diag = np.array(pdo_op.c(XX)).reshape(N,)
            S = diags(c_diag, 0, shape=(N,N))
            A += S
    return A

#######          GRID FOR 2D and 3D      #########
# Given box geometry, generates the grid points in a rectangle.
class FDDiscretization(AbstractPDESolver):

    def __init__(self,pdo,geom,h,kh=0):

        self._geom = geom
        self._npoints_dim, self._XX, self.J_C, self.J_X = grid(self.geom.bounds,h)

        self.pdo       = pdo

        self.A         = assemble_sparse(self.pdo,self.npoints_dim,self._XX)

    @property
    def XX(self):
        return self._XX

    @property
    def p(self):
        return 2

    @property
    def geom(self):
        return self._geom

    @property
    def npoints_dim(self):
        return self._npoints_dim

    @property
    def I_C(self):
        return self.J_C

    @property
    def I_X(self):
        return self.J_X
    
    @property
    def A_CC(self):
        return self.A[self.I_C][:,self.I_C]

    @property
    def A_CX(self):
        return self.A[self.I_C][:,self.I_X]

    @property
    def A_XX(self):
        return self.A[self.I_X][:,self.I_X]

    @property
    def A_XC(self):
        return self.A[self.I_X][:,self.I_C]