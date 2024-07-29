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