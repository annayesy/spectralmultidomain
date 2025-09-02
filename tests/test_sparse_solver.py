# test_sparse_solver.py
import os
import numpy as np
import pytest
from scipy.sparse import diags, csr_matrix
from numpy.linalg import norm

from hps.sparse_utils import SparseSolver

# ------------------
# Helpers & fixtures
# ------------------

def spd_tridiag(n=20, dtype=float):
    """Simple SPD tridiagonal 1D Poisson operator (Dirichlet)."""
    main = 2.0 * np.ones(n, dtype=dtype)
    off  = -1.0 * np.ones(n - 1, dtype=dtype)
    A = diags([off, main, off], [-1, 0, 1], shape=(n, n), dtype=dtype)
    return csr_matrix(A)

def nonsym_strict(n=20, seed=0, dtype=float):
    """Well-conditioned nonsymmetric matrix."""
    rng = np.random.default_rng(seed)
    # start with SPD and add a small strictly upper component to break symmetry
    A = spd_tridiag(n, dtype=dtype).astype(dtype).toarray()
    U = np.triu(rng.normal(scale=1e-2, size=(n, n)), 1)
    A = A + U
    return csr_matrix(A)

def random_rhs(n, m=1, seed=123, dtype=float):
    rng = np.random.default_rng(seed)
    if m == 1:
        return rng.normal(size=n).astype(dtype)
    return rng.normal(size=(n, m)).astype(dtype)

def reverse_perm(n):
    return np.arange(n-1, -1, -1, dtype=int)

# PETSc capability checks (safe for environments without petsc4py)
def _petsc_imported():
    try:
        from petsc4py import PETSc  # noqa: F401
        return True
    except Exception:
        return False

def _petsc_has_mumps():
    if not _petsc_imported():
        return False
    from petsc4py import PETSc
    try:
        # Minimal capability probe: create a dummy LU(MUMPS) PC
        A = spd_tridiag(3)
        pA = PETSc.Mat().createAIJ(A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create()
        pc = ksp.getPC()
        ksp.setOperators(pA)
        ksp.setType("preonly")
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setUp()
        # clean
        pA.destroy()
        return True
    except Exception:
        return False

# ------------------
# SciPy backend tests
# ------------------

@pytest.mark.parametrize("n", [10, 25])
@pytest.mark.parametrize("use_perm", [False, True])
def test_scipy_symmetric_vector_and_matmat(n, use_perm):
    A = spd_tridiag(n)
    b = random_rhs(n)
    B = random_rhs(n, m=3)

    perm = reverse_perm(n) if use_perm else None
    solver = SparseSolver(A, use_approx=False, perm=perm)

    # vector solve
    x = solver.solve_op.matvec(b)
    rel = norm(A @ x - b) / max(1.0, norm(b))
    assert rel < 1e-10

    # matmat solve (3 RHS)
    X = solver.solve_op.matmat(B)
    relM = norm(A @ X - B) / max(1.0, norm(B))
    assert relM < 1e-10

@pytest.mark.parametrize("n", [12])
@pytest.mark.parametrize("use_perm", [False, True])
def test_scipy_nonsymmetric_vector_matmat_and_transpose(n, use_perm):
    A = nonsym_strict(n)
    b = random_rhs(n)
    B = random_rhs(n, m=2)

    perm = reverse_perm(n) if use_perm else None
    solver = SparseSolver(A, use_approx=False, perm=perm)

    # A x = b
    x = solver.solve_op.matvec(b)
    rel = norm(A @ x - b) / max(1.0, norm(b))
    assert rel < 1e-10

    # A X = B
    X = solver.solve_op.matmat(B)
    relM = norm(A @ X - B) / max(1.0, norm(B))
    assert relM < 1e-10

    # (A^T) y = c  via rmatvec / rmatmat
    c = random_rhs(n, seed=999)
    y = solver.solve_op.rmatvec(c)
    relT = norm(A.T @ y - c) / max(1.0, norm(c))
    assert relT < 1e-10

    C = random_rhs(n, m=2, seed=1001)
    Y = solver.solve_op.rmatmat(C)
    relTM = norm(A.T @ Y - C) / max(1.0, norm(C))
    assert relTM < 1e-10

# ------------------
# PETSc backend tests (skip if unavailable)
# ------------------

petsc = pytest.mark.skipif(not _petsc_imported(), reason="petsc4py not available")
mumps = pytest.mark.skipif(not _petsc_has_mumps(), reason="PETSc MUMPS not available")

@petsc
@mumps
@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("use_perm", [False, True])
def test_petsc_mumps_symmetric_vector_and_matmat(n, use_perm):
    A = spd_tridiag(n)
    b = random_rhs(n)
    B = random_rhs(n, m=3)

    perm = reverse_perm(n) if use_perm else None
    solver = SparseSolver(A, use_approx=False, perm=perm)

    # vector
    x = solver.solve_op.matvec(b)
    rel = norm(A @ x - b) / max(1.0, norm(b))
    assert rel < 1e-12

    # multi-RHS
    X = solver.solve_op.matmat(B)
    relM = norm(A @ X - B) / max(1.0, norm(B))
    assert relM < 1e-12

@petsc
@mumps
@pytest.mark.parametrize("n", [12])
@pytest.mark.parametrize("use_perm", [False, True])
def test_petsc_mumps_nonsymmetric_vector_matmat_and_transpose(n, use_perm):
    A = nonsym_strict(n)
    b = random_rhs(n)
    B = random_rhs(n, m=2)

    perm = reverse_perm(n) if use_perm else None
    solver = SparseSolver(A, use_approx=False, perm=perm)

    # A x = b
    x = solver.solve_op.matvec(b)
    rel = norm(A @ x - b) / max(1.0, norm(b))
    assert rel < 1e-12

    # A X = B
    X = solver.solve_op.matmat(B)
    relM = norm(A @ X - B) / max(1.0, norm(B))
    assert relM < 1e-12

    # (A^T) y = c
    c = random_rhs(n, seed=2024)
    y = solver.solve_op.rmatvec(c)
    relT = norm(A.T @ y - c) / max(1.0, norm(c))
    assert relT < 1e-12

    C = random_rhs(n, m=2, seed=2025)
    Y = solver.solve_op.rmatmat(C)
    relTM = norm(A.T @ Y - C) / max(1.0, norm(C))
    assert relTM < 1e-12

# ------------------
# (Optional) PETSc iterative mode
# ------------------
# Your code sets Hypre as the PC in approximate mode. If PETSc is not built with Hypre,
# KSPSetUp will error. We therefore *xfail* this test if it raises a PETSc error.

@petsc
@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("use_perm", [False, True])
def test_petsc_iterative_gmres_hypre_if_available(n, use_perm):
    from petsc4py import PETSc
    A = spd_tridiag(n)  # SPD is easy for GMRES+Hypre
    b = random_rhs(n)
    perm = reverse_perm(n) if use_perm else None

    try:
        solver = SparseSolver(A, use_approx=True, perm=perm)
    except PETSc.Error:
        pytest.xfail("PETSc Hypre not available in this build")

    x = solver.solve_op.matvec(b)
    rel = norm(A @ x - b) / max(1.0, norm(b))
    assert rel < 1e-8  # a bit looser for iterative


