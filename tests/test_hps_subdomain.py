import numpy as np
import matplotlib.pyplot as plt

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

def helper_test_bounds(xx,box_geom):
	ndim             = box_geom.shape[-1]
	cond1 = xx[:,0] >= box_geom[0,0] 
	cond2 = xx[:,0] <= box_geom[1,0]
	cond3 = xx[:,1] >= box_geom[0,1]
	cond4 = xx[:,1] <= box_geom[1,1]

	assert np.all(cond1); assert np.all(cond2)
	assert np.all(cond3); assert np.all(cond4)

	if (ndim == 3):
		cond5 = xx[:,2] >= box_geom[0,2] 
		cond6 = xx[:,2] <= box_geom[1,2]
		assert np.all(cond5); assert np.all(cond6)

def solve_helmholtz_on_patch(box_geom, a, p, kh=0, savefig=False):

	ndim           = box_geom.shape[-1]
	patch_utils    = PatchUtils(np.ones(ndim) * a,p,ndim=ndim)

	if (ndim == 2):
		pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	else:
		pdo = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2)) 
	leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

	helper_test_bounds(leaf_subdomain.xxloc_int,box_geom)
	helper_test_bounds(leaf_subdomain.xxloc_ext,box_geom)

	uu_exact = get_known_greens(leaf_subdomain.xxloc_int,kh)
	uu_dir   = get_known_greens(leaf_subdomain.xxloc_ext,kh)

	uu_sol   = leaf_subdomain.solve_dir(uu_dir)

	err    = uu_exact - uu_sol
	relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact)
	return relerr

def test_laplace_2dpatch():

	box_geom = np.array([[0,0],[1,1]]); a = np.array([0.5,0.5])
	p = 8
	relerr = solve_helmholtz_on_patch(box_geom,a,p,savefig=False)
	assert relerr < 1e-10

def test_helmholtz_2dpatch():
	box_geom = np.array([[0.25,0],[0.75,0.5]]); a = np.array([0.25,0.25])
	p = 18
	relerr = solve_helmholtz_on_patch(box_geom,a,p,kh=10,savefig=True)
	assert relerr < 1e-10

def test_2dconvergence_plot():

	a = 0.5; kh = 20
	p_range      = np.arange(6,30)
	relerr_range = np.zeros(p_range.shape)

	box_geom = np.array([[0,0],[1,1]])

	for (j,p) in enumerate(p_range):
		relerr_range[j] = solve_helmholtz_on_patch(box_geom,a,p,kh)

	plt.figure()
	plt.semilogy(p_range,relerr_range)
	plt.axis("equal")
	plt.xlabel("p")
	plt.ylabel("relerr")
	plt.title("Relative error in solution for Helmholtz with kh="+\
		"%5.2f\n on square patch of size %5.2f for various p" % (kh,2*a))
	plt.savefig("figures/relerr_range2d.pdf")

def test_laplace_3dpatch():

	box_geom = np.array([[0,0,0],[1,1,1]]); a = 0.5
	p = 8
	relerr = solve_helmholtz_on_patch(box_geom,a,p,savefig=False)
	assert relerr < 1e-10

def test_helmholtz_3dpatch():
	box_geom = np.array([[0.25,0,0.5],[0.75,0.5,1]]); a = 0.25
	p = 18
	relerr = solve_helmholtz_on_patch(box_geom,a,p,kh=10,savefig=True)
	assert relerr < 1e-10

def test_3dconvergence_plot():

	a = 0.5; kh = 20
	p_range      = np.arange(6,20)
	relerr_range = np.zeros(p_range.shape)

	box_geom = np.array([[0,0,0],[1,1,1]])

	for (j,p) in enumerate(p_range):
		relerr_range[j] = solve_helmholtz_on_patch(box_geom,a,p,kh)

	plt.figure()
	plt.semilogy(p_range,relerr_range)
	plt.axis("equal")
	plt.xlabel("p")
	plt.ylabel("relerr")
	plt.title("Relative error in solution for Helmholtz with kh="+\
		"%5.2f\n on cube patch of size %5.2f for various p" % (kh,2*a))
	plt.savefig("figures/relerr_range3d.pdf")

import numpy as np
import pytest

from hps.pdo import PDO2d, PDO3d, const, get_known_greens
from hps.hps_patch_utils import PatchUtils
from hps.hps_subdomain import LeafSubdomain

@pytest.mark.parametrize("ndim", [2, 3])
def test_leaf_subdomain_projection_and_solution(ndim):
    if ndim == 2:
        box_geom = np.array([[0.5, 0.25], [1.0, 0.75]])
        a = np.array([0.25,0.25])
        p = 20
        kh = 0
        patch_utils = PatchUtils(a, p, ndim=2)
        pdo = PDO2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    else:
        box_geom = np.array([[0.5, 0.25, 0.25], [1.0, 0.75, 0.75]])
        a = np.array([0.25,0.25,0.25])
        p = 10
        kh = 0
        patch_utils = PatchUtils(a, p, ndim=3)
        pdo = PDO3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(-kh**2))

    leaf = LeafSubdomain(box_geom, pdo, patch_utils)
    xx_int = leaf.xxloc_int
    xx_ext = leaf.xxloc_ext

    uu_ext = get_known_greens(xx_ext, kh)
    uu_int = get_known_greens(xx_int, kh)

    Ji = leaf.JJ_int.Ji
    if ndim == 2:
        Jx = np.hstack((leaf.JJ_int.Jl, leaf.JJ_int.Jr,
                        leaf.JJ_int.Jd, leaf.JJ_int.Ju))
    else:
        Jx = np.hstack((leaf.JJ_int.Jl, leaf.JJ_int.Jr,
                        leaf.JJ_int.Jd, leaf.JJ_int.Ju,
                        leaf.JJ_int.Jb, leaf.JJ_int.Jf))

    # Projection: Legendre <- Chebyshev
    uu_proj_ext = patch_utils.legfcheb_exterior_mat @ uu_int[Jx]
    assert np.linalg.norm(uu_proj_ext - uu_ext) < 1e-10

    # Projection: Chebyshev <- Legendre, solve
    Aloc = leaf.Aloc
    uu_calc = np.zeros_like(uu_int)
    Jx_set = np.setdiff1d(np.arange(xx_int.shape[0]), Ji)

    uu_calc[Jx] = patch_utils.chebfleg_exterior_mat @ uu_ext
    uu_calc[Ji] = -np.linalg.solve(Aloc[Ji][:, Ji], Aloc[Ji][:, Jx_set] @ uu_calc[Jx_set])

    err_Jx = np.linalg.norm(uu_calc[Jx] - uu_int[Jx])
    err_Ji = np.linalg.norm(uu_calc[Ji] - uu_int[Ji])
    assert err_Jx < 1e-10
    assert err_Ji < 1e-10

    # Dirichlet solve: uu_ext should give uu_int
    uu_solved = leaf.solve_dir(uu_ext)
    assert np.linalg.norm(uu_solved - uu_int) < 1e-7


@pytest.mark.parametrize("ndim", [2])
def test_leaf_subdomain_dtn_neumann(ndim):
    # Only defined for 2D in the original block
    box_geom = np.array([[0.5, 0.25], [1.0, 0.75]])
    a = np.array([0.25,0.25])
    p = 20
    kh = 0
    patch_utils = PatchUtils(a, p, ndim=2)
    pdo = PDO2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))

    leaf = LeafSubdomain(box_geom, pdo, patch_utils)
    xx_ext = leaf.xxloc_ext
    uu_ext = get_known_greens(xx_ext, kh=0, center=np.zeros(2,))

    dtn_result = leaf.DtN @ uu_ext

    def r_sq(xx):
        return (xx[:, 0]**2 + xx[:, 1]**2).reshape(xx.shape[0], 1)

    neu_true = np.zeros_like(dtn_result)
    Jl, Jr, Jd, Ju = leaf.JJ_ext.Jl, leaf.JJ_ext.Jr, leaf.JJ_ext.Jd, leaf.JJ_ext.Ju
    neu_true[:p]     = -box_geom[0, 0] / r_sq(xx_ext[Jl])
    neu_true[p:2*p]  = +box_geom[1, 0] / r_sq(xx_ext[Jr])
    neu_true[2*p:3*p]= -box_geom[0, 1] / r_sq(xx_ext[Jd])
    neu_true[3*p:]   = +box_geom[1, 1] / r_sq(xx_ext[Ju])

    assert np.linalg.norm(dtn_result - neu_true) < 1e-12

