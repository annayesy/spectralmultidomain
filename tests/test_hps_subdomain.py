import numpy as np
import matplotlib.pyplot as plt

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

def solve_helmholtz_on_patch(box_geom, a, p, kh=0, savefig=False):

	ndim           = box_geom.shape[-1]
	patch_utils    = PatchUtils(a,p,ndim=ndim)

	if (ndim == 2):
		pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	else:
		pdo = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2)) 
	leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

	cond1 = leaf_subdomain.xxloc[:,0] >= box_geom[0,0] 
	cond2 = leaf_subdomain.xxloc[:,0] <= box_geom[1,0]
	cond3 = leaf_subdomain.xxloc[:,1] >= box_geom[0,1]
	cond4 = leaf_subdomain.xxloc[:,1] <= box_geom[1,1]

	assert np.all(cond1)
	assert np.all(cond2)
	assert np.all(cond3)
	assert np.all(cond4)

	if (ndim == 3):
		cond5 = leaf_subdomain.xxloc[:,2] >= box_geom[0,2] 
		cond6 = leaf_subdomain.xxloc[:,2] <= box_geom[1,2]
		assert np.all(cond5)
		assert np.all(cond6)

	uu_exact = get_known_greens(leaf_subdomain.xxloc,kh)
	uu_sol   = leaf_subdomain.solve_dir(uu_exact[leaf_subdomain.Jx])

	if (savefig):

		XX = leaf_subdomain.xxloc
		Jc = leaf_subdomain.Jc
		Jx = leaf_subdomain.Jx

		assert  Jc.shape[0] == (p-2)**ndim
		assert  Jx.shape[0] == (p-2)**(ndim - 1) * (4 if ndim == 2 else 6)

		fig = plt.figure()
		ax  = fig.add_subplot() if box_geom.shape[-1] == 2 else fig.add_subplot(projection='3d')

		if (box_geom.shape[-1] ==2):
			ax.scatter(XX[Jc,0], XX[Jc,1],color='tab:blue')
			ax.scatter(XX[Jx,0], XX[Jx,1],color='tab:red')
			ax.set_aspect('equal','box')
		else:
			ax.scatter(XX[Jc,0], XX[Jc,1], XX[Jc,2],color='tab:blue')
			ax.scatter(XX[Jx,0], XX[Jx,1], XX[Jx,2],color='tab:red')
			ax.set_box_aspect([1,1,1])
		plt.savefig('figures/leaf_%dd.pdf' % box_geom.shape[-1])

	err    = uu_exact[leaf_subdomain.Jc] - uu_sol
	relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact[leaf_subdomain.Jc])
	return relerr

def test_laplace_2dpatch():

	box_geom = np.array([[0,0],[1,1]]); a = 0.5
	p = 8
	relerr = solve_helmholtz_on_patch(box_geom,a,p,savefig=False)
	assert relerr < 1e-10

def test_helmholtz_2dpatch():
	box_geom = np.array([[0.25,0],[0.75,0.5]]); a = 0.25
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