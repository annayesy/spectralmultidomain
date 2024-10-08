import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

from hps.pdo             import PDO2d,const
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

def get_known_greens(xx,kh):

	if (xx.shape[-1] == 2):
		ddsq     = np.multiply(xx[:,0]-3,xx[:,0]-3) + np.multiply(xx[:,1]-3,xx[:,1]-3)
		r_points = np.sqrt(ddsq)

		if (kh == 0):
			uu_exact = (1/(2*np.pi)) * np.log(r_points)
		else:
			uu_exact = j0(kh * r_points)
	else:
		raise ValueError
	
	if (uu_exact.ndim == 1):
		uu_exact = uu_exact[:,np.newaxis]
	return uu_exact

def solve_helmholtz_on_patch(box_geom, a, p, kh=0, savefig=False):

	patch_utils    = PatchUtils(a,p)

	pdo            = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

	cond1 = leaf_subdomain.xxloc[:,0] >= box_geom[0,0] 
	cond2 = leaf_subdomain.xxloc[:,0] <= box_geom[1,0]
	cond3 = leaf_subdomain.xxloc[:,1] >= box_geom[0,1] 
	cond4 = leaf_subdomain.xxloc[:,1] <= box_geom[1,1]

	assert np.all(cond1)
	assert np.all(cond2)
	assert np.all(cond3)
	assert np.all(cond4)

	if (box_geom.shape[-1] == 3):
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

		plt.scatter(XX[: ,0], XX[: ,1],color='black')
		plt.scatter(XX[Jc,0], XX[Jc,1],color='tab:blue')
		plt.scatter(XX[Jx,0], XX[Jx,1],color='tab:red')
		plt.savefig('figures/leaf.pdf')

	err    = uu_exact[leaf_subdomain.Jc] - uu_sol
	relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact[leaf_subdomain.Jc])
	return relerr

def test_laplace_patch():

	box_geom = np.array([[0,0],[1,1]]); a = 0.5
	p = 8
	relerr = solve_helmholtz_on_patch(box_geom,a,p,savefig=False)
	assert relerr < 1e-10

def test_helmholtz_patch():
	box_geom = np.array([[0.25,0],[0.75,0.5]]); a = 0.25
	p = 18
	relerr = solve_helmholtz_on_patch(box_geom,a,p,kh=10,savefig=True)
	assert relerr < 1e-10

def test_convergence_plot():

	a = 0.5; kh = 20
	p_range      = np.arange(8,30)
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
	plt.savefig("figures/relerr_range.pdf")