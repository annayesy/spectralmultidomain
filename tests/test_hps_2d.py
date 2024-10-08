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

def solve_helmholtz_on_patch(a,p,kh=0,savefig=False):

	patch_utils    = PatchUtils(a,p)
	box_geom       = np.array([[-a,a],[-a,a]])

	pdo            = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

	uu_exact = get_known_greens(leaf_subdomain.xxloc,kh)
	uu_sol   = leaf_subdomain.solve_dir(uu_exact[leaf_subdomain.Jx])

	err    = uu_exact[leaf_subdomain.Jc] - uu_sol
	relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact)
	return relerr

def test_laplace_patch():
	a = 0.5; p = 8
	relerr = solve_helmholtz_on_patch(a,p,savefig=False)
	assert relerr < 1e-10

def test_helmholtz_patch():
	a = 0.5; p = 18
	relerr = solve_helmholtz_on_patch(a,p,kh=10,savefig=True)
	assert relerr < 1e-10

def test_convergence_plot():

	a = 0.5; kh = 20
	p_range      = np.arange(8,30)
	relerr_range = np.zeros(p_range.shape)

	for (j,p) in enumerate(p_range):
		relerr_range[j] = solve_helmholtz_on_patch(a,p,kh)

	plt.figure()
	plt.semilogy(p_range,relerr_range)
	plt.axis("equal")
	plt.xlabel("p")
	plt.ylabel("relerr")
	plt.title("Relative error in solution for Helmholtz with kh="+\
		"%5.2f\n on square patch of size %5.2f for various p" % (kh,2*a))
	plt.savefig("figures/relerr_range.pdf")
