import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

from hps2d.pdo import PDO2d,const
from hps2d.hps_leaf_ops import get_Aloc
from hps2d.hps_patch import Patch

def get_known_greens(xx,kh):
	ddsq     = np.multiply(xx[:,0]-3,xx[:,0]-3) \
	+ np.multiply(xx[:,1]-3,xx[:,1]-3)
	r_points = np.sqrt(ddsq)

	if (kh == 0):
		uu_exact = (1/(2*np.pi)) * np.log(r_points)
	else:
		uu_exact = j0(kh * r_points)
	return uu_exact


def solve_helmholtz_on_patch(a,p,kh=0,savefig=False):
	hps_patch = Patch(a,p)

	# Discretization is on a patch of size [-a,a] x [-a,a] with p x p Chebyshev nodes
	zzloc = hps_patch.zz;
	Jx      = hps_patch.JJ.Jx
	Jc      = hps_patch.JJ.Jc
	Jcorner = np.setdiff1d(np.arange(p**2),np.union1d(Jx,Jc))

	if (savefig):
		plt.figure()
		plt.scatter(zzloc[:,0],zzloc[:,1])
		plt.scatter(zzloc[Jc,0],zzloc[Jc,1],label='Jc')
		plt.scatter(zzloc[Jx,0],zzloc[Jx,1],label='Jx')
		plt.scatter(zzloc[Jcorner,0],zzloc[Jcorner,1],label='Jcorner')

		plt.legend()
		plt.axis('equal')
		plt.title('HPS patch, a=%5.2f, p=%2.0d' % (a,p))
		plt.savefig("figures/zzloc.pdf")

	# Discretize the Laplace equation
	diff_ops = hps_patch.Ds

	pdo  = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	Aloc = get_Aloc(pdo,zzloc,diff_ops)

	uu_exact = get_known_greens(zzloc,kh)

	uu_sol = np.zeros(uu_exact.shape)
	uu_sol[Jx]      = uu_exact[Jx]
	uu_sol[Jcorner] = uu_exact[Jcorner]

	# when the PDO has no cross derivatives, computing with the corner points is not necessary
	uu_sol[Jc]      = -np.linalg.solve(Aloc[Jc][:,Jc],Aloc[Jc][:,Jx] @ uu_exact[Jx])

	err    = uu_exact - uu_sol
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
