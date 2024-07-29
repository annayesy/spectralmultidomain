import numpy as np
from numpy_hps.hps_patch import Patch
import matplotlib.pyplot as plt

def solve_laplace_on_patch(a,p,savefig=False):
	hps_patch = Patch(a,p)

	# Discretization is on a patch of size [-a,a] x [-a,a] with p x p Chebyshev nodes
	zzloc = hps_patch.zz;
	Jx      = hps_patch.JJ.Jx
	Jc      = hps_patch.JJ.Jc
	Jcorner = np.setdiff1d(np.arange(p**2),np.union1d(Jx,Jc))

	if (savefig):
		plt.figure()
		plt.scatter(zzloc[0],zzloc[1])
		plt.scatter(zzloc[0,Jc],zzloc[1,Jc],label='Jc')
		plt.scatter(zzloc[0,Jx],zzloc[1,Jx],label='Jx')
		plt.scatter(zzloc[0,Jcorner],zzloc[1,Jcorner],label='Jcorner')

		plt.legend()
		plt.axis('equal')
		plt.title('HPS patch, a=%5.2f, p=%2.0d' % (a,p))
		plt.savefig("figures/zzloc.pdf")

	# Discretize the Laplace equation
	diff_ops = hps_patch.Ds
	Aloc     = - diff_ops.D11 - diff_ops.D22

	# Green's function centered at (3,3) which is away from the patch
	ddsq     = np.multiply(zzloc[0]-3,zzloc[0]-3) + np.multiply(zzloc[1]-3,zzloc[1]-3)
	r_points = np.sqrt(ddsq)

	uu_exact = (1/(2*np.pi)) * np.log(r_points)

	uu_sol = np.zeros(uu_exact.shape)
	uu_sol[Jx]      = uu_exact[Jx]
	uu_sol[Jcorner] = uu_exact[Jcorner]

	# when the PDO has no cross derivatives, computing with the corner points is not necessary
	uu_sol[Jc]      = -np.linalg.solve(Aloc[Jc][:,Jc],Aloc[Jc][:,Jx] @ uu_exact[Jx])

	err    = uu_exact - uu_sol
	relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact)
	return relerr

def test_patch():
	a = 0.5; p = 8
	relerr = solve_laplace_on_patch(a,p,savefig=True)
	assert relerr < 1e-10

def test_convergence_plot():

	a = 0.5; 
	p_range      = np.arange(4,20)
	relerr_range = np.zeros(p_range.shape)

	for (j,p) in enumerate(p_range):
		relerr_range[j] = solve_laplace_on_patch(a,p)

	plt.figure()
	plt.semilogy(p_range,relerr_range)
	plt.axis("equal")
	plt.xlabel("p")
	plt.ylabel("relerr")
	plt.savefig("figures/relerr_range.pdf")
	assert True