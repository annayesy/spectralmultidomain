import numpy as np

from hps.pdo               import PDO2d,PDO3d,const,get_known_greens
from hps.geom              import BoxGeometry
from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization

import matplotlib.pyplot as plt

def get_discretization_relerr(a,p,kh,ndim,elongated_x=False,elongated_y=False):

	if (ndim == 2):
		pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
		if (elongated_x):
			box         = np.array([[0,0],[2*a,1.0]])
		elif (elongated_y):
			box         = np.array([[0,0],[1.0,2*a]])
		else:
			box         = np.array([[0,0],[1.0,0.5]])
	else:
		pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
		box         = np.array([[0,0,0],[0.5,1.0,0.25]])
	geom = BoxGeometry(box)

	if (p > 2):
		solver    = HPSMultidomain(pdo,geom,a,p)
	else:
		solver    = FDDiscretization(pdo,geom,a)

	if (p > 2):
		assert np.linalg.norm(solver.XX[solver._Jcopy1] - solver.XX[solver._Jcopy2]) < 1e-15

	if (p > 2 and not (elongated_x or elongated_y)):
		plt.scatter(solver.XX[solver._Jx,0],solver.XX[solver._Jx,1])
		plt.scatter(solver.XX[solver._Jcopy1,0],solver.XX[solver._Jcopy1,1])
		plt.axis('equal')
		plt.show()
		
		print(np.unique(solver.XX[solver._Jcopy1,0]))

		d = np.linalg.svd(solver._Aii.todense(),compute_uv=False)
		print(d)


	return  solver.verify_discretization(kh)

def test_fd_2d():

	a = 1/40; p = 2; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 4e-10

	kh = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 5e-5


def test_fd_3d():

	a = 1/20; p = 2; kh = 0; ndim = 3
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 5e-10

	kh = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 5e-5


def test_hps_2d_elongated():

	a = 1/8; p = 20; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim,elongated_x=True)
	assert relerr < 1e-12

	kh = 10
	relerr = get_discretization_relerr(a,p,kh,ndim,elongated_y=True)
	assert relerr < 1e-12


def test_hps_2d():

	a = 1/4; p = 8; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 1e-12

	kh = 10
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 1e-12


def test_hps_3d():

	a = 1/16; p = 7; kh = 0; ndim = 3
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 1e-12

	kh = 10
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 5e-8