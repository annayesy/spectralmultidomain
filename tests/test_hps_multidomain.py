import numpy as np

from hps.pdo               import PDO2d,PDO3d,const,get_known_greens

from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization

def get_discretization_relerr(a,p,kh,ndim):

	if (ndim == 2):
		pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
		box_geom    = np.array([[0,0],[1.0,0.5]])
	else:
		pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
		box_geom    = np.array([[0,0,0],[0.5,1.0,0.25]])

	if (p > 2):
		solver    = HPSMultidomain(pdo,box_geom,a,p)
	else:
		solver    = FDDiscretization(pdo,box_geom,a)

	return  solver.verify_discretization(kh)

def test_fd_2d():

	a = 1/20; p = 2; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	assert relerr < 5e-10

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


def test_hps_2d():

	a = 1/8; p = 20; kh = 0; ndim = 2
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