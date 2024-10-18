import numpy as np

from hps.pdo               import PDO2d,PDO3d,const,get_known_greens

from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from time import time
from matplotlib import pyplot as plt

a = 1/8; p = 10; kh = 8; ndim = 3

if (ndim == 2):
	pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	box_geom    = np.array([[0,0],[1.0,0.5]])
else:
	pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
	box_geom    = np.array([[0,0,0],[0.5,0.5,1]])

if (p > 2):

	tic       = time()  
	solver    = HPSMultidomain(pdo,box_geom,a,p)
	toc_dtn   = time() - tic

else:
	tic       = time()  
	solver    = FDDiscretization(pdo,box_geom,a)
	toc_dtn   = time() - tic

######################################################################

tic       = time()
solver.setup_solver_CC()
toc_setup = time() - tic

relerr    = solver.verify_discretization(kh)

print("Ntot = %d, p=%d, kh = %5.2f" % (np.prod(solver.npoints_dim),solver.p,kh))
print("\t Points on each dim   ",solver.npoints_dim)

if (kh > 0):
	print("\t Points per wavelength",solver.get_ppw_ndim(kh))

print ("\t Time to ( get A , setup solver) = (%5.2f,%5.2f) s" % \
	(toc_dtn,toc_setup))
print("\t Relative error %2.5e" % relerr)

######################################################################

fig = plt.figure()
ax  = fig.add_subplot() if ndim == 2 else fig.add_subplot(projection='3d')

if (ndim == 2):

	ax.scatter(solver.XX[solver.I_X,0],solver.XX[solver.I_X,1])
	ax.scatter(solver.XX[solver.I_C,0],solver.XX[solver.I_C,1])
	ax.set_aspect('equal','box')

else:
	
	ax.scatter(solver.XX[solver.I_X,0],solver.XX[solver.I_X,1],\
		solver.XX[solver.I_X,2])
	ax.scatter(solver.XX[solver.I_C,0],solver.XX[solver.I_C,1],\
		solver.XX[solver.I_C,2])
	ax.set_box_aspect([1,1,1])

plt.show()