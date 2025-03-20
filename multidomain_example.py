import numpy as np

from hps.pdo               import PDO2d,PDO3d,const,get_known_greens
from hps.geom              import BoxGeometry

from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from time import time
from matplotlib import pyplot as plt

a = 1/8; p = 10; kh = 8; ndim = 2

if (ndim == 2):
	pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0],[1.0,0.5]])
	geom        = BoxGeometry(box)
else:
	pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0,0],[0.5,0.5,1]])
	geom        = BoxGeometry(box)

if (p > 2):

	tic       = time()
	solver    = HPSMultidomain(pdo,geom,a,p)
	toc_dtn   = time() - tic

else:
	tic       = time()
	solver    = FDDiscretization(pdo,geom,a)
	toc_dtn   = time() - tic

######################################################################

tic       = time()
solver.setup_solver_Aii()
toc_setup = time() - tic

relerr    = solver.verify_discretization(kh)

print("Ntot = %d, p=%d, kh = %5.2f" % (np.prod(solver.npoints_dim),solver.p,kh))
print("\t Points on each dim   ",solver.npoints_dim)

if (kh > 0):
	print("\t Nwaves on each dim   ",solver.get_nwaves_dim(kh))

print ("\t Time to ( get A , setup solver) = (%5.2f,%5.2f) s" % \
	(toc_dtn,toc_setup))
print("\t Relative error %2.5e" % relerr)

######################################################################

fig = plt.figure()
ax  = fig.add_subplot() if ndim == 2 else fig.add_subplot(projection='3d')

if (ndim == 2):

	ax.scatter(solver.XX[solver.Jx,0],solver.XX[solver.Jx,1])
	ax.scatter(solver.XX[solver.Ji,0],solver.XX[solver.Ji,1])
	ax.set_aspect('equal','box')

else:

	ax.scatter(solver.XX[solver.Jx,0],solver.XX[solver.Jx,1],\
		solver.XX[solver.Jx,2])
	ax.scatter(solver.XX[solver.Ji,0],solver.XX[solver.Ji,1],\
		solver.XX[solver.Ji,2])
	ax.set_box_aspect([1,1,1])

plt.show()
