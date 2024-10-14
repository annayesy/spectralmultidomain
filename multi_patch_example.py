import numpy as np

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

from hps.hps_multidomain import Multidomain
from time import time
from matplotlib import pyplot as plt

a = 1/4; p = 20; kh = 0; ndim = 2

if (ndim == 2):
	pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	box_geom    = np.array([[0,0],[1.0,0.5]])
else:
	pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
	box_geom    = np.array([[0,0,0],[1,1,1]])

tic       = time()  
multi     = Multidomain(pdo,box_geom,a,p)
toc_dtn   = time() - tic

######################################################################

assert( np.linalg.norm (multi.XX[multi.I_copy1] - multi.XX[multi.I_copy2]) == 0)

fig = plt.figure()
ax  = fig.add_subplot() if ndim == 2 else fig.add_subplot(projection='3d')

if (ndim == 2):

	ax.scatter(multi.XX[multi.I_X,0],multi.XX[multi.I_X,1])
	ax.scatter(multi.XX[multi.I_copy1,0],multi.XX[multi.I_copy1,1])
	ax.set_aspect('equal','box')

else:
	
	ax.scatter(multi.XX[multi.I_X,0],multi.XX[multi.I_X,1],\
		multi.XX[multi.I_X,2])
	ax.scatter(multi.XX[multi.I_copy1,0],multi.XX[multi.I_copy1,1],\
		multi.XX[multi.I_copy1,2])

plt.savefig("multi.pdf")

######################################################################

tic       = time()
multi.setup()
toc_setup = time() - tic

uu      = get_known_greens(multi.XX,kh)
uu_sol  = multi.solve_dir(uu[multi.I_X])
uu_true = uu[multi.I_copy1]

relerr = np.linalg.norm(uu_sol - uu_true ,ord=2)/ np.linalg.norm(uu_true,ord=2)
print ("Multi-npans",multi.npan_dim, "p=%d"%multi.p)
print ("Time to ( get DtNs , setup multilevel solver) = (%5.2f,%5.2f) s" % \
	(toc_dtn,toc_setup))
print("Relative error %2.5e" % relerr)

print("Condition number of ACC %5.2f" % np.linalg.cond(multi.A_CC.todense()))