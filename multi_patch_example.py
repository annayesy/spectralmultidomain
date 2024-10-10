import numpy as np

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

from hps.hps_multidomain import Multidomain

from scipy.linalg import block_diag

from matplotlib import pyplot as plt

a = 1/8; p = 10; kh = 2

patch_utils = PatchUtils(a,p,ndim=2)
pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
box_geom    = np.array([[0,0],[1,1]])
multi       = Multidomain(pdo,box_geom,a,p)

######################################################################

fig = plt.figure()
ax  = fig.add_subplot()

#ax.scatter(multi.XX[:,0],multi.XX[:,1])
ax.scatter(multi.XX[multi.Icopy1,0],multi.XX[multi.Icopy1,1])
ax.scatter(multi.XX[multi.I_X,0],multi.XX[multi.I_X,1])
plt.savefig("multi.pdf")

assert( np.linalg.norm (multi.XX[multi.Icopy1] - multi.XX[multi.Icopy2]) == 0)

A    = multi.A.todense()

A_CC  = A[multi.Icopy1][:,multi.Icopy1] + A[multi.Icopy2][:,multi.Icopy2]
A_CC += A[multi.Icopy1][:,multi.Icopy2] + A[multi.Icopy2][:,multi.Icopy1]

A_CX  = A[multi.Icopy1][:,multi.I_X]    + A[multi.Icopy2][:,multi.I_X]

uu      = get_known_greens(multi.XX,kh)
uu_sol  = - np.linalg.solve(A_CC, A_CX @ uu[multi.I_X])
uu_true = uu[multi.Icopy1]

relerr = np.linalg.norm(uu_sol - uu_true ,ord=2)/ np.linalg.norm(uu_true,ord=2)
print(relerr)