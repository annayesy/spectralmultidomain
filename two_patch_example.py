import numpy as np

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

from hps.hps_multidomain import Multidomain

from scipy.linalg import block_diag

from matplotlib import pyplot as plt

a = 0.25; p = 10; kh = 2

patch_utils = PatchUtils(a,p,ndim=2)
pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

leaf_left = LeafSubdomain( np.array([[0.0,0.0],[0.5,0.5]]),pdo,patch_utils)
leaf_right= LeafSubdomain( np.array([[0.5,0.0],[1.0,0.5]]),pdo,patch_utils)

#########################################################################
# assemble sparse system

DtN_list = np.stack((leaf_left.DtN,  leaf_right.DtN))
xx_list  = np.stack((leaf_left.xxloc[leaf_left.Jx],\
	leaf_right.xxloc[leaf_right.Jx]))

xx_tot   = xx_list.reshape(DtN_list.shape[0] * DtN_list.shape[1],2)

########
# enforce continuity

bndsize = (p-2)
DtNsize = 4*bndsize

Icopy1 = np.arange(bndsize) + bndsize
Icopy2 = np.arange(bndsize) + DtNsize

########

A = block_diag(*DtN_list)

########

Ictot = np.hstack((Icopy1,Icopy2))
Idir  = np.setdiff1d(np.arange(A.shape[0]), Ictot)

# Solve the PDE

Acc = A[Icopy1][:,Icopy1] + A[Icopy2][:,Icopy2]
Acx = A[Icopy1][:,Idir]   + A[Icopy2][:,Idir]

uu_true = get_known_greens(xx_tot,kh)
uu_calc = - np.linalg.solve(Acc, Acx @ uu_true[Idir])

relerr = np.linalg.norm(uu_calc - uu_true[Icopy1],\
	ord=2) / np.linalg.norm(uu_true[Icopy1],ord=2)
print(relerr)

#########################################################################

a = 1/8; p = 10; kh = 2

patch_utils = PatchUtils(a,p,ndim=2)
pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

box_geom    = np.array([[0,0],[1,1]])

multi       = Multidomain(pdo,box_geom,a,p)

fig = plt.figure()
ax  = fig.add_subplot()

for j in range(multi.xx_list.shape[0]):
	ax.scatter(multi.xx_list[j,:,0], multi.xx_list[j,:,1])

plt.savefig("multi.pdf")