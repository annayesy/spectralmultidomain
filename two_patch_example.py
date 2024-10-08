import numpy as np

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

from matplotlib import pyplot as plt

a = 0.25; p = 10; kh = 2

patch_utils = PatchUtils(a,p,ndim=2)
pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

leaf_left = LeafSubdomain( np.array([[0.0,0.0],[0.5,0.5]]),pdo,patch_utils)
leaf_right= LeafSubdomain( np.array([[0.5,0.0],[1.0,0.5]]),pdo,patch_utils)


fig = plt.figure()
ax  = fig.add_subplot()

ax.scatter(leaf_left.xxloc[:,0], leaf_left.xxloc[:,1])
ax.scatter(leaf_right.xxloc[:,0],leaf_right.xxloc[:,1])
ax.set_aspect("equal","box")

plt.savefig("two_patch.pdf")

######################################################################################
# assemble sparse system

DtNleft  = leaf_left.DtN
DtNright = leaf_right.DtN

# TODO 