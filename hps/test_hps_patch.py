import numpy as np
import matplotlib.pyplot as plt

from hps.pdo             import PDO2d,PDO3d,const,get_known_greens
from hps.hps_subdomain   import LeafSubdomain
from hps.hps_patch_utils import PatchUtils

box_geom = np.array([[0.25,0],[0.75,0.5]]); a = 0.25; p = 18; kh = 10
savefig  = True

ndim           = box_geom.shape[-1]
patch_utils    = PatchUtils(a,p,ndim=ndim)

pdo = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))

leaf_subdomain = LeafSubdomain(box_geom, pdo, patch_utils)

cond1 = leaf_subdomain.xxloc[:,0] >= box_geom[0,0] 
cond2 = leaf_subdomain.xxloc[:,0] <= box_geom[1,0]
cond3 = leaf_subdomain.xxloc[:,1] >= box_geom[0,1]
cond4 = leaf_subdomain.xxloc[:,1] <= box_geom[1,1]

assert np.all(cond1)
assert np.all(cond2)
assert np.all(cond3)
assert np.all(cond4)

uu_exact = get_known_greens(leaf_subdomain.xxloc,kh)
uu_sol   = leaf_subdomain.solve_dir(uu_exact[leaf_subdomain.Jx])

if (savefig):

	XX = leaf_subdomain.xxloc
	Jc = leaf_subdomain.Jc
	Jx = leaf_subdomain.Jx

	assert  Jc.shape[0] == (p-2)**ndim
	assert  Jx.shape[0] == (p-2)**(ndim - 1) * (4 if ndim == 2 else 6)

	fig = plt.figure()
	ax  = fig.add_subplot()

	ax.scatter(XX[Jc,0], XX[Jc,1],color='tab:blue')
	ax.scatter(XX[Jx,0], XX[Jx,1],color='tab:red')
	ax.set_aspect('equal','box')

	plt.savefig("xxloc.png",transparent=True,dpi=300)

err    = uu_exact[leaf_subdomain.Jc] - uu_sol
relerr = np.linalg.norm(err) / np.linalg.norm(uu_exact[leaf_subdomain.Jc])
print(relerr)