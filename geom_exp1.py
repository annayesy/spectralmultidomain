import numpy as np

from hps.pdo               import PDO2d, PDO3d, const, get_known_greens
from hps.geom              import BoxGeometry

from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from time                  import time
from matplotlib            import pyplot as plt

# --------------------------------------------------------------------------------------
# PARAMETERS: polynomial degree p, wavenumber*scale kh, spatial dimension ndim
#---------------------------------------------------------------------------------------
p    = 10    # if p > 2, use HPS; otherwise, use FD
kh   = 30    # Helmholtz parameter (kh = 0 would reduce to Laplace)
ndim = 2     # 2 or 3

# --------------------------------------------------------------------------------------
# DEFINE PDE OPERATOR AND GEOMETRY BASED ON DIMENSION
#---------------------------------------------------------------------------------------
if ndim == 2:
    # In 2D, build a PDO2d: c11 = c22 = 1, zero mixed term, no first derivatives, c = -kh^2
    pdo  = PDO2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    # Define a rectangular domain [0,1] x [0,0.5]
    box  = np.array([[0, 0], [1.0, 0.5]])
    geom = BoxGeometry(box)
else:
    # In 3D, build a PDO3d: c11 = c22 = c33 = 1, no mixed or first derivatives, c = -kh^2
    pdo  = PDO3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(-kh**2))
    # Define a rectangular box [0,0.5] x [0,0.5] x [0,1]
    box  = np.array([[0, 0, 0], [0.5, 0.5, 1]])
    geom = BoxGeometry(box)

# --------------------------------------------------------------------------------------
# CHOOSE DISCRETIZATION METHOD: HPS IF p > 2, OTHERWISE FINITE DIFFERENCES
#---------------------------------------------------------------------------------------
if p > 2:
    # HPS requires a vector of half-lengths `a` per dimension
    # For 2D, a might be [width_x/1024, width_y/512], for 3D, similar scale
    a = np.array([1/32, 1/16, 1/32]) if ndim == 3 else np.array([1/1024, 1/512])

    tic = time()
    # Build the HPS multidomain solver: this computes per-patch DtN maps and assembles sparse blocks
    solver = HPSMultidomain(pdo, geom, a, p, verbose=True)
    toc_dtn = time() - tic

else:
    # For low p, use simple finite-difference discretization with mesh spacing a
    a = 1/100  # uniform grid spacing in all dimensions

    tic = time()
    solver = FDDiscretization(pdo, geom, a)
    toc_dtn = time() - tic

# --------------------------------------------------------------------------------------
# REPORT PROBLEM SIZE AND TIMINGS
#---------------------------------------------------------------------------------------
Ntot = np.prod(solver.npoints_dim)
print("Ntot = %d, p=%d, kh = %5.2f" % (Ntot, solver.p, kh))

if p > 2:
    # For HPS, also report time to build sparse Schur complement (stats from solver)
    print(
        "\t Time to (get DtNs, assemble sparse) = (%5.2f, %5.2f) s" 
        % (solver.stats['toc_dtn'], solver.stats['toc_sparse'])
    )

# --------------------------------------------------------------------------------------
# FACTORIZE OR SET UP THE Aii SOLVER
#---------------------------------------------------------------------------------------
tic = time()
solver.setup_solver_Aii()  # builds or factorizes the interior Schur block
toc_setup = time() - tic
print("\t Time to ( factorize sparse system ) = (%5.2f) s" % (toc_setup,))

# --------------------------------------------------------------------------------------
# VERIFY DISCRETIZATION USING A KNOWN GREEN'S FUNCTION SOLUTION
#---------------------------------------------------------------------------------------
relerr = solver.verify_discretization(kh)
print("\t Points on each dim   ", solver.npoints_dim)
if kh > 0:
    print("\t Nwaves on each dim   ", solver.get_nwaves_dim(kh))
print("\t Relative error %2.5e" % relerr)

# --------------------------------------------------------------------------------------
# PLOT GRID POINTS, DISTINGUISHING INTERIOR VS BOUNDARY
#---------------------------------------------------------------------------------------
fig = plt.figure()
if ndim == 2:
    ax = fig.add_subplot()  # 2D scatter plot
else:
    ax = fig.add_subplot(projection='3d')  # 3D scatter plot

if ndim == 2:
    # Plot boundary points in one color, interior in another
    ax.scatter(solver.XX[solver.Jx, 0], solver.XX[solver.Jx, 1], label='Boundary')
    ax.scatter(solver.XX[solver.Ji, 0], solver.XX[solver.Ji, 1], label='Interior')
    ax.set_aspect('equal', 'box')
else:
    ax.scatter(
        solver.XX[solver.Jx, 0],
        solver.XX[solver.Jx, 1],
        solver.XX[solver.Jx, 2],
        label='Boundary'
    )
    ax.scatter(
        solver.XX[solver.Ji, 0],
        solver.XX[solver.Ji, 1],
        solver.XX[solver.Ji, 2],
        label='Interior'
    )
    ax.set_box_aspect([1, 1, 1])

ax.legend()
plt.show()
