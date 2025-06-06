import jax
# Enable 64-bit (double) precision for all JAX computations
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hps.pdo import PDO2d, PDO3d, const, get_known_greens
from hps.geom import BoxGeometry, ParametrizedGeometry2D, ParametrizedGeometry3D

from hps.hps_multidomain import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from time import time
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------
# SETUP PARAMETERS
# --------------------------------------------------------------------------------------
a  = 1/8    # Half-width of each HPS patch in each dimension
p  = 10     # Polynomial degree (number of Chebyshev nodes per direction on each patch)
kh = 0      # Wavenumber times scale; kh=0 → Laplace; kh>0 → Helmholtz

# --------------------------------------------------------------------------------------
# DEFINE A CURVED 3D GEOMETRY VIA PARAMETRIZATION
# This mimics the Jupyter notebook demo but in standalone Python form.
# We will warp the unit cube by “shearing” the y-coordinate using a sine function in x.
# --------------------------------------------------------------------------------------
mag   = 0.25  # amplitude of the sinusoidal warping in y-direction

# Base warping function ψ(x) = 1 – mag * sin(6x)
psi   = lambda z: 1 - mag * jnp.sin(6 * z)
# First derivative ψ'(x) = –mag * 6 * cos(6x)
dpsi  = lambda z: -mag * 6 * jnp.cos(6 * z)
# Second derivative ψ''(x) = mag * 36 * sin(6x)
ddpsi = lambda z: mag * 36 * jnp.sin(6 * z)

# --------------------------------------------------------------------------------------
# FORWARD MAP z : (x_ref, y_ref, z_ref) → (x_curved, y_curved, z_curved)
# --------------------------------------------------------------------------------------
# In this example:
#   x_curved = x_ref
#   y_curved = y_ref / ψ(x_ref)   (i.e., we “stretch” y by 1/ψ(x))
#   z_curved = z_ref
z1 = lambda xx: xx[..., 0]
z2 = lambda xx: jnp.divide(xx[..., 1], psi(xx[..., 0]))
z3 = lambda xx: xx[..., 2]

# --------------------------------------------------------------------------------------
# INVERSE MAP y : (x_curved, y_curved, z_curved) → (x_ref, y_ref, z_ref)
# --------------------------------------------------------------------------------------
# Inverting the above:
#   x_ref = x_curved
#   y_ref = y_curved * ψ(x_curved)
#   z_ref = z_curved
y1 = lambda xx: xx[..., 0]
y2 = lambda xx: jnp.multiply(xx[..., 1], psi(xx[..., 0]))
y3 = lambda xx: xx[..., 2]

# --------------------------------------------------------------------------------------
# FIRST DERIVATIVES OF THE INVERSE MAP (∂y_i/∂x_j) for needed indices:
#   y1_d1 = ∂(x_ref)/∂(x_curved) = 1
#   y2_d1 = ∂(y_ref)/∂(x_curved) = y_curved * ψ'(x_curved)
#   y2_d2 = ∂(y_ref)/∂(y_curved) = ψ(x_curved)
#   y3_d3 = ∂(z_ref)/∂(z_curved) = 1
y1_d1 = lambda xx: jnp.ones(xx[..., 0].shape)
y2_d1 = lambda xx: jnp.multiply(xx[..., 1], dpsi(xx[..., 0]))
y2_d2 = lambda xx: psi(xx[..., 0])
y3_d3 = lambda xx: jnp.ones(xx[..., 2].shape)

# --------------------------------------------------------------------------------------
# SECOND DERIVATIVE FOR y2: ∂²(y_ref)/∂(x_curved)² = y_curved * ψ''(x_curved)
#   (Only this second derivative is nonzero for our example)
y2_d1d1 = lambda xx: jnp.multiply(xx[..., 1], ddpsi(xx[..., 0]))

# --------------------------------------------------------------------------------------
# WE WISH TO SOLVE THE CONSTANT-COEFFICIENT HELMHOLTZ (or LAPLACE) EQUATION
# ON THE CURVED GEOMETRY. kh=0 → Laplace; else Helmholtz.
# Build a ParametrizedGeometry3D with our forward/inverse maps and their derivatives.
# --------------------------------------------------------------------------------------
box_geom = jnp.array([[0, 0, 0], [1.0, 1.0, 1.0]])  # reference unit cube in 3D

param_geom = ParametrizedGeometry3D(
    box_geom,
    z1, z2, z3,         # forward mapping functions
    y1, y2, y3,         # inverse mapping functions
    y1_d1=y1_d1,        # ∂y1/∂x_curved
    y2_d1=y2_d1,        # ∂y2/∂x_curved
    y3_d3=y3_d3,        # ∂y3/∂z_curved
    y2_d2=y2_d2,        # ∂y2/∂y_curved
    y2_d1d1=y2_d1d1     # ∂²y2/∂x_curved²
)

# --------------------------------------------------------------------------------------
# DEFINE bfield(yy, kh) = -(kh^2) * 1  → constant-coefficient Helmholtz/Laplace source term
# --------------------------------------------------------------------------------------
def bfield_constant(xx, kh):
    # Return an array of shape (...) filled with –(kh^2)
    return -(kh**2) * jnp.ones(xx[..., 0].shape)

# --------------------------------------------------------------------------------------
# TRANSFORM THE HELMHOLTZ PDO FROM THE REFERENCE TO THE CURVED DOMAIN
# This returns a PDO3d object whose c11,c22,c33,c1,c2,c3,c12,c13,c23,c are functions of x_curved.
# --------------------------------------------------------------------------------------
pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

# --------------------------------------------------------------------------------------
# BUILD AND TIME THE HPS MULTIDOMAIN SOLVER ON THE CURVED DOMAIN
# --------------------------------------------------------------------------------------
tic    = time()
solver = HPSMultidomain(pdo_mod, param_geom, a, p)
toc_dtn = time() - tic

# --------------------------------------------------------------------------------------
# FACTORIZE THE INTERIOR SCHUR BLOCK (Aii) AND MEASURE TIME
# --------------------------------------------------------------------------------------
tic    = time()
solver.setup_solver_Aii()
toc_sp = time() - tic

# --------------------------------------------------------------------------------------
# VERIFY DISCRETIZATION USING A KNOWN GREEN’S FUNCTION (kh=0 → Laplace Green’s function)
# --------------------------------------------------------------------------------------
relerr = solver.verify_discretization(kh)

print(
    "\t Setup time (DtNs + assembly = %5.2f s, factorization = %5.2f s);"
    " relerror in 3D curved domain %5.2e"
    % (toc_dtn, toc_sp, relerr)
)

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')  # 3D scatter plot

XX_curved = solver.geom.parameter_map(solver.XX)

ax.scatter(
    XX_curved[solver.Jx, 0],
    XX_curved[solver.Jx, 1],
    XX_curved[solver.Jx, 2],
    label='Boundary'
)
ax.scatter(
    XX_curved[solver.Ji, 0],
    XX_curved[solver.Ji, 1],
    XX_curved[solver.Ji, 2],
    label='Interior'
)
ax.set_box_aspect([1, 1, 1])
plt.show()