import jax
# Enable 64-bit (double) precision globally for JAX computations
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import functools
import numpy as np


def const(c=1):
    """
    Returns a function that produces a constant coefficient c at any set of spatial locations.

    The returned function `const_func(xxloc)` inspects the shape of `xxloc` (which should be
    either a NumPy ndarray or JAX DeviceArray of shape (..., 2) or (..., 3)) and returns an array
    of the same leading (“spatial”) shape filled with the constant value c.

    Parameters:
    - c: Scalar constant to return at each location.

    Returns:
    - const_func: a callable that, given xxloc of shape (..., ndim), returns an array of shape (...)
                  filled with c, matching dtype (and device for JAX) of xxloc.
    """
    def const_func(xxloc):
        # Determine the “spatial” output shape from xxloc[..., 0]
        out_shape = xxloc[..., 0].shape

        # Ensure the last axis of xxloc is 2 or 3 (2D or 3D coordinates)
        assert xxloc.shape[-1] in (2, 3), "xxloc must be provided as (..., 2) or (..., 3)."

        if isinstance(xxloc, np.ndarray):
            # If xxloc is a NumPy array, produce a NumPy array of shape out_shape with value c
            return c * np.ones(out_shape, dtype=xxloc.dtype)
        else:
            # Otherwise (e.g., a JAX DeviceArray), produce a JAX array of shape out_shape with value c
            return c * jnp.ones(out_shape, dtype=xxloc.dtype)

    return const_func


class PDO2d:
    """
    Represents a 2D Partial Differential Operator (PDO) with variable coefficients.

    The operator acts on a scalar function u(x,y) as:

        A[u] = - c11(x,y) * ∂²u/∂x²
               - c22(x,y) * ∂²u/∂y²
               - 2 c12(x,y) * ∂²u/∂x∂y
               + c1(x,y)  * ∂u/∂x
               + c2(x,y)  * ∂u/∂y
               + c(x,y)   * u.

    We assume:
      1. The coefficient matrix [[c11, c12], [c12, c22]] is positive definite (ellipticity).
      2. All coefficient functions are sufficiently smooth on the domain.

    Attributes:
    - c11, c22: Functions (or arrays) giving second-derivative coefficients.
    - c12:       Cross-derivative coefficient (optional).
    - c1, c2:    First-derivative coefficients (optional).
    - c:         Zeroth-order coefficient (optional).
    """

    def __init__(self, c11, c22, c12=None, c1=None, c2=None, c=None):
        # Store coefficient functions/arrays
        self.c11 = c11
        self.c22 = c22
        self.c12 = c12
        self.c1 = c1
        self.c2 = c2
        self.c = c


class PDO3d:
    """
    Represents a 3D Partial Differential Operator (PDO) with variable coefficients.

    The operator acts on a scalar function u(x,y,z) as:

        A[u] = - c11(x) ∂²u/∂x²
               - c22(x) ∂²u/∂y²
               - c33(x) ∂²u/∂z²
               - 2 c12(x) ∂²u/∂x∂y
               - 2 c13(x) ∂²u/∂x∂z
               - 2 c23(x) ∂²u/∂y∂z
               + c1(x)   ∂u/∂x
               + c2(x)   ∂u/∂y
               + c3(x)   ∂u/∂z
               + c(x)    u.

    We assume ellipticity: the 3×3 matrix of second-derivative coefficients is positive definite
    at each point. These assumptions follow from the 3D HPS method (see Hao & Martinsson 2016).

    Attributes:
    - c11, c22, c33: Second-derivative coefficients for ∂²/∂x², ∂²/∂y², ∂²/∂z² respectively.
    - c12, c13, c23: Cross-derivative coefficients (optional).
    - c1, c2, c3:    First-derivative coefficients (optional).
    - c:             Zeroth-order coefficient (optional).
    """

    def __init__(
        self,
        c11,
        c22,
        c33,
        c12=None,
        c13=None,
        c23=None,
        c1=None,
        c2=None,
        c3=None,
        c=None
    ):
        # Store coefficient functions/arrays
        self.c11 = c11
        self.c22 = c22
        self.c33 = c33
        self.c12 = c12
        self.c13 = c13
        self.c23 = c23
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c = c


def get_known_greens(xx, kh, center=None):
    """
    Evaluate a known “Green's-type” solution at points xx for testing discretization.

    For 2D:
      - If kh == 0:  u_exact = log(r)
      - If kh != 0:  u_exact = J0(kh * r) (Bessel j0), where r = distance to source.

    For 3D:
      - If kh == 0:  u_exact = 1 / r
      - If kh != 0:  u_exact = sin(kh * r) / r, where r = distance to source.

    Parameters:
    - xx: numpy.ndarray of shape (N, ndim)
          Points at which to evaluate the Green’s function.
    - kh: float
          Wavenumber times reference length; determines Helmholtz vs Poisson.
    - center: array-like of length ndim (optional)
          Location of the Green's source. If None, defaults to [10, 10] or [10,10,10].

    Returns:
    - uu_exact: numpy.ndarray of shape (N, 1)
          Column vector of evaluated Green’s function values at each row of xx.
    """
    import numpy as np
    from scipy.special import j0

    xx_tmp = xx.copy()
    ndim = xx_tmp.shape[-1]

    # Default source location far outside the domain if not provided
    if center is None:
        center = np.ones(ndim) * 10

    if ndim == 2:
        # Compute distances r in 2D
        xx_d0 = xx_tmp[:, 0] - center[0]
        xx_d1 = xx_tmp[:, 1] - center[1]
        ddsq = np.multiply(xx_d0, xx_d0) + np.multiply(xx_d1, xx_d1)
        rr = np.sqrt(ddsq)

        if kh == 0:
            # Laplace 2D Green's function: log(r)
            uu_exact = np.log(rr)
        else:
            # Helmholtz 2D Green's function: J0(kh * r)
            uu_exact = j0(kh * rr)

    else:
        # ndim == 3: compute distances r in 3D
        xx_d0 = xx_tmp[:, 0] - center[0]
        xx_d1 = xx_tmp[:, 1] - center[1]
        xx_d2 = xx_tmp[:, 2] - center[2]
        ddsq = (
            np.multiply(xx_d0, xx_d0)
            + np.multiply(xx_d1, xx_d1)
            + np.multiply(xx_d2, xx_d2)
        )
        rr = np.sqrt(ddsq)

        if kh == 0:
            # Laplace 3D Green's function: 1/r
            uu_exact = 1.0 / rr
        else:
            # Helmholtz 3D Green's function: sin(kh * r) / r
            uu_exact = np.sin(kh * rr) / rr

    # Ensure result is a column vector of shape (N, 1)
    if uu_exact.ndim == 1:
        uu_exact = uu_exact[:, np.newaxis]
    return uu_exact
