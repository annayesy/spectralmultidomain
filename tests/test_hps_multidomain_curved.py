import jax.numpy as jnp
import pytest

from hps.pdo import PDO2d, const,get_known_greens
from hps.geom import ParametrizedGeometry2D
from hps.geom import ParametrizedGeometry3D
from hps.hps_multidomain import HPSMultidomain


def test_hps_multidomain_curved_2d():
    a = 1 / 8
    p = 16
    kh = 0

    # Warp functions for the 2D geometry
    mag = 0.3
    psi = lambda x: 1 - mag * jnp.sin(4 * x)
    dpsi = lambda x: -mag * 4 * jnp.cos(4 * x)
    ddpsi = lambda x: mag * 16 * jnp.sin(4 * x)

    z1 = lambda xx: xx[..., 0]
    z2 = lambda xx: xx[..., 1] / psi(xx[..., 0])

    y1 = lambda xx: xx[..., 0]
    y2 = lambda xx: xx[..., 1] * psi(xx[..., 0])

    y1_d1 = lambda xx: jnp.ones_like(xx[..., 0])
    y2_d1 = lambda xx: xx[..., 1] * dpsi(xx[..., 0])
    y2_d2 = lambda xx: psi(xx[..., 0])
    y2_d1d1 = lambda xx: xx[..., 1] * ddpsi(xx[..., 0])

    # Box and parametrized geometry
    box_geom = jnp.array([[0, 0], [1.0, 1.0]])
    param_geom = ParametrizedGeometry2D(
        box_geom,
        z1, z2, y1, y2,
        y1_d1=y1_d1,
        y2_d1=y2_d1,
        y2_d2=y2_d2,
        y2_d1d1=y2_d1d1
    )

    # Constant-coefficient Helmholtz operator
    def bfield_constant(xx, kh):
        return -(kh ** 2) * jnp.ones(xx[...,0].shape)

    pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

    # Run the HPS solver
    solver = HPSMultidomain(pdo_mod, param_geom, a, p)
    relerr = solver.verify_discretization(kh)

    # Validate error
    assert relerr < 1e-6, f"Relative error too high in 2D: {relerr:.2e}"

    import numpy as np

    points_bnd       = solver.geom.parameter_map(solver.XX)
    points_full      = solver.geom.parameter_map(solver._XXfull)

    uu_full     = get_known_greens(points_full,kh,center = solver.geom.bounds[1]+10)
    uu_bnd      = get_known_greens(points_bnd,kh,center  = solver.geom.bounds[1]+10)

    uu_sol      = solver.solve_dir_full(uu_bnd[solver.Jx]) # solution on all HPS points
    relerr      = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-10, f"Relative error too high in 2D: {relerr:.2e}"

    assert kh == 0

    def get_mms(points):
        # u(x, y) = sin(pi x) * cos(pi y)
        mms = np.sin(np.pi * points[:,0]) * np.cos(np.pi * points[:,1])
        return mms[:, np.newaxis]

    def get_body_load(points):
        # f(x, y) = 2 * pi^2 * sin(pi x) * cos(pi y)
        ff = 2 * (np.pi**2) * get_mms(points)
        return ff[:, np.newaxis]


    ########### Get the solution on the full domain (with body load)
    uu_full     = get_mms(points_full)
    uu_bnd      = get_mms(points_bnd)

    uu_sol      = solver.solve_dir_full(uu_bnd[solver.Jx], get_body_load(points_full)) # solution on all HPS points
    relerr      = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-7, f"Relative error too high in 2D: {relerr:.2e}"


def test_hps_multidomain_curved_3d():
    a = 1 / 8
    p = 10
    kh = 0

    # Geometry warping functions
    mag = 0.25
    psi = lambda z: 1 - mag * jnp.sin(6 * z)
    dpsi = lambda z: -mag * 6 * jnp.cos(6 * z)
    ddpsi = lambda z: mag * 36 * jnp.sin(6 * z)

    z1 = lambda xx: xx[..., 0]
    z2 = lambda xx: xx[..., 1] / psi(xx[..., 0])
    z3 = lambda xx: xx[..., 2]

    y1 = lambda xx: xx[..., 0]
    y2 = lambda xx: xx[..., 1] * psi(xx[..., 0])
    y3 = lambda xx: xx[..., 2]

    y1_d1 = lambda xx: jnp.ones_like(xx[..., 0])
    y2_d1 = lambda xx: xx[..., 1] * dpsi(xx[..., 0])
    y2_d2 = lambda xx: psi(xx[..., 0])
    y3_d3 = lambda xx: jnp.ones_like(xx[..., 2])
    y2_d1d1 = lambda xx: xx[..., 1] * ddpsi(xx[..., 0])

    # Parametrized geometry
    box_geom = jnp.array([[0, 0, 0], [1.0, 1.0, 1.0]])
    param_geom = ParametrizedGeometry3D(
        box_geom,
        z1, z2, z3, y1, y2, y3,
        y1_d1=y1_d1, y2_d1=y2_d1, y3_d3=y3_d3,
        y2_d2=y2_d2, y2_d1d1=y2_d1d1
    )

    # Helmholtz PDE operator on mapped geometry
    def bfield_constant(xx, kh):
        return -(kh ** 2) * jnp.ones(xx[...,0].shape)

    pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

    # Solve and verify
    solver = HPSMultidomain(pdo_mod, param_geom, a, p)
    relerr = solver.verify_discretization(kh)

    # Assert relative error is within a tight tolerance
    assert relerr < 1e-6, f"Relative error too high: {relerr:.2e}"