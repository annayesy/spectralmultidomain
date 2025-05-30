import numpy as np
import pytest

from hps.pdo import PDO2d, const
from hps.geom import ParametrizedGeometry2D
from hps.geom import ParametrizedGeometry3D
from hps.hps_multidomain import HPSMultidomain


def test_hps_multidomain_curved_2d():
    a = 1 / 8
    p = 16
    kh = 0

    # Warp functions for the 2D geometry
    mag = 0.3
    psi = lambda x: 1 - mag * np.sin(4 * x)
    dpsi = lambda x: -mag * 4 * np.cos(4 * x)
    ddpsi = lambda x: mag * 16 * np.sin(4 * x)

    z1 = lambda xx: xx[..., 0]
    z2 = lambda xx: xx[..., 1] / psi(xx[..., 0])

    y1 = lambda xx: xx[..., 0]
    y2 = lambda xx: xx[..., 1] * psi(xx[..., 0])

    y1_d1 = lambda xx: np.ones_like(xx[..., 0])
    y2_d1 = lambda xx: xx[..., 1] * dpsi(xx[..., 0])
    y2_d2 = lambda xx: psi(xx[..., 0])
    y2_d1d1 = lambda xx: xx[..., 1] * ddpsi(xx[..., 0])

    # Box and parametrized geometry
    box_geom = np.array([[0, 0], [1.0, 1.0]])
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
        return -(kh ** 2) * np.ones(xx[...,0].shape)

    pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

    # Run the HPS solver
    solver = HPSMultidomain(pdo_mod, param_geom, a, p)
    relerr = solver.verify_discretization(kh)

    # Validate error
    assert relerr < 1e-6, f"Relative error too high in 2D: {relerr:.2e}"


def test_hps_multidomain_curved_3d():
    a = 1 / 8
    p = 10
    kh = 0

    # Geometry warping functions
    mag = 0.25
    psi = lambda z: 1 - mag * np.sin(6 * z)
    dpsi = lambda z: -mag * 6 * np.cos(6 * z)
    ddpsi = lambda z: mag * 36 * np.sin(6 * z)

    z1 = lambda xx: xx[..., 0]
    z2 = lambda xx: xx[..., 1] / psi(xx[..., 0])
    z3 = lambda xx: xx[..., 2]

    y1 = lambda xx: xx[..., 0]
    y2 = lambda xx: xx[..., 1] * psi(xx[..., 0])
    y3 = lambda xx: xx[..., 2]

    y1_d1 = lambda xx: np.ones_like(xx[..., 0])
    y2_d1 = lambda xx: xx[..., 1] * dpsi(xx[..., 0])
    y2_d2 = lambda xx: psi(xx[..., 0])
    y3_d3 = lambda xx: np.ones_like(xx[..., 2])
    y2_d1d1 = lambda xx: xx[..., 1] * ddpsi(xx[..., 0])

    # Parametrized geometry
    box_geom = np.array([[0, 0, 0], [1.0, 1.0, 1.0]])
    param_geom = ParametrizedGeometry3D(
        box_geom,
        z1, z2, z3, y1, y2, y3,
        y1_d1=y1_d1, y2_d1=y2_d1, y3_d3=y3_d3,
        y2_d2=y2_d2, y2_d1d1=y2_d1d1
    )

    # Helmholtz PDE operator on mapped geometry
    def bfield_constant(xx, kh):
        return -(kh ** 2) * np.ones(xx[...,0].shape)

    pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

    # Solve and verify
    solver = HPSMultidomain(pdo_mod, param_geom, a, p)
    relerr = solver.verify_discretization(kh)

    # Assert relative error is within a tight tolerance
    assert relerr < 1e-6, f"Relative error too high: {relerr:.2e}"