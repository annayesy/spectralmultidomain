import warnings

import numpy as np
import pytest

from hps.fd_discretization import FDDiscretization
from hps.pdo import PDO2d, PDO3d


class BoxGeometry:
    def __init__(self, bounds):
        self.bounds = bounds


def const(c):
    def const_func(xxloc):
        return c * np.ones(xxloc[..., 0].shape, dtype=xxloc.dtype)

    return const_func


@pytest.mark.parametrize("ndim", [2, 3])
def test_fd_discretization_uses_float_stencils(ndim):
    if ndim == 2:
        bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        pdo = PDO2d(c11=const(1.0), c22=const(1.0), c=const(0.0))
    else:
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pdo = PDO3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(0.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        disc = FDDiscretization(pdo, BoxGeometry(bounds), h=0.5)

    assert disc.A.shape == (disc.XX.shape[0], disc.XX.shape[0])
