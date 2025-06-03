# hps

A simple, extensible implementation of the Hierarchical Poincaré-Steklov scheme (HPS) in 2 and 3 dimensions.  HPS is a spectral multidomain collocation scheme that enables the use of high local polynomial order `p`, while interfacing well with sparse direct solvers. When `p` is set as 2, the discretization used second-order finite difference.
The package works with minimal dependencies (listed in `setup.py`), but the solver is much faster when `PetSc` is installed.

## Installation
To install the package, clone the repository and run the following command:
```
conda create -n hpsenv python=3.12 petsc petsc4py -c defaults -c conda-forge

pip install jax[cuda12] # for GPU-capable machines, install the cuda version of jax
pip install -e .        # otherwise, the CPU-only version is in the setup.py file
```
To verify the installation, run:
```
pytest
```