# hps

A simple, extensible implementation of HPS in 2 and 3 dimensions for rectangular geometries.
For example usage, see `multidomain_example.py`. When `p` is set as 2, the discretization used second-order finite difference.
The package works with minimal dependencies (e.g. only `numpy` and `scipy`), but the solver is much faster when `PetSc` is installed.

## Installation
To install the package, clone the repository and run the following command:
```
pip install -e .
```
To verify the installation, run:
```
pytest
```
