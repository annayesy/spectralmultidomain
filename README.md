# hps

A simple, extensible implementation of HPS in 2 and 3 dimensions for rectangular geometries.
For example usage, see `multidomain_example.py`. When `p` is set as 2, the discretization used second-order finite difference.
The package works with minimal dependencies (e.g. only `numpy` and `scipy`), but the solver is much faster when `PetSc` is installed.

## Installation
To install the package, clone the repository and run the following command:
```
conda create -n hpsenv python=3.12 petsc petsc4py -c defaults -c conda-forge  
pip install -e .
```
To verify the installation, run:
```
pytest
```

## On the Oden machines
Ssh and navigate to the appropriate directory.
```
ssh username@machine.oden.utexas.edu
cd /workspace/username
```
Install miniconda3 in the `/workspace/username` directory. After the install, check that package data is stored in `/workspace/username`.
```
# Add settings
conda config --file /workspace/username/miniconda3/.condarc \
  --add pkgs_dirs /workspace/username/miniconda3/pkgs

conda config --file /workspace/username/miniconda3/.condarc \
  --add envs_dirs /workspace/username/miniconda3/envs

# Verify
conda config --file /workspace/username/miniconda3/.condarc \
  --show pkgs_dirs envs_dirs
```
Install the package in a conda environment as described above.
