import numpy as np

from hps.pdo               import PDO2d,PDO3d,const,get_known_greens
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D

from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from time import time
from matplotlib import pyplot as plt

a = 1/8; p = 10; kh = 0

#############################################################

### Function z maps from reference domain (rectangle) to curved domain
mag    = 0.25
psi    = lambda z: 1 - mag * np.sin(6*z)
dpsi   = lambda z:   - mag*6 * np.cos(6*z)
ddpsi  = lambda z:     mag*36  * np.sin(6*z)

z1   = lambda xx: xx[:,0]
z2   = lambda xx: np.divide(xx[:,1],psi(xx[:,0]))

### Function y is the inverse map
### Compute 2 partial derivatives wrt y (some are zero and omitted).
y1   = lambda xx: xx[:,0]
y2   = lambda xx: np.multiply(xx[:,1],psi(xx[:,0]))

y1_d1  = lambda xx: np.ones(xx[:,0].shape)
y2_d1  = lambda xx: np.multiply(xx[:,1], dpsi(xx[:,0]))
y2_d2  = lambda xx: psi(xx[:,0])

y2_d1d1  = lambda xx: np.multiply(xx[:,1], ddpsi(xx[:,0]))

#############################################################
## We would like to solve the constant-coefficient Helmholtz equation
## on the parameterized geometry.

box_geom   = np.array([[0,0],[1.0,1.0]])
param_geom = ParametrizedGeometry2D(box_geom, z1, z2, y1, y2,\
    y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2,y2_d1d1=y2_d1d1)

def bfield_constant(xx,kh):
    return -(kh**2) * np.ones(xx.shape[0])

pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

solver  = HPSMultidomain(pdo_mod,param_geom,a,p)
relerr  = solver.verify_discretization(kh)
print("relerror in 2D curved domain %5.2e" % relerr)


#############################################################

### Function z maps from reference domain (rectangle) to curved domain
z1   = lambda xx: xx[:,0]
z2   = lambda xx: np.divide(xx[:,1],psi(xx[:,0]))
z3   = lambda xx: xx[:,2]

### Function y is the inverse map
### Compute 2 partial derivatives wrt y (some are zero and omitted).
y1   = lambda xx: xx[:,0]
y2   = lambda xx: np.multiply(xx[:,1],psi(xx[:,0]))
y3   = lambda xx: xx[:,2]

y1_d1  = lambda xx: np.ones(xx[:,0].shape)
y2_d1  = lambda xx: np.multiply(xx[:,1], dpsi(xx[:,0]))
y2_d2  = lambda xx: psi(xx[:,0])
y3_d3  = lambda xx: np.ones(xx[:,2].shape)

y2_d1d1  = lambda xx: np.multiply(xx[:,1], ddpsi(xx[:,0]))

#############################################################
## We would like to solve the constant-coefficient Helmholtz equation
## on the parameterized geometry.

box_geom   = np.array([[0,0,0],[1.0,1.0,1.0]])
param_geom = ParametrizedGeometry3D(box_geom, z1, z2, z3, y1, y2, y3,\
    y1_d1=y1_d1, y2_d1=y2_d1, y3_d3=y3_d3, y2_d2=y2_d2, y2_d1d1=y2_d1d1)

def bfield_constant(xx,kh):
    return -(kh**2) * np.ones(xx.shape[0])

pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

solver  = HPSMultidomain(pdo_mod,param_geom,a,p)
relerr  = solver.verify_discretization(kh)
print("relerror in 3D curved domain %5.2e" % relerr)