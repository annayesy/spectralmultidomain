from   abc   import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from   hps.pdo import PDO2d,PDO3d


############################################################################################

class AbstractGeometry(metaclass=ABCMeta):

    @abstractproperty
    def bounds(self):
        pass

class BoxGeometry(AbstractGeometry):
    def __init__(self,box_geom):
        self.box_geom = box_geom

    @property
    def bounds(self):
        return self.box_geom


#############################################################################################


class ParametrizedGeometry2D(AbstractGeometry):

    ## zz_func maps values on the reference domain to the curved domain
    ## yy_func maps values on the curved domain    to the reference.
    ## The reference domain is a rectangle defined by box_geom.
    def __init__(self,box_geom,z1,z2,y1,y2,\
        y1_d1=None, y1_d2=None, y2_d1=None, y2_d2=None,\
        y1_d1d1=None, y1_d2d2=None, y2_d1d1=None, y2_d2d2=None):

        self.box_geom = box_geom
        self.zz       = (z1,z2)
        self.yy       = (y1,y2)
        self.yy_deriv = (y1_d1, y1_d2, y2_d1, y2_d2,\
            y1_d1d1, y1_d2d2, y2_d1d1, y2_d2d2)

    @property
    def bounds(self):
        return self.box_geom

    @property
    def parameter_map(self):
        def param_map(xx):
            (z1,z2) = self.zz
            ZZ = xx.copy()
            ZZ[:,0] = z1(xx)
            ZZ[:,1] = z2(xx)
            return ZZ
        return param_map

    @property
    def inv_parameter_map(self):
        def inv_param_map(xx):
            (y1,y2) = self.yy
            YY = xx.copy()
            YY[:,0] = y1(xx)
            YY[:,1] = y2(xx)
            return YY
        return inv_param_map

    def transform_helmholtz_pdo(self,bfield,kh):

        (y1_d1, y1_d2, y2_d1, y2_d2,\
        y1_d1d1, y1_d2d2, y2_d1d1, y2_d2d2) = self.yy_deriv

        def helper_double_deriv(derivs):

            if all(d is None for d in derivs):
                func = None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return sum(d(yy)**2 for d in derivs if d is not None)
            return func

        def helper_single_deriv(derivs):

            if all(d is None for d in derivs):
                func = None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return -sum(d(yy) for d in derivs if d is not None)
            return func

        c11 = helper_double_deriv([y1_d1,  y1_d2])
        c22 = helper_double_deriv([y2_d1,  y2_d2])
        c1  = helper_single_deriv([y1_d1d1,y1_d2d2])
        c2  = helper_single_deriv([y2_d1d1,y2_d2d2])

        pairs = [(y1_d1, y2_d1), (y1_d2, y2_d2)]
        if not any(a is not None and b is not None for a, b in pairs):
            c12 = None
        else:
            def c12(xx):
                yy = self.parameter_map(xx)
                result = 0
                for a, b in pairs:
                    if a is not None and b is not None:
                        result += np.multiply(a(yy), b(yy))
                return result
            
        def c(xx):
            return bfield(self.parameter_map(xx),kh)
    
        return PDO2d(c11=c11,c22=c22,c1=c1,c2=c2,c12=c12,c=c)
    
class ParametrizedGeometry3D(AbstractGeometry):

    ## zz_func maps values on the reference domain to the curved domain
    ## yy_func maps values on the curved domain    to the reference.
    ## The reference domain is a rectangle defined by box_geom.
    def __init__(self,box_geom,z1,z2,z3,y1,y2,y3,\
        y1_d1=None, y1_d2=None, y1_d3=None,\
        y2_d1=None, y2_d2=None, y2_d3=None,\
        y3_d1=None, y3_d2=None, y3_d3=None,
        y1_d1d1=None, y1_d2d2=None,y1_d3d3=None,\
        y2_d1d1=None, y2_d2d2=None,y2_d3d3=None,\
        y3_d1d1=None, y3_d2d2=None,y3_d3d3=None):

        self.box_geom = box_geom
        self.zz       = (z1,z2,z3)
        self.yy       = (y1,y2,y3)
        self.yy_deriv = (y1_d1, y1_d2, y1_d3,\
            y2_d1, y2_d2, y2_d3,\
            y3_d1, y3_d2, y3_d3,
            y1_d1d1, y1_d2d2,y1_d3d3,\
            y2_d1d1, y2_d2d2,y2_d3d3,\
            y3_d1d1, y3_d2d2,y3_d3d3)

    @property
    def bounds(self):
        return self.box_geom

    @property
    def parameter_map(self):
        def param_map(xx):
            (z1,z2,z3) = self.zz
            ZZ = xx.copy()
            ZZ[:,0] = z1(xx)
            ZZ[:,1] = z2(xx)
            ZZ[:,2] = z3(xx)
            return ZZ
        return param_map

    @property
    def inv_parameter_map(self):
        def inv_param_map(xx):
            (y1,y2,y3) = self.yy
            YY = xx.copy()
            YY[:,0] = y1(xx)
            YY[:,1] = y2(xx)
            YY[:,2] = y3(xx)
            return YY
        return inv_param_map

    def transform_helmholtz_pdo(self,bfield,kh):

        (y1_d1, y1_d2, y1_d3,y2_d1, y2_d2, y2_d3, y3_d1, y3_d2, y3_d3,\
            y1_d1d1, y1_d2d2,y1_d3d3,y2_d1d1,y2_d2d2,y2_d3d3,y3_d1d1, y3_d2d2,y3_d3d3) \
        = self.yy_deriv

        def helper_double_deriv(derivs):

            if all(d is None for d in derivs):
                func = None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return sum(d(yy)**2 for d in derivs if d is not None)
            return func

        def helper_single_deriv(derivs):

            if all(d is None for d in derivs):
                func = None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return -sum(d(yy) for d in derivs if d is not None)
            return func

        def helper_mixed_deriv(pairs):
            if not any(f is not None and g is not None for f, g in pairs):
                func = None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    result = 0
                    for f, g in pairs:
                        if f is not None and g is not None:
                            result += np.multiply(f(yy), g(yy))
                    return result
                return func

        c11 = helper_double_deriv([y1_d1,  y1_d2,  y1_d3])
        c22 = helper_double_deriv([y2_d1,  y2_d2,  y2_d3])
        c33 = helper_double_deriv([y3_d1,  y3_d2,  y3_d3])
        c1  = helper_single_deriv([y1_d1d1,y1_d2d2,y1_d3d3])
        c2  = helper_single_deriv([y2_d1d1,y2_d2d2,y2_d3d3])
        c3  = helper_single_deriv([y3_d1d1,y3_d2d2,y3_d3d3])
        c12 = helper_mixed_deriv ([(y1_d1, y2_d1), (y1_d2, y2_d2), (y1_d3, y2_d3)])
        c13 = helper_mixed_deriv ([(y1_d1, y3_d1), (y1_d2, y3_d2), (y1_d3, y3_d3)])
        c23 = helper_mixed_deriv ([(y2_d1, y3_d1), (y2_d2, y3_d2), (y2_d3, y3_d3)])

        def c(xx):
            return bfield(self.parameter_map(xx),kh)
    
        return PDO3d(c11=c11,c22=c22,c33=c33,\
            c1=c1,c2=c2,c3=c3,\
            c12=c12,c13=c13,c23=c23,c=c)