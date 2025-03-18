from   abc   import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from   hps.pdo import PDO2d


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

        if ((y1_d1 is None) and (y1_d2 is None)):
            c11 = None
        else:
            def c11(xx):
                yy = self.parameter_map(xx)

                result = 0
                if (y1_d1 is not None):
                    result += y1_d1(yy)**2
                if (y1_d2 is not None):
                    result += y1_d2(yy)**2
                return result

        if ((y2_d1 is None) and (y2_d2 is None)):
            c22 = None
        else:

            def c22(xx):
                yy = self.parameter_map(xx)

                result = 0
                if (y2_d1 is not None):
                    result += y2_d1(yy)**2
                if (y2_d2 is not None):
                    result += y2_d2(yy)**2
                return result

        if ((y1_d1d1 is None) and (y1_d2d2 is None)):
            c1 = None
        else:

            def c1(xx):
                yy = self.parameter_map(xx)

                result = 0
                if (y1_d1d1 is not None):
                    result -= y1_d1d1(yy)
                if (y1_d2d2 is not None):
                    result -= y1_d2d2(yy)
                return result

        if ((y2_d1d1 is None) and (y2_d2d2 is None)):
            c2 = None
        else:

            def c2(xx):
                yy = self.parameter_map(xx)

                result = 0
                if (y2_d1d1 is not None):
                    result -= y2_d1d1(yy)
                if (y2_d2d2 is not None):
                    result -= y2_d2d2(yy)
                return result

        bool_expr1 = (y1_d1 is not None) and (y2_d1 is not None)
        bool_expr2 = (y1_d2 is not None) and (y2_d2 is not None) 
        if ((not bool_expr1) and (not bool_expr2)):
            c12 = None
        else:

            def c12(xx):
                yy = self.parameter_map(xx)

                result = 0
                if (bool_expr1):
                    result += np.multiply(y1_d1(yy),y2_d1(yy))
                if (bool_expr2):
                    result += np.multiply(y1_d2(yy),y2_d2(yy))
                return result
    
        def c(xx):
            return bfield(self.parameter_map(xx),kh)
    
        return PDO2d(c11=c11,c22=c22,c1=c1,c2=c2,c12=c12,c=c)
    
