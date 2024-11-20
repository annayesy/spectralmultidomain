from   abc   import ABCMeta, abstractmethod, abstractproperty
import numpy as np

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