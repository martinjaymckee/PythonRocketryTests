

from . import utils


# TODO: PAD SHOULD HAVE A RAIL LENGTH, RAIL TILT (N AND E), ETC.

class LaunchPad:
    def __init__(self, pos):
        self.__pos = pos.copy()
        self.__guide_height = 1.8 # meters
        
    @property
    def pos(self):
        return self.__pos
    
    @property
    def guide_height(self):
        return self.__guide_height
    
