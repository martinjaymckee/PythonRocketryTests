

from . import utils


# TODO: PAD SHOULD HAVE A RAIL LENGTH, RAIL TILT (N AND E), ETC.

class LaunchPad:
    def __init__(self, pos):
        self.__pos = pos
        
    @property
    def pos(self):
        return self.__pos
    
