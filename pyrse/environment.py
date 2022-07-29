import math

import ambiance
import numpy as np

from . import utils

class Environment:
    def __init__(self, air_density=1.225, kinematic_viscosity=1.81e-5):
        self.__g0 = 9.80665
        self.__air_density = air_density
        self.__kinematic_viscosity = kinematic_viscosity
        # TODO: CALCULATE THE HEIGHT FROM THE GEOID
        
    def g(self, pos, frame='ECEF'):
        # TODO: MAKE THIS WORK BETTER USING A BETTER MODEL
        # TODO: RETURN A GRAVITY VECTOR
        ellip = pos.ellipsoid
        Ravg = math.sqrt((ellip.a**2 + ellip.b**2) / 2.0)  # Elliptical Quadratic Mean
        R_2 = np.sum(pos.ecef**2)
        mag = (self.__g0 * (Ravg**2)) / R_2
        if frame is None:
            return mag
        if frame == 'ECEF':
            print('Gravity Normal Vector = {}'.format(pos.ecef / math.sqrt(R_2)))
            return -mag * (pos.ecef / math.sqrt(R_2))
        assert False, 'Gravity calculation error: unknown frame {}'.format(frame)
        
    
    def air_density(self, pos):
        return self.__air_density
    
    def kinematic_viscosity(self, pos):
        return self.__kinematic_viscosity
    
    
    

