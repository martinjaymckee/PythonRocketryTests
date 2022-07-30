import math

import ambiance
import numpy as np

from . import numpy_utils as npu
from . import utils

class EnvironmentException(Exception):
    def __init__(self, msg='Environment Error'):
        Exception.__init__(self, msg)
        

class EnvironmentSample:
    def __init__(self, g, density, P, T, kinematic_viscosity):
        self.g = g        
        self.density = density
        self.P = P
        self.T = T
        self.kinematic_viscosity = kinematic_viscosity
        
    def __str__(self):
        return 'g = {}m/s^2, rho = {}kg/m^3, P = {}Pa, T = {}K, mu = {} m^2/s'.format(self.g, self.density, self.P, self.T, self.kinematic_viscosity)
    
    
class Environment:
    def __init__(self, pos, T=288.15, P=101325, centripetal_correction=False, terrain=None):
        self.__pos = pos
        self.__T = T
        self.__P = P
        self.__centripetal_correction = centripetal_correction
        self.__terrain = terrain
        self.__g0 = 9.80665
        self.__omega_earth_rotation = 7.2921150e-5  # rad / s        
        self.__omega_earth_rotation_2 = self.__omega_earth_rotation**2  # rad^2 / s^2
        self.__density = None
        self.__kinematic_viscosity = None
        self.__R_avg = None
        self.__g = None
        self.__h = None
        self.pos = pos

    @property
    def pos(self):
        return self.__pos
    
    @pos.setter
    def pos(self, _pos):
        if not isinstance(_pos, utils.GeographicPosition):
            raise utils.PositionException('Invalid position of type {}'.format(type(_pos)))

        self.__pos = _pos
        
        #
        # Calculate Gravity
        #
        ellip = self.__pos.ellipsoid

        # Calculate Base Gravity -- NOTE: THIS IS A DIFFERENT IMPLEMENTATION THAN THAT FOUND IN AMBIANCE NEITHER IS PERFECT
        ecef = self.__pos.ecef
        self.__Ravg = math.sqrt((ellip.a**2 + ellip.b**2) / 2.0)  # Elliptical Quadratic Mean
        R_2 = np.sum(ecef**2)
        mag = (self.__g0 * (self.__Ravg**2)) / R_2
        g0 = -mag * (ecef / math.sqrt(R_2))
        
        gc = np.array([0, 0, 0])
        if self.__centripetal_correction:
            # Calculate Centripital Correction
            v = ecef.copy()
            v[2] = 0  # Zero the Z component to oint orthagonal to the rotation axis
            R_off = npu.magnitude(v)
            v = npu.normalized(v)
            mag = R_off * self.__omega_earth_rotation_2
            gc = mag * v

        self.__g = g0 + gc
        
        #
        # Calculate Atmospheric Characteristics
        #
        self.__h = utils.heightAboveGround(self.__pos, self.__terrain)
        
        atmos = ambiance.Atmosphere(self.__h)
        self.__density = atmos.density[0]
        self.__kinematic_viscosity = atmos.kinematic_viscosity[0]
        self.__T = atmos.temperature[0]
        self.__P = atmos.pressure[0]
        
        return self.pos
    
    @property
    def h(self):
        return self.__h

    @property
    def density(self):
        return self.__density
    
    @property
    def kinematic_viscosity(self):
        return self.__kinematic_viscosity
    
    def g(self, frame='ECEF'):
        if frame is None:
            return npu.magnitude(self.__g)
        elif frame == 'ECEF':
            return self.__g
        assert False, 'Gravity calculation error: unknown frame {}'.format(frame)
        
    def sample(self, frame='ECEF'):
        return EnvironmentSample(
            self.g(frame),
            self.__density,
            self.__P,
            self.__T,
            self.__kinematic_viscosity
        )
        
    
    

