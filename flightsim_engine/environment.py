from . import utils

class Environment:
    def __init__(self, g=9.80665, air_density=1.225, kinematic_viscosity=1.81e-5):
        self.__g = g
        self.__air_density = air_density
        self.__kinematic_viscosity = kinematic_viscosity

        # TODO: CALCULATE THE HEIGHT FROM THE GEOID
        
    def g(self, pos):
        GeographicPosition()
        return 9.80665
    
    def air_density(self, pos):
        return self.__air_density
    
    def kinematic_viscosity(self, pos):
        return self.__kinematic_viscosity
    
    
    

