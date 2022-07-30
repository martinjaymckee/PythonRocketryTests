import numpy as np

from . import environment
from . import pad
from . import utils

class SimulationException(Exception):
    def __init__(self, msg='Simulation Error'):
        Exception.__init__(self, msg)
        

class SimResult:
    def __init__(self, pos=None, vel=None, accel=None, orientation=None, rate=None):
        self.pos = utils.GeographicPosition() if pos is None else pos
        self.vel = utils.VelocityVector3D() if vel is None else vel
        self.accel = utils.AccelerationVector3D() if accel is None else accel
        self.rate = utils.AngularRateVector3D() if rate is None else rate
        self.orientation = None  # TODO: INITIALIZE ORIENTATION
        self.mmoi = None # TODO: INITIALIZE MASS MOMENT OF INERTIA        
        self.mass = None # TODO: INITIALIZE MASS
        self.forces = []
        self.moments = []
        self.environment = None
        
        
class SimulationCore:
    def __init__(self, env, pad, model):
        self.__env = env
        self.__pad = pad
        self.__models = [model]

    @property
    def environment(self):
        return self.__env
    
    @property
    def pad(self):
        return self.__pad
    
    @property
    def models(self):
        return self.__models
    
    @property
    def surface_normal(self):
        # TODO: THIS MAY NOT BE THE BEST IMPLEMENTATION... IT SHOULD WORK, HOWEVER
        pos = self.__pad.pos
        llh = pos.llh
        pos2 = utils.GeographicPosition.LLH(llh[0], llh[1], llh[2] + 1.0)
        # print('pos = {}'.format(pos))
        # print('pos2 = {}'.format(pos2))
        return pos2.ecef - self.__pad.pos.ecef
    

class Simulation1D(SimulationCore):

    def __init__(self, env, pad, models):
        SimulationCore.__init__(self, env, pad, models)
        print('Initialize 1D Simulation')
        self.__states = [np.array([0]*3)] * len(self.models)
        print('__states= {}'.format(self.__states))
        
    
        
class Simulation3D(SimulationCore):
    def __init__(self, env, pad, model):
        SimulationCore.__init__(self, env, pad, model)
        print('Initialize 3D Simulation')
        

class Simulation6D(SimulationCore):
    def __init__(self, env, pad, model):
        SimulationCore.__init__(self, env, pad, model)        
        print('Initialize 6D Simulation')
        