from pyrse import utils

class SimulationException(Exception):
    def __init__(self, msg='Simulation Error'):
        Exception.__init__(self, msg)        

        
class SimState:
    def __init__(self, t = 0, dt = 0, pos=None, vel=None, accel=None, orientation=None, rate=None):
        self.t = 0
        self.dt = 0
        self.pos = utils.GeographicPosition() if pos is None else pos
        self.vel = utils.VelocityVector3D() if vel is None else vel
        self.accel = utils.AccelerationVector3D() if accel is None else accel
        self.rate = utils.AngularRateVector3D() if rate is None else rate
        self.orientation = orientation

    def copy(self):
        return self.__class__(
            t = self.t, 
            dt = self.dt, 
            pos = self.pos.copy(), 
            vel = self.vel.copy(),
            accel = self.accel.copy(),
            orientation = self.orientation.copy(),
            rate = self.rate.copy()
        )


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value


class SimResult:
    @classmethod
    def FromState(cls, state):
        result = cls()
        result.state = state
        return result
    
    def __init__(self, t = 0, dt = 0, pos=None, vel=None, accel=None, orientation=None, rate=None):
        self.t = 0
        self.dt = dt
        self.pos = utils.GeographicPosition() if pos is None else pos
        self.vel = utils.VelocityVector3D() if vel is None else vel
        self.accel = utils.AccelerationVector3D() if accel is None else accel
        self.rate = utils.AngularRateVector3D() if rate is None else rate
        self.orientation = orientation
        self.mmoi = None # TODO: INITIALIZE MASS MOMENT OF INERTIA        
        self.mass = None # TODO: INITIALIZE MASS
        self.forces = AttributeDict()
        self.moments = []
        self.environment = None
        self.events = []
        
    @property
    def state(self):
        return SimState(
            self.t, 
            self.dt, 
            self.pos.copy(), 
            self.vel.copy(), 
            self.accel.copy(), 
            None if self.orientation is None else self.orientation.copy(), 
            None if self.rate is None else self.rate.copy()
        )
    
    @state.setter
    def state(self, _state):
        self.t = _state.t
        self.dt = _state.dt
        self.pos = _state.pos.copy()
        self.vel = _state.vel.copy()
        self.accel = _state.accel.copy()
        self.rate = None if _state.rate is None else _state.rate.copy()
        self.orientation = None if _state.orientation is None else _state.orientation.copy()
        
        
        
        
        
        