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
        
        
class SimState:
    def __init__(self, pos=None, vel=None, accel=None, orientation=None, rate=None):
        self.pos = utils.GeographicPosition() if pos is None else pos
        self.vel = utils.VelocityVector3D() if vel is None else vel
        self.accel = utils.AccelerationVector3D() if accel is None else accel
        self.rate = utils.AngularRateVector3D() if rate is None else rate
        self.orientation = None  # TODO: INITIALIZE ORIENTATION
        
        
class SimulationCore:
    def __init__(self, env, pad, model):
        self.__envs = [env.copy() for _ in range(len(self.models))
        self.__pad = pad
        self.__models = [model]
        self.__states =  [SimState()] * len(self.models)
        self.__logs = [[]] * len(self.models)
        self.reset()
        
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
    
    def reset(self):
        for idx, (state, env) in enumerate(zip(self.__states, self.__envs)):
            state.pos = self.__env.pos
            state.vel = utils.VelocityVector3D()
            state.accel = utils.AccelerationVector3D()
            state.rate = utils.AngularRateVector3D()
            state.orientation = None
            self.__logs[idx] = []


class Simulation1D(SimulationCore):
    def __init__(self, env, pad, models):
        SimulationCore.__init__(self, env, pad, models)
        print('Initialize 1D Simulation')
    

        # def simulate_launch(dt, eng, m_rocket, launch_detect=None, burnout_detect=None, g=9.80665):    
    def update(self):
        for model in self.models:
            
            vel = 0
            S = math.pi * (0.041/2)**2
            Cd = 0.54
            rho = 1.225
            t = -eng.burn_time-random.uniform(0, 1/5)
            ts = []
            ms = []
            Ts = []
            Ds = []
            accels = []
            vels = []
            t_launch = None
            t_delta = None
            t_burnout = None
            launch_detected = False
            launch_detect.reset()
            burnout_detect.reset()
            while t < (1.3 * eng.burn_time):
                T = eng.thrust(t)
                m_eng = eng.mass(t)
                D = 0.5 * rho * S * Cd * vel**2
                m_tot = m_rocket + m_eng
                accel = ((T-D) / m_tot) - g
                vel = vel + dt * accel if t > 0 else 0
                accel_measured = random.gauss(accel, accel_sd)
                ts.append(t)
                ms.append(m_tot)
                Ts.append(T)
                Ds.append(D)
                accels.append(accel_measured)
                vels.append(vel)
                if not launch_detected and launch_detect is not None:
                    detected, t_delta = launch_detect(t, accel_measured)
                    if detected:
                        launch_detected = True
                        t_launch = t
                        # print('Launch Detected at v = {:0.2f} m/s, t = {:0.2f} s, with a t_delta = {:0.2f} s'.format(vel, t_launch, t_delta))
                if launch_detected and (burnout_detect is not None):
                    detected = burnout_detect(t, accel_measured)
                    if detected:
                        t_burnout = t
                        # print('Burnout Detected at t = {:0.2f} s'.format(t_burnout))
                t += dt
            return ts, ms, Ts, Ds, accels, vels, t_launch, t_delta, t_burnout        
    
        
class Simulation3D(SimulationCore):
    def __init__(self, env, pad, model):
        SimulationCore.__init__(self, env, pad, model)
        print('Initialize 3D Simulation')
        

class Simulation6D(SimulationCore):
    def __init__(self, env, pad, model):
        SimulationCore.__init__(self, env, pad, model)        
        print('Initialize 6D Simulation')
        
        
def RunSimulation(sim):
    pass

        