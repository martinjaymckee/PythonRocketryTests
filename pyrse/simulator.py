import numpy as np

from pyrse.simulator_utils import *

from pyrse import engines
from pyrse import environment
from pyrse import numpy_utils as npu
from pyrse import pad
from pyrse import triggers
from pyrse import utils
         
#        
# General Notes:
#
#   -- If there is no trigger defined for an engine, it should start immediately at the beginning of the simulation
#
 
class SimulationCore:
    def __init__(self, env, pad, model, dt_min = 0.01, triggers = [], integrator = 'Euler', dof=None, update_translations=True, update_rotations=False):
        self.__pad = pad
        self.__models = [model] # NOTE: CURRENTLY THIS IS WORKING WITH ONLY A SINGLE MODEL AS INPUT
        self.__dt_min = dt_min
        self.__envs = [env.copy() for _ in range(len(self.models))]
        self.__states =  [SimState()] * len(self.models)
        self.__logs = [[]] * len(self.models)
        self.__triggers = triggers
        self.__integrator = None # TODO: CREATE THE INTEGRATOR OBJECT FROM THE "integrator" DESCRIPTION
        self.__engines = None
        self.__dof = dof
        self.__update_translations = update_translations
        self.__update_rotations = update_rotations
        self.find_engines()
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
    def states(self):
        return self.__states
    
    @property
    def logs(self): # TODO: ADD SIM LOGS AND SPLIT THESE OUT AS DATA LOGS
        return self.__logs
    
    @property
    def envs(self):
        return self.__envs
    
    @property
    def engines(self):
        return self.__engines

    @property
    def triggers(self):
        return self.__triggers

    @property
    def surface_normal(self):
        # TODO: THIS MAY NOT BE THE BEST IMPLEMENTATION... IT SHOULD BASICALLY WORK, HOWEVER
        pos = self.__pad.pos
        llh = pos.llh
        pos2 = utils.GeographicPosition.LLH(llh[0], llh[1], llh[2] + 1.0)
        # print('pos = {}'.format(pos))
        # print('pos2 = {}'.format(pos2))
        return npu.normalized(pos2.ecef - self.__pad.pos.ecef)
    
    @property
    def dof(self):
        return self.__dof
    
    @property
    def translation_updated(self):
        return self.__update_translation
    
    @property
    def rotation_updated(self):
        return self.__update_rotation
    
    def reset(self):
        for idx, (state, env) in enumerate(zip(self.__states, self.__envs)):
            state.t = 0
            state.pos = self.__pad.pos.copy()
            state.vel = utils.VelocityVector3D()
            state.accel = utils.AccelerationVector3D()
            state.rate = utils.AngularRateVector3D()
            state.orientation = None
            self.__logs[idx] = []
            
        self.init_engines()

    def find_model_engines(self, model): # TODO: MAKE THIS HANDLE ENGINES AT MUTIPLE LEVELS. AT THIS POINT, IT WILL ONLY DETECT FIRST LEVEL CHILDREN.
                            #   NOTE: THIS SHOULD BE DONE BY IMPLEMENTING A "SELECT" METHOD ON THE ROCKET COMPONENTS 
        engs = set()
        for child in model.children:
            if isinstance(child, engines.Engine):
                engs.add(child)
        return engs
    
    def find_engines(self):
        self.__engines = set()
        for model in self.models:
            self.__engines.update(self.find_model_engines(model))
        
    def init_engines(self):
        engine_triggers = {}
        for trigger in self.__triggers:
            if isinstance(trigger.component, engines.Engine):
                engine_triggers[trigger.component.id] = trigger
                
        for engine in self.__engines:
            if not engine.id in engine_triggers.keys():
                engine.start(0)
            
    def model_simulation_complete(self, model, state):
        for trigger in self.__triggers:
            triggered, action, _ = trigger(model, state)
            if triggered and (action == 'end simulation'):
                return True
        return False
    
    def completed(self):
        done = True
        for model, state in zip(self.__models, self.__states):
            if not self.model_simulation_complete(model, state):
                done = False
        return done
    
    def process_triggers(self):
        for trigger in self.__triggers:
            pass
        
        return []
        
    def model_thrust(self, model, state): # TODO: HANDLE CANTED ENGINE THRUST
        engs = self.find_model_engines(model)
        t = state.t
        T = 0 # TODO: HANDLE CANTED ENGINE THRUST
        for eng in engs:
            T += eng.thrust(t)
        return T
        
    def update(self):
        done = True
        events = []

        # TODO: ADD HANDLING OF SIMULATOR LOOP BEGIN CALLBACKS
            
        for idx in range(len(self.models)):
            model = self.models[idx]
            state = self.states[idx]
            
            if not self.model_simulation_complete(model, state):
                # TODO: ADD HANDLING OF MODEL LOOP BEGIN CALLBACKS -- ENSURE THAT THESE CAN BE RUN MULTIPLE TIMES WITHOUT ERROR
                env = self.envs[idx]                
                dt = self.__dt_min
                t = state.t
                state.dt = dt
                trigger_events = self.process_triggers()
                events += trigger_events
                
                T = self.surface_normal * self.model_thrust(model, state) # TODO: THIS NEEDS TO BE ROTATED INTO THE CORRECT COORDINATES
                T_mag = npu.magnitude(T)
                m = model.mass(t)
                # TODO: THE VELOCITY HERE SHOULD BE PASSED IN IN BODY COORDINATES....
                D = npu.normalized(state.vel.ecef) * model.drag(state.vel, env) # TODO: HANDLE NON-AXIAL DRAG CALCULATIONS....
                F = T - D # TODO: THIS NEEDS TO BE GENERALIZED TO HANDLE ADDITIONAL FORCES....
                accel = self.calc_linear_accel(env, F, m)
                state.accel = accel    
               
                # State Update - Euler Integration
                dv = accel.timestep(dt) if t >= 0 else utils.VelocityVector3D()
                new_v = state.vel + dv
                state.vel = new_v
                dpos = new_v.timestep(dt)
                state.pos += dpos
                
                env.pos = state.pos # TODO: FIGURE OUT IF UPDATING THE ENVIRONMENT POSITION IS CAUSING A PROBLEM
                
                result = SimResult.FromState(state)
                result.mass = m
                result.forces['T'] = T
                result.forces['D'] = D
                result.environment = env.sample()
                result.events = events
                self.logs[idx].append(result)
                self.states[idx].t = t + dt
                # TODO: ADD MODEL LOOP END CALLBACK
            # TODO: ADD TIMESTEP CHANGED CALLBACK
        # TODO: ADD SIMULATION LOOP END CALLBACK
        return events
    
    def calc_linear_accel(self, env, F, m):
        raise Exception('Error: {}.calc_linear_acceleration() is undefined!'.format(self.__class__.__name__))

    def calc_angular_accel(self, env, Q, mmoi):
        raise Exception('Error: {}.calc_linear_acceleration() is undefined!'.format(self.__class__.__name__))
        
        
class Simulation1D(SimulationCore):
    def __init__(self, env, pad, models, dt_min=0.01, triggers = [], **kwargs):
        SimulationCore.__init__(self, env, pad, models, dt_min=dt_min, triggers=triggers, dof=1, **kwargs)

        
    def calc_linear_accel(self, env, F, m):
        return utils.AccelerationVector3D((F / m) + env.g())
    
        
class Simulation3D(SimulationCore):
    def __init__(self, env, pad, model, dt_min=0.01, triggers = [], **kwargs):
        SimulationCore.__init__(self, env, pad, model, dt_min=dt_min, triggers=triggers, dof=3, **kwargs)


class Simulation6D(Simulation3D):
    def __init__(self, env, pad, model, dt_min=0.01, triggers = [], **kwargs):
        Simulation3D.__init__(self, env, pad, model, dt_min=dt_min, triggers=triggers, dof=6, update_translations=True, update_rotations=True)        
        
        
class SimulationRunStatus:
    def __init__(self):
        self.iterations = 0
        self.t_max = 0
        self.num_models = 1


def RunSimulation(sim, t0 = 0): # TODO: USE T0 TO INITIALIZE THE SIMULATION
    done = False
    status = SimulationRunStatus()
   
    i = 0
    while not (sim.completed() or i > 2500):
        status.iterations += 1
        events = sim.update()
        if len(events) > 0:
            print(events)
    
        i = i + 1
        
    return status

        