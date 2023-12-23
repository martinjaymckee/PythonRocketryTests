from pyrse import coordinates

from pyrse.simulator_utils import *


class SimActions:
    StartSim = 'start simulation'
    EndSim = 'end simulation'
    StartEngine = 'start engine'
    
    
class SimTriggerBase:
    def __init__(self, component, action):
        self.__component = component
        self.__action = action
        
    @property
    def component(self):
        return self.__component
    
    @property
    def action(self):
        return self.__action
     
    def __call__(self, model, state):
        return false, self.action, model

 
class TimedEngineStartTrigger(SimTriggerBase):
    def __init__(self, eng, t=10):
        SimTriggerBase.__init__(self, eng, SimActions.StartEngine)
        self.__t = float(t)
        
    def __call__(self, model, state):
        return state.t >= self.__t, self.action, model # TODO: MAKE THIS TRIGGER ONLY IF THE MODEL CONTAINS THE COMPONENT
       
        
class SimulationTimedEndTrigger(SimTriggerBase):
    def __init__(self, t=10):
        SimTriggerBase.__init__(self, None, SimActions.EndSim)
        self.__t = float(t)
        
    def __call__(self, model, state):
        return state.t >= self.__t, self.action, model


class ModelLandedEndTrigger(SimTriggerBase):
    def __init__(self, model, pad):
        SimTriggerBase.__init__(self, model, SimActions.EndSim) #TODO: THIS IS A HACK THAT DOESN'T REALLY WORK AS IT WILL END THE SIMULATION IMMEDIATELY... OR KEEP PROCESSING THE MODEL
        self.__model = model
        self.__pad = pad
        self.__running = False
        
    def __call__(self, model, state):
        if not model == self.__model:
            return False, self.action, model
        pos_enu = coordinates.ECEFToENU(state.pos.ecef, self.__pad.pos.ecef)
        if self.__running:
#            print('pad pos = {}, pos = {}, u = {}'.format(self.__pad.pos.ecef, state.pos.ecef, pos_enu[2]))
            return (pos_enu[2] < 0), self.action, model
        self.__running = pos_enu[2] > 0
        return False, self.action, model


class LiftoffTrigger(SimTriggerBase):
    def __init__(self, model, pad, event):
        SimTriggerBase.__init__(self, model, event) #TODO: THIS IS A HACK THAT DOESN'T REALLY WORK AS IT WILL END THE SIMULATION IMMEDIATELY... OR KEEP PROCESSING THE MODEL
        self.__model = model
        self.__pad = pad
        self.__triggered = False
        
    def __call__(self, model, state):
        if not model == self.__model:
            return False, self.action, model
        pos_enu = coordinates.ECEFToENU(state.pos.ecef, self.__pad.pos.ecef)
        if not self.__triggered:
            triggered = (pos_enu[2] > 0)
            self.__triggered = triggered
            return triggered, self.action, model
        return False, self.action, model        
        
    
class PadClearedTrigger(SimTriggerBase):
    def __init__(self, model, pad, event):
        SimTriggerBase.__init__(self, model, event) #TODO: THIS IS A HACK THAT DOESN'T REALLY WORK AS IT WILL END THE SIMULATION IMMEDIATELY... OR KEEP PROCESSING THE MODEL
        self.__model = model
        self.__pad = pad
        self.__triggered = False
        self.__ref_pos = None
        
    def __call__(self, model, state):
        if not model == self.__model:
            return False, self.action, model
        pos_enu = coordinates.ECEFToENU(state.pos.ecef, self.__pad.pos.ecef)
        if not self.__triggered:
            triggered = (pos_enu[2] > self.__pad.guide_height)
            self.__triggered = triggered
            return triggered, self.action, model
        return False, self.action, model        

    
class LandedTrigger(SimTriggerBase):
    def __init__(self, model, pad, event):
        SimTriggerBase.__init__(self, model, event) #TODO: THIS IS A HACK THAT DOESN'T REALLY WORK AS IT WILL END THE SIMULATION IMMEDIATELY... OR KEEP PROCESSING THE MODEL
        self.__model = model
        self.__running = False
        self.__ref_pos = None
        
    def __call__(self, model, state):
        if not model == self.__model:
            return False, self.action, model
        if self.__ref_pos is None:
            self.__ref_pos = state.pos.copy()
        pos_enu = coordinates.ECEFToENU(state.pos.ecef, self.__ref_pos.ecef)
        if self.__running:
            return (pos_enu[2] < 0), self.action, model
        self.__running = pos_enu[2] > 0
        return False, self.action, model


class ApogeeTrigger(SimTriggerBase):
    def __init__(self, model, event):
        SimTriggerBase.__init__(self, model, event) #TODO: THIS IS A HACK THAT DOESN'T REALLY WORK AS IT WILL END THE SIMULATION IMMEDIATELY... OR KEEP PROCESSING THE MODEL
        self.__model = model
        self.__running = False
        self.__h_last = None
        self.__ref_pos = None
        
    def __call__(self, model, state):
        if not model == self.__model:
            return False, self.action, model
        if self.__ref_pos is None:
            self.__ref_pos = state.pos.copy()        
        pos_enu = coordinates.ECEFToENU(state.pos.ecef, self.__ref_pos.ecef)
        h = pos_enu[2]
        if self.__running:
            triggered = h < self.__h_last
            self.__h_last = h
            return triggered, self.action, model
        self.__running = h > 0
        self.__h_last = h
        return False, self.action, model