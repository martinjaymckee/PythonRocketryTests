import pyrse.environment
import pyrse.pad
import pyrse.engines
import pyrse.rocket_components

class EiffelTowerModel(pyrse.rocket_components.Component):
    def __init__(self, cd, empty_mass, engine):
        pyrse.rocket_components.Component.__init__(self)
        self.__cd = cd
        self.__empty_mass = empty_mass
        self.__engine = engine
        self.add(engine)

    @property
    def cd(self):
        return self.__cd
    
    @cd.setter
    def cd(self, _cd):
        self.__cd = _cd
        return self.__cd
    
    def calc_mass(self, t0):
        return self.__empty_mass

    def calc_cd(self, v, env):
        return self.__cd


def run_flight_simulation(env, pad, model):
    simulator = pyrse.simulator.Simulation1D(env, pad, [model])

    
def main():
    engs = pyrse.engines.EngineDirectory('./Engines')
    engine = engs.load_first("Estes Industries, Inc.", "E16") 
    model = EiffelTowerModel(1, 0.1, engine)
    pad = pyrse.pad.LaunchPad((38.155458, -104.808906, 1663.5))

if __name__ == '__main__':
    main()