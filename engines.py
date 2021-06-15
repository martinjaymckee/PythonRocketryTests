import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate

import rocket_components


class Engine(rocket_components.Component):
    @classmethod
    def RASP(cls, path, manufacturer=None, model=None):
        ts, Ts = [0], [0]
        header_parsed = False
        propellent_mass = None
        empty_mass = None
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(';') or line == '':
                    continue
                if not header_parsed:
                    args = line.split()
                    if not len(args) == 7:
                        raise Exception('RASP header line wrong length ({}).format(len(args))')
                    model = args[0] if model is None else model
                    manufacturer = args[6] if manufacturer is None else manufacturer
                    propellent_mass = float(args[4])
                    empty_mass = float(args[5]) - propellent_mass
                    header_parsed = True
                else:
                    args = line.split()
                    if not len(args) == 2:
                        raise Exception('RASP data point line wrong length ({}).format(len(args))')
                    ts.append(float(args[0]))
                    Ts.append(float(args[1]))
        kwargs = {}
        kwargs['ts'] = ts
        kwargs['Ts'] = Ts
        kwargs['manufacturer'] = manufacturer
        kwargs['model'] = model
        kwargs['propellent_mass'] = propellent_mass
        kwargs['empty_mass'] = empty_mass
        return Engine(**kwargs)

    def __init__(self, ts=[], Ts=[], manufacturer=None, model=None, propellent_mass=0, empty_mass=0, t_start=0):
        rocket_components.Component.__init__(self, np.array([0.0, 0.0, 0.0]), 0.0)
        self.__ts = np.array(ts)
        self.__Ts = np.array(Ts)
        self.__manufacturer = manufacturer
        self.__model = model
        self.__propellent_mass = propellent_mass
        self.__empty_mass = empty_mass
        self.__t_burn = np.max(self.__ts)
        self.__t_start = t_start
        self.__max_thrust = np.max(self.__Ts)
        self.__total_impulse = np.trapz(self.__Ts, self.__ts)
        self.__thrust_spline = scipy.interpolate.UnivariateSpline(self.__ts, self.__Ts, s=0, k=1)
        # self.__thrust_spline = scipy.interpolate.interp1d(self.__ts, self.__Ts, kind='cubic')
        self.__ueq = self.__total_impulse / self.__propellent_mass

    @property
    def burn_time(self): return self.__t_burn

    @property
    def total_impulse(self): return self.__total_impulse

    @property
    def max_thrust(self): return self.__max_thrust

    @property
    def average_thrust(self): return self.total_impulse / self.burn_time

    @property
    def empty_mass(self): return self.__empty_mass

    @property
    def propellent_mass(self): return self.__propellent_mass

    @property
    def total_mass(self): return self.empty_mass + self.propellent_mass

    @property
    def specific_impulse(self): return self.__ueq / 9.08665

    @property
    def manufacturer(self): return self.__manufacturer

    @property
    def model(self): return self.__model

    def thrust(self, t):
        if (t < self.__t_start) or (t > (self.__t_start + self.__t_burn)):
            return 0
        return self.__thrust_spline(t)

    def spent_impulse(self, t):
        if t < self.__t_start:
            return 0
        elif t > self.__t_start+self.__t_burn:
            return self.__total_impulse
        return self.__thrust_spline.integral(0, t-self.__t_start)

    def spent_mass(self, t):
        return self.__propellent_mass * (self.spent_impulse(t) / self.__total_impulse)

    def calc_mass(self, t):
        propellent_mass = self.__propellent_mass - self.spent_mass(t)
        total_mass = self.__empty_mass + propellent_mass
        return total_mass

    def cg(self, t0):  # Overload Component Value
        return self.pos


def load_engine_files(directory=None):
    directory = './' if directory is None else directory
    engines = {}
    for file in os.listdir(os.path.abspath(directory)):
        try:
            path = os.path.abspath(os.path.join(directory, file))
            name, ext = os.path.splitext(file)
            manufacturer, _, model = name.partition('_')
            if ext == '.eng':
                eng = Engine.RASP(path, manufacturer=manufacturer)
                engines[(manufacturer, model)] = eng
            else:
                pass
        except Exception as e:
            print(e)
    return engines


class EngineDirectory:
    __manufacturer_map = {
        'AT': 'Aerotech'
    }

    def __init__(self, directory=None):
        self.__engines = {}
        directory = './' if directory is None else directory
        for file in os.listdir(os.path.abspath(directory)):
            try:
                path = os.path.abspath(os.path.join(directory, file))
                name, ext = os.path.splitext(file)
                manufacturer, _, model = name.partition('_')
                manufacturer = manufacturer.capitalize()
                model = model.upper()
                if ext == '.eng':
                    eng = Engine.RASP(path, manufacturer=manufacturer)
                    self.__engines[(manufacturer, model)] = eng
                else:
                    pass
            except Exception as e:
                print(e)

    def load(self, manufacturer, model):  # NOTE: THIS SHOULD ALSO CHECK FOR NEAR MATCHES
        try:
            if manufacturer in EngineDirectory.__manufacturer_map:
                manufacturer = EngineDirectory.__manufacturer_map[manufacturer]
            eng = self.__engines[(manufacturer.capitalize(), model.upper())]
            return eng.duplicate()
        except Exception as e:
            print(e)
        return None


if __name__ == '__main__':
    dt = 0.01
    engines = load_engine_files('./engines')

    fig, ax = plt.subplots(1, figsize=(15, 12), sharex=True)
    fig.tight_layout()
    ax.set_title('Engine Thrust curves')
    for idx, (manufacturer, model) in enumerate(engines.keys()):
        eng = engines[(manufacturer, model)]
        ts = np.arange(0, eng.burn_time, dt)
        Ts = np.array([eng.thrust(t) for t in ts])
        label = '{} {}'.format(manufacturer, model)
        ax.plot(ts, Ts, alpha=0.5, label=label)
    ax.legend()
    plt.show()
