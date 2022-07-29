import math
import os
import os.path
import xml.etree

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
        diameter = None
        length = None
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
                    diameter = float(args[1])
                    length = float(args[2])
                    propellent_mass = float(args[4])
                    empty_mass = float(args[5]) - propellent_mass
                    header_parsed = True
                else:
                    args = line.split()
                    if not len(args) == 2:
                        raise Exception('RASP data point line wrong length ({}).format(len(args))')
                    ts.append(float(args[0]))
                    Ts.append(float(args[1]))
        kwargs = {'src': 'RASP'}
        kwargs['ts'] = ts
        kwargs['Ts'] = Ts
        kwargs['manufacturer'] = manufacturer
        kwargs['model'] = model
        kwargs['diameter'] = diameter
        kwargs['length'] = length
        kwargs['propellent_mass'] = propellent_mass
        kwargs['empty_mass'] = empty_mass
        return Engine(**kwargs)

    @classmethod
    def RSE(cls, path, raw_data=False, manufacturer=None, model=None):
        tree = xml.etree.ElementTree.parse(path) if not raw_data else xml.etree.ElementTree(xml.etree.ElementTree.fromstring(path))
        root = tree.getroot()
        engines = None
        if not root.tag == 'engine-database':
            return engines

        for engine_list in root.findall('engine-list'):
            for engine_data in engine_list:
                try:
                    kwargs = {'src': 'RSE'}
                    file_manufacturer = engine_data.get('mfg') 
                    file_model = engine_data.get('code')
                    kwargs['manufacturer'] = (file_manufacturer if manufacturer is None else manufacturer).capitalize()
                    kwargs['model'] = (file_model if model is None else model).upper()
                    dia = engine_data.get('dia')
                    kwargs['diameter'] = None if dia is None else float(dia)
                    length = engine_data.get('len')                    
                    kwargs['length'] = None if length is None else float(length)
                    total_mass = engine_data.get('initWt')
                    assert total_mass is not None, 'Engine file did not contain total mass'
                    prop_mass = engine_data.get('propWt')
                    assert prop_mass is not None, 'Engine file did not contain propellant mass'                    
                    kwargs['propellent_mass'] = float(prop_mass)
                    kwargs['empty_mass'] = float(total_mass) - kwargs['propellent_mass']
                    comments_element = engine_data.find('comments')
                    kwargs['comments'] = None if comments_element is None else comments_element.text.replace('\n', ' ')

                    ts, Ts = [], []
                    data_element = engine_data.find('data')                    
                    for data_line in data_element:
                        ts.append(float(data_line.get('t')))
                        Ts.append(float(data_line.get('f')))
                    if ts[0] > 0.0:
                        ts = [0] + ts
                        Ts = [0] + Ts
                    kwargs['ts'] = ts
                    kwargs['Ts'] = Ts

                    if engines is None:
                        engines = []
                    engines.append(Engine(**kwargs))
                except Exception as e:
                    print('Failure when parsing, {} - {}'.format(path, e))
        if engines is None:
            return None
        return engines[0] if len(engines) == 1 else engines
    
    def __init__(self, ts=[], Ts=[], manufacturer=None, model=None, diameter=None, length=None, delays=None, propellent_mass=0, empty_mass=0, t_start=0, comments=None, src=None):
        rocket_components.Component.__init__(self, np.array([0.0, 0.0, 0.0]), 0.0)
        # print(manufacturer, model)
        self.__ts = np.array(ts)
        self.__Ts = np.array(Ts)
        self.__manufacturer = manufacturer.title()
        self.__model = model
        self.__delays = delays
        self.__diameter = diameter
        self.__length = length
        self.__propellent_mass = propellent_mass
        self.__empty_mass = empty_mass
        self.__comments = comments
        self.__src = src
        self.__t_burn = np.max(self.__ts)
        self.__t_start = t_start
        self.__max_thrust = np.max(self.__Ts)
        self.__total_impulse = np.trapz(self.__Ts, self.__ts)
        self.__impulse_class = self.__get_impulse_class(self.__total_impulse)
        self.__thrust_spline = scipy.interpolate.UnivariateSpline(self.__ts, self.__Ts, s=0, k=1)
        # self.__thrust_spline = scipy.interpolate.interp1d(self.__ts, self.__Ts, kind='cubic')
        self.__ueq = self.__total_impulse / self.__propellent_mass

    def __str__(self):
        if self.src is None:
            return 'Engine({} {})'.format(self.manufacturer, self.model)
        return 'Engine({} {} {})'.format(self.manufacturer, self.model, self.src)
    
    @property
    def diameter(self): return self.__diameter
    
    @property
    def length(self): return self.__length
    
    @property
    def burn_time(self): return self.__t_burn

    @property
    def total_impulse(self): return self.__total_impulse
    
    @property
    def impulse_class(self): return self.__impulse_class

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

    @property
    def src(self): return self.__src
    
    @property
    def delays(self): return self.__delays
    
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

    def properties(self, t):
        pass
    
    def __get_impulse_class(self, total_impulse):
        if total_impulse <= 0.3125:
            return '1/8A'
        elif total_impulse <= 0.625:
            return '1/4A'
        elif total_impulse <= 1.25:
            return '1/2A'
        elif total_impulse <= 2.5:
            return 'A'
        
        N = math.ceil(math.log2(total_impulse/2.5))
        impulse_class = chr(ord('A') + N)
        return impulse_class
        # print('total_impulse = {}, N = {}, impulse class = {}'.format(total_impulse, N, impulse_class))
        

# def load_engine_files(directory=None):
#     directory = './' if directory is None else directory
#     engines = {}
#     for file in os.listdir(os.path.abspath(directory)):
#         try:
#             path = os.path.abspath(os.path.join(directory, file))
#             name, ext = os.path.splitext(file)
#             manufacturer, _, model = name.partition('_')
#             if ext == '.eng':
#                 eng = Engine.RASP(path, manufacturer=manufacturer)
#                 engines[(manufacturer, model)] = eng
#             elif ext == '.rse':
#                 eng = Engine.RSE(path)
#                 engines[(eng.manufacturer, eng.model)] = eng                
#             else:
#                 pass
#         except Exception as e:
#             print(e)
#     return engines


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
                eng = None
                if ext == '.eng':
                    eng = Engine.RASP(path, manufacturer=manufacturer)
                elif ext == '.rse':
                    eng = Engine.RSE(path)
                if eng is not None:
                    if (eng.manufacturer, eng.model) in self.__engines:
                        if isinstance(self.__engines[(eng.manufacturer, eng.model)], list):
                            self.__engines[(eng.manufacturer, eng.model)].append(eng)                                                    
                        else:
                            last = self.__engines[(eng.manufacturer, eng.model)]
                            self.__engines[(eng.manufacturer, eng.model)] = [last, eng]
                    else:
                        self.__engines[(eng.manufacturer, eng.model)] = eng                    
            except Exception as e:
                print(e)

    @property
    def directory(self):
        return self.__engines.keys()
    
    def load(self, manufacturer, model, src=None):  # NOTE: THIS SHOULD ALSO CHECK FOR NEAR MATCHES
        try:
            if manufacturer in EngineDirectory.__manufacturer_map:
                manufacturer = EngineDirectory.__manufacturer_map[manufacturer]
            eng = self.__engines[(manufacturer.title(), model.upper())]
            if isinstance(eng, list):
                # print([str(e) for e in eng])
                return [e.duplicate() for e in eng if (src is None) or (e.src == src)]
            # print(str(eng))
            return eng.duplicate() if (src is None) or (eng.src == src) else None
        except Exception as e:
            print(e)
        return None


if __name__ == '__main__':
    dt = 0.01
    engine_directory = EngineDirectory('../Engines')

    fig, ax = plt.subplots(1, figsize=(15, 12), sharex=True)
    fig.tight_layout()
    ax.set_title('Engine Thrust curves')
    for idx, (manufacturer, model) in enumerate(engine_directory.directory):
        engs = engine_directory.load(manufacturer, model, src=None)  # NOTE: THIS IS PROPERLY FILTERING BY SOURCE TYPE
        if engs is None:
            continue
        if not isinstance(engs, list):
            engs = [engs]
        for eng in engs:
            if eng is not None: # NOTE: IF WORKING PROPERLY, THE ENGINE SHOULD NEVER BE NULL
                # if 1.25 <= eng.total_impulse <= 2.5:
                # if eng.impulse_class == '1/2A':                
                if True:
                    ts = np.arange(0, eng.burn_time, dt)
                    Ts = np.array([eng.thrust(t) for t in ts])
                    label = '{} {} {} ({}mm x {}mm)'.format(manufacturer, model, eng.src, eng.diameter, eng.length)
                    ax.plot(ts, Ts, alpha=0.5, label=label)
            else:
                print('Invalid engine found with engs = {}, manufacturer = {}, model = {}'.format(engs, manufacturer, model))
    ax.legend()
    plt.show()
