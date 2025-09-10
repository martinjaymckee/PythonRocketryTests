import itertools
import math
import os
import os.path
import random
import re
import xml.etree
import xml.etree.ElementTree

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate

from pyrse import rocket_components


# TODO: ADD EQUALITY CHECK TO THE ENGINE OBJECT
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
        exceptions = []
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
                    kwargs['propellent_mass'] = float(prop_mass) / 1000.0
                    kwargs['empty_mass'] = (float(total_mass) / 1000.0) - kwargs['propellent_mass']
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
                    exceptions.append(e)
        
        if len(exceptions) > 0:
            raise(exceptions[0])
        
        if engines is None:
            return None
        return engines[0] if len(engines) == 1 else engines
    
    def __init__(self, ts=[], Ts=[], manufacturer=None, model=None, diameter=None, length=None, delays=None, propellent_mass=0, empty_mass=0, t_start=None, comments=None, src=None):
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
        self.__thrust_spline = scipy.interpolate.UnivariateSpline(self.__ts, self.__Ts, s=1, k=1)
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
    def mass_fraction(self):
        return 100 * self.propellent_mass / self.total_mass
    
    @property
    def delays(self): return self.__delays
    
    @property
    def reference_thrust_curve(self):
        return self.__ts[:], self.__Ts

    def thrust(self, t):
        if (self.__t_start is None) or (t < self.__t_start) or (t > (self.__t_start + self.__t_burn)):
            return 0
        return float(self.__thrust_spline(t-self.__t_start))

    def spent_impulse(self, t):
        if (self.__t_start is None) or (t < self.__t_start):
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
    
    def start(self, t):
        self.__t_start = t
        
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
        
    def Scaled(self, impulse_multiplier = 1, burn_rate_multiplier = 1, noise_sd=0):
        thrust_multiplier = impulse_multiplier / burn_rate_multiplier

        new_ts = np.array([burn_rate_multiplier * t for t in self.__ts])        
        new_Ts = []
        #noise_sd = noise_sd * np.max(self.__Ts)
        for T in self.__Ts:
            T_noise = 0 if noise_sd == 0 else T * random.gauss(0, noise_sd)
            T = max(0, (thrust_multiplier * T) + T_noise)
            new_Ts.append(T)
        new_Ts = np.array(new_Ts)
        
        return Engine(
            new_ts, 
            new_Ts, 
            manufacturer=self.__manufacturer, 
            model=self.__model, 
            diameter=self.__diameter, 
            length=self.__length, 
            delays=self.__delays, 
            propellent_mass=self.__propellent_mass, 
            empty_mass=self.__empty_mass, 
            t_start=self.__t_start, 
            comments=('' if self.__comments is None else self.__comments) + ' --> Scaled with impulse = x {}, burn rate = x {}, thrust = x {}, std(noise) = {}'.format(impulse_multiplier, burn_rate_multiplier, thrust_multiplier, noise_sd), 
            src=self.__src
        )

    def Clone(self):
        return Engine(
            self.__ts, 
            self.__Ts, 
            manufacturer=self.__manufacturer, 
            model=self.__model, 
            diameter=self.__diameter, 
            length=self.__length, 
            delays=self.__delays, 
            propellent_mass=self.__propellent_mass, 
            empty_mass=self.__empty_mass, 
            t_start=self.__t_start, 
            comments=('' if self.__comments is None else self.__comments), 
            src=self.__src
        )


class EngineDirectory:
    __manufacturer_map = {
        'AT': 'Aerotech'
    }

    def __init__(self, directory=None, fail_on_parsing=False):
        self.__engines = {}
        self.__num_files = 0
        self.__fail_on_parsing = fail_on_parsing
        self.__model_regex = re.compile('[a-sA-S][1-9][0-9]*')
        directory = './' if directory is None else directory
        self.reload(directory, force_clean=True)

    def __len__(self):
        return len(self.__engines.keys())
    
    @property
    def num_files(self):
        return self.__num_files
    
    @property
    def directory(self):
        return list(self.__engines.keys())
    
    def reload(self, directory, force_clean=False):
        if force_clean:
            self.__engines = {}
            self.__num_files = 0
        
        # TODO: WITHOUT FORCE_CLEAN THIS WILL RESULT IN DUPLICATE ENGINES LOADED
        #   AS SUCH, THIS NEEDS TO BE ABLE TO CHECK THAT THE ACTUAL DEFINITION OF
        #   AN ENGINE FILE MATCHES ONE ALREADY STORED
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
                            self.__num_files += 1
                        else:
                            last = self.__engines[(eng.manufacturer, eng.model)]
                            self.__engines[(eng.manufacturer, eng.model)] = [last, eng]
                            self.__num_files += 1                            
                    else:
                        self.__engines[(eng.manufacturer, eng.model)] = eng                    
                        self.__num_files += 1                        
            except Exception as e:
                if self.__fail_on_parsing:
                    raise(e)

    def load(self, manufacturer, model='', src=None, approx_match=True):  # NOTE: THIS SHOULD ALSO CHECK FOR NEAR MATCHES
        try:
            eng = None
            if approx_match:
                tokens = [t.strip() for t in manufacturer.split()]
                manufacturer_tokens = []
                model_tokens = []
                for token in tokens:
                    if self.__is_model(token):
                        model_tokens.append(self.__strip_model(token).upper())
                    if self.__is_manufacturer(token):
                        manufacturer_tokens.append(token.title())
                for manufacturer, model in itertools.product(manufacturer_tokens, model_tokens):
                    for manufacturer_key, model_key in self.__engines.keys():
                        if (manufacturer in manufacturer_key) and (model in model_key):
                            new_eng = self.__engines[(manufacturer_key, model_key)]
                            if eng is None:
                                eng = new_eng
                            elif isinstance(eng, list):
                                eng.append(new_eng)
                            else:
                                eng = [eng, new_eng]
            else:
                if manufacturer in EngineDirectory.__manufacturer_map:
                    manufacturer = EngineDirectory.__manufacturer_map[manufacturer]
                eng = self.__engines[(manufacturer.title(), model.upper())]
            if isinstance(eng, list):
                return [e.duplicate() for e in eng if (src is None) or (e.src == src)]
            return eng.duplicate() if (src is None) or (eng.src == src) else None
        except Exception as e:
            if self.__fail_on_parsing:
                raise(e)

        return None
    
    def load_first(self, manufacturer, model='', src=None, approx_match=True):
        engs = self.load(manufacturer, model, src=src, approx_match=True)
        if isinstance(engs, list):
            return engs[0]
        return engs

    def __is_model(self, model):
        return self.__model_regex.match(model) is not None

    def __strip_model(self, model):
        model, _, delay = model.partition('-')
        return model

    def __is_manufacturer(self, manufacturer):
        return True

    
class EngineRandomizer:
    def __init__(self, ref_eng, impulse_range=0.005, burn_rate_range=0.0025, noise_sd=0.0005):
        self.__ref_eng = ref_eng
        self.__impulse_range = impulse_range
        self.__burn_rate_range = burn_rate_range
        self.__noise_sd = noise_sd
        
    def __call__(self):
        impulse_multiplier = random.uniform(1-self.__impulse_range, 1+self.__impulse_range)
        burn_rate_multiplier = random.uniform(1-self.__burn_rate_range, 1+self.__burn_rate_range)
        return self.__ref_eng.Scaled(impulse_multiplier, burn_rate_multiplier, self.__noise_sd)
        

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
