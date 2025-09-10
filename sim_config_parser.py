import numbers
import random

import pyrse.engines


class VarianceConfig:
    def __init__(self, value, percent=False):
        self.__value = value 
        self.__percent = percent

    def __str__(self):
        if self.__percent:
            return 'sd({}%)'.format(self.__value)
        return 'sd({})'.format(self.__value)

    @property
    def value(self):
        return self.__value
    
    @property
    def percent(self):
        return self.__percent

    def __mul__(self, c):
        return VarianceConfig(c*self.__value, self.__percent)
        
    def __rmul__(self, c):
        return self.__mul__(c)


class MotorConfig:
    def __init__(self, eng, variance):
        self.__eng = eng
        total_impulse = self.__eng.total_impulse
        self.__sd = 0 if variance is None else (total_impulse * variance.value / 100 if variance.percent else variance.value)
        self.__is_constant = (self.__sd == 0)
        self.__eng_randomizer = pyrse.engines.EngineRandomizer(self.__eng, impulse_range = self.__sd, burn_rate_range = 0.02, noise_sd=0.0015)

    def __str__(self):
        if self.__is_constant:
            return '{}'.format(self.__eng)
        return '{} +- sd({})'.format(self.__eng, self.__sd)

    @property
    def eng(self):
        return self.__eng.Scaled()

    @property
    def is_constant(self):
        return self.__is_constant

    @property
    def sd(self):
        return self.__sd

    def __call__(self, idx=None, N=None):
        if self.is_constant:
            return self.__eng.Scaled()
        return self.__eng_randomizer()


class NumericValueConfig:
    def __init__(self, mean, variance):
        self.__mean = mean
        self.__sd = 0 if variance is None else (mean * variance.value / 100 if variance.percent else variance.value)
        self.__is_constant = (self.__sd == 0)

    def __str__(self):
        if self.__is_constant:
            return '{}'.format(self.__mean)
        return '{} +- sd({})'.format(self.__mean, self.__sd)

    @property
    def mean(self):
        return self.__mean

    @property
    def is_constant(self):
        return self.__is_constant

    @property
    def sd(self):
        return self.__sd

    def __call__(self, idx=None, N=None):
        if self.is_constant:
            return self.__mean
        return random.gauss(self.__mean, self.__sd)


class FunctionalValueConfig:
    def __init__(self, variance):
        self.__variance = variance

    def __call__(self, idx=1, N=1):
        return self
    
    def transform(self, val):
        assert False, 'Error: {}.transform(val) is undefined'.format(self.__class__.__name__)


class StepValueConfig(FunctionalValueConfig):
    def __init__(self, initial, final, threshold, variance = 0):
        super().__init__(variance)
        self.__initial = initial
        self.__final = final
        self.__threshold = threshold

    def __call__(self, idx=1, N=1):
        return self

    def transform(self, val):
        if val > self.__threshold:
            return self.__final
        return self.__initial


class NumericValueRangeConfig:
    def __init__(self, a, b, variance, is_log=False):
        self.__a = a
        self.__b = b
        self.__variance = variance
        mean = (a + b) / 2 # TODO: FIGURE OUT IF THERE IS A BETTER WAY TO CALCULATE SD....
        self.__sd = 0 if variance is None else (mean * variance.value / 100 if variance.percent else variance.value)
        self.__is_log = is_log
        self.__is_constant = (self.__sd == 0) and (self.__a == self.__b)

    def __str__(self):
        if self.__is_constant:
            return '{}|{}'.format(self.__a, self.__b)
        return '{}|{} +- sd({})'.format(self.__a, self.__b, self.__sd)

    @property
    def minimum(self):
        return min(self.__a, self.__b)

    @property
    def maximum(self):
        return max(self.__a, self.__b)

    @property
    def mean(self):
        return (self.__a + self.__b) / 2

    @property
    def is_constant(self):
        return self.__is_constant

    @property
    def is_log(self):
        return self.__is_log

    @property
    def sd(self):
        return self.__sd

    def __call__(self, idx=1, N=1):
        mean = self.mean
        if (N == 1) or self.is_constant:
            return mean
        mean = 0
        if self.__is_log:
            C = (self.__b / self.__a) ** (1 / (N-1))
            mean = self.__a * (C ** idx)
        else:
            step = (self.__b - self.__a) / (N-1)
            mean = self.__a + (idx * step)
        return random.gauss(mean, self.__sd)

    def __mul__(self, c):
        return NumericValueRangeConfig(c*self.__a, c*self.__b, c*self.__variance, self.__is_log)
        
    def __rmul__(self, c):
        return self.__mul__(c)


class SimulationConfig:
    def __init__(self, N=1, eng=None, m=None, cd=None, S=None, plots=None, **kwargs):
        self.__N = N
        self.__eng = eng
        self.__m = m
        self.__cd = cd
        self.__S = S 
        self.__plots = plots 
        self.__kwargs = kwargs

    @property
    def N(self):
        return self.__N

    @property
    def eng(self):
        return self.__eng

    @property
    def m(self):
        return self.__m

    @property
    def cd(self):
        return self.__cd

    @property
    def S(self):
        return self.__S

    @property
    def plots(self):
        return self.__plots

    @property
    def kwargs(self):
        return self.__kwargs


class SimulationMotorParser:
    def __init__(self, engine_directory):
        self.__engine_directory = engine_directory
        self.__engs = pyrse.engines.EngineDirectory('./Engines')

    def __call__(self, motor):
        eng = self.__engs.load_first(motor, approx_match=True) 
        return eng


# class SimulationCdParser:
#     def __init__(self):
#         pass

#     def __call__(self, cd):
#         try:
#             return float(cd)
#         except:
#             pass
#         return cd

# def log_range(a, b):
#     return NumericValueRangeConfig(a, b, None, is_log=True)

# def linear_range(a, b):
#     return NumericValueRangeConfig(a, b, None, is_log=False)

# def step_cd(cd0, cd1, RE_thresh):
#     return StepValueConfig(cd0, cd1, RE_thresh)

class PythonParser:
    __suffix_map = {
        'kg': 1,        
        'g': .001,
        'm': 1,
        'cm': .01,
        'mm': .001,
        'm_2': 1,
        'cm_2': 1 / (100 * 100),
        'mm_2': 1 / (1000 * 1000)
    }

    def __init__(self):
        self.__globals = {
            'log_range': lambda a, b: NumericValueRangeConfig(a, b, None, is_log=True),
            'linear_range': lambda a, b: NumericValueRangeConfig(a, b, None, is_log=False),
            'step_cd': lambda cd0, cd1, RE_thresh: StepValueConfig(cd0, cd1, RE_thresh)
        }
        self.__suffix_list = list(reversed(sorted(PythonParser.__suffix_map.keys(), key=lambda x:len(x))))
        self.__locals = {}
        
    def __call__(self, value):
        value = value.strip()
        multiplier = None
        for suffix in self.__suffix_list:
            if value.endswith(suffix):
                multiplier = PythonParser.__suffix_map[suffix]
                value = value[:-int(len(suffix))].strip()
        value = '__result = ' + value
        exec(value, self.__globals)
        result = self.__globals['__result']
        if multiplier is not None:
            result = result * multiplier
        return result


class SimulationConfigParser:
    Idle = 0
    KeyParse = 1
    ValueParse = 2
    VarianceParse = 3
    CommentParse = 4
    IntervalueCommentParse = 5

    whitespace = ['\n', ' ', '\t', '\r']

    def __init__(self, engine_directory='Engines'):
        self.__char_buffer = ''
        self.__list_buffer = []
        self.__mode = SimulationConfigParser.Idle
        self.__bracket_stack = []
        self.__key = None
        self.__value = None
        self.__variance = None
        self.__func_name = None
        self.__results = {}
        self.__python_parser = PythonParser()
        self.__value_parsers = {
            'sims': lambda x: int(self.__python_parser(x)),
            'dt': self.__python_parser,
            'mass': self.__python_parser,
            'motor': SimulationMotorParser(engine_directory),
            'cd': self.__python_parser,
            'ref_area': self.__python_parser
        }

    def __call__(self, path):
        with open(path, 'r') as file:
            for c in file.read():
                if self.__mode == SimulationConfigParser.Idle:
                    if c == ';':
                        self.__char_buffer = ''
                        self.__mode = SimulationConfigParser.CommentParse
                    elif not c in SimulationConfigParser.whitespace:
                        self.__char_buffer = c
                        self.__mode = SimulationConfigParser.KeyParse
                elif self.__mode == SimulationConfigParser.KeyParse:
                    if c == '\n':
                        assert False, 'Error: Newline during parsing of configuration Key! key = {}'.format(self.__char_buffer)
                    elif c == ':':
                        self.__process_key()
                        self.__mode = SimulationConfigParser.ValueParse
                    else:
                        self.__char_buffer += c
                elif self.__mode == SimulationConfigParser.ValueParse:
                    if c == '[':
                        self.__bracket_stack.append(c)
                    elif c == ']':
                        if len(self.__bracket_stack) < 1:
                            assert False, 'Error: closing bracket found when bracket stack is empty'
                        self.__bracket_stack = self.__bracket_stack[:-1] # Drop the last bracket
                    elif c == ';':
                        if len(self.__bracket_stack) > 0:
                            self.__mode = SimulationConfigParser.IntervalueCommentParse
                        else:
                            self.__process_value()                            
                            self.__mode = SimulationConfigParser.CommentParse
                    # elif c == ',':
                    #     if len(self.__bracket_stack) > 0:
                    #         pass # TODO: USE THIS FOR LIST PROCESSING 
                    elif c == '@':
                        if len(self.__bracket_stack) > 0:
                            assert False, "Error: attempting to parse a variance when a list value has not been closed (len(bracket_stack) == {})".format(len(self.__bracket_stack))
                        self.__process_value()
                        self.__mode = SimulationConfigParser.VarianceParse
                    else:
                        if (c == '\n') and (len(self.__bracket_stack) == 0):
                            self.__process_value()
                            self.__finalize_line_parse()
                        else:
                            self.__char_buffer += c
                elif self.__mode == SimulationConfigParser.VarianceParse:
                    if c == '\n':
                        self.__process_variance()
                        self.__finalize_line_parse()
                        self.__mode = SimulationConfigParser.Idle
                    elif c == ';':
                        self.__process_variance()
                        self.__mode = SimulationConfigParser.CommentParse
                    else:
                        self.__char_buffer += c
                elif self.__mode == SimulationConfigParser.CommentParse:
                    if (c == '\n'):
                        self.__finalize_line_parse()
                elif self.__mode == SimulationConfigParser.IntervalueCommentParse:
                    if (c == '\n'):
                        self.__mode = SimulationConfigParser.ValueParse
            results = self.__results
            self.__results = {}
            return results
        return None

    def __process_key(self):
        key = self.__char_buffer.strip()
        self.__char_buffer = ''
        self.__key = key 

    def __process_value(self):
        value = self.__char_buffer.strip()
        self.__char_buffer = ''
        if self.__key in self.__value_parsers:
            func = self.__value_parsers[self.__key]
            value = func(value) 
        self.__value = value            

    def __process_variance(self):
        variance = self.__char_buffer.strip()
        self.__char_buffer = ''

        percent = False
        if variance[-1] == '%':
            variance = variance[:-1]
            percent = True
        variance = self.__python_parser(variance)
        self.__variance = VarianceConfig(variance, percent)

    def __finalize_line_parse(self):
        if (self.__key is not None) and (self.__value is not None):
            self.__mode = SimulationConfigParser.Idle
            if isinstance(self.__value, numbers.Number):
                self.__results[self.__key] = NumericValueConfig(self.__value, self.__variance)
            elif isinstance(self.__value, pyrse.engines.Engine):
                self.__results[self.__key] = MotorConfig(self.__value, self.__variance)
            elif isinstance(self.__value, NumericValueRangeConfig): # TODO: FIX THIS TO USE THE VARIANCE ALSO
                self.__results[self.__key] = self.__value
            elif isinstance(self.__value, StepValueConfig):
                self.__results[self.__key] = self.__value
            elif isinstance(self.__value, tuple):
                if isinstance(self.__value[0], numbers.Number):
                    self.__results[self.__key] = NumericValueRangeConfig(self.__value[0], self.__value[1], self.__variance)
                else: # TODO: FIX THE FACT THAT THIS IS SIMPLY IGNORING VARIANCE ON ANYTHING OTHER THAN A NUMBER
                    self.__results[self.__key] = self.__value[0]
            else:
                self.__results[self.__key] = (self.__value, self.__variance)
        self.__char_buffer = ''
        self.__bracket_stack = []
        self.__mode = SimulationConfigParser.Idle
        self.__key = None
        self.__value = None
        self.__variance = None 


if __name__ == '__main__':
    import os
    import os.path

    directory = r"D:\Workspace\Rockets\PythonRocketryTests\Simulation Configurations"
    #filename = "test_sim.cfg"
    filename = "anomoloy_detection_sim.cfg"

    path = os.path.join(directory, filename)
    path = os.path.abspath(path)

    python_parser = PythonParser()
    python_parser('log_range(.001, .1)')
    python_parser('step_cd(0.8, 0.65, 1.5E6)')

    dt_range = NumericValueRangeConfig(0.001, 0.1, None, is_log=True)
    dts = []
    N = 5
    for idx in range(N):
        dts.append(dt_range(idx, N))
    print(dts)

    cfg_parser = SimulationConfigParser()

    sim_cfg = cfg_parser(path)

    for key, value_cfg in sim_cfg.items():
        print('{}: {}'.format(key, value_cfg))
