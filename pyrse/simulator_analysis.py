import numbers

import numpy as np


from pyrse import numpy_utils as npu
from pyrse import utils


class LogResults:
    def __init__(self, results, debug=False):
        # TODO: CONVERT VALUES TO NP.ARRAY OBJECTS AS LONG AS THEY ARE NUMERIC
        # TODO: CHECK THAT ALL VALUES LISTS ARE OF THE SAME LENGTH
        num_values = None
        self.__keys = []
        for name, values in results.items():
            self.__keys.append(name)
            if num_values is None:
                num_values = len(values)
            assert num_values == len(values), 'Error: Attempting to build a LogResult object with value list of heterogeneous length.'
            setattr(self, name, values)
    
    def __str__(self):
        return 'LogResults({})'.format(', '.join(self.__keys))

    @property
    def keys(self):
        return self.__keys[:]

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return None


class AST_Base:
    __last_id = 0
    def __init__(self, name, debug=False):
        self.__name = name
        self.__debug = debug
        self.__id = AST_Base.__last_id
        AST_Base.__last_id += 1
        
    @property
    def id(self):
        return self.__id
    
    @property
    def name(self):
        if self.__name is None:
            preferred = self.get_preferred_name()
            if preferred is not None:
                return preferred
            return 'node_{}'.format(self.__id)
        return self.__name
    
    @property
    def debug(self):
        return self.__debug
    
    def get_preferred_name(self):
        return None
    
    
class AST_Const(AST_Base):
    def __init__(self, value, name=None, debug=False):
        AST_Base.__init__(self, name, debug)
        self.value = value
        
    def __str__(self):
        if self.debug:
            return '{}:Const[{}]'.format(self.name, self.value)
        return '{}:{}'.format(self.name, self.value)
    
    def __call__(self, value, ctx, _):
        return self.value
    

class AST_Path(AST_Base):
    def __init__(self, path, name=None, debug=False):
        AST_Base.__init__(self, name, debug)
        self.path = path
           
    def __str__(self):
        if self.debug:
            return '{}:Path[{}]'.format(self.name, '->'.join(self.path))
        return '{}:{}'.format(self.name, '.'.join(self.path))
    
    def __call__(self, value, ctx, transformers):
        obj = value
        for attr in self.path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return None
        return obj
 
    def get_preferred_name(self):
        return '.'.join(self.path)

    
class AST_Transform(AST_Base):
    def __init__(self, func, args=[], name=None, debug=False):
        AST_Base.__init__(self, name, debug)
        self.func = func
        self.args = args
        
    def __str__(self):
        formatted_args = ', '.join([str(arg) for arg in self.args])
        if self.debug:
            return '{}:Transform[{}({})]'.format(self.name, self.func, formatted_args)
        return '{}:{}({})'.format(self.name, self.func, formatted_args)
    
    def __call__(self, value, ctx, transformers):
        if not self.func in transformers:
            return None
        
        transformed_args = []
        for arg in self.args:
            if isinstance(arg, numbers.Number):
                transformed_args.append(arg)
            else:
                transformed_args.append(arg(value, ctx, transformers))
                
        return transformers[self.func](ctx, *transformed_args)
            
    
class AST_Parser:
    Idle = 0
    ParsePath = 1
    ParseTransformArgs = 2
    ParseNumber = 3
    
    def __init__(self, debug=False):
        self.__debug = debug
    
    
    def __call__(self, fmts):
        asts = []
        
        for fmt in fmts:
            ast = self.__parse_fmt(fmt)
            asts.append(ast)
        return asts
    

    def __parse_fmt(self, fmt):
        fmt = ''.join(fmt.split()) # remove white space from the format
        ast = None
        tag = None
        t0 = ''
        t1 = ''
        stack = []
        args = []
        parens_count = 0
        mode = AST_Parser.Idle
        first = True

        for idx, c in enumerate(fmt):
            if mode == AST_Parser.ParseTransformArgs:
                if c == ',':
                    args.append(t1)
                    t1 = ''
                elif c == ')':
                    if len(t1) > 0:
                        args.append(t1)
                    ast_args = [self.__parse_fmt(arg) for arg in args]
                    ast = AST_Transform(t0, ast_args, tag, debug=self.__debug)
                    t0 = ''
                    t1 = ''
                else:
                    t1 += c
            elif mode == AST_Parser.ParseNumber:
                t0 += c
            else:
                if c == '.':
                    assert len(t0) > 0, 'Error: Attempting to parse an empty name in fmt = {}'.format(self.__mark_err_loc(fmt, idx))
                    args.append(t0)
                    t0 = ''
                    first = False
                    mode = AST_Parser.ParsePath
                elif c == ':':
                    if first:
                        assert len(t0) > 0, 'Error: Attempting to set empty value tag in fmt = {}'.format(self.__mark_err_loc(fmt, idx))
                        tag = t0
                        t0 = ''
                        first = False
                elif c == ',':
                    raise Exception('Error: Argument separator out of place in fmt = {}'.format(self.__mark_err_loc(fmt, idx)))
                    return None
                elif c == '(':
                    assert mode == AST_Parser.Idle, 'Error: Rogue parentheses found in path parse at idx = {} ({})'.format(idx, self.__mark_err_loc(fmt, idx))
                    mode = AST_Parser.ParseTransformArgs
                    args = []
                    t1 = ''
                    first = False
                    parens_count += 1
                elif c == ')':
                    raise Exception('Error: Incorrect mode for closing argument list')
                    return None
                else:
                    if (len(t0) == 0) and c.isnumeric():
                        mode = AST_Parser.ParseNumber
                    t0 += c
        if mode == AST_Parser.Idle:
            ast = AST_Path([t0], tag, debug=self.__debug)
        elif mode == AST_Parser.ParsePath:
            ast = AST_Path(args+[t0], tag, debug=self.__debug)
        elif mode == AST_Parser.ParseNumber:
            ast = AST_Const(float(t0), tag, debug=self.__debug)
        return ast
    
    def __mark_err_loc(self, fmt, idx):
        pre = fmt[:idx]
        post = fmt[idx:]
        return '{}^{}'.format(pre, post)
    
    
class SimulationLogExtractor:
    default_fmts = ['t', 'accel', 'vel', 'pos']
    default_transforms = {
        'magnitude': lambda _, v: npu.magnitude(v.ecef if isinstance(v, utils.Vector3D) else v),
        'abs': lambda _, a: abs(a),
        'enu': lambda _, a: a.enu,
        'ecef': lambda _, a: a.ecef,
        'alt': lambda sim, a: a.enu(sim.pad.pos)[2],
    }
    
    def __init__(self, fmts=None, debug=False):
        self.__fmts = SimulationLogExtractor.default_fmts if fmts is None else fmts
        self.__asts = AST_Parser(debug=debug)(fmts)
        self.__transforms = SimulationLogExtractor.default_transforms # TODO: MAKE THIS CONFIGURABLE
        
    @property
    def asts(self):
        return self.__asts
    
    def __call__(self, sim): # TODO: ADD SOME DEBUGGING INFORMATION INTO THIS....
        logs = sim.logs
        results = []
        data = {}
    
        for ast in self.__asts:
            data[ast.name] = []
    
        for log in logs:         
            for sample in log:
                for ast in self.__asts:
                    try:
                        result = ast(sample, sim, self.__transforms)
                        data[ast.name].append(result)
                    except Exception as e:
                        print(e)
            results.append(LogResults(data))
        return results