import numpy as np


import pyrse.simulator_analysis as psa
import pyrse.numpy_utils as npu

transforms = {
    'magnitude': lambda v: npu.magnitude(v.ecef if isinstance(v, utils.Vector3D) else v),
    'add': lambda a, b: a+b,
    'sum': sum,
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'std': np.std
}

def test_extractor_parse(expected, fmt, obj, tgt):
    # TODO: PROPERLY HANDLE EXPECTED RESULTS
    global transforms
    try:
        extractor = psa.SimulationLogExtractor([fmt], debug=True)
        print('Successfully parsed  {}'.format(fmt))
        ast = extractor.asts[0]
        print('\tAST = {}'.format(ast))
        # TODO: WHEN EXPECTED == TRUE AND PARSE SUCCEEDED, RUN THE EXTRACTOR AND PRINT THE RESULTS... COMPARE AGAINST THE TARGET
        result = ast(obj, transforms)
        print('\tResult = {}'.format(result))
    except Exception as e:
        print('Error when parsing {} -- ({})'.format(fmt, e))


if __name__ == '__main__':
    class Bar:
        def __init__(self):
            self.a = np.array([1, 2, 3, 4])
            self.b = np.array([5, 6, 7, 8])
            
    class B:
        def __init__(self):
            self.bar = Bar()
            self.v = np.array([1, 3, 5, 7, 9])
            
    class TestObj:
        def __init__(self):
            self.foo = B()
            self.fizz = np.array([0, 1, 1])
            self.buzz = 1
            self.fizzbuzz = 2
            
    obj = TestObj()
    
    #
    # Successful parses
    #
    test_extractor_parse(True, 'foo', obj, obj.foo)
    test_extractor_parse(True, 'c:foo.bar', obj, obj.foo.bar)
    test_extractor_parse(True, '3.14159', obj, 3.14159)
    test_extractor_parse(True, 'magnitude(fizz)', obj, 1.0)
    test_extractor_parse(True, 'add(buzz, fizzbuzz)', obj, 3.0)
    test_extractor_parse(True, 'add(foo.bar.a, foo.bar.b)', obj, np.array([6, 8, 10, 12]))
    test_extractor_parse(True, 'add(foo.bar.a, 10)', obj, np.array([11, 12, 13, 14]))
    test_extractor_parse(True, 'sum(foo.v)', obj, 25)
#    test_extractor_parse(True, 'mult(2.5, add(buzz, fizzbuz))', obj, 7.5) # TODO: THIS IS NOT CORRECTLY HANDLING THE NESTED FUNCTION
    
    print()
    
    #
    # Add tests for parses with errors
    #
    test_extractor_parse(False, 'foo)', obj, None) # Out of place closing parens
    test_extractor_parse(False, 'foo,', obj, None) # Out of place argument separator
    test_extractor_parse(False, 'foo.', obj, None) # Empty Path Component -- ERROR: THIS ONE IS CURRENTLY SUCCEEDING
    test_extractor_parse(False, 'foo..', obj, None) # Empty Path Component
    