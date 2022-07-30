import ambiance
import numpy as np

import pyrse
import pyrse.engines
import pyrse.simulator
import pyrse.utils

if __name__ == '__main__':
    #
    # Testing Utils
    #
    pos = pyrse.utils.GeographicPosition.LLH(38.2535, -105.1234, 2300)
    offset = pyrse.utils.OffsetVector3D.ECEF((1000, 1000, 100))
    pos2 = pos + offset
    print(offset)
    print(pos)
    print(pos2)
    print()
    pos.fmt = 'LLH_dec'
    pos2.fmt = 'LLH_dec'    
    print(pos)    
    print(pos2)
    print()
    pos.fmt = 'LLH_dms'
    pos2.fmt = 'LLH_dms'
    print(pos)
    print(pos2)
    print()
    
    #
    # Testing Engines
    #
    engine_directory = pyrse.engines.EngineDirectory('./Engines')
    print('Number of engines found - {}'.format(len(engine_directory)))
    print('Number of files loaded - {}'.format(engine_directory.num_files))    

    for key in engine_directory.directory:
        print(key)

    engs = engine_directory.load('Estes', 'A10T')
    for eng in engs:
        print(eng)
    
    print()
    
    #
    # Testing Simulator
    #
    # pad_pos = pyrse.utils.GeographicPosition.LLH(90, 0, 0) # North Pole
    pad_pos = pyrse.utils.GeographicPosition.LLH(38, -105, 2300)
    env = pyrse.environment.Environment(pad_pos)
    pad = pyrse.pad.LaunchPad(pad_pos)
    model = None
    sim = pyrse.simulator.Simulation1D(env, pad, model)
    print('Pad Surface Normal = {}'.format(sim.surface_normal))
    g0 = env.g()
    print('g0 (ECEF) = {}'.format(g0))
    g0 = env.g(frame=None)    
    print('g0 (mag) = {}'.format(g0))
    atmos = ambiance.Atmosphere(2300)
    g0 = atmos.grav_accel
    print('g0 (mag - ambiance) = {}'.format(g0))
    h = env.h
    print('h = {}m'.format(h))
    print('density = {} kg/m^3'.format(env.density))
    print('kinematic viscosity = {} m^2 / s'.format(env.kinematic_viscosity))
    print(env.sample())
    # print('eps = {}'.format(np.finfo(type(1.0)).eps))    
    
    