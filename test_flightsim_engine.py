
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
    env = pyrse.environment.Environment()
    pad = pyrse.pad.LaunchPad(pad_pos)
    model = None    
    sim = pyrse.simulator.Simulator1D(env, pad, model)
    print('Pad Surface Normal = {}'.format(sim.surface_normal))
    g0 = env.g(pad_pos)
    print('g0 (ECEF) = {}'.format(g0))
    g0 = env.g(pad_pos, frame=None)    
    print('g0 (mag) = {}'.format(g0))    
    
    