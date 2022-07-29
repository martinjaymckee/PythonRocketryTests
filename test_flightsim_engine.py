
import flightsim_engine
import flightsim_engine.utils

if __name__ == '__main__':
    pos = flightsim_engine.utils.GeographicPosition.LLH(38.2535, -105.1234, 2300)
    offset = flightsim_engine.utils.OffsetVector3D.ECEF((1000, 1000, 100))
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
        