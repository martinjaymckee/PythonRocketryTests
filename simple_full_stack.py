import copy

import numpy as np

import pyrse.engines as engines
import pyrse.rocket_components as rocket_components


def getFullStackShuttleModel():
    cardboard = rocket_components.Material('Cardboard', 1.0)  # TODO: SET THE CORRECT MATERIAL
    foam = rocket_components.Material('Readiboard', 1.0)  # TODO: SET THE CORRECT MATERIAL

    engs = engines.EngineDirectory('./Engines')

    orbiter_engine = engs.load_first('Aerotech', 'D2.3')
    srb_right_engine = engs.load_first('Quest', 'C6-0')
    srb_left_engine = srb_right_engine.duplicate()

    orbiter = rocket_components.EmptyComponent()
    external_tank = rocket_components.EmptyComponent()
    srb_left = rocket_components.EmptyComponent()
    srb_right = rocket_components.EmptyComponent()

    # external_tank = rocket_components.AeroBody(.75, .147, .3, .295, pos=np.array([-.375, 0, 0]))
    # srb_left = rocket_components.AeroBody(.726, .066, .15, .295, pos=np.array([-.539, -0.132, 0]))
    # srb_right = rocket_components.AeroBody(.726, .066, .15, .295, pos=np.array([-.539, 0.132, 0]))
    # orbiter = rocket_components.AeroBody(.594, .099, .35, .31, pos=np.array([-.572, 0, .1375]))
    #
    # orbiter_engine.pos = np.array([-.242, 0, 0])
    #
    # srb_right_engine.pos = np.array([-.35, 0, 0])
    # srb_left_engine.pos = np.array([-.35, 0, 0])

    orbiter.add(orbiter_engine)
    srb_left.add(srb_left_engine)
    srb_right.add(srb_right_engine)

    full_stack = rocket_components.EmptyComponent(children=[external_tank, srb_left, srb_right, orbiter])
    sustainer = rocket_components.EmptyComponent(children=[external_tank, orbiter])

    configurations = [
        (srb_left_engine.burn_time+0.25, full_stack),
        (10.0, sustainer)
    ]

    return configurations


if __name__ == '__main__':
    configurations = getFullStackShuttleModel()

    for t_end, config in configurations:
        print('End Time = {:0.1f} s, Configuration ID = {}'.format(t_end, config.id))
        print('\tMass = {:0.1f} g'.format(1000*config.mass(0)))
