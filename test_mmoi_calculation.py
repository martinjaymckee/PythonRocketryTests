import numpy as np

import pyrse.rocket_components as rc

if __name__ == '__main__':
    body = rc.Component(mass=1)
    m1 = rc.Component(pos=np.array([0, 0.5, 1.5]), mass=1)
    m2 = rc.Component(pos=np.array([0, 0.5, -1.5]), mass=1)
    m3 = rc.Component(pos=np.array([0, -0.5, 0]), mass=2)
    m2.add(m3)

    body.add(m1)
    body.add(m2)
    #  CG at (0, 0, 0)
    print('MMOI = {}'.format(body.mmoi(0)))
    print('cg = {}'.format(body.cg(0)))
    print('mass = {}'.format(body.mass(0)))
