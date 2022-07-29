import math

import numpy as np

import coordinates


def hav(x):
    return (1 - np.cos(x)) / 2


def surfaceDistance(llh_src, llh_dest, method='Euclidian'):
    C_equator = 40.075e6
    d = None
    theta = None

    if method == 'Naive':
        C_src = C_equator + (2*math.pi*llh_src[2])
        deg_per_m_lat = (360 / C_src)
        deg_per_m_lon = deg_per_m_lat / math.cos(math.radians(llh_src[0]))
        m_per_deg = (1.0/deg_per_m_lat, 1.0/deg_per_m_lon)
        d = np.sqrt(
            (m_per_deg[0] * (llh_dest[0] - llh_src[0]))**2
            + (m_per_deg[1] * (llh_dest[1] - llh_src[1]))**2
        )
    elif method == 'Euclidian':
        xyz_src = coordinates.LLHToECEF(llh_src)
        xyz_dest = coordinates.LLHToECEF(llh_dest)
        dx = xyz_dest[0] - xyz_src[0]
        dy = xyz_dest[1] - xyz_src[1]
        dz = xyz_dest[2] - xyz_src[2]
        d = math.sqrt(dx**2 + dy**2 + dz**2)
    elif method == 'Haversine':
        pass
    elif method == 'Vicenty':
        pass
    return d, theta


if __name__ == '__main__':
    pass
