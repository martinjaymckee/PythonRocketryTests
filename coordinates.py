import math


class GRS80:
    def __init__(self):
        self.__a = 6378137  # m
        self.__b = 6356752.314140347  # m
        self.__e2 = (self.__a**2 - self.__b**2) / (self.__a**2)
        self.__e2_prime = (self.__a**2 - self.__b**2) / (self.__b**2)

    @property
    def a(self): return self.__a

    @property
    def b(self): return self.__b

    @property
    def e2(self): return self.__e2

    @property
    def e2_prime(self): return self.__e2_prime


class WGS84:
    def __init__(self):
        self.__a = 6378137  # m
        self.__b = 6356752.314245  # m
        self.__e2 = (self.__a**2 - self.__b**2) / (self.__a**2)
        self.__e2_prime = (self.__a**2 - self.__b**2) / (self.__b**2)

    @property
    def a(self): return self.__a

    @property
    def b(self): return self.__b

    @property
    def e2(self): return self.__e2

    @property
    def e2_prime(self): return self.__e2_prime


def LLHToECEF(llh, ellipsoid=WGS84()):
    lat, lon, h = llh
    lat = math.radians(lat)
    lon = math.radians(lon)
    N = ellipsoid.a / math.sqrt(1 - ellipsoid.e2 * (math.sin(lat)**2))
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = ((1 - ellipsoid.e2) * N + h) * math.sin(lat)
    return (x, y, z)


def ECEFToLLH(ecef, ellipsoid=WGS84(), algo='Newton-Raphson', iters=4):
    def calcN(lat):
        return ellipsoid.a / math.sqrt(1 - ellipsoid.e2 * (math.sin(lat)**2))

    x, y, z = ecef
    if algo == 'Newton-Raphson':
        lon = math.atan2(y, x)
        p = math.sqrt(x**2 + y**2)
        lat = math.atan2(p, z)
        h = (p / math.cos(lat)) - calcN(lat)
        for _ in range(iters):
            N = calcN(lat)
            h = (p / math.cos(lat)) - N
            lat = math.atan((z / p) * (1 / (1 - ellipsoid.e2 * (N / (N + h)))))
        return (math.degrees(lat), math.degrees(lon), h)
    return (0, 0, 0)


def DMSToDecDeg(deg, min=0, sec=0):
    return deg + (min / 60) + (sec / 3600)


if __name__ == '__main__':
    import numpy as np
    deg_to_m = 40.075e6 / 360

    lat = 38
    lon = -104
    h = 1500
    llh = (lat, lon, h)
    # llh = (34.12771308, -117.82545897, 250.640)
    ecef = LLHToECEF(llh)
    # ecef_tgt = (-2467178.313, -4674384.341, 3558322.813)
    # ecef_tgt = (-1217.741e3, -4884.092e3, 3906.367e3)
    ecef_tgt = (-1217740.797, -4884091.569, 3906367.461)
    ecef_err = np.array(ecef_tgt) - np.array(ecef)
    llh_convert = ECEFToLLH(ecef)
    llh_err = np.array(llh) - np.array(llh_convert)
    print('Using WGS84, LLH({} deg, {} deg, {} m) ->\tECEF({} m, {} m, {} m)'.format(*llh, *ecef))
    print("\tECEF Error -> ({} m, {} m, {} m)".format(*ecef_err))
    print()
    print('LLH = ({} deg, {} deg, {} m)'.format(*llh_convert))
    print("\tLLH Error -> ({} m, {} m, {} m)".format(deg_to_m*llh_err[0], deg_to_m*llh_err[1], llh_err[2]))
    print()
    print()
