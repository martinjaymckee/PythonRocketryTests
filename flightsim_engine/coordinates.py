import math

import numpy as np


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

# TODO: CONVERT TO WORKING WITH NP ARRAYS
def LLHToECEF(llh, ellipsoid=WGS84()):
    lat, lon, h = llh
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = ellipsoid.a / np.sqrt(1 - ellipsoid.e2 * (np.sin(lat)**2))
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = ((1 - ellipsoid.e2) * N + h) * math.sin(lat)
    return (x, y, z)


# TODO: CONVERT TO WORKING WITH ARRAYS...
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
    elif algo == 'Ferrari':
        # Based on formulation on Wikipedia, variable names taken from there.
        #   Note: Currently, this is not working.  The latitude is approximately ~30m off, but the height is completely wrong.
        p = math.sqrt(x**2 + y**2)
        F = 54 * ellipsoid.b**2 * z**2
        G = p**2 + (1-ellipsoid.e2)*z**2 - ellipsoid.e2*(ellipsoid.a**2 - ellipsoid.b**2)
        c = (ellipsoid.e2**2 * F * p**2) / (G**3)
        s = pow((1 + c + math.sqrt(c**2 + 2*c)), 1/3)
        k = s + 1 + 1/s
        P = F / (3 * k**2 * G**2)
        Q = math.sqrt(1 + (2 * ellipsoid.e2**2 * P))
        # print('Q = {}'.format(Q))
        r0 = ((-P*ellipsoid.e2*p)/(1+Q)) + math.sqrt(0.5*ellipsoid.a**2*(1 + 1/Q) - ((P*(1-ellipsoid.e2)*z**2)/(Q*(1+Q)) - (0.5*P*p**2)))
        # print('r0 = {}'.format(r0))
        U = math.sqrt((p - ellipsoid.e2*r0)**2 + z**2)
        V = math.sqrt((p - ellipsoid.e2*r0)**2 + (1-ellipsoid.e2)*z**2)
        z0 = (ellipsoid.b**2 * z) / (ellipsoid.a * V)
        h = U * (1 - (ellipsoid.b**2 / (ellipsoid.a * V)))
        # print('a = {}, b = {}, U = {}, V = {}'.format(ellipsoid.a, ellipsoid.b, U, V))
        # print('b^2 / aV = {}'.format((ellipsoid.b**2 / (ellipsoid.a * V))))
        lat = math.atan2((z + ellipsoid.e2_prime*z0), p)
        lon = math.atan2(y, x)
        return (math.degrees(lat), math.degrees(lon), h)
    return (0, 0, 0)


def DMSToDecDeg(deg, minutes=0, seconds=0):
    return deg + (minutes / 60.0) + (seconds / 3600.0)

def DecDegToDMS(deg):
    whole_deg = int(deg)
    frac = deg - whole_deg
    minutes = int(60.0 * frac)
    frac = (60.0 * frac) - minutes
    seconds = 60.0 * frac
    return whole_deg, minutes, seconds


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
    d, m, s = 45, 10, 5
    deg = DMSToDecDeg(d, m, s)
    print(deg)
    d1, m1, s1 = DecDegToDMS(deg)
    print("{}deg {}' {}\"".format(d1, m1, s1))
