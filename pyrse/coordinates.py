import math


class Ellipsoid:
    def __init__(self, a, b):
        a, b = float(a), float(b)
        self.__a = a  # m
        self.__a2 = a * a
        self.__b = b  # m
        self.__b2= b * b
        self.__e2 = (self.__a2 - self.__b2) / (self.__a2)
        self.__e2_prime = (self.__a2 - self.__b2) / (self.__b2)

    @property
    def a(self): return self.__a
    
    @property
    def a2(self): return self.__a2

    @property
    def b(self): return self.__b
    
    @property
    def b2(self): return self.__b2

    @property
    def e2(self): return self.__e2

    @property
    def e2_prime(self): return self.__e2_prime

        
GRS80 = Ellipsoid(6378137, 6356752.314140347)
WGS84 = Ellipsoid(6378137, 6356752.314245)
         

def LLHToECEF(llh, ellipsoid=WGS84):
    lat, lon, h = llh
    lat = math.radians(lat)
    lon = math.radians(lon)
    N = ellipsoid.a / math.sqrt(1 - ellipsoid.e2 * (math.sin(lat)**2))
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = ((1 - ellipsoid.e2) * N + h) * math.sin(lat)
    return (x, y, z)


# TODO: ADD IMPLMENTATION OF ENHANCED ZHU'S ALGORITHM FROM https://hal.archives-ouvertes.fr/hal-01704943v2/document
#   Accurate Conversion of Earth-Fixed Earth-Centered Coordinates to Geodetic Coordinates -- Karl Osen
def ECEFToLLH(ecef, ellipsoid=WGS84, algo='Newton-Raphson', iters=5):
    def calcN(lat):
        return ellipsoid.a / math.sqrt(1 - ellipsoid.e2 * (math.sin(lat)**2))

    x, y, z = ecef
    if algo == 'Newton-Raphson':
        lon = math.atan2(y, x)
        p = math.sqrt(x**2 + y**2)
        pole = (abs(x) < 1e-9) and (abs(y) < 1e-9)
        lat = math.radians(90 if z > 0 else -90) if pole else math.atan2(p, z)
        h = (p / math.cos(lat)) - calcN(lat)
        for _ in range(iters):
            N = calcN(lat)
            h = (p / math.cos(lat)) - N
            if not pole:
                lat = math.atan((z / p) * (1 / (1 - ellipsoid.e2 * (N / (N + h)))))
        return (math.degrees(lat), math.degrees(lon), h)
    elif algo == 'Ferrari':    
        # Based on formulation on Wikipedia, variable names taken from there.
        #   Note: Currently, this is not working.  The latitude is approximately ~30m off, but the height is completely wrong.
        p = math.sqrt(x**2 + y**2)
        F = 54 * ellipsoid.b2 * (z**2)
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


def ECEFToENU(p1, p0, **kwargs):
    llh = ECEFToLLH(p0, **kwargs)
    
    s_lat, c_lat = math.sin(math.radians(llh[0])), math.cos(math.radians(llh[0]))
    s_lon, c_lon = math.sin(math.radians(llh[1])), math.cos(math.radians(llh[1]))

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]    
    dz = p1[2] - p0[2]
    
#    print('dx = {}, dy = {}, dz = {}'.format(dx, dy, dz))
    
    x = (-s_lon * dx) + (c_lon * dy)
    y = (-s_lat * c_lon * dx) + (-s_lat * s_lon * dy) + (c_lat * dz)
    z = (c_lat * c_lon * dx) + (c_lat * s_lon * dy) + (s_lat * dz)
    
    return (x, y, z)


def ENUToECEF(enu, p0, **kwargs):
    llh = ECEFToLLH(p0, **kwargs)
    
    s_lat, c_lat = math.sin(math.radians(llh[0])), math.cos(math.radians(llh[0]))
    s_lon, c_lon = math.sin(math.radians(llh[1])), math.cos(math.radians(llh[1]))
   
    x = (-s_lon * enu[0]) + (-s_lat * c_lon * enu[1]) + (c_lat * c_lon * enu[2])
    y = (c_lon * enu[0]) + (-s_lat * s_lon * enu[1]) + (c_lat * s_lon * enu[2])
    z = (c_lat * enu[1]) + (s_lat * enu[2])
    
    return (x + p0[0], y + p0[1], z + p0[2])


def LLHToENU(llh, p0, **kwargs):
    ecef = LLHToECEF(llh, **kwargs)
    return ECEFToLLH(ecef, **kwargs)


def ENUToLLH(enu, p0, **kwargs):
    ecef = ENUToECEF(enu, p0, **kwargs)
    return ECEFToLLH(ecef, **kwargs)


# TODO: IMPLEMENT CONVERSION BETWEEN ECEF AND ENU COORDINATES USING THE FORMULAS
#   PAGE IS Transformations between ECEF and ENU coordinates
#   AVAILABLE AT https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates#:~:text=From%20the%20figure%201%20it,axis%20with%20the%20z%2Daxis.
#   ADDITIONAL INFO AT https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf


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
    ecef_err = ((ecef_tgt[0] - ecef[0]), (ecef_tgt[1] - ecef[1]), (ecef_tgt[2] - ecef[2]))
    llh_convert = ECEFToLLH(ecef)

 #   print(ecef, llh_convert)
    
    llh_err = ((llh[0] - llh_convert[0]), (llh[1] - llh_convert[1]), (llh[2] - llh_convert[2]))    
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
    
    lat = 38.0001
    lon = -104.0001
    h = 1500.5
    llh2 = (lat, lon, h)
    ecef2 = LLHToECEF(llh2)
    enu = ECEFToENU(ecef2, ecef)
    ecef3 = ENUToECEF(enu, ecef)
    print('p0 = {}, p1 = {}'.format(ecef, ecef2))
    print('ENU(p1, p0) = {} -> ECEF = {}'.format(enu, ecef3))

