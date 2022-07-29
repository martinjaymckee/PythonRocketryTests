
import numpy as np

from . import coordinates

class VectorException(Exception):
    def __init__(self, msg='Vector Error'):
        Exception.__init__(self, msg)
        
class PositionException(Exception):
    def __init__(self, msg='Position Error'):
        Exception.__init__(self, msg)


class Vector3D:
    __valid_frames = (
        'ECEF',
        'Body'
    )
    
    def __init__(self, vec, label=None, frame='ECEF', units=''):
        self.__vec = np.array(vec)  # TODO: IMPLEMENT PROPER CONVERSIONS HERE....
        self.__label = label
        if frame not in Vector3D.__valid_frames:
            raise VectorException('Attempting to create vector with invalid frame -> {}'.format(frame))
        self.__frame = frame
        self.__units = units
        
    def __str__(self):
        x, y, z = self.__vec[0], self.__vec[1], self.__vec[2]
        if self.__label is None:
            return 'Vector3D({:0.2f} {units}, {:0.2f} {units}, {:0.2f} {units})'.format(x, y, z, units=self.__units)
        return '{}({:0.2f} {units}, {:0.2f} {units}, {:0.2f} {units})'.format(self.__label, x, y, z, units=self.__units)

    @property
    def ecef(self):
        return self.__vec
    
    @property
    def x(self):
        return self.__vec[0]

    @x.setter
    def x(self, _x):
        self.__vec[0] = _x
        return self.x

    @property
    def y(self):
        return self.__vec[1]

    @y.setter
    def y(self, _y):
        self.__vec[1] = _y
        return self.y

    @property
    def z(self):
        return self.__vec[2]

    @z.setter
    def z(self, _z):
        self.__vec[2] = _z
        return self.z
    
    def copy(self):
        return self.__class__(self.__vec) 
    
    
class OffsetVector3D(Vector3D):
    @classmethod
    def ECEF(cls, vec):
        return cls(vec)
    
    @classmethod
    def LLH(cls, vec):
        assert False, 'Error: currently the OffsetVector3D.LLH() constructor is not implemented'
        # TODO: CONVERT TO ECEF VECTOR
        cls(None)
        
    def __init__(self, vec):
        Vector3D.__init__(self, vec, label='Offset', units='m')
        
    # TODO: ADD GETTERS AND SETTERS FOR LATITUDE, LONGITUDE AND HEIGHT
        
class VelocityVector3D(Vector3D):
    def __init__(self, vec):
        Vector3D.__init__(self, vec, label='Velocity', units='m/s')


class AccelerationVector3D(Vector3D):
    def __init__(self, vec):
        Vector3D.__init__(self, vec, label='Acceleration', units='m/s^2')


class AngularRateVector3D(Vector3D):
    def __init__(self, vec):
        Vector3D.__init__(self, vec, label='AngularRate', units='rad/s')
        
        
class GeographicPosition:
    default_position = np.array([0, 0, 0])
    
    @classmethod
    def LLH(cls, lat, lon, h, ellipsoid = coordinates.WGS84()):
        # TODO: VALIDATE THE INPUTS AND CONVERT TO XYZ COORDINATES
        ecef = coordinates.LLHToECEF((lat, lon, h), ellipsoid=ellipsoid)
        return GeographicPosition(ecef, ellipsoid=ellipsoid)
    
    @classmethod
    def ECEF(cls, x, y, z, ellipsoid = coordinates.WGS84()):
        # TODO: IT WOULD BE GOOD TO VALIDATE THIS HERE....
        return GeographicPosition((x, y, z), ellpisoid=ellipsoid)
    
    def __init__(self, src, ellipsoid = coordinates.WGS84(), fmt='ECEF', iters=10):
        self.__ecef = np.array(GeographicPosition.default_position)
        self.__ellipsoid = ellipsoid
        self.__fmt = fmt
        self.__iters = iters
        if src is not None:
           if isinstance(src, GeographicPosition):
               self.__ecef = src.ecef.copy() 
               self.__ellipsoid = src.ellipsoid
               self.__fmt = src.fmt
           elif isinstance(src, np.ndarray):
               self.__ecef = src.copy()
           elif len(src) == 3:
               self.__ecef = np.array(src)
           else:
               self.__conversion_exception(src)

    def __str__(self):
        def fmtDMS(deg):
            d, m, s = coordinates.DecDegToDMS(deg)
            return "{:d}\u00B0{:d}'{:0.2f}\"".format(d, m, s)            
        if self.__fmt == 'ECEF':
            return 'Location({:0.1f} m, {:0.1f} m, {:0.1f} m)'.format(self.__ecef[0], self.__ecef[1], self.__ecef[2])
        else:
            llh = self.llh
            if self.__fmt == 'LLH_dec':
                return 'Location({:0.5f}, {:0.5f} deg, {:0.1f} m)'.format(llh[0], llh[1], llh[2])
            elif self.__fmt == 'LLH_dms':  
                lat_hemi = 'N' if llh[0] > 0 else 'S'
                lon_hemi = 'E' if llh[1] > 0 else 'W'
                return 'Location({}{}, {}{}, {:0.1f} m)'.format(fmtDMS(abs(llh[0])), lat_hemi, fmtDMS(abs(llh[1])), lon_hemi, llh[2])
        raise PositionException('Format, {}, not implemented!'.format(self.__format))
        
    def __add__(self, v):
        ecef = self.ecef + v.ecef
        return self.__class__(ecef, self.ellipsoid, self.fmt)

    def __radd__(self, v):
        return self.__add__(v)
    
    def __iadd__(self, v):
        self.__ecef += np.array([v.x, v.y, v.z])
        return self
    
    @property
    def ellipsoid(self):
        return self.__ellipsoid
    
    @ellipsoid.setter
    def ellipsoid(self, _ellipsoid):
        # TODO: VALIDATE THE ELLIPSOID
        self.__ellipsoid = _ellipsoid
        return self.__ellipsoid
    
    @property
    def fmt(self):
        return self.__fmt
    
    @fmt.setter
    def fmt(self, _fmt):
        # TODO: VALIDATE THE FORMAT
        self.__fmt = _fmt
        return self.__fmt
    
    @property
    def ecef(self):
        return self.__ecef
    
    @ecef.setter
    def ecef(self, src):
        if isinstance(src, np.ndarray):
            self.__ecef = src.copy()
        elif len(src) == 3:
            self.__ecef = np.array(src)
        else:
            self.__conversion_exception(src)
        return self.ecef
    
    @property
    def x(self):
        return self.__ecef[0]
    
    @x.setter
    def x(self, _x):
        self.__ecef[0] = _x
        return self.x

    @property
    def y(self):
        return self.__ecef[1]
    
    @y.setter
    def y(self, _y):
        self.__ecef[1] = _y
        return self.y

    @property
    def z(self):
        return self.__ecef[2]
    
    @z.setter
    def z(self, _z):
        self.__ecef[2] = _z
        return self.z
    
    @property
    def llh(self):
        return np.array(coordinates.ECEFToLLH(self.ecef, ellipsoid=self.__ellipsoid, iters=self.__iters))
        
    # TODO: IMPLEMENT LLH SETTERS
    def __conversion_exception(self, src):
        raise PositionException('Position conversion error with invalid type {}'.format(type(src)))
        

def heightAboveGeoid(pos):
    llh = pos.llh
    return llh[0]

    
def heightAboveGround(pos, terrain=None):
    llh = pos.llh
    h_terrain = 0 if terrain is None else terrain.h(llh[0], llh[1])
    return heightAboveGeoid(pos) - h_terrain