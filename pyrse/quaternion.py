import math

from pyrse import vector3d

class Quaternion:
    @classmethod
    def Vector3D(self, v):
        return Quaternion(0, v.x, v.y, v.z)

    @classmethod
    def AngleAxis(self, theta, v):
        c_theta, s_theta = math.cos(theta/2), math.sin(theta/2)
        w = c_theta
        x = s_theta * v.x
        y = s_theta * v.y
        z = s_theta * v.z
        return Quaternion(w, x, y, z)
    
    def __init__(self, w=1, x=0, y=0, z=0):
        self.__w = w
        self.__x = x
        self.__y = y
        self.__z = z

    def copy(self):
        return Quaternion(self.__w, self.__x, self.__y, self.__z)

    def __str__(self):
        return 'Quaternion({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(self.__w, self.__x, self.__y, self.__z)

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val
        return self.__w

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, val):
        self.__x = val
        return self.__x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, val):
        self.__y = val
        return self.__y

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, val):
        self.__z = val
        return self.__z

    def __add__(self, other):
        return Quaternion(self.__w + other.w, self.__x + other.x, self.__y + other.y, self.__z + other.z)

    def __sub__(self, other):
        return Quaterion(self.__w - other.w, self.__x - other.x, self.__y - other.y, self.__z - other.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w_new = (self.w*other.w) - (self.x*other.x) - (self.y*other.y) - (self.z*other.z)
            x_new = (self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y)
            y_new = (self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x)
            z_new = (self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w)
            return Quaternion(w_new, x_new, y_new, z_new)
        return Quaternion(other * self.__w, other * self.__x, other * self.__y, other * self.__z)

    def __rmul__(self, scalar):
        return Quaternion(scalar * self.__w, scalar * self.__x, scalar * self.__y, scalar * self.__z)

    def __iadd__(self, other):
        self.__w += other.w
        self.__x += other.x
        self.__y += other.y
        self.__z += other.z
        return self

    def __isub__(self, other):
        self.__w -= other.w
        self.__x -= other.x
        self.__y -= other.y
        self.__z -= other.z
        return self

    def __imul__(self, other):
        if isinstance(other, Quaternion):
            pass
        else:
            self.__w *= other
            self.__x *= other
            self.__y *= other
            self.__z *= other
        return self

    def norm(self):
        return math.sqrt((self.__w ** 2) + (self.__x ** 2) + (self.__y ** 2) + (self.__z ** 2))

    def normalize(self):
        m = self.norm()**2
        self.__w /= m
        self.__x /= m
        self.__y /= m
        self.__z /= m
        return self

    def conjugate(self):
        return Quaternion(self.__w, -self.__x, -self.__y, -self.__z)

    def inverse(self):
        q = self.copy()
        q.normalize()
        return q.conjugate()


def normalized(q):
    q = q.copy()
    q.normalize()
    return q


def rotate_vector(q, v):
    q = normalized(q)
    q_v = Quaternion.Vector3D(v)
    q_r = q * q_v * q.conjugate()
    return vector3d.Vector3D(q_r.x, q_r.y, q_r.z)


def angle_between(q0, q1):
#θ = cos-1 [ (a · b) / (|a| |b|) ]
    v = vector3d.Vector3D(1, 0, 0)
    v0 = rotate_vector(q0, v)
    v1 = rotate_vector(q1, v)
    # print('norm(v0) = {}, norm(v1) = {}'.format(v0.norm(), v1.norm()))
    # print('{}'.format(vector3d.dot(v0, v1) / (v0.magnitude() * v1.magnitude())))
    return math.acos(vector3d.dot(v0, v1) / (v0.magnitude() * v1.magnitude()))    
    # q = q0.conjugate() * q1
    # q.normalize()
    # return 2 * math.atan2(q.norm(), q.w)


if __name__ == '__main__':
    p = Quaternion(3, 1, -2, 1)
    q = Quaternion(2, -1, 2, 3)
    u = vector3d.Vector3D(1, 0, 0)
    v = vector3d.Vector3D(1, 0, 1)
    theta = math.radians(90)

    print('p = {}, q = {}'.format(p, q))
    print('norm(p) = {}\nnorm(q) = {}'.format(p.norm(), q.norm()))
    print('pq = {}'.format(p * q))

    print('normalized(p) = {}\nnormalized(q) = {}'.format(normalized(p), normalized(q)))
    print('norm(normalized(p)) = {}\nnorm(normalized(q)) = {}'.format(normalized(p).norm(), normalized(q).norm()))

    print('inverse(p) = {}\ninverse(q) = {}'.format(p.inverse(), q.inverse()))
    print('p * inverse(p) = {}\nq * inverse(q) = {}'.format(p * p.inverse(), q * q.inverse()))

    q_r = Quaternion.AngleAxis(theta, u)
    print('theta = {}, u = {}, q_r = {}'.format(theta, u, q_r))
    print('w*v*q_inv = {}'.format(rotate_vector(q_r, v)))

    print('angle_between(p, q) = {} deg'.format(math.degrees(angle_between(p, q))))




