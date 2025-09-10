import math

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.__x = x
        self.__y = y
        self.__z = z

    def __str__(self):
        return '[{}, {}, {}]'.format(self.__x, self.__y, self.__z)

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
        return Vector3D(self.__x + other.x, self.__y + other.y, self.__z + other.z)

    def __sub__(self, other):
        return Vector3D(self.__x - other.x, self.__y - other.y, self.__z - other.z)
        
    def __mul__(self, scalar):
        return Vector3D(scalar * self.__x, scalar * self.__y, scalar * self.__z)

    def __rmul__(self, scalar):
        return Vector3D(scalar * self.__x, scalar * self.__y, scalar * self.__z)

    def __iadd__(self, other):
        self.__x += other.x
        self.__y += other.y
        self.__z += other.z
        return self

    def __isub__(self, other):
        self.__x -= other.x
        self.__y -= other.y
        self.__z -= other.z
        return self

    def __imul__(self, other):
        self.__x *= other
        self.__y *= other
        self.__z *= other
        return self

    def norm(self):
        return (self.__x ** 2) + (self.__y ** 2) + (self.__z ** 2)

    def magnitude(self):
        return math.sqrt(self.norm())

        
def dot(v0, v1):
    return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z)

def cross(v0, v1):
#    print('v0 = {}, v1 = {}'.format(v0, v1))
    return Vector3D(v0.y*v1.z-v0.z*v1.y, v0.z*v1.x-v0.x*v1.z, v0.x*v1.y-v0.y*v1.x)


if __name__ == '__main__':
    A = Vector3D(1, 2, 3)
    B = Vector3D(3, 1, 2)

    print('A = {}, B = {}'.format(A, B))
    print('norm(A) = {}, norm(B) = {}'.format(A.norm(), B.norm()))    
    print('magnitude(A) = {}, magnitude(B) = {}'.format(A.magnitude(), B.magnitude()))

    ApB = A+B
    AmB = A-B
    three_A = 3 * A
    B_four = B * 4
    print('A + B = {}'.format(ApB))
    print('A - B = {}'.format(AmB))
    print('3 * A = {}'.format(three_A))
    print('B * 4 = {}'.format(B_four))    

    print('A dot B = {}'.format(dot(A, B)))
    print('A x B = {}'.format(cross(A, B)))


