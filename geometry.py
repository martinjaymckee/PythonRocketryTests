import math

import numpy as np


class Plane3D:
    @classmethod
    def FromPoints(cls, *ps):
        if len(ps) == 3:
            p0, p1, p2 = ps
            LA = p1-p0
            LB = p2-p0
            N = np.cross(LA, LB)
            d = -p0.dot(N)
            # print('LA = {}, LB = {}, N = {}'.format(LA, LB, N))
            print('LA.dot(N) = {}, LB.dot(N) = {}'.format(LA.dot(N), LB.dot(N)))
            return cls(N[0], N[1], N[2], d, origin=p0, normal=N)
        else:
            assert False, 'Error: Attempting to create a plane from {} points'.format(len(ps))

    def __init__(self, a, b, c, d, origin=None, normal=None):
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__origin = np.array([0, 0, -d / c]) if origin is None else origin
        self.__N = np.array([a, b, c]) if normal is None else normal

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def d(self):
        return self.__d

    @property
    def origin(self):
        return self.__origin

    @property
    def normal(self):
        return self.__N

    def origin_xy(self, x, y):
        self.__origin = np.array([x, y, (-self.d - self.a * x - self.b * y) / self.c])
        return self.origin


if __name__ == '__main__':
    import mpl_toolkits.mplot3d
    import matplotlib.pyplot as plt

    N = 15
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([1, 1, 1])
    # p2 = np.array([1, 0.5, 0])

    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 1, 1])
    p2 = np.array([1, 0, 1])

    # p0 = np.array([3, 1, 1])
    # p1 = np.array([1, 4, 2])
    # p2 = np.array([1, 3, 4])

    plane = Plane3D.FromPoints(p0, p1, p2)
    plane.origin_xy(0.5, 0.5)
    # plane.origin_xy(2, 2)

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]], alpha=0.5, color='b')
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    # X, Y = np.meshgrid(np.linspace(1, 4, N), np.linspace(1, 4, N))

    Z = (-plane.d - plane.a * X - plane.b * Y) / plane.c

    # Plot the plane
    ax.plot_wireframe(X, Y, Z, color='k', lw=0.5)

    # Plot the origin
    print('Origin = {}'.format(plane.origin))
    ax.scatter([plane.origin[0]], [plane.origin[1]], [plane.origin[2]], alpha=0.5, color='r')

    # Plot the normal
    print('Normal = {}'.format(plane.normal))
    ax.quiver([plane.origin[0]], [plane.origin[1]], [plane.origin[2]], [plane.normal[0]], [plane.normal[1]], [plane.normal[2]], alpha=0.5, length=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    # print(X, Y)
