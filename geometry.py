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


class Line3D:
    @classmethod
    def FromPoints(cls, *ps):
        if len(ps) == 2:
            return cls(ps[0], ps[1]-ps[0])
        else:
            assert False, 'Error: Attempting to create a line from {} points'.format(len(ps))

    def __init__(self, origin, direction):
        self.__origin = origin
        self.__direction = direction

    @property
    def origin(self):
        return self.__origin

    @property
    def direction(self):
        return self.__direction


def LinePlaneIntersect3D(line, plane):
    p = line.origin
    v = line.direction
    a, b, c, d = plane.a, plane.b, plane.c, plane.d
    t = -(a*p[0] + b*p[1] + c*p[2] + d) / (a*v[0] + b*v[1] + c*v[2])
    if t is None:
        return None
    return p + t*v


def plotPoint3D(point, ax, **kwargs):
    ax.scatter([point[0]], [point[1]], [point[2]], **kwargs)


def plotVector3D(direction, ax, offset=None, **kwargs):
    offset = np.array([0, 0, 0]) if offset is None else offset
    ax.quiver([offset[0]], [offset[1]], [offset[2]], [direction[0]], [direction[1]], [direction[2]], **kwargs)


def plotPlane3D(plane, xlim, ylim, ax, N=10, **kwargs):
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], N), np.linspace(ylim[0], ylim[1], N))
    Z = (-plane.d - plane.a * X - plane.b * Y) / plane.c
    ax.plot_wireframe(X, Y, Z, **kwargs)


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

    p3 = np.array([1, 1, 0])

    # p0 = np.array([3, 1, 1])
    # p1 = np.array([1, 4, 2])
    # p2 = np.array([1, 3, 4])

    plane = Plane3D.FromPoints(p0, p1, p2)
    plane.origin_xy(0.5, 0.5)
    # plane.origin_xy(2, 2)

    p_end =  np.array([0.33, 0.1, .75])
    line = Line3D.FromPoints(p3, p_end)

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    # ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]], alpha=0.5, color='b')

    p4 = LinePlaneIntersect3D(line, plane)
    plotPlane3D(plane, (0, 1), (0, 1), ax, color='k', lw=0.5)

    # Plot the origin
    print('Plane Origin = {}'.format(plane.origin))
    plotPoint3D(plane.origin, ax, alpha=0.5, color='r')

    # Plot the normal
    print('Plane Normal = {}'.format(plane.normal))
    plotVector3D(plane.normal, ax, offset=plane.origin, alpha=0.5, length=0.5)

    # Plot the line points
    plotPoint3D(p3, ax, alpha=0.5, color='m')
    plotPoint3D(p_end, ax, alpha=0.5, color='m')

    # Plot the line direction
    print('Line Direction = {}'.format(line.direction))
    plotVector3D(line.direction, ax, offset=line.origin, alpha=0.5, length=0.5)

    # Plot the Intersection
    print('Intersection = {}'.format(p4))
    plotPoint3D(p4, ax, alpha=0.5, color='g')


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    print('Intersection = {}'.format(LinePlaneIntersect3D(line, plane)))
    # print(X, Y)
