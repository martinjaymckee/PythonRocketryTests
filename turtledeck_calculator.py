import math


def calcCircularTurtledeck(L, r1, r2, theta1=180, theta2=180):
    assert r2 > r1, 'Error: r2 <= r1'
    c1 = (math.pi * r1 * theta1) / 180
    c2 = (math.pi * r2 * theta2) / 180
    alpha = (180 / (math.pi * L)) * (c2 - c1)
    r = ((180 * c2) / (math.pi * alpha)) - L
    R = L + r
    # R = 180 * w2 / alpha
    # r = 180 * w1 / alpha
    Lx = 2 * R * math.sin(math.radians(alpha/2))
    return alpha, R, r, Lx


if __name__ == '__main__':
    L = 292.5         # mm
    r1 = 14.961         # mm
    r2 = 35             # mm

    alpha, R, r, Lx = calcCircularTurtledeck(L, r1, r2)

    print('alpha = {:0.2f} deg, R = {:0.2f} mm, r = {:0.2f} mm, Lx = {:0.2f} mm'.format(alpha, R, r, Lx))
