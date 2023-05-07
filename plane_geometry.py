

def genPoints(xys, params):
    a, b, c, d = [float(param) for param in params]
    coords = []
    for x, y in xys:
        x = float(x)
        y = float(y)
        z = (d - a*x - b*y) / c
        coords.append((x, y, z))
    return coords


def genIntersectingLinesPoints(origin, ts, vs):
    x0, y0, z0 = [float(c) for c in origin]
    points = []
    for v in vs:
        i, j, k = [float(c) for c in v]
        new_points = []
        for t in ts:
            t = float(t)
            new_points.append((x0 + t*i, y0 + t*j, z0 + t*k))
        points.append(new_points)
    return points


if __name__ == '__main__':
    xys = [(-6, 0), (12, 12), (0, 6), (6, 6)]
    plane_points = genPoints(xys, (-1/2, 3, 1/3, -5))
    print('plane points = {}'.format(plane_points))
    p_inter = plane_points[-1]
    print('intersection point = {}'.format(p_inter))
    ts = (-10, 15)
    vs = ((1, 1, 1), (2, 1/2, 1/4))
    lines_points = genIntersectingLinesPoints(p_inter, ts, vs)
    print('lines points = {}'.format(lines_points))
