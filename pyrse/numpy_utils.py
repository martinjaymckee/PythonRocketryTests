import math
import numpy as np


def magnitude(v):
    return math.sqrt(np.sum(v**2))


def normalized(v, order=2):
    denom = pow(np.sum(v**order), 1/order)
    if denom == 0:
        return np.zeros(v.shape)
    return v / denom


def signs(v):
    return np.array([1 if x >= 0 else -1 for x in v])

# def normalized(a, axis=-1, order=2):
#     l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
#     l2[l2==0] = 1
#     return a / np.expand_dims(l2, axis)