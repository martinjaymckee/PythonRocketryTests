import math
import numpy as np

def normalized(v, order=2):
    return v / pow(np.sum(v**order), 1/order)

def magnitude(v):
    return math.sqrt(np.sum(v**2))


# def normalized(a, axis=-1, order=2):
#     l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
#     l2[l2==0] = 1
#     return a / np.expand_dims(l2, axis)