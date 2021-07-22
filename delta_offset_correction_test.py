import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as linalg
import seaborn as sns


def createDeltaCorrection5Samples(odr=5, V=25e-3, dV=5e-5, Voff_0=1e-3, M_off=-2.5e-6):
    dt = 1 / odr
    ts = dt * np.array([0, 1, 2, 3, 4])
    Vadcs = np.zeros(5)
    Vadcs[0] = random.gauss(V, dV) + (0*M_off*dt + Voff_0)
    Vadcs[1] = -random.gauss(V, dV) + (1*M_off*dt + Voff_0)
    Vadcs[2] = random.gauss(V, dV) + (2*M_off*dt + Voff_0)
    Vadcs[3] = -random.gauss(V, dV) + (3*M_off*dt + Voff_0)
    Vadcs[4] = random.gauss(V, dV) + (4*M_off*dt + Voff_0)
    return ts, Vadcs


def fitDeltaCorrection5(ts, Vadcs):
    V = 0
    Voff_0 = 0
    M_off = 0
    A = np.array([
        [1, ts[0], 1],
        [-1, ts[1], 1],
        [1, ts[2], 1],
        [-1, ts[3], 1],
        [1, ts[4], 1]
    ])

    Vadcs = Vadcs.reshape((5,1))
    a = np.matmul(A.T, A)
    b = np.matmul(A.T, Vadcs)

    X = linalg.solve(a, b)

    return X[0, 0], X[1, 0], X[2, 0]


if __name__ == '__main__':
    do_plot = False
    N = 25
    odr = 5
    V = 25e-3
    dV = 5e-5
    Voff_0 = 1e-3
    M_off = -2.5e-4
    fig, ax = plt.subplots(1, figsize=(16, 9))
    err_max = 0
    Vs = []
    for _ in range(N):
        ts, Vadcs = createDeltaCorrection5Samples(odr, V, dV, Voff_0, M_off)
        if do_plot:
            sns.regplot(x=ts, y=Vadcs, ax=ax)
        results = fitDeltaCorrection5(ts, Vadcs)
        # print('V = {:0.3f} mv, M = {:0.1f} uv/s, b = {:0.1f} uv'.format(1e3 * results[0], 1e6 * results[1], 1e6 * results[2]))
        err = 100 * (V - results[0]) / V
        Vs.append(results[0])
        if abs(err) > err_max:
            err_max = err
        # print('\terror(V) = {:0.3f} %'.format(err))
    if do_plot:
        plt.show()
    print('Maximum Error = {:0.3f} %'.format(err_max))
    Vs = np.array(Vs)
    print('Reading Mean = {:0.4f} mv, S.D. = {:0.1f} uv'.format(1e3 * np.mean(Vs), 1e6 * np.std(Vs)))
    print('Reading Noise = {:0.1f} uV'.format(1e6 * dV))
