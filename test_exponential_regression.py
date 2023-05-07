# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:22:40 2022

@author: marti
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


def generateSignal(t_max, y_final, N, e=None):
    e = 0.025
    ts = np.linspace(0, t_max, N)
    y = 0
    ys = []
    for t in ts:
        ys.append(y)
        y = y + (e * (y_final - y))
    ys = np.array(ys)      
    return ts, ys


def generateNoise(sd_t, sd, N):
    dts = np.array([random.gauss(0, sd_t) for _ in range(N)])
    dys = np.array([random.gauss(0, sd) for _ in range(N)])
    return dts, dys


def main():
    dt = 0.5
    t_max = 100
    y_final = 10
    N = int(t_max / dt)
    W = 5
    sd = 0.25
    sd_t = dt / 100
    ts, ys = generateSignal(t_max, y_final, N)
    dtys, dys = generateNoise(sd_t, sd, N)
    
    ys_noisy = ys + dys
    ts_noisy = ts + dtys
    
    t_est = 15.0
    ys_est = []
    for idx in range(N - W):
        ys_windowed = []
        ts_windowed = []
        X = []
        Y = []
        t0 = ts_noisy[idx]
        for offset in range(W):
            y = ys_noisy[idx+offset]
            t = ts_noisy[idx+offset]
            X.append( (t-t0, 1) )
            Y.append([0 if y < 0 else math.log(y)])
        X = np.array(X)
        Y = np.array(Y)
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
        K_est = math.exp(beta[1])
        y_final_est = K_est * math.exp(beta[0] * t_est)
        ys_est.append(y_final_est)        
        
    ys_est = np.array(ys_est)
    ys_est[ys_est > 50] = 50
    plt.plot(ts, ys_noisy)
    plt.plot(ts[:N-W], ys_est)
    plt.show()
    
    return


if __name__ == '__main__':
    main()
