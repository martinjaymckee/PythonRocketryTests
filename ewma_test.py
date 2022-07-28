import math

import matplotlib.pyplot as plt
import numpy as np


def calcAlpha(fs, tau):
    return math.exp(-1 / (tau * fs))


if __name__ == '__main__':
    t_min = 0
    t_max = 30
    tau_min = 0.1
    tau_max = 5.0
    fs = 2000
    ts = np.linspace(t_min, t_max, int(fs * (t_max - t_min)))
    taus = np.linspace(tau_min, tau_max, 10)
    zs = np.zeros(len(ts))
    step_offset = int(len(ts)/5)
    # t_step_offset = step_offset / fs
    zs[step_offset:] = 1.0

    fig, ax = plt.subplots(1, constrained_layout=True)
    ax.plot(ts, zs, c='k', alpha=0.5)

    for tau in taus:
        ys = []
        y = 0
        alpha = calcAlpha(fs, tau)
        for z in zs:
            y = (alpha * y) + ((1 - alpha) * z)
            ys.append(y)
        g = ax.plot(ts, ys, alpha=0.75, label='tau = {:0.2f} (alpha = {:0.6G})'.format(tau, alpha))
        # ax.axvline(tau + t_step_offset, c=g[-1].get_color(), alpha=0.1)
    # ax.axhline(-math.log(0.5))
    ax.legend()
    plt.show()
