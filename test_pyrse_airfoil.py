import math

import matplotlib.pyplot as plt
import numpy as np

import pyrse.airfoil as airfoil


if __name__ == '__main__':
    t = 0.125
    c = 2.5
    max_alpha = 100
    flatplate_airfoil = airfoil.FlatPlateAirfoil(t/c)

    N_alpha = 100
    N_re = 5

    alphas = np.linspace(-math.radians(max_alpha), math.radians(max_alpha), N_alpha)
    Res = np.linspace(1e4, 1e6, N_re)

    fig, axs = plt.subplots(2, layout='constrained', sharex=True)
    for Re in Res:
        Cls = [flatplate_airfoil.Cl(alpha, Re) for alpha in alphas]
        Cds = [flatplate_airfoil.Cd(alpha, Re) for alpha in alphas]

        axs[0].plot(alphas, Cls, alpha=0.5, label='Re = {}'.format(Re))
        axs[1].plot(alphas, Cds, alpha=0.5, label='Re = {}'.format(Re))

    for ax in axs:
        ax.legend()

    plt.show()