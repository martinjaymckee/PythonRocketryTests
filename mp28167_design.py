
def calcInductor(Vout, Vin, Imax, fsw=500e3, dI_scalar=0.3):
    dIL = dI_scalar * Imax
    if Vin > Vout:
        return (Vout / (fsw * dIL)) * (1 - (Vout / Vin))
    return (Vin * (Vout - Vin)) / (Vout * fsw * dIL)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    Vin = 5
    Vout = (1, 9)
    Imax = 0.5
    fsw = 750e3

    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    Ls = []
    Vs = []
    for V in np.linspace(*Vout, 100):
        L = calcInductor(V, Vin, Imax, fsw)
        Vs.append(V)
        Ls.append(L)
    Ls = 1e6 * np.array(Ls)
    axs[0].plot(Vs, Ls)

    plt.show()
