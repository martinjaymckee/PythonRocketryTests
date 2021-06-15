import matplotlib.pyplot as plt

if __name__ == '__main__':
    dt = 0.01
    t = 0
    h = 500
    V = 0
    V_terminal = -25
    g = -9.80655
    C = g / (V_terminal**2)

    ts = [t]
    hs = [h]
    vs = [V]
    dvs = [g]
    while h > 0:
        t += dt
        h = h + (dt * V)
        dV = (g - (C*V**2))
        V = V + (dt * dV)
        ts.append(t)
        hs.append(h)
        dvs.append(dV)
        vs.append(V)

    fig, axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    axs[0].plot(ts, hs)
    axs[1].plot(ts, vs)
    axs[2].plot(ts, dvs)

    fig.tight_layout()
    plt.show()
