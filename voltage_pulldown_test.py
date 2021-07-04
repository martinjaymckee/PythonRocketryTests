
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parseVoltagePulldownData(file, dt=1):
    ts = []
    firing = []
    Vbats = []
    Vmcus = []
    Vpyros = []
    t = 0
    for line in file.readlines():
        params = line.split(',')
        if len(params) == 6:
            ts.append(t)
            t += dt
            firing.append(int(params[2]))
            Vbats.append(float(params[3]))
            Vmcus.append(float(params[4]))
            Vpyros.append(float(params[5]))
    ts = np.array(ts)
    firing = np.array(firing)
    Vbats = np.array(Vbats)
    Vmcus = np.array(Vmcus)
    Vpyros = np.array(Vpyros)
    return ts, firing, Vbats, Vmcus, Vpyros


if __name__ == '__main__':
    file = open('voltage_pulldown_data_1.csv', 'r')
    Vcritical = 3.65
    ts, firing, Vbats, Vmcus, Vpyros = parseVoltagePulldownData(file, dt=1/300)

    fig, ax = plt.subplots(1, figsize=(16, 9), sharex=True)

    sns.lineplot(x=ts, y=Vbats, ax=ax, label='Battery Voltage')
    sns.lineplot(x=ts, y=Vpyros, ax=ax, label='Pyro Voltage')
    sns.lineplot(x=ts, y=Vmcus, ax=ax, label='MCU Voltage')

    ax.axhline(Vcritical, c='r', linestyle='dashed', alpha=0.5)
    cells = math.ceil(np.max(Vbats) / 4.2)
    for idx in range(cells):
        ax.axhline((1+idx) * 4.2, c='k', alpha=0.25)

    t_start = None
    t_end = None
    t_last = 0
    for idx, t in enumerate(ts):
        # print('firing[{}] = {}'.format(idx, firing[idx]))
        if t_start is None:
            if firing[idx] == 1:
                t_start = t_last
        else:
            if firing[idx] == 0:
                t_end = t_last
                break
        t_last = t

    ax.axvspan(t_start, t_end, facecolor='r', alpha=0.1)
    ax.set_ylim(0, 4.2*cells + 0.2)

    ax.annotate(
        'Vbat = {:0.2f}v, Vmcu = {:0.2f}v'.format(Vbats[0], Vmcus[0]),
        xy=(ts[0], Vmcus[0]),
        xytext=(ts[5], Vmcus[0]-0.6),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
    )

    ax.annotate(
        'Vbat = {:0.2f}v, Vmcu = {:0.2f}v'.format(Vbats[-1], Vmcus[-1]),
        xy=(ts[-1], Vmcus[-1]),
        xytext=(ts[-45], Vmcus[-45]-0.6),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
    )

    fig.tight_layout()

    plt.show()
