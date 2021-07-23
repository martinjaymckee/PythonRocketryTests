import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parseVoltagePulldownData(file, dt=1):
    firing = []
    Vbats = []
    Vmcus = []
    Vpyros = []
    vars = {}
    for line in file.readlines():
        line = line.strip()
        if len(line) > 0:
            if line[0] == '#':
                try:
                    key, _, value = line[1:].partition('=')
                    vars[key.strip()] = value.strip()
                except Exception as e:
                    print(e)
            else:
                params = line.split(',')
                if len(params) == 6:
                    firing.append(int(params[2]))
                    Vbats.append(float(params[3]))
                    Vmcus.append(float(params[4]))
                    Vpyros.append(float(params[5]))
    firing = np.array(firing)
    Vbats = np.array(Vbats)
    Vmcus = np.array(Vmcus)
    Vpyros = np.array(Vpyros)

    dt = float(vars['dt']) if 'dt' in vars else dt
    ts = np.array([idx * dt for idx in range(len(firing))])
    start_idx = np.argmax(firing)
    ts -= ts[start_idx]
    return ts, firing, Vbats, Vmcus, Vpyros, vars


def plotVoltagePulldownData(ts, firing, Vbats, Vmcus, Vpyros, vars, Vcritical=3.6, ax=None):
    ax.set_title(vars['title'] if 'title' in vars else vars['filename'])
    sns.lineplot(x=ts, y=Vbats, ax=ax, label='Vbat')
    sns.lineplot(x=ts, y=Vpyros, ax=ax, label='Vpyro')
    sns.lineplot(x=ts, y=Vmcus, ax=ax, label='Vmcu')

    ax.axhline(Vcritical, c='r', linestyle='dashed', alpha=0.5)
    cells = math.ceil(np.max(Vbats) / 4.2)
    for idx in range(cells):
        ax.axhline((1+idx) * 4.2, c='k', alpha=0.25)

    t_start = None
    t_end = None
    t_last = 0
    for idx, t in enumerate(ts):
        if t_start is None:
            if firing[idx] == 1:
                t_start = t_last
        else:
            if firing[idx] == 0:
                t_end = t_last
                break
        t_last = t

    ax.axvspan(t_start, t_end, facecolor='crimson', alpha=0.05)
    Vmin, Vmax = Vcritical - 0.2, 4.2*cells + 0.2
    ax.set_ylim(Vmin, Vmax)

    ax.annotate(
        'Vbat = {:0.2f}v\nVmcu = {:0.2f}v'.format(Vbats[0], Vmcus[0]),
        xy=(ts[0], Vmcus[0]),
        xytext=(ts[5], Vmcus[0]-1.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
    )

    ax.annotate(
        'Vbat = {:0.2f}v\nVmcu = {:0.2f}v'.format(Vbats[-1], Vmcus[-1]),
        xy=(ts[-1], Vmcus[-1]),
        xytext=(ts[-40], Vmcus[-40]-1.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
    )

    idx_min = np.argmin(Vmcus)
    ax.annotate(
        'Vmcu = {:0.2f}v'.format(Vmcus[idx_min]),
        xy=(ts[idx_min], Vmcus[idx_min]),
        xytext=(ts[idx_min + 5], Vmcus[idx_min]-0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
    )

    if 'description' in vars:
        V = Vmin + 0.33 * (Vmax - Vmin)
        ax.annotate(vars['description'], xy=(ts[int(len(ts)/20)], V), c='b')


def plotVoltagePulldownTests(filenames, dt=1/300, Vcritical=3.65):
    if (filenames is None) or (len(filenames) < 1):
        return
    fig, axs = plt.subplots(len(filenames), figsize=(16, 9), sharex=True)
    if len(filenames) == 1:
        axs = [axs]
    for idx, filename in enumerate(filenames):
        with open(filename, 'r') as file:
            values = parseVoltagePulldownData(file, dt=dt)
            values[-1]['filename'] = filename
            plotVoltagePulldownData(*values, Vcritical=Vcritical, ax=axs[idx])
    fig.tight_layout()
    mgr = fig.canvas.manager
    mgr.window.showMaximized()


def estimateSourceResistance(Vpyros, Rl):
    # print('V[0] = {}, V[-1] = {}'.format(Vpyros[0], Vpyros[-1]))
    Vin = Vpyros[-1]  # Use the recovered battery voltage as the reference
    idx = np.argmin(Vpyros)
    Vsense = Vpyros[idx]
    Rs = Rl * ((Vin - Vsense) / Vsense)
    Il = Vsense / Rl
    return Rs, Il


if __name__ == '__main__':
    filenames = [
        'voltage_pulldown_data_1.csv',
        'voltage_pulldown_data_2.csv',
        'voltage_pulldown_data_3.csv'
    ]
    # filenames = [
    #     'voltage_pulldown_data_1.csv',
    #     'voltage_pulldown_data_2.csv',
    #     'voltage_pulldown_data_3.csv',
    #     'voltage_pulldown_data_4.csv',
    #     'voltage_pulldown_data_5.csv',
    #     'voltage_pulldown_data_6.csv'
    # ]
    # filenames = [
    #     'voltage_pulldown_data_4.csv',
    #     'voltage_pulldown_data_5.csv',
    #     'voltage_pulldown_data_6.csv'
    # ]
    filenames = [
        'voltage_pulldown_data_7.csv'
    ]
    plotVoltagePulldownTests(filenames)
    plt.show()

    filenames = [
        ('voltage_pulldown_data_1.csv', 1),
        ('voltage_pulldown_data_2.csv', 0.5),
        ('voltage_pulldown_data_7.csv', 0.5)
    ]

    for (filename, Rl) in filenames:
        print('filename = {}, R1 = {}'.format(filename, Rl))
        with open(filename, 'r') as file:
            ts, firing, Vbats, Vmcus, Vpyros, vars = parseVoltagePulldownData(file)
            Rs, Il = estimateSourceResistance(Vpyros, Rl)
            print('Rs = {:0.4f} ohm, I = {:0.2f} A'.format(Rs, Il))
