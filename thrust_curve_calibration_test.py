import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import seaborn as sns


def loadCalibrationData(filename, directory=None):
    directory = 'data' if directory is None else directory
    path = os.path.join(directory, filename)

    with open(path, 'r') as file:
        Ws = []
        Ls = []
        Ts = []

        for line in file.readlines():
            line = line.strip()
            if len(line) > 0:
                if line.startswith('#'):
                    pass # TODO: PARSE THE PROPERTIES
                else:
                    tokens = [float(t.strip()) for t in line.split(',')]
                    if len(tokens) == 3:
                        Ws.append(tokens[0])
                        Ls.append(tokens[1])
                        Ts.append(tokens[2])
                    else:
                        Ws.append(tokens[0])
                        Ls.append(tokens[1])
                        Ts.append(tokens[3])
        Ws = np.array(Ws)
        Ls = np.array(Ls)
        Ts = np.array(Ts)
        return Ws, Ls, Ts
    return None, None, None


def splitByWeight(Ws, Ls, Ts):
    data = {}

    for W, L, T in zip(Ws, Ls, Ts):
        if W in data:
            data[W][0].append(L)
            data[W][1].append(T)
        else:
            data[W] = [[L], [T]]
    return data


def plotADCvsTemp(Ws, Ls, Ts, ax=None, title=None):
    data = splitByWeight(Ws, Ls, Ts)
    if ax is None:
        _, ax = plt.subplots(1, constrained_layout=True)

    for W, (Ls, Ts) in data.items():
        Ls = np.array(Ls)
        dLs = (Ls - Ls.mean()) / Ls.std()
        sns.regplot(x=Ts, y=dLs, ax=ax, label='{} g'.format(1000 * W))
    ax.legend()
    if title is not None:
        ax.set_title(title)


def ols(X, b):
    X_T = X.transpose()
    y = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T), b)
    return y


def fitADCCorrector(Ws, Ls, Ts):
    X = []
    b = []
    for W, L, T in zip(Ws, Ls, Ts):
        X.append([L*T, L, T, 1])
        b.append([W])
    X = np.array(X)
    b = np.array(b)
    y = ols(X, b)
    print('First Weight Approximator: y = {}'.format(y))
    return lambda L, T: float(y[0]*L*T + y[1]*L + y[2]*T + y[3])


def fitPhysicalNonlinearityCorrector(Ws, Lcs):
    X = []
    b = []
    for W, Lc in zip(Ws, Lcs):
        X.append([Lc*Lc*Lc, Lc*Lc, Lc, 1])
        b.append([W])
    X = np.array(X)
    b = np.array(b)
    y = ols(X, b)
    print('Nonlinearity Weight Approximator: y = {}'.format(y))
    return lambda L: float(y[0]*L*L*L + y[1]*L*L + y[2]*L + y[3])


def manualCorrector(L, T):
    A = 1.159073e-13
    B = -8.381903e-09
    C = -4.567206e-06
    D = -0.003112793
    E = -0.3710938
    return A*L*L + B*L*T + C*L + D*T + E


def weightError(Ws, Wcs):
    Ws = np.array(Ws)
    Wcs = np.array(Wcs)
    return np.std(Ws - Wcs)


if __name__ == '__main__':
    # Ws, Ls, Ts = loadCalibrationData('nonlinearity_backup.csv')
    Ws, Ls, Ts = loadCalibrationData('nonlinearity.csv')

    # fig, axs = plt.subplots(3, constrained_layout=True)
    # sns.histplot(Ts, ax=axs[0])
    # sns.jointplot(x=Ts, y=Ls, ax=axs[1])
    #
    # sns.scatterplot(x=Ws, y=Ls, hue=Ts, cmap='viridis', ax=axs[2])

    fig, axs = plt.subplots(3, sharex=True, sharey=True, constrained_layout=True)
    plotADCvsTemp(Ws, Ls, Ts, ax=axs[0], title='Uncorrected ADC Readings')

    corrector = fitADCCorrector(Ws, Ls, Ts)  # Note: this should only use the bottom N weights to fit (and therefore reduce nonlinearity)
    # corrector = manualCorrector
    Lcs = [corrector(L, T) for L, T in zip(Ls, Ts)]
    plotADCvsTemp(Ws, Lcs, Ts, ax=axs[1], title='Temperature Corrected ADC Readings')
    print('Temperature Corrected Weight S.D. = {}'.format(weightError(Ws, Lcs)))

    nonlinearCorrector = fitPhysicalNonlinearityCorrector(Ws, Lcs)
    Wcs = [nonlinearCorrector(L) for L in Lcs]

    plotADCvsTemp(Ws, Wcs, Ts, ax=axs[2], title='Fully Corrected Weights Readings')
    print('Nonlinearity Corrected Weight S.D. = {}'.format(weightError(Ws, Wcs)))

    axs[1].set_xlabel('Temperature (C)')
    axs[1].set_ylim(-1.5, 1.5)

    fig, ax = plt.subplots(1, constrained_layout=True)
    ax.axhline(0, c='k')
    sns.regplot(x=Ws, y=1000 * (Ws - Lcs), ax=ax, label='Temperature Corrected')
    sns.regplot(x=Ws, y=1000 * (Ws - Wcs), ax=ax, label='Fully Corrected')
    ax.set_ylabel('Error (g)')
    ax.legend()

    plt.show()
