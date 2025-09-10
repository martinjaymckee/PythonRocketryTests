import matplotlib.pyplot as plt

import randx

if __name__ == '__main__':
    N = 10000
    ys = [randx.gauss(0, 1) for _ in range(N)]

    fig, ax = plt.subplots(1, constrained_layout=True)
    ax.hist(ys, bins=100)

    plt.show()