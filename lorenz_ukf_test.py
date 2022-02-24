

class LorenzSystem:
    def __init__(self, sigma=10, beta=8/3, rho=28):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.x = 1
        self.y = 0
        self.z = rho
        self.t = 0

    def __call__(self, dt=1e-3):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x*self.y - self.beta*self.z
        self.x += dt * dx
        self.y += dt * dy
        self.z += dt * dz
        self.t += dt
        return self.t, self.x, self.y, self.z


if __name__ == '__main__':
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np

    ts = []
    xs = []
    ys = []
    zs = []
    lorenz = LorenzSystem()

    for _ in range(50000):
        t, x, y, z = lorenz(0.0075)
        ts.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    cs = np.array(ts)
    fig = plt.figure(constrained_layout=True)
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs, ys, zs, c=cm.viridis(np.array(ts)/ts[-1]), alpha=0.75, edgecolor=None, s=2)
    plt.show()
