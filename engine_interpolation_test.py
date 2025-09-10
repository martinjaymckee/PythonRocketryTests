import random

import matplotlib.pyplot as plt
import numpy as np

import pyrse.engines as engines    



def main():
    random.seed(12345)

    N = 5
    impulse_sd_percent = 5
    burn_rate_sd_percent = 10
    thrustcurve_noise_sd_percent = 5

    engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_I65W.rse"
    ref_eng = engines.Engine.RSE(engine_path)
    
    t_ignition = -0.15
    ts = np.linspace(-1, 10, 1000)
    fig, axs = plt.subplots(2, layout='constrained', sharex=True)

    for idx in range(N):
        eng = ref_eng.Scaled(impulse_multiplier = random.gauss(1, impulse_sd_percent/100.0), burn_rate_multiplier = random.gauss(1, burn_rate_sd_percent/100.0), noise_sd=random.gauss(0, thrustcurve_noise_sd_percent/100))
        eng.start(t_ignition)

        Ts = np.array([eng.thrust(t) for t in ts])
        ms = np.array([eng.calc_mass(t) for t in ts])
        
        axs[0].plot(ts, Ts, c='r', alpha=0.25)
        axs[1].plot(ts, ms, c='r', alpha=0.25)

    plt.show()


if __name__ == '__main__':
    main()