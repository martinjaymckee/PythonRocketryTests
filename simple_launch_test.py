import math
import random

import matplotlib.pyplot as plt
import numpy as np

import pyrse.engines as engines
from pyrse.flight_events import BurnoutDetect, LaunchDetect

accel_fs = 100
accel_sd = 5 * 130e-6 * accel_fs * 9.80665  # Five times LSM6DSM with bandwidth of accel_fs

def getCMColors(name, N, scale_range=(0, 1)):
    cmap = plt.cm.get_cmap(name)
    vals = np.linspace(*scale_range, N)
    return [cmap(val) for val in vals]


def simulate_launch(dt, eng, m_rocket, launch_detect=None, burnout_detect=None, g=9.80665):
    vel = 0
    S = math.pi * (0.041/2)**2
    Cd = 0.54
    rho = 1.225
    t = -eng.burn_time-random.uniform(0, 1/5)
    ts = []
    ms = []
    Ts = []
    Ds = []
    accels = []
    vels = []
    t_launch = None
    t_delta = None
    t_burnout = None
    launch_detected = False
    launch_detect.reset()
    burnout_detect.reset()
    while t < (1.3 * eng.burn_time):
        T = eng.thrust(t)
        m_eng = eng.mass(t)
        D = 0.5 * rho * S * Cd * vel**2
        m_tot = m_rocket + m_eng
        accel = ((T-D) / m_tot) - g
        vel = vel + dt * accel if t > 0 else 0
        accel_measured = random.gauss(accel, accel_sd)
        ts.append(t)
        ms.append(m_tot)
        Ts.append(T)
        Ds.append(D)
        accels.append(accel_measured)
        vels.append(vel)
        if not launch_detected and launch_detect is not None:
            detected, t_delta = launch_detect(t, accel_measured)
            if detected:
                launch_detected = True
                t_launch = t
                # print('Launch Detected at v = {:0.2f} m/s, t = {:0.2f} s, with a t_delta = {:0.2f} s'.format(vel, t_launch, t_delta))
        if launch_detected and (burnout_detect is not None):
            detected = burnout_detect(t, accel_measured)
            if detected:
                t_burnout = t
                # print('Burnout Detected at t = {:0.2f} s'.format(t_burnout))
        t += dt
    return ts, ms, Ts, Ds, accels, vels, t_launch, t_delta, t_burnout


if __name__ == '__main__':
    cs = getCMColors('viridis', 5)
    print(cs)

    dt = 1/100
    m_rocket = 0.15  # kg
    v_launch = 15
    engs = engines.EngineDirectory('./Engines')
    eng = engs.load_first("Aerotech", "D13W")

    launch_detect = LaunchDetect(v_launch, t_buffer=0.64, N=32)
    burnout_detect = BurnoutDetect(-5, -20, e=0.9)

    do_sim_plot = True
    fig, sim_axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    # fig.tight_layout()
    sim_ax0 = sim_axs[0].twinx()
    sim_ax1 = sim_axs[1].twinx()

    t_launchs = []
    t_deltas = []
    t_burnouts = []
    t_errors = []
    for idx in range(50):
        ts, ms, Ts, Ds, accels, vels, t_launch, t_delta, t_burnout = simulate_launch(dt, eng, m_rocket, launch_detect=launch_detect, burnout_detect=burnout_detect)
        t_error = None
        if t_launch is not None and t_delta is not None:
            t_error = t_launch-t_delta
        if t_launch is not None:
            t_launchs.append(t_launch)
        if t_delta is not None:
            t_deltas.append(t_delta)
        if t_burnout is not None:
            t_burnouts.append(t_burnout)
        if t_error is not None:
            t_errors.append(t_error)

        if do_sim_plot:
            sim_axs[0].plot(ts, Ts, c=cs[0], alpha=0.1)
            sim_ax0.plot(ts, Ds, c=cs[1], alpha=0.1)
            sim_axs[1].plot(ts, accels, c=cs[0], alpha=0.1)
            sim_axs[1].plot(launch_detect.ts, launch_detect.accels, c=cs[2], alpha=0.1)
            sim_ax1.plot(burnout_detect.ts, burnout_detect.jerks, c=cs[1], alpha=0.1)
            sim_axs[2].plot(ts, vels, c=cs[0], alpha=0.1)
            sim_axs[2].plot(launch_detect.ts, launch_detect.vels, c=cs[2], alpha=0.15)
            for ax in sim_axs:
                if t_launch is not None:
                    ax.axvline(t_launch, c=cs[0], alpha=0.1)
                if t_error is not None:
                    ax.axvline(t_error, c=cs[1], alpha=0.1)
                if t_burnout is not None:
                    ax.axvline(t_burnout, c=cs[2], alpha=0.1)

    if do_sim_plot:
        sim_axs[0].set_ylabel('Thrust (N)')
        sim_axs[0].yaxis.label.set_color(cs[0])
        sim_ax0.set_ylabel('Drag (N)')
        sim_ax0.yaxis.label.set_color(cs[0])

        sim_axs[1].set_ylabel('Acceleration ($m/s^2$)')
        sim_axs[1].yaxis.label.set_color(cs[0])
        sim_ax1.set_ylabel('Jerk ($m/s^3$)')
        sim_ax1.yaxis.label.set_color(cs[0])

        sim_axs[2].set_ylabel('Velocity ($m/s$)')
        sim_axs[2].yaxis.label.set_color(cs[0])

        sim_axs[1].axhline(0, c='k', alpha=0.5)
        sim_axs[1].axhline(-5, c=cs[3], alpha=0.5)

        sim_axs[2].axhline(0, c='k', alpha=0.5)
        sim_axs[2].axhline(v_launch, c=cs[3], alpha=0.5)

        t_min = min([0, *t_errors])-0.25
        t_max = max([*t_burnouts, *t_launchs, eng.burn_time])+0.25
        # t_max = max(t_launchs) + 0.25
        for ax in sim_axs:
            ax.set_xlim(t_min, t_max)
            ax.axvline(0, c='k', alpha=0.5)
            ax.axvline(eng.burn_time, c=cs[3], alpha=0.5)
        plt.show()
    print('Launch Estimate Error Mean = {:0.4f} s'.format(np.mean(t_errors)))
    print('Launch Estimate Error Standard Deviation = {:0.4f} s'.format(np.std(t_errors)))
    print('Number of Launch Detects = {}'.format(len(t_launchs)))
    print('Number of Burnout Detects = {}'.format(len(t_burnouts)))
