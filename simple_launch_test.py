import math
import random

import matplotlib.pyplot as plt
import numpy as np

import engines

accel_fs = 100
accel_sd = 5 * 130e-6 * accel_fs * 9.80665  # Five times LSM6DSM with bandwidth of accel_fs

def getCMColors(name, N, scale_range=(0, 1)):
    cmap = plt.cm.get_cmap(name)
    vals = np.linspace(*scale_range, N)
    return [cmap(val) for val in vals]

class BurnoutDetect:
    def __init__(self, accel_threshold, jerk_threshold, e=0.9):
        self.__t_last = None
        self.__accel_last = 0
        self.__jerk_last = 0
        self.__accel_threshold = accel_threshold
        self.__jerk_threshold = jerk_threshold
        self.__detected = False
        self.__ts = []
        self.__accels = []
        self.__jerks = []
        self.__e = e

    def reset(self):
        self.__accel_last = 0
        self.__jerk_last = 0
        self.__detected = False
        self.__ts = []
        self.__accels = []
        self.__jerks = []

    @property
    def ts(self): return self.__ts

    @property
    def jerks(self): return self.__jerks

    def __call__(self, t, accel):
        detected = False
        if self.__t_last is not None:
            dt = t - self.__t_last
            new_jerk = (accel - self.__accel_last) / dt
            jerk = self.__e * self.__jerk_last + (1 - self.__e) * new_jerk
            accel_event = accel < self.__accel_threshold
            jerk_event = jerk < self.__jerk_threshold
            self.__ts.append(t)
            self.__accels.append(accel)
            self.__jerks.append(jerk)
            self.__jerk_last = jerk
            if not self.__detected:
                self.__detected = accel_event and jerk_event
                detected = self.__detected
                if self.__detected:
                    pass
                    # print('Burnout with accel = {:0.2f} m/s^2, jerk = {:0.2f} m/s^3'.format(accel, jerk))
        self.__accel_last = accel
        self.__t_last = t
        return detected


class LaunchDetect:
    def __init__(self, v_launch, t_buffer=1.0, N=10):
        self.__N = N
        self.__dt = t_buffer / N
        self.__buffer = []
        self.__v_launch = v_launch
        self.__t_last = None
        self.__detected = False
        self.__ts = []
        self.__accels = []
        self.__vels = []
        self.__last_accel = 0
        self.__accel_min = 9.80665 * 0.75
        self.__dv_min = self.__accel_min * self.__dt
        # print('Launch Detect.dv_min = {:0.2f} m/s'.format(self.__dv_min))

    def reset(self):
        self.__ts = []
        self.__accels = []
        self.__vels = []
        self.__t_last = None
        self.__last_accel = 0
        self.__detected = False
        self.__buffer = []

    @property
    def ts(self): return self.__ts

    @property
    def accels(self): return self.__accels

    @property
    def vels(self): return self.__vels

    @property
    def v_launch(self):
        return self.__v_launch

    @property
    def detected(self):
        return self.__detected

    def __call__(self, t, accel):
        t_delta = None
        if self.detected:
            return False, 0
        if self.__t_last is None or (t - self.__t_last) > self.__dt:
            avg_accel = accel  #(self.__last_accel + accel) / 2
            self.__buffer.append(0 if avg_accel < 0 else self.__dt * avg_accel)
            if len(self.__buffer) > self.__N:
                self.__buffer = self.__buffer[-self.__N:]
            self.__t_last = t if self.__t_last is None else self.__t_last + self.__dt
            self.__ts.append(self.__t_last)
            self.__accels.append(avg_accel)
            self.__last_accel = accel
            vel_est = sum(self.__buffer)
            self.__vels.append(vel_est)
            self.__detected = vel_est > self.__v_launch
            if self.detected:
#                print('Calculate t_delta:')
                t_delta = -self.__dt
                v = vel_est
                for dv in reversed(self.__buffer):
                    # print('\tt_delta = {:0.4f} s, v = {:0.2f} m/s, dv = {:0.2f} m/s'.format(t_delta, v, dv))
                    if v > dv:
                        t_delta += self.__dt
                        if dv < self.__dv_min:
                            break
                    else:
                        dt = self.__dt * (v / dv)
                        # print('\t\tdt = {:0.4f} * ( {:0.2f} / {:0.2f}) = {:0.4f}'.format(self.__dt, v, dv, dt))
                        t_delta += dt
                        v -= (dt / self.__dt) * dv
                        break
                    v -= dv
                # print('\tt_delta = {:0.4f} s, v = {:0.2f} m/s'.format(t_delta, v))
        return self.detected, t_delta


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
    engs = engines.load_engine_files('./engines')
    print(engs)
    eng = engs[("AeroTech", "D13W")]

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
