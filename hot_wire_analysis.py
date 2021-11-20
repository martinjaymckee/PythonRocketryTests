import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hot_wire_design as hwd
import normal_rvs as nrvs


def convergenceAndStability(data, Tset, Tstabilized=5, plot=True, centigrade=True):
    T0 = 273.15 if centigrade else 0
    Tset -= T0
    ts = np.array(data['ts'])
    Ts = np.array([T.mean for T in data['Ts']]) - T0
    Ts_est = np.array([T.mean for T in data['Ts_est']]) - T0
    Ts_est_sd = np.array([T.standard_deviation for T in data['Ts_est']])
    Ts_offset = Ts - Tset
    Ts_est_err = Ts - Ts_est
    N = np.argmax((Tset - Ts_est) < Tstabilized)
    tconv = ts[N]
    offset_sd = np.std(Ts_offset[N:])
    offset_mean = np.mean(Ts_offset[N:])
    err_sd = np.std(Ts_est_err[N:])
    err_mean = np.mean(Ts_est_err[N:])
    stabilized = np.max(np.abs(Ts_offset[N:])) <= Tstabilized
    # assert stabilized, 'Unstabilized system, max offset = {}'.format(np.max(Ts_offset[N:]))

    if plot:
        temp_unit = r'$^{\circ}C$' if centigrade else 'K'
        fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True, constrained_layout=True)
        fig.suptitle('Simulation Convergence and Stability')
        axs[0].plot(ts, Ts, c='k', alpha=0.5, label='Physical')
        axs[0].scatter(ts, Ts_est, c='c', alpha=0.75, s=1, label='Estimated')
        axs[0].errorbar(ts, Ts_est, yerr=Ts_est_sd, alpha=0.075, color='c', fmt=' ', zorder=-1)
        axs[0].axhline(Tset, c='k', alpha=0.25)
        axs[0].set_title('Absolute')
        axs[0].set_ylabel('Temperature ({})'.format(temp_unit))
        axs[0].legend()
        axs[0].annotate('Convergence at t = {:0.2f}s'.format(tconv),xy=(tconv, Tset - Tstabilized), xytext=(tconv+2.5,  0.8 * Tset), arrowprops={'arrowstyle': '->'})
        c_track = (1.0, 0, 1.0, 1.0)
        c_err = (0.5, 0.25, 0, 1.0)
        axs[1].plot(ts, Ts_offset, c=c_track, alpha=0.25, label='Tracking Error')
        axs[1].plot(ts, Ts_est_err, c=c_err, alpha=0.25, label='Estimation Error')
        axs[1].set_title('Errors')
        axs[1].set_ylabel('Temperature ({})'.format(temp_unit))
        y_sd = max(offset_sd, err_sd)
        ymin = min(offset_mean, err_mean) - 5 * y_sd
        ymax = max(offset_mean, err_mean) + 5 * y_sd
        axs[1].set_ylim(ymin, ymax)
        axs[1].legend()
        xmin, xmax = axs[0].get_xlim()
        axs[0].axvline(tconv, c='k', alpha=0.1)
        axs[1].axvline(tconv, c='k', alpha=0.1)
        axs[0].axvspan(xmin, tconv, color='k', alpha=0.05)
        axs[1].axvspan(xmin, tconv, color='k', alpha=0.05)
    offset = nrvs.NRV(offset_mean, sd=offset_sd)
    err = nrvs.NRV(err_mean, sd=err_sd)
    return tconv, stabilized, offset, err


def pidGainStability(hw_sys, Kp=None, Ki=None, Kd=None, Tset=273.15+315, t_max=120, Tstabilized=5, steps=15, plot=True, seed=123456789):
    hw_sim = hwd.HotwireSimulator(t_max=t_max, seed=seed)
    Kps = np.linspace(*Kp, steps)
    Kds = np.linspace(*Kd, steps)
    Kis = []
    ts_conv = []
    offsets = []
    idxs = []
    Ki = 0 if Ki is None else Ki
    rand_state = random.getstate()
    for idx_Kp, idx_Kd in itertools.product(range(steps), repeat=2):
        print('Kp = {}, Ki = {}, Kd = {}'.format(Kp, Ki, Kd))
        idxs.append((idx_Kp, idx_Kd))
        Kp = Kps[idx_Kp]
        Kd = Kds[idx_Kd]
        Kis.append(Ki)
        hw_sys.controller.pid.Kp = Kp
        hw_sys.controller.pid.Ki = Ki
        hw_sys.controller.pid.Kd = Kd
        random.setstate(rand_state)
        sim_data = hw_sim.run_sim(hw_sys,  Tset=Tset, Tinit=Tamb)
        tconv, stabilized, offset, err = convergenceAndStability(sim_data, Tset, Tstabilized=Tstabilized, plot=False)
        ts_conv.append(tconv)
        offsets.append(offset)
    if plot:
        cmap = 'viridis'
        fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        offset_means = np.array([v.mean for v in offsets])
        offset_sds = np.array([v.standard_deviation for v in offsets])
        #
        # Plot convergence time vs. Kp/Kd
        #
        data_conv = np.zeros(shape=(steps, steps))
        data_offset_mean = np.zeros(shape=(steps, steps))
        data_offset_sds = np.zeros(shape=(steps, steps))
        for idx, (column, row) in enumerate(idxs):
            data_conv[row, column] = ts_conv[idx]
            data_offset_mean[row, column] = offset_means[idx]
            data_offset_sds[row, column] = offset_sds[idx]
        Kp_labels = ['{:0.3f}'.format(v) for v in Kps]
        Kd_labels = ['{:0.3f}'.format(v) for v in Kds]

        sns.heatmap(data_conv, ax=axs[0][0], cmap=cmap, xticklabels=Kp_labels, yticklabels=Kd_labels)
        sns.heatmap(data_offset_mean, ax=axs[1][0], cmap=cmap, xticklabels=Kp_labels, yticklabels=Kd_labels)
        sns.heatmap(data_offset_sds, ax=axs[1][1], cmap=cmap, xticklabels=Kp_labels, yticklabels=Kd_labels)
        axs[0][0].set_xlabel('Kp')
        axs[0][0].set_ylabel('Kd')
        axs[1][0].set_xlabel('Kp')
        axs[1][0].set_ylabel('Kd')
        axs[1][1].set_xlabel('Kp')
        axs[1][1].set_ylabel('Kd')


if __name__ == '__main__':
    T0 = 273.15
    Tamb = T0 + 30
    Tset = T0 + 315

    def testSystem():
        f_update = 100
        Vin = 24
        wire = hwd.HotWire('316L', 30, 1.3)
        # wire = hwd.HotWire('Ni200', 30, 1.3)

        # pid_kws = {'Kp': 2, 'Ki': 0.001, 'Kd': 0.075}
        # pid_kws = {'Kp': 0.1, 'Ki': 0.001, 'Kd': 0.01}
        # pid_kws = {'Kp': 0.005, 'Ki': 0.00015, 'Kd': 0.0075}
        pid_kws = {'Kp': 0.245, 'Ki': 0.001, 'Kd': 0.01}

        ctrl_type = hwd.PredictiveHotwireController
        hw_sys = hwd.HotwireSystem(Vin, wire, ctrl_type, ctrl_kws=pid_kws, f_update=f_update)
        return hw_sys

    def testSim(t_max=30):
        hw_sim = hwd.HotwireSimulator(t_max=t_max, seed=123456789)
        sim_data = hw_sim.run_sim(testSystem(),  Tset=Tset, Tinit=Tamb)
        return sim_data

    sim_data = testSim(t_max=120)
    print([str(v) for v in convergenceAndStability(sim_data, Tset)])
    plt.show()

    # pidGainStability(testSystem(), Kp=(0.2, 0.5), Kd=(0.01, 0.05), steps=15)
    # plt.show()
