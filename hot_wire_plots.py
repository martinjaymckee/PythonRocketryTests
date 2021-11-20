import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#
# Analog Front End (AFE) Measurements Plot
#
def plotHotwireAFEMeasurements(hotwire_data, fig=None, ax=None):
    fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True, constrained_layout=True)
    fig.suptitle('ADC Measurement Data')
    ts = hotwire_data['ts']
    Imax = hotwire_data['Imax']
    Vmax = hotwire_data['Vmax']

    #   Voltage Measurements
    axs[0].plot(ts, hotwire_data['Vs_drv'], alpha=0.75, label='Drive')
    axs[0].plot(ts, hotwire_data['Vs_hw'], alpha=0.75, label='Hotwire')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].axhline(Vmax, c='r', alpha=0.25)
    axs[0].set_ylim(0, 1.05 * Vmax)
    axs[0].legend()
    #   Current Measurements
    axs[1].plot(ts, hotwire_data['Is'], alpha=0.75)
    axs[1].set_ylabel('Current (A)')
    axs[1].set_xlabel('Time (s)')
    axs[1].axhline(Imax, c='r', alpha=0.25)
    axs[1].set_ylim(0, 1.05 * Imax)
    return fig, axs


#
# Driver Characteristics Plot
#
def plotHotwireDriverCharacteristics(hotwire_data, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
    fig.suptitle('Driver Output')
    ts = hotwire_data['ts']
    ax.plot(ts, hotwire_data['Vs_drv'], alpha=0.75, label='Actual')
    ax.plot(ts, hotwire_data['Vs_drv_requested'], alpha=0.75, label='Requested')
    ax.plot(ts, hotwire_data['Vs_drv_quantized'], alpha=0.75, label='Quantized')
    ax.set_ylabel('Voltage (V)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    return fig, ax


#
# Hotwire System Response Plot
#
def plotHotwireSystemResponse(hotwire_data, fig=None, ax=None):
    fig, axs = plt.subplots(3, figsize=(16, 9), sharex=True, constrained_layout=True)
    fig.suptitle('Hotwire Response')
    #   Temperature Response
    # axs[0].plot(ts, Ts - T0, alpha=0.75, label='Physical')
    # axs[0].plot(ts, Ts_est - T0, alpha=0.75, label='Measured')
    # axs[0].set_ylabel(r'Temperature ($^{\circ}C$)')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].axhline(Tset - T0, c='k', alpha=0.15)
    # axs[0].legend()
    # ax0 = axs[0].twinx()
    # ax0.plot(ts, Ts - Ts_est, alpha=0.15, c='g', label='Estimation')
    # ax0.plot(ts, Ts - Tset, alpha=0.15, c='m', label='Tracking')
    # N = min(np.argmax((Tset - Ts_est) < 5), int(len(Ts_est) / 2))
    # sd_T_err = np.std(Ts[N:]-Tset)
    # sd_T_err_mean = np.mean(Ts[N:]-Tset)
    # # print('Index = {}'.format(N))
    # if math.isnan(sd_T_err) or math.isinf(sd_T_err):
    #     # print('sd_T_err = {}'.format(sd_T_err))
    #     sd_T_err = 100
    #     sd_T_err_mean = 0
    # ax0.set_ylabel(r'Temperature Error ($^{\circ}C$)')
    # ax0.set_ylim(-5*sd_T_err + sd_T_err_mean, 5*sd_T_err + sd_T_err_mean)
    # ax0.legend()
    # #   Power Response
    # axs[1].plot(ts, Vs_drv * Is_reading, alpha=0.75, label='Wire Power')
    # axs[1].plot(ts, loads, alpha=0.75, label='Cut Power')
    # axs[1].set_ylabel('Power (W)')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].legend()
    # ax1 = axs[1].twinx()
    # ax1.plot(ts, effs, alpha=0.15, c='g')
    # ax1.set_ylabel('Drive Efficiency (%)')
    # print('Maximum Sense Resistor Power = {} W'.format(np.max(Ps_RI)))
    # print('Maximum Drive Current = {} A'.format(np.max(I_hw_afe)))
    # # print('Efficiency min = {} %, max = {} %'.format(100 * np.min(effs), 100 * np.max(effs)))
    # #   Resistance Response
    # axs[2].plot(ts, Rs, alpha=0.75, label='Physical')
    # axs[2].plot(ts, Rs_est, alpha=0.75, label='Measured')
    axs[2].set_ylabel(r'Resistance ($\Omega$)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    return (fig, axs)
