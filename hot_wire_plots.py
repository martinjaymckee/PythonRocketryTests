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
