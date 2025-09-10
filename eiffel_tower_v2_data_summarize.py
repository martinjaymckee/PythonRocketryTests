import os
import os.path
import time

import ambiance
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import seaborn as sns

import pyrse.engines as engines
from pyrse.flight_data import loadBlueRavenLog
import pyrse.flight_data_plotting as fdp


def plotFlightDebugData(flight_log):
        fdp.plotFlightOverview(flight_log)                

        fig_a, ax_a = plt.subplots(1, layout='constrained', sharex=True)
        fig_a.suptitle('Accelerations (Body Frame)')
        ts = flight_log['t'].values
        accels = flight_log['a'].values
        fdp.plotVectors(ax_a, ts, accels, 'a{axis}')

        fig_g, ax_g = plt.subplots(1, layout='constrained', sharex=True)
        fig_g.suptitle('Rotation Rates (Body Frame)')
        ts = flight_log['t'].values
        gyros = flight_log['g'].values
        fdp.plotVectors(ax_g, ts, gyros, 'g{axis}')

        fig_a, ax_q = plt.subplots(1, layout='constrained', sharex=True)
        fig_a.suptitle('Quaternion Orientation')
        ts = flight_log['t'].values
        qs = flight_log['q'].values
        fdp.plotQuaternions(ax_q, ts, qs, 'q{axis}')

        fig_off_vertical, ax = plt.subplots(1, constrained_layout=True, sharex=True)
        fig_off_vertical.suptitle('Angle from Vertical')
        ts = flight_log['t'].values
        alphas = br_flight_log['off_vertical'].values
        ax.plot(ts, 57.3 * alphas, c='g')
        ax.axhline(0, c='c', alpha=0.5)


def load_flight_data_list(filename):
    flight_data_list = None
    global_vars = {}
    local_vars = {}
    with open(filename) as src:
        code = compile(src.read(), filename, 'exec')
        exec(code, global_vars, local_vars)
        flight_data_list = local_vars['flight_data_list']
    return flight_data_list


if __name__ == '__main__':
    g0 = 9.80665

    output_dir = r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images'
    flight_data_list = load_flight_data_list("eiffel_tower_maiden_data_list.dat")

    fig_basic, axs_basic = plt.subplots(3, layout='constrained', sharex=True, figsize=(8, 6))
    fig_basic.suptitle('Basic Flight Summary')
    axs_basic[0].set_ylabel('Acceleration ($m s^{-2}$)')
    axs_basic[1].set_ylabel('Velocity ($m s^{-1}$)')
    axs_basic[2].set_xlabel('Time ($s$)')
    axs_basic[2].set_ylabel('Altitude ($m$)')

    fig_offaxis, ax_offaxis = plt.subplots(1, layout='constrained', figsize=(8, 6))
    fig_offaxis.suptitle('Flight Off-Axis Summary')
    ax_offaxis.set_xlabel('Time ($s$)')
    ax_offaxis.set_ylabel('Angle from Vertical (deg)')

    for ax in axs_basic:
         ax.axvline(0, c='k', alpha=0.5)
    ax_offaxis.axvline(0, c='k', alpha=0.5)

    basic_filename = 'basic_summary'
    offaxis_filename = 'offaxis_summary'
    for flight in flight_data_list:
        t_start = time.time_ns()    
        br_flight_log = loadBlueRavenLog(flight['summary_path'], flight['low_rate_path'], flight['high_rate_path'])
        t_new = time.time_ns() - t_start
        print('{} Parse Time = {:0.2f} ms'.format(flight['name'], t_new / 1e6))

        if br_flight_log is not None:
            ref_eng = engines.Engine.RSE(flight['engine_path'])            
            eng_name = ref_eng.model
            basic_filename += '_{}'.format(eng_name)
            offaxis_filename += '_{}'.format(eng_name)

            br_flight_log.updateEvents(force=True)

            events = br_flight_log.events
            t_ignition = 0 if not 'Ignition' in events else events['Ignition'].t        
            t_liftoff = 0 if not 'Liftoff' in events else events['Liftoff'].t
            t_apogee = None if not 'Apogee' in events else events['Apogee'].t
            # print('t_apogee = {}'.format(t_apogee))  

            ts = br_flight_log['t'].values
            t_end = t_apogee + 1.0
            idx_end = np.searchsorted(ts, t_end, side='right')
            t_start = t_ignition - 1.0
            idx_start = max(0, np.searchsorted(ts, t_start, side='left'))
            azs = -1.0*br_flight_log['az'].values[idx_start:idx_end] # - g0
            # azs_filt = signal.savgol_filter(azs, window_length=25, polyorder=3, delta=0.002)
            off_axis = br_flight_log['off_vertical'].values[idx_start:idx_end]        

            Vzs = br_flight_log['Vz'].values[idx_start:idx_end]
            Vhs = br_flight_log['Vh'].values[idx_start:idx_end]
            hs = br_flight_log['h'].values[idx_start:idx_end]
            rhos = ambiance.Atmosphere(hs+flight['h0']).density
            ts = ts[idx_start:idx_end] - t_ignition

            name = ref_eng.model
            c = flight['color']

            axs_basic[0].plot(ts, azs, c=c, alpha=0.5, label=name)
            axs_basic[1].plot(ts, Vzs, c=c, alpha=0.5, label=name)
            #axs_basic[1].plot(ts, Vhs, ':', c=c, alpha=0.5)
            axs_basic[2].plot(ts, hs, c=c, alpha=0.5, label=name)

            ax_offaxis.plot(ts, 57.3*off_axis, c=c, alpha=0.5, label=name)

            for ax in axs_basic:
                ax.axvline(t_apogee-t_ignition, c=c, alpha=0.5)

            ax_offaxis.axvline(t_apogee-t_ignition, c=c, alpha=0.25)

            print('{} (on {}) - Apogee Altitude = {} m, Maximum Velocity = {} m/s'.format(flight['name'], ref_eng.model, np.max(hs), np.max(Vzs)))
        else:
            print('\tError: Flight Parsing Failed')            

    axs_basic[2].legend()
    # for ax in axs_basic:
    #     ax.legend()

    ax_offaxis.legend()

    basic_filename += '.png'.format(eng_name)
    offaxis_filename += '.png'.format(eng_name)

    basic_filename = os.path.join(output_dir, basic_filename)
    offaxis_filename = os.path.join(output_dir, offaxis_filename)

    fig_basic.savefig(basic_filename, dpi=300)
    fig_offaxis.savefig(offaxis_filename, dpi=300)    

    plt.show()