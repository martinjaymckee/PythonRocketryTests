import math
import time

import ambiance
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import scipy.optimize as optimize
import seaborn as sns


import pyrse.engines as engines
from pyrse.flight_data import loadBlueRavenLog


import eiffel_tower_v2_drag_est_processing as cd_est_proc
from eiffel_tower_v2_data_summarize import load_flight_data_list


class CdEstRegressor:
    def __init__(self, results, debug=False):
        self.__processed = None
        self.__Vmin = None
        self.__Vmax = None
        #self.__cd_func = lambda x, a, b, c, d, e, f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f          
        self.__cd_func = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d        
        #self.__cd_func = lambda x, a, b, c: a*x**2 + b*x + c        
        #self.__cd_func = lambda x, a, b, c: a*np.exp(b*x + c)        
        
        self.__params = None
        self.__debug = debug

        self.__do_regression(results)
    
    @property
    def valid(self):
        return self.__processed

    @property
    def func(self):
        class wrapped_func:
            def __init__(self, Vmin, Vmax, func):
                self.__Vmin = Vmin
                self.__Vmax = Vmax
                self.__func = func

            def __call__(self, V):
                if V > self.__Vmax:
                    V = self.__Vmax
                if V < self.__Vmin:
                    V = self.__Vmin
                return self.__func(V)
            
        return wrapped_func(self.__Vmin, self.__Vmax, lambda V: self.__cd_func(V, *self.__params))

    def __do_regression(self, results, N=50000):
        xresults = []
        yresults = []
        fig, ax = None, None

        if self.__debug:
            fig, ax = plt.subplots(1, layout='constrained')

            fig.suptitle('$C_d$ vs. $V$ Regression')
            ax.set_xlabel(r'Velocity ($m s^{-1}$)')
            ax.set_ylabel(r'Coefficient of Drag')

        for result in results:
            cfg, ts, cds, Vs, Ts = result            
            if self.__debug:
                sns.scatterplot(x=Vs, y=cds, color=(0.80, 0.80, 0.80), alpha=0.1, ax=ax)

            Vmin, Vmax = np.min(Vs), np.max(Vs)
            if (self.__Vmin is None) or (Vmin < self.__Vmin):
                self.__Vmin = Vmin
            if (self.__Vmax is None) or (Vmax > self.__Vmax):
                self.__Vmax = Vmax

            xresults.append(np.array(Vs))
            yresults.append(np.array(cds))
        
        xdata = np.concatenate(xresults)
        ydata = np.concatenate(yresults)
        popt, pcov = optimize.curve_fit(self.__cd_func, xdata, ydata, maxfev=N)
        # print(popt)

        if self.__debug:
            Vs = np.linspace(self.__Vmin, self.__Vmax, 100)
            est_cds = [self.__cd_func(V, *popt) for V in Vs]
            sns.lineplot(x=Vs, y=est_cds, color='g', ax=ax)
            eq = ''
            print(popt)
            for idx, coeff in enumerate(popt):
                idx = len(popt) - idx - 1
                eq += (' + ' if coeff > 0 else ' - ') if len(eq) > 0 else ''
                coeff = abs(coeff) if len(eq) > 0 else coeff
                if idx == 0:
                    eq += '{:f}'.format(coeff)
                elif idx == 1:
                    eq += '{:f} V'.format(coeff)
                else:
                    eq += '{:f} V^{:d}'.format(coeff, idx)
            ax.text(0.2, 0.8, '$C_{{d}} = {}$'.format(eq), transform=ax.transAxes)
            #fig.savefig(r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images\part_II_regression_I65_J180_J145.png', dpi=300)


        self.__processed = True
        self.__params = popt


def sim_3d_flight(dt, t_ignition, eng, m0, Sref, vToCd, h0=1600, off_axis=None):
    g0 = -9.80665
    h = h0
    h_last = h
    Vx = 0
    Vz = 0
    az = 0
    t = t_ignition
    eng.start(t_ignition)

    ts = []
    azs = []
    Vxs = []
    Vzs = []
    hs = []
    ms = []
    alphas = []
    Ts = []
    Ds = []
    cds = []
    done = False
    ascent = False
    idx = 0
    t_burn = eng.burn_time
    while not done:
        m = eng.calc_mass(t) + m0
        T = eng.thrust(t)
        alpha = 0 if off_axis is None else off_axis[idx]
        ascent = True if not ascent and T > m*g0 else ascent
        D = 0
        cd = 0
        if ascent:
            #print(h)
            rho = ambiance.Atmosphere(h).density[0]
            V_mag = math.sqrt(Vx**2 + Vz**2)
            cd = vToCd(V_mag)
            D = 0.5 * rho * (V_mag**2) * Sref * cd
            a_aero = (T-D) / m
            az_aero = a_aero * math.cos(alpha)
            ax_aero = a_aero * math.sin(alpha)
            az = az_aero + g0
            ax = ax_aero
            #print('cd = {}, alpha = {}, m = {}, T = {}, D = {}, az = {}, Vz = {}'.format(cd, 57.3 * alpha, m, T, D, az, Vz))
            h = h + dt * Vz + (dt**2 / 2) * az
            Vx = Vx + dt * ax
            Vz = Vz + dt * az
            done = (t > t_burn) and (h < h_last)
            h_last = h
        ts.append(t)
        hs.append(h)
        azs.append(az+g0)
        Vxs.append(Vx)
        Vzs.append(Vz)
        ms.append(m)
        Ts.append(T)
        Ds.append(D)
        alphas.append(alpha)
        cds.append(cd)
        t += dt
        idx += 1
    ts = np.array(ts)
    azs = np.array(azs)
    Vxs = np.array(Vxs)
    Vzs = np.array(Vzs)
    hs = np.array(hs)
    alphas = np.array(alphas)
    Ts = np.array(Ts)
    Ds = np.array(Ds)
    cds = np.array(cds)

    return ts, azs, Vzs, hs, Ts, Ds, cds, alphas 


def interpolate_values(ts_in, vs_in, ts_out):
    # print(len(ts_in), len(vs_in), len(ts_out))
    interpolator = scipy.interpolate.interp1d(
        ts_in,
        vs_in,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=False  # ensure it handles unsorted input
    )

    return interpolator(ts_out)


if __name__ == '__main__':
    estimation_output_dir = 'D:\Workspace\Rockets\HPR\Eiffel Tower v2\Cd Estimation Outputs'  
    reference_flight = load_flight_data_list("eiffel_tower_altitude_flight_data_list.dat") [-1]
    input_handler = cd_est_proc.CdEstFileHandler()

    cd_est_results = input_handler.load_dir(estimation_output_dir)

    regressor = CdEstRegressor(cd_est_results)

    dt = .02
    t_ignition = 0
    ref_eng = engines.Engine.RSE(reference_flight['engine_path'])

    m0 = 953/1000 #750 / 1000 # kg
    Sref = .155 * .155 # m^2


    t_start = time.time_ns()    
    br_flight_log = loadBlueRavenLog(reference_flight['summary_path'], reference_flight['low_rate_path'], reference_flight['high_rate_path'])
    t_new = time.time_ns() - t_start
    print('{} Parse Time = {:0.2f} ms'.format(reference_flight['name'], t_new / 1e6))

    ref_ts = None
    ref_azs = None
    ref_off_axis = None
    ref_Vzs = None
    ref_hs = None
    if br_flight_log is not None:
        br_flight_log.updateEvents(force=True)

        events = br_flight_log.events
        t_ignition = 0 if not 'Ignition' in events else events['Ignition'].t        
        t_liftoff = 0 if not 'Liftoff' in events else events['Liftoff'].t
        t_apogee = None if not 'Apogee' in events else events['Apogee'].t
            
        print('\tLiftoff at {} s, Apogee at {} s'.format(t_liftoff, t_apogee))
        ref_ts = br_flight_log['t'].values
        idx_start = max(0, np.searchsorted(ref_ts, t_ignition, side='left'))
        idx_end = max(0, np.searchsorted(ref_ts, t_apogee, side='left'))
        print('t_ignition = {} s'.format(t_ignition))
        ref_ts = ref_ts[idx_start:idx_end]
        #ref_ts -= t_ignition
        ref_azs = -1.0*br_flight_log['az'].values[idx_start:idx_end] - 9.80665
        ref_off_axis = br_flight_log['off_vertical'].values[idx_start:idx_end]

        ref_Vzs = br_flight_log['Vz'].values[idx_start:idx_end]
        #Vhs = br_flight_log['Vh'].values[idx_start:idx_end]
        ref_hs = br_flight_log['h'].values[idx_start:idx_end]
        #rhos = ambiance.Atmosphere(hs+reference_flight['h0']).density
    print(ref_ts)
    # TODO: FIGURE OUT IF THERE IS A WAY TO USE OFF-AXIS HERE....
    ts, azs, Vzs, hs, Ts, Ds, cds, alphas = sim_3d_flight(dt, t_ignition, ref_eng, m0, Sref, regressor.func, h0=reference_flight['h0'])#, off_axis=alphas) 

    hs = hs - reference_flight['h0']
    azs_err = interpolate_values(ref_ts, ref_azs, ts) - azs
    Vzs_err = interpolate_values(ref_ts, ref_Vzs, ts) - Vzs
    hs_err = interpolate_values(ref_ts, ref_hs, ts) - hs

    print('Maximum Altitude (agl) = {} ft, Maximum Velocity = {} m/s'.format(3.28 * np.max(hs), np.max(Vzs)))

    fig, axs = plt.subplots(3, layout='constrained', sharex=True, figsize=(8, 10))
    fig.suptitle('{} Simulation vs {} Actual Flight'.format(ref_eng.model, ref_eng.model))
    axs[0].set_ylabel('Acceleration ($m s^{-2}$)')
    axs[1].set_ylabel('Velocity ($m s^{-1}$)')
    axs[2].set_xlabel('Time ($s$)')
    axs[2].set_ylabel('Altitude ($m$)')

    ax0_err = axs[0].twinx()
    ax1_err = axs[1].twinx()
    ax2_err = axs[2].twinx()

    ax0_err.set_ylabel('Acceleration Error ($m s^{-2}$)')
    ax1_err.set_ylabel('Velocity Error ($m s^{-1}$)')
    ax2_err.set_ylabel('Altitude Error ($m$)')

    axs[0].plot(ts, azs, 'g', alpha=0.5, label='Simulated')
    axs[1].plot(ts, Vzs, 'g', alpha=0.5, label='Simulated')
    axs[2].plot(ts, hs, 'g', alpha=0.5, label='Simulated')

    axs[0].plot(ref_ts, ref_azs, 'k', alpha=0.5, label='Actual')
    axs[1].plot(ref_ts, ref_Vzs, 'k', alpha=0.5, label='Actual')
    axs[2].plot(ref_ts, ref_hs, 'k', alpha=0.5, label='Actual')

    ax0_err.plot(ts, azs_err, 'r', alpha=0.25)
    azs_mean = np.mean(azs_err)
    azs_std = np.std(azs_err)
    ax0_err.set_ylim(azs_mean - 3*azs_std, azs_mean + 3*azs_std)
    
    ax1_err.plot(ts, Vzs_err, 'r', alpha=0.25)
    Vzs_mean = np.mean(Vzs_err)
    Vzs_std = np.std(Vzs_err)
    ax1_err.set_ylim(Vzs_mean - 3*Vzs_std, Vzs_mean + 3*Vzs_std)

    ax2_err.plot(ts, hs_err, 'r', alpha=0.25)
    hs_mean = np.mean(hs_err)
    hs_std = np.std(hs_err)
    ax2_err.set_ylim(hs_mean - 3*hs_std, hs_mean + 3*hs_std)

    for ax in axs:
        ax.legend()

    fig.savefig(r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images\part_II_actual_vs_simulation.png', dpi=300)


    plt.show()