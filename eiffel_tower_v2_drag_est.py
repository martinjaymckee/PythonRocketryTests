import math
import os
import os.path
import random
import time

import ambiance
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal
import seaborn as sns

import pyrse.engines as engines
from pyrse.flight_data import FlightData, loadBlueRavenLog, loadOpenRocketExport
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


def run_cd_est(ts, azs, Vzs, Vhs, rhos, off_axis, m0, eng, Sref):
    g0 = -9.80665    
    Ts = np.array([eng.thrust(t) for t in ts])
    ms = np.array([eng.calc_mass(t) for t in ts])
    cds = []
    ms_tot = ms + m0
    as_eff = []
    print('{}, {}, {}, {}, {}, {}, {}, {}'.format(len(ts), len(azs), len(Vzs), len(Vhs), len(rhos), len(off_axis), len(Ts), len(ms_tot)))
    for t, az, Vz, Vh, rho, alpha, T, m in zip(ts, azs, Vzs, Vhs, rhos, off_axis, Ts, ms_tot):
        a_eff = g0+az
        Vmag = math.sqrt(Vz**2 + Vh**2)
        cd = ((2 * (T*math.cos(alpha) - (m * a_eff))) / (rho * Sref * (Vmag**2) * math.cos(alpha)))
        cds.append(0.75*cd)
        as_eff.append(a_eff)
    cds = np.array(cds)
    as_eff = np.array(as_eff)
    return ts, Vzs, cds, ms_tot, as_eff, Ts


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

class CdEstFileHandler:
    @classmethod
    def Load(cls, filename):
        handler = CdEstFileHandler(filename)
        handler.load()
        return handler


    def __init__(self, filename=None):
        self.__filename = filename


    def save(self, cfg, ts, cds, Vs, filename=None, eng=None):
        filename = self.__filename if filename is None else filename
        t0 = 0 if 't0' not in cfg else cfg['t0']

        with open(filename, 'w') as file:
            file.write('# Base Configuration:\n')
            file.write('m0 = {}\n'.format(cfg['m0']))
            file.write('Sref = {}\n'.format(cfg['Sref']))
            file.write('h0 = {}\n'.format(cfg['h0']))
            file.write('t0 = {}\n'.format(cfg['t0']))
            file.write('ts = {}\n'.format([t for t in ts]))
            t0 = cfg['t0']

            file.write('\n# Engine Properties\n')
            if eng is None:
                file.write('eng_manufacturer = None\n')
                file.write('eng_model = None\n')
                file.write('Ts = None\n')
            else:
                eng = eng.Clone()
                eng.start(t0)
                Ts = []
                for t in ts:
                    Ts.append(eng.thrust(t))
                file.write('eng_manufacturer = {}\n'.format(repr(eng.manufacturer)))
                file.write('eng_model = {}\n'.format(repr(eng.model)))
                file.write('Ts = {}\n'.format([T for T in Ts]))

            file.write('\n# Cd Estimate Results\n')
            file.write('cds = {}\n'.format([cd for cd in cds]))
            file.write('Vs = {}\n'.format([V for V in Vs]))

    def load(self, filename=None):
        filename = self.__filename if filename is None else filename
        cfg = {}
        ts = []
        cds = []
        Vs = []
        Ts = []

        global_vars = {}
        local_vars = {}
        with open(filename) as src:
            code = compile(src.read(), filename, 'exec')
            exec(code, global_vars, local_vars)
            cfg['t0'] = local_vars['t0']
            cfg['m0'] = local_vars['m0']
            cfg['h0'] = local_vars['h0']
            cfg['Sref'] = local_vars['Sref']
            ts = local_vars['ts']
            cds = local_vars['cds']
            Vs = local_vars['Vs']
            Ts = local_vars['Ts']

        return cfg, ts, cds, Vs, Ts

    def load_dir(self, directory):
        cd_est_results = []
        for filename in os.listdir(directory):  
            path = os.path.join(directory, filename)
            results = self.load(path)
            cd_est_results.append(result)
        return cd_est_results


if __name__ == '__main__':
    random.seed(12345)

    estimation_output_dir = 'D:\Workspace\Rockets\HPR\Eiffel Tower v2\Cd Estimation Outputs'
    estimation_output_fmt = '{manufacturer}-{model}-{run}.cd_est'

    h0 = 1600
    m0 = 750 / 1000 # kg
    Sref = .155 * .155 # m^2

    N = 15
    impulse_sd_percent = 5
    burn_rate_sd_percent = 2.5
    thrustcurve_noise_sd_percent = 10
    est_percent_of_ascent_start = 0
    est_percent_of_ascent_end = 85

    debug = False
    engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_I65.rse"
    summary_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250503_I65\BBXI VC_summary_05-03-2025_09_45_29_.csv"
    low_rate_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250503_I65\BBXI VC LR_05-03-2025_09_45_29.csv"
    high_rate_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250503_I65\BBXI VC HR_05-03-2025_09_45_29.csv"

    # engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_J180.rse"
    # summary_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250517_J180\eiffeltowerj180\Eiffel Tower_summary_05-17-2025_10_36_05_.csv"
    # low_rate_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250517_J180\eiffeltowerj180\Eiffel Tower LR_05-17-2025_10_36_05.csv"
    # high_rate_path = r"D:\Workspace\Rockets\HPR\Eiffel Tower v2\Data\20250517_J180\eiffeltowerj180\Eiffel Tower HR_05-17-2025_10_36_05.csv"

    t_start = time.time_ns()    
    br_flight_log = loadBlueRavenLog(summary_path, low_rate_path, high_rate_path)
    t_new = time.time_ns() - t_start
    print('Parse Time = {:0.2f} ms'.format(t_new / 1e6))
    t_start = time.time_ns()
    br_flight_log.updateEvents(force=True)
    t_new = time.time_ns() - t_start
    print('Event Processing Time = {:0.2f} ms'.format(t_new / 1e6))

    print('Events:')
    for name, event in br_flight_log.events.items():
        print('\t{}'.format(event))
    engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_I65W.rse"
    ref_eng = engines.Engine.RSE(engine_path)

    if br_flight_log is not None:
        if debug:
            print(br_flight_log.summary)  
            plotFlightDebugData(br_flight_log)

        events = br_flight_log.events
        t_ignition = 0 if not 'Ignition' in events else events['Ignition'].t        
        t_liftoff = 0 if not 'Liftoff' in events else events['Liftoff'].t
        t_apogee = None if not 'Apogee' in events else events['Apogee'].t
            
        print('Liftoff at {} s, Apogee at {} s'.format(t_liftoff, t_apogee))
        ts = br_flight_log['t'].values
        t_end = t_liftoff + (est_percent_of_ascent_end * (t_apogee - t_liftoff) / 100.0)
        idx_end = np.searchsorted(ts, t_end, side='right')
        t_start = t_liftoff + (est_percent_of_ascent_start * (t_apogee - t_liftoff) / 100.0)
        idx_start = max(0, np.searchsorted(ts, t_start, side='left'))

        print('Start index = {} (t = {} s), End index = {} (t = {} s)'.format(idx_start, t_liftoff, idx_end, t_end))            
        ts = ts[idx_start:idx_end]
        azs = -1.0*br_flight_log['az'].values[idx_start:idx_end] - 9.80665
        azs_filt = signal.savgol_filter(azs, window_length=25, polyorder=3, delta=0.002)
        off_axis = br_flight_log['off_vertical'].values[idx_start:idx_end]        
        fig_az, axs_az = plt.subplots(2, layout='constrained', sharex=True)
        axs_az[0].plot(ts, azs, c='k', alpha=0.5)
        axs_az[0].plot(ts, azs_filt, c='g', alpha=0.5)
        axs_az[1].plot(ts, 57.3 * off_axis, c='k', alpha=0.5)

        Vzs = br_flight_log['Vz'].values[idx_start:idx_end]
        Vhs = br_flight_log['Vh'].values[idx_start:idx_end]
        hs = br_flight_log['h'].values[idx_start:idx_end]
        rhos = ambiance.Atmosphere(hs+h0).density

        fig_cd, ax_cd = plt.subplots(1, layout='constrained')
        run_results = []
        for idx in range(N):
            eng = ref_eng.Scaled(impulse_multiplier = random.gauss(1, impulse_sd_percent/100.0), burn_rate_multiplier = random.gauss(1, burn_rate_sd_percent/100.0), noise_sd=random.gauss(0, thrustcurve_noise_sd_percent/100))
            eng.start(t_ignition)
            ts, Vzs_ref, cds, ms_tot, as_eff, Ts = run_cd_est(ts, azs_filt, Vzs, Vhs, rhos, off_axis, m0, eng, Sref) 
            sns.scatterplot(x=Vzs_ref, y=cds, alpha=0.5, ax=ax_cd)

            run_results.append( (ts, Vzs_ref, cds, ms_tot, as_eff, Ts, eng) )

        output_handler = CdEstFileHandler()
        output_cfg = {
            'm0': m0,
            'Sref': Sref,
            'h0': h0,
            't0': t_ignition
        }

        for idx, result in enumerate(run_results):
            ts, Vzs_ref, cds, _, _, Ts, eng = result
          
            output_path = os.path.join(estimation_output_dir, estimation_output_fmt.format(
                        manufacturer=eng.manufacturer,
                        model = eng.model,
                        run = idx
                    )
                )

            output_handler.save(output_cfg, ts, cds, Vzs_ref, eng=eng, filename=output_path)

        input_handler = CdEstFileHandler()
        est_cd_results = input_handler.load_dir(estimation_output_dir)

        print(est_cd_results)

        def cd_func(x, a, b, c, d, e):
            return a*x**3 + b*x**2 + c*x**1 + d*x + e

        xdata = np.concatenate([result[1] for result in run_results])
        ydata = np.concatenate([result[2] for result in run_results])
        popt, pcov = optimize.curve_fit(cd_func, xdata, ydata)
        print(popt)

        est_cds = [cd_func(Vz, *popt) for Vz in Vzs]
        sns.lineplot(x=Vzs, y=est_cds, ax=ax_cd)

        def vToCd_basic(V):
            return min(50, max(5, cd_func(V, *popt)))

        dt = 0.002
        h0 = 1600
        vToCd = vToCd_basic

        engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_I65W.rse"        
        #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_K185W.rse"
        # engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_J180T.rse"
        #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\Cesaroni_699J145-19A.rse"

        eng = engines.Engine.RSE(engine_path)        
        t_ignition = 0 if not 'Ignition' in events else events['Ignition'].t        

        idx_start = np.searchsorted(ts, t_ignition, side='left')
        off_axis = None #br_flight_log['off_vertical'].values[idx_start:] 
        ts, azs, Vzs, hs, Ts, Ds, cds, alphas = sim_3d_flight(dt, t_ignition, eng, m0, Sref, vToCd, h0=h0, off_axis=off_axis)
        t_apogee = 0 if not 'Apogee' in events else events['Apogee'].t 
        print('Time of Apogee = {} s'.format(t_apogee))       
        idx_end = np.searchsorted(br_flight_log['t'].values, t_apogee, side='right')
        ts_ref = br_flight_log['t'].values[idx_start:idx_end]
        azs_ref = br_flight_log['az'].values[idx_start:idx_end]
        Vzs_ref = br_flight_log['Vz'].values[idx_start:idx_end]
        hs_ref = br_flight_log['h'].values[idx_start:idx_end]

        fig, axs = plt.subplots(3, layout='constrained')
        axs[0].plot(ts, azs+9.80665, c='g')
        axs[0].plot(ts_ref, -azs_ref, c='k', alpha=0.5)
        # ax0 = axs[0].twinx()
        # ax0.plot(ts, Ts, c='r')
        # ax0.plot(ts, Ds, c='b')
        axs[1].plot(ts, Vzs, c='g')
        axs[1].plot(ts_ref, Vzs_ref, c='k', alpha=0.5)
        axs[2].plot(ts, hs-h0, c='g')
        axs[2].plot(ts_ref, hs_ref, c='k', alpha=0.5)
        # axs[3].plot(ts, cds, c='g')
        # axs[4].plot(ts, 57.3*alphas, c='k')
        h_max = np.max(hs-h0)
        v_max = np.max(Vzs)
        print('Maximum Altitude = {} m, {} ft'.format(h_max, 3.28*h_max))
        print('Maximum Velocity = {} m/s, {} ft/s'.format(v_max, 3.28*v_max))
        plt.show()
