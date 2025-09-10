
import numpy as np
import scipy.signal as signal

import pyrse.flight_data as flight_data

class BlenderResampler:
    def __init__(self):
        pass

    def export_csv(self, filename, dataset, eng=None, fps=30, t_pre=0, t_delay=0):
        success = False
        output_dataset = self.resample(dataset, eng, fps, t_pre, t_delay)
        with open(filename, 'w') as file:
            file.write('t,dx,dy,theta,T,burn,smoke\n')
            for t, dx, dy, theta, T, burn, smoke in zip(output_dataset['t'], output_dataset['dx'], output_dataset['dy'], output_dataset['theta'], output_dataset['T'], output_dataset['burn'], output_dataset['smoke']):
                file.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n'.format(t, dx, dy, theta, T, burn, smoke))
            success = True
        return success, output_dataset
    
    def resample(self, dataset, eng, fps=30, t_pre=0, t_delay=0, smoothing_window_multiplier=5, min_burning=0.25):
        t_max = np.max(dataset['t'].values)
        dataset.updateEvents()
        t_ignition = 0
        try:
            t_ignition = dataset.events['Ignition'].t
        except:
            pass

        ref_eng = None
        if eng is not None:
            ref_eng = eng.Scaled()
            ref_eng.start(0)
        dt = 1 / float(fps)
        N_pre = int(t_pre * fps)
        t = -N_pre * dt
        dx = 0
        ts = []
        dxs = []
        dys = []
        thetas = []
        burning = []
        smoking = []
        Ts = None if ref_eng is None else []
        t_burntime = None if ref_eng is None else ref_eng.burn_time

        # find the values at the first sample
        values = dataset.values_at(t_ignition)
        h0 = values['h']

        # Fill the N_pre copies
        for _ in range(N_pre):
            ts.append(t)
            dxs.append(0)
            dys.append(0)
            thetas.append(values['off_vertical'])
            Ts.append(0)
            burning.append(0)
            smoking.append(0)
            t += dt

        # Fill from t_ignition to the end with the nearest samples to passed fps
        # ts, dxs, dys, Ts, thetas, burning, smoking
        t_sample = t_ignition
        while t_sample < t_max:
            values = dataset.values_at(t_sample)
            ts.append(t)
            dxs.append(dx)
            dys.append(max(0, values['h'] - h0))
            thetas.append(values['off_vertical'])
            Ts.append(0 if ref_eng is None else ref_eng.thrust(t))
            burning.append(1 if 0 <= t < t_burntime else 0)
            smoking.append(1 if 0 <= t < t_burntime + t_delay else 0)
            dx += dt * values['Vh']
            t += dt
            t_sample += dt
            
        # Smooth data and handle value constraints
        dys_smooth = signal.savgol_filter(np.array(dys), window_length=int(smoothing_window_multiplier*fps), polyorder=3, delta=dt)
        dys_smooth[dys_smooth < 0] = 0
        Ts_smooth = signal.savgol_filter(np.array(Ts), window_length=int(smoothing_window_multiplier*fps), polyorder=3, delta=dt)
        Ts_smooth[Ts_smooth < 0] = 0

        # Calculate scaled burning rate from thrust with offset correction
        burning = np.array(burning)
        burning = (burning * ((1 - min_burning) * Ts_smooth / np.max(Ts_smooth))) + (burning * min_burning)

        return {
            't': np.array(ts),
            'dx': signal.savgol_filter(np.array(dxs), window_length=int(smoothing_window_multiplier*fps), polyorder=3, delta=dt),
            'dy': dys_smooth,
            'theta': signal.savgol_filter(np.array(thetas), window_length=int(smoothing_window_multiplier*fps), polyorder=3, delta=dt),
            'T': Ts_smooth,
            'burn': burning,
            'smoke': np.array(smoking)
        }




