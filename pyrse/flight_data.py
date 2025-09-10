
import math
import os
import os.path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import scipy.interpolate as interpolate
import scipy.signal as signal

from pyrse import flight_data_events
from pyrse import flight_data_filters
from pyrse import quaternion
from pyrse import vector3d

#
# The module depends upon a number of standard modules including:
#   * numpy - Numpy arrays are used for internal storage in the FlightData class to ensure high performance
#   * pandas - Pandas is used primarily to ease parsing of CSV type files
#
# The FlightData class is intended to be a superset of aa variety of model rocket data formats. Parsers are available for
# soureces such as flight computrers (Altus Metrum, FeatherwWeight, etc.) as well as flight simulators (RockSim, OpenRocket,
# and RASAero) By converting multiple files into a single format, it becomes possible to more directly compare a combination
# of flight records and simulations.
#

# The minimal set of data that a FlightData class contains is:
#   * t - sample times
#   * h - altitude (pad relative)
#   * Vz - velocity (axial)
#   * az - acceleration (axial)

# Beyond the base storage type, the module contains a number of useful utility functions for handling flight data. These incllude
# operations such as:
#   * alignment of flight data
#   * combination of flight data
#
# The related module -  flight_plotting - allows for creating advanced plots of imported flight data.
#  

class FlightDataUnitConverter:
    __unit_map = {
        'm' : 1.0, 'Feet' : 1 / 3.28,
        'Miles / Hour' : 0.44704, 'mph' : 0.44704, 'fps' : 0.3048,
        '°' : 1 / 57.2958, 'deg' : 1 / 57.2958, 'Degrees' : 1 / 57.2958, 'rad' : 1.0, 'Radians' : 1.0,
        '°/s' : 1 / 57.2958, 'Radians / Second' : 1.0, 'rad/s' : 1.0, 'Degrees / Second' : 1 / 57.2958, 'deg/s' :  1 / 57.2958,
        'Gees' : 9.80665, 'm/s²' : 1.0, 'm/s^2': 1.0, 'ft/s^2' : 0.3048,
        'N' : 1.0, 'lbs' : 4.44822,
        'g' : 1 / 1000, 'Grams' : 1 / 1000, 'kg' : 1.0
    }

    @classmethod
    def standard_scale(cls, src):
        try:
            return FlightDataUnitConverter.__unit_map[src]
        except:
            return 1.0

    @classmethod
    def convert_to_standard(cls, val, src):
        return FlightDataUnitConverter.standard_scale(src) * val


class FlightData(dict):
    valid_sources = ['Measured', 'Calculated', 'Unknown']

    class Series:

        def __init__(self, parent, name, values = None, errors=None, interpolation_mask=None, source='Unknown', description=None, ts=None, dx=None, ddx=None):
            self.__parent = parent
            self.__name = name
            self.__values = np.array(values) if values is not None else None
            self.__errors = np.array(errors) if errors is not None else None
            self.__interpolation_mask = np.array(interpolation_mask, dtype=bool) if interpolation_mask is not None else None
            self.__source = source            
            self.__description = description
            self.__ts = ts
            self.__dx = dx
            self.__ddx = ddx

# class Series:
    # def __init__(self, times: np.ndarray, values: np.ndarray,
    #              first_derivative: np.ndarray = None,
    #              second_derivative: np.ndarray = None):
    #     self.times = np.asarray(times)
    #     self.values = np.asarray(values)
    #     self.first_derivative = np.asarray(first_derivative) if first_derivative is not None else None
    #     self.second_derivative = np.asarray(second_derivative) if second_derivative is not None else None

    #     # Sort by time if necessary
    #     if np.any(np.diff(self.times) < 0):
    #         idx = np.argsort(self.times)
    #         self.times = self.times[idx]
    #         self.values = self.values[idx]
    #         if self.first_derivative is not None:
    #             self.first_derivative = self.first_derivative[idx]
    #         if self.second_derivative is not None:
    #             self.second_derivative = self.second_derivative[idx]

    # def at(self, t: float) -> float:
    #     """
    #     Interpolate the series at time t using value, first derivative, and optionally second derivative.
    #     """
    #     # locate interval
    #     if t <= self.times[0]:
    #         idx = 0
    #     elif t >= self.times[-1]:
    #         idx = -2
    #     else:
    #         idx = np.searchsorted(self.times, t) - 1

    #     t0, t1 = self.times[idx], self.times[idx + 1]
    #     y0, y1 = self.values[idx], self.values[idx + 1]
    #     dt = t1 - t0
    #     alpha = (t - t0) / dt

    #     # estimate first derivative if missing
    #     if self.first_derivative is not None:
    #         dy0 = self.first_derivative[idx]
    #         dy1 = self.first_derivative[idx + 1]
    #     else:
    #         dy0 = (y1 - y0) / dt
    #         dy1 = dy0

    #     # check if second derivative is available
    #     if self.second_derivative is not None:
    #         ddy0 = self.second_derivative[idx]
    #         ddy1 = self.second_derivative[idx + 1]
    #         # Hermite cubic with acceleration (Taylor expansion at endpoints)
    #         h00 = 1 - 3*alpha**2 + 2*alpha**3
    #         h10 = alpha - 2*alpha**2 + alpha**3
    #         h20 = 0.5*(alpha**2 - alpha**3)
    #         h01 = 3*alpha**2 - 2*alpha**3
    #         h11 = -alpha**2 + alpha**3
    #         h21 = 0.5*(alpha**2 - alpha**3)

    #         y_interp = (h00*y0 + h10*dt*dy0 + h20*dt**2*ddy0 +
    #                     h01*y1 + h11*dt*dy1 + h21*dt**2*ddy1)
    #     else:
    #         # standard cubic Hermite using value and first derivative
    #         h00 = 2*alpha**3 - 3*alpha**2 + 1
    #         h10 = alpha**3 - 2*alpha**2 + alpha
    #         h01 = -2*alpha**3 + 3*alpha**2
    #         h11 = alpha**3 - alpha**2
    #         y_interp = h00*y0 + h10*dt*dy0 + h01*y1 + h11*dt*dy1

    #     return y_interp

        def __bool__(self):
            return self.valid

        def __len__(self):
            return len(self.__values)

        def __str__(self):
            source = self.__source
            description = self.__description
            if description is None:
                return 'FlightData.Series({}[{}] ({}))'.format(self.__name, len(self.__values), source)
            return 'FlightData.Series({}[{}] ({}) - {})'.format(self.__name, len(self.__values), source, description)   

        @property
        def name(self):
            return self.__name

        @property
        def valid(self):
            if (not self.__source in FlightData.valid_sources) or (self.__values is None):
                return False 
            if self.__errors is None:
                return True
            return len(self.__values) == len(self.__errors)

        @property
        def values(self):
            return self.__values # TODO: THIS SHOULD RETURN A CLONE

        @values.setter
        def values(self, vals):
            self.__values = np.array(vals)

        @property
        def errors(self):
            return self.__errors # TODO: THIS SHOULD RETURN A CLONE

        @errors.setter
        def errors(self, errs):
            self.__errors = np.array(errs)

        @property
        def interpolation_mask(self):
            return self.__interpolation_mask # TODO: THIS SHOULD RETURN A CLONE

        @interpolation_mask.setter
        def interpolation_mask(self, mask):
            self.__interpolation_mask = np.array(mask, dtype=bool)

        @property
        def source(self):
            return self.__source
        
        @property
        def ts(self):
            return self.__ts

        @ts.setter
        def ts(self, new_ts):
            if isinstance(new_ts, FlightData.Series):
                self.__ts = new_ts
            return self.__ts
        
        @property
        def dx(self):
            return self.__dx

        @dx.setter
        def dx(self, new_dx):
            if isinstance(new_dx, FlightData.Series):
                self.__dx = new_dx
            return self.__dx

        @property
        def ddx(self):
            return self.__ddx

        @ddx.setter
        def ddx(self, new_ddx):
            if isinstance(new_ddx, FlightData.Series):
                self.__ddx = new_ddx
            return self.__ddx
                        
        def set_parent(self, parent):
            self.__parent = parent

        def at(self, t: float) -> float:
            """
            Interpolate the series at time t using value, first derivative, and optionally second derivative.
            """
            ts = self.__ts.values
            dxs = self.__dx
            ddxs = self.__ddx
            values = self.__values

            # locate interval
            if t <= ts[0]:
                idx = 0
            elif t >= ts[-1]:
                idx = -2
            else:
                idx = np.searchsorted(ts, t) - 1

            t0, t1 = ts[idx], ts[idx + 1]
            y0, y1 = values[idx], values[idx + 1]
            dt = t1 - t0
            alpha = (t - t0) / dt

            # estimate first derivative if missing
            if dxs is not None:
                dy0 = dxs.values[idx]
                dy1 = dxs.values[idx + 1]
            else:
                dy0 = (y1 - y0) / dt
                dy1 = dy0

            # check if second derivative is available
            if ddxs is not None:
                ddy0 = ddxs.values[idx]
                ddy1 = ddxs.values[idx + 1]
                # Hermite cubic with acceleration (Taylor expansion at endpoints)
                h00 = 1 - 3*alpha**2 + 2*alpha**3
                h10 = alpha - 2*alpha**2 + alpha**3
                h20 = 0.5*(alpha**2 - alpha**3)
                h01 = 3*alpha**2 - 2*alpha**3
                h11 = -alpha**2 + alpha**3
                h21 = 0.5*(alpha**2 - alpha**3)

                y_interp = (h00*y0 + h10*dt*dy0 + h20*dt**2*ddy0 +
                            h01*y1 + h11*dt*dy1 + h21*dt**2*ddy1)
            else:
                # standard cubic Hermite using value and first derivative
                h00 = 2*alpha**3 - 3*alpha**2 + 1
                h10 = alpha**3 - 2*alpha**2 + alpha
                h01 = -2*alpha**3 + 3*alpha**2
                h11 = alpha**3 - alpha**2
                y_interp = h00*y0 + h10*dt*dy0 + h01*y1 + h11*dt*dy1
            return y_interp

        def error_at(self, t, extrapolate=False):
            pass

    class Summary:
        def __init__(self):
            self.keys = []
            self.files = None
            self.apogee_altitude = 0
            self.max_velocity = 0
            self.max_acceleration = 0
            self.t_range = None

        def __str__(self):
            t_min = None if self.t_range is None else self.t_range[0]
            t_max = None if self.t_range is None else self.t_range[1]
            return 'FlightData.Summary(keys={}, t = [{}, {}], max(h) = {:0.1f} m, max(v) = {:0.1f} m/s, max(a) = {:0.1f} m/s^2)'.format(self.keys, t_min, t_max, self.apogee_altitude, self.max_velocity, self.max_acceleration)

    class Event:
        predefined_events = ['Ignition', 'Launch', 'Liftoff', 'Cleared Guide', 'Burnout', 'Preapogee', 'Apogee', 'Charge', 'Deployment', 'Ground Hit']
        predefined_colors = {'Ignition':'r', 'Launch':'c', 'Liftoff':'g', 'Cleared Guide':'c', 'Burnout':'g', 'Preapogee':'c', 'Apogee':'g', 'Charge':'r', 'Deployment':'c', 'Ground Hit':'g'}
        default_color = 'k'

        def __init__(self, name, t, fixed=False):
            self.__name = name
            self.__t = t
            self.__fixed = fixed

        def __str__(self):
            return 'Event({}) @ {:0.3f} s'.format(self.__name, self.__t)

        @property
        def name(self):
            return self.__name

        @property
        def t(self):
            return self.__t

        @property
        def fixed(self):
            return self.__fixed

        @property
        def predefined(self):
            return self.__name in FlightData.Event.predefined_events

        @property
        def color(self):
            if self.__name in FlightData.Event.predefined_colors:
                return FlightData.Event.predefined_colors[self.__name]
            return FlightData.Event.default_color

    def __init__(self):
        super().__init__()
        self.__files = None
        self.__events = {}

    def __getitem__(self, k):

        # TODO: ASSEMBLE QUATERNIONS
        # TODO: ASSEMBLE ACCEL VECTORS
        # TODO: ASSEMBLE WORLD ACCEL VECTORS
        # TODO: ASSEMBLE GYRO VECTORS        

        if k == 'a': # Get accelerations in world coordinates
            if all([kv in self for kv in ['ax', 'ay', 'az']]):
                axs = super().__getitem__('ax').values
                ays = super().__getitem__('ay').values
                azs_series = super().__getitem__('az')
                azs = azs_series.values
                a_mask = azs_series.interpolation_mask
                a_src = azs_series.source
                accels = [vector3d.Vector3D(x, y, z) for x, y, z in zip(axs, ays, azs)]
                return FlightData.Series(self, 'a', accels, interpolation_mask=a_mask, source=a_src, description='Vector objects representation the accelerations in world coordinates')

        elif k == 'a_body': # Get accelerations in body coordinates
            pass
        elif k == 'g': # Get rotation rates in world coordinates
            if all([kv in self for kv in ['gx', 'gy', 'gz']]):
                gxs = super().__getitem__('gx').values
                gys = super().__getitem__('gy').values
                gzs_series = super().__getitem__('gz')
                gzs = gzs_series.values
                g_mask = gzs_series.interpolation_mask
                g_src = gzs_series.source
                gyros = [vector3d.Vector3D(x, y, z) for x, y, z in zip(gxs, gys, gzs)]
                return FlightData.Series(self, 'g', gyros, interpolation_mask=g_mask, source=g_src, description='Vector objects representation the rotation rates in world coordinates')
        elif k == 'g_body': # Get rotation rates in body coordinates
            pass
        elif k == 'q': # Get orientation quaternions
            if all([kv in self for kv in ['qw', 'qx', 'qy', 'qz']]):
                qws = super().__getitem__('qw').values
                qxs = super().__getitem__('qx').values
                qys = super().__getitem__('qy').values
                qzs_series = super().__getitem__('qz')
                qzs = qzs_series.values
                q_mask = qzs_series.interpolation_mask
                q_src = qzs_series.source
                qs = [quaternion.Quaternion(w, x, y, z) for w, x, y, z in zip(qws, qxs, qys, qzs)]
                return FlightData.Series(self, 'q', qs, interpolation_mask=q_mask, source=q_src, description='Quaternion objects representation the rotation from body coordinates into world coordinates')
        else:
            return super().__getitem__(k)
        raise KeyError('FlightData series, {}, unknown'.format(k))
            
    def __setitem__(self, k, data):
        new_series = None
        if isinstance(data, FlightData.Series):
            # print('Add Series Directly')
            new_series = data
            new_series.set_parent(self)
        elif isinstance(data, dict):
            # print('Create series from Dictionary')
            values = data['values']
            errors = data['errors'] if 'errors' in data else None
            interpolation_mask = data['interpolation_mask'] if 'interpolation_mask' in data else None
            description = data['description'] if 'description' in data else None
            source = data['source'] if 'source' in data else 'Unknown'
            new_series = FlightData.Series(self, k, values=values, errors=errors, interpolation_mask=interpolation_mask, source=source, description=description)
        else: # If data is just a list of values, create a series directly.
            # print('Create Series')
            new_series = FlightData.Series(self, k, data)

        super().__setitem__(k, new_series)

    @property
    def files(self):
        return self.__files

    @files.setter
    def files(self, files):
        self.__files = files

    @property
    def summary(self):
        result = FlightData.Summary()
        result.keys = list(self.keys())
        result.files = list(self.__files)
        if 't' in self:
            result.t_range = (np.min(self['t'].values), np.max(self['t'].values))
        result.apogee_altitude = np.max(self['h'].values)
        result.max_velocity = np.max(self['Vz'].values)
        result.max_acceleration = np.max(self['az'].values)
        return result

    @property
    def events(self):
        return self.__events

    def values_at(self, t_sample):
        #idx = 0
        ts = self['t'].values        
        # for t in ts:
        #     if t > t_sample:
        #         break
        #     idx += 1
        idx = np.searchsorted(ts, t_sample, side='left')        
        values = {}
        for k, s in self.items():
            if not k == 't':
                values[k] = s.values[idx]
        return values

    def addEvents(self, events, force=False):
        for evt_def in events:
            name = None
            t = None
            fixed = False
            if isinstance(evt_def, FlightData.Event):
                name, t = evt_def.name, evt_def.t
            elif len(evt_def) == 2:
                name, t = evt_def
            elif len(evt_def) == 3:
                name, t, fixed = evt_def
            
            if name is not None:
                update = True
                if name in self.__events.keys():
                    update = force or (not self.__events[name].fixed)
                if update:
                    self.__events[name] = FlightData.Event(name, t, fixed=fixed)

    def updateEvents(self, force=True):
        liftoff_detector = flight_data_events.FlightDataLiftoffDetector()
        burnout_detector = flight_data_events.FlightDataBurnoutDetector()
        preapogee_detector = flight_data_events.FlightDataPreapogeeDetector()
        apogee_detector = flight_data_events.FlightDataApogeeDetector()
        # TODO: CREATE A LANDING DETECTOR
        events = []

        ts = self['t'].values
        hs = self['h'].values
        Vzs = self['Vzraw'].values # TODO: CHECK HOW THE FILTERED (INTERPOLATED) VALUES ARE CALCULATED....
        use_quaternions = ('qw' in self)
        azs = self['az'].values
        if use_quaternions: # TODO: CALCULATE THESE FROM THE AX/AY/AZ VALUES AND QW/QX/QY/QZ DATA, IF AVAILABLE....
            vs = []
            axs = self['ax'].values
            ays = self['ay'].values
            for ax, ay, az in zip(axs, ays, azs): # TODO: ADD "ACCELERATION VECTORS" PROPERTY
                vs.append(vector3d.Vector3D(ax, ay, az))
            vs = np.array(vs)

            qs = []
            qws = self['qw'].values
            qxs = self['qx'].values
            qys = self['qy'].values
            qzs = self['qz'].values
            for qw, qx, qy, qz in zip(qws, qxs, qys, qzs):
                qs.append(quaternion.Quaternion(qw, qx, qy, qz))
            qs = np.array(qs)
            azs_rot = []
            for v, q in zip(vs, qs):
                azs_rot.append(-quaternion.rotate_vector(q, v).z)
        azs_rot = np.array(azs_rot) # TODO: THIS IS JUST THE WORLD COORDINATE ACCELERATIONS....
        # fig, axs = plt.subplots(2, layout='constrained', sharex=True)
        # axs[0].plot(ts, azs)
        # axs[0].plot(ts, azs_rot)
        # axs[1].plot(ts, Vzs)

        t_last = ts[0]
        for t, h, Vz, az in zip(ts, hs, Vzs, azs_rot):
            az -= 9.80665
            #print('t = {}, h = {}, Vz = {}, az = {}'.format(t, h, Vz, az))
            dt = t - t_last
            liftoff_event = liftoff_detector(dt, az)
            if liftoff_event.detected:
                events.append(FlightData.Event('Launch', t))
                events.append(FlightData.Event('Ignition', t - liftoff_event.t)) # TODO: ADD THE ABILITY TO TRACK AIRSTART EVENTS
                burnout_detector.start()
                preapogee_detector.start()
                apogee_detector.start()
            burnout_event = burnout_detector(dt, az)
            if burnout_event.detected:
                events.append(FlightData.Event('Burnout', t)) # TODO: ADD AN ID TO EVENTS TO ALLOW FOR MULTIPLE IGNITION AND BURNOUT EVENTS                
            preapogee_event = preapogee_detector(dt, Vz)
            if preapogee_event.detected:
                events.append(FlightData.Event('Preapogee', t))
            apogee_event = apogee_detector(dt, Vz, h)
            if apogee_event.detected:
                events.append(FlightData.Event('Apogee', t))
                burnout_detector.stop()
                preapogee_detector.stop()
                # TODO: START THE LANDING DETECTOR
            t_last = t
        
        self.addEvents(events, force=force)
        # for evt in events:
        #     print(evt)


# def calcDerivedVelAccel(ts, hs, filter_hs=True, accel_max=490, debug=False):
#     """
#     Filter altitudes that violate max acceleration, then compute velocity and
#     acceleration with Savitzky–Golay smoothing/derivatives.

#     Parameters
#     ----------
#     ts : np.ndarray
#         1D array of times (s), approximately uniform sampling.
#     hs : np.ndarray
#         1D array of altitudes (m).
#     filter_hs : bool, default=True
#         If True, replace altitude samples that imply |a| > accel_max.
#     accel_max : float, default=490
#         Maximum plausible acceleration (m/s^2).

#     Returns
#     -------
#     hs_sg : np.ndarray
#         Smoothed altitude (m).
#     vs_sg : np.ndarray
#         Smoothed velocity from SG derivative (m/s).
#     accs_sg : np.ndarray
#         Smoothed acceleration from SG second derivative (m/s^2).
#     extrap_used : bool
#         True if any end samples required extrapolation during validity fix-up.
#     sg_info : dict
#         {'window_length': int, 'polyorder': int, 'dt': float, 'fs': float}
#     """
#     ts = np.asarray(ts, dtype=float)
#     hs = np.asarray(hs, dtype=float)
#     if ts.ndim != 1 or hs.ndim != 1 or len(ts) != len(hs):
#         raise ValueError("ts and hs must be 1D arrays of equal length")
#     n = len(ts)
#     if n < 7:
#         raise ValueError("Need at least 7 samples for stable Savitzky–Golay smoothing.")

#     # --- sampling stats (SG assumes roughly uniform dt)
#     dts = np.diff(ts)
#     dt = np.median(dts)
#     fs = 1.0 / dt
#     # warn (soft) if irregular timing; we still proceed using median dt
#     if np.std(dts) / dt > 0.05:
#         # You might want to resample to a uniform grid upstream.
#         pass

#     # --- quick finite-difference pass to find impossible |a| > accel_max
#     v0 = np.gradient(hs, ts)
#     a0 = np.gradient(v0, ts)
#     invalid = np.abs(a0) > accel_max

#     hs_filt = hs.copy()
#     extrap_used = False
#     if filter_hs and np.any(invalid):
#         valid = ~invalid
#         vidx = np.flatnonzero(valid)
#         if vidx.size == 0:
#             # fallback: straight line fit if everything looked invalid
#             coeffs = np.polyfit(ts, hs, 1)
#             hs_filt = np.polyval(coeffs, ts)
#             extrap_used = True
#         else:
#             f = interpolate.interp1d(ts[valid], hs[valid], kind="linear", fill_value="extrapolate", assume_sorted=True)
#             bad_idx = np.flatnonzero(invalid)
#             hs_filt[bad_idx] = f(ts[bad_idx])
#             if np.any(bad_idx < vidx[0]) or np.any(bad_idx > vidx[-1]):
#                 extrap_used = True

#         # optional: one more finite-difference pass after repair (not SG yet)
#         v0 = np.gradient(hs_filt, ts)
#         a0 = np.gradient(v0, ts)

#     # ---------------------------------------------------------------------
#     # Automatic Savitzky–Golay window selection
#     #
#     # Idea:
#     #   - Let J be a robust estimate of |jerk| = |da/dt| from repaired data.
#     #   - Real acceleration changes on timescale tau ~ accel_scale / J.
#     #   - Choose SG cutoff f_c ~ 1/(2π tau).
#     #   - For SG (p=2..3), an effective rule-of-thumb: f_c ≈ k_p * fs / N,
#     #     so N ≈ k_p * fs / f_c. We use k_p ≈ 0.6 and p=3 by default.
#     #   - Then round N to an odd integer within [p+2, n-1].
#     #
#     # Notes:
#     #   - If jerk is tiny (coasting), window grows; clamp to a reasonable max.
#     #   - If jerk is large (aggressive maneuvers), window shrinks toward min.
#     # ---------------------------------------------------------------------

#     def _odd_clamp(x, lo, hi):
#         x = int(np.clip(int(round(x)), lo, hi))
#         # make odd
#         if x % 2 == 0:
#             x = min(x + 1, hi) if x < hi else max(x - 1, lo)
#         return x

#     # Robust jerk estimate from the repaired signal
#     v_est = np.gradient(hs_filt, ts)
#     a_est = np.gradient(v_est, ts)
#     j_est = np.gradient(a_est, ts)
#     J = np.percentile(np.abs(j_est), 90)  # robust scale
#     if not np.isfinite(J) or J <= 0:
#         # fallback: relate jerk to accel_max over a few samples
#         J = accel_max * fs  # "accel_max change over ~1 s" conservative fallback

#     # Set a "tolerable" fractional acceleration change across the window.
#     # Smaller alpha -> larger window. alpha ~ 0.2 is a good default.
#     alpha = 0.2
#     tau = (alpha * accel_max) / J  # seconds
#     tau = np.clip(tau, 3 * dt, 10.0)  # keep within reasonable [3*dt, 10 s] span

#     fc = 1.0 / (2.0 * np.pi * tau)   # Hz
#     kp = 0.6                         # SG order 3 constant ~0.55–0.65 works well
#     N_target = kp * fs / max(fc, 1e-6)

#     polyorder = 3
#     N = _odd_clamp(N_target, polyorder + 2, max(polyorder + 2, n - (n + 1) % 2))

#     # If the data are very short, ensure N fits
#     N = min(N, n - 1 if (n - 1) % 2 == 1 else n - 2)
#     N = max(N, polyorder + 2)
#     if N % 2 == 0:  # ensure odd
#         N = max(N - 1, polyorder + 2)

#     # --- Final SG smoothing + derivatives
#     hs_sg  = signal.savgol_filter(hs_filt, window_length=N, polyorder=polyorder, deriv=0, delta=dt, mode="interp")
#     vs_sg  = signal.savgol_filter(hs_sg, window_length=2*N, polyorder=polyorder, deriv=1, delta=dt, mode="interp")
#     accs_sg = signal.savgol_filter(hs_sg, window_length=4*N, polyorder=polyorder, deriv=2, delta=dt, mode="interp")

#     sg_info = {"window_length": int(N), "polyorder": int(polyorder), "dt": float(dt), "fs": float(fs)}
#     print(sg_info) # HACK: THIS IS JUST FOR INITIAL DEBUGGING
#     if debug:
#         return hs_sg, vs_sg, accs_sg, extrap_used, invalid, sg_info
#     return hs_sg, vs_sg, accs_sg, extrap_used

# def calcDerivedVelAccel(ts, hs, filter_hs=True, accel_max=490):
#     """
#     Fix altitudes that violate |a| <= accel_max, then fit a cubic smoothing spline
#     to hs and return spline-based velocity and acceleration. The spline smoothness
#     is chosen from a noise estimate and (if needed) relaxed upward until the
#     resulting acceleration respects accel_max (with a small margin).
#     """
#     ts = np.asarray(ts, dtype=float)
#     hs = np.asarray(hs, dtype=float)
#     if ts.ndim != 1 or hs.ndim != 1 or len(ts) != len(hs):
#         raise ValueError("ts and hs must be 1D arrays of equal length")
#     n = len(ts)
#     if n < 7:
#         raise ValueError("Need at least 7 samples")

#     # --- step 1: repair impossible altitudes using finite-difference acceleration
#     v0 = np.gradient(hs, ts)
#     a0 = np.gradient(v0, ts)
#     invalid = np.abs(a0) > accel_max

#     hs_repaired = hs.copy()
#     extrap_used = False
#     if filter_hs and np.any(invalid):
#         valid = ~invalid
#         vidx = np.flatnonzero(valid)
#         if vidx.size == 0:
#             # fallback to straight line if everything looked invalid
#             coeffs = np.polyfit(ts, hs, 1)
#             hs_repaired = np.polyval(coeffs, ts)
#             extrap_used = True
#         else:
#             f = interpolate.interp1d(ts[valid], hs[valid], kind="linear", fill_value="extrapolate", assume_sorted=True)
#             bad = np.flatnonzero(invalid)
#             hs_repaired[bad] = f(ts[bad])
#             extrap_used = np.any(bad < vidx[0]) or np.any(bad > vidx[-1])

#     # --- step 2: estimate measurement noise on altitude (robust)
#     # Use median absolute deviation of first differences (accounts for drift)
#     dh = np.diff(hs_repaired)
#     if dh.size >= 5:
#         mad = np.median(np.abs(dh - np.median(dh)))
#         sigma_h = 1.4826 * mad / np.sqrt(2.0)  # per-sample noise std on h
#         if not np.isfinite(sigma_h) or sigma_h == 0:
#             sigma_h = np.std(hs_repaired) * 1e-3 + 1e-9
#     else:
#         sigma_h = np.std(hs_repaired) * 1e-3 + 1e-9

#     # --- step 3: fit a cubic smoothing spline with adaptive smoothness
#     # UnivariateSpline minimizes sum(w_i^2*(h_i - s(t_i))^2) + λ * ∫ (s''(t))^2 dt
#     # SciPy exposes this via "s" (the target residual sum). Start with s ≈ n*sigma_h^2,
#     # then increase until the acceleration obeys the bound (within margin).
#     base_s = n * (sigma_h ** 2)
#     s = base_s
#     margin = 1.10  # allow ~10% slack over accel_max to avoid over-smoothing
#     max_iters = 12

#     # weights are 1/sigma per point; if your noise varies per point, pass a vector instead
#     w = np.full(n, 1.0 / max(sigma_h, 1e-12))

#     # helper to fit & check acceleration bound
#     def fit_and_check(curr_s):
#         spl = interpolate.UnivariateSpline(ts, hs_repaired, w=w, s=curr_s, k=3)
#         a = spl.derivative(2)(ts)
#         return spl, a

#     spl, acc = fit_and_check(s)
#     # If the accel is still too spiky, progressively increase smoothness
#     it = 0
#     while np.nanpercentile(np.abs(acc), 99) > margin * accel_max and it < max_iters:
#         s *= 2.5  # multiplicative bump; converges quickly
#         spl, acc = fit_and_check(s)
#         it += 1

#     # --- outputs
#     hs_s = spl(ts)
#     vs_s = spl.derivative(1)(ts)
#     accs_s = acc  # already computed

#     return hs_s, vs_s, accs_s, extrap_used

import numpy as np
from scipy.interpolate import interp1d

def calcDerivedVelAccel(ts, hs, filter_hs=True, accel_max=980, debug=True):
    """
    Repairs impossible altitudes using an acceleration cap, then estimates
    altitude, velocity, and acceleration with a [h, v, a] Kalman RTS smoother
    driven by white jerk. Robust Huber weighting reduces oscillations from
    outliers. The jerk spectral density q is adapted so |a| stays plausible.
    
    Returns
    -------
    h_s : np.ndarray
    v_s : np.ndarray
    a_s : np.ndarray
    extrap_used : bool
    """
    ts = np.asarray(ts, float)
    hs = np.asarray(hs, float)
    if ts.ndim != 1 or hs.ndim != 1 or len(ts) != len(hs):
        raise ValueError("ts and hs must be 1D arrays of equal length")
    n = len(ts)
    if n < 5:
        raise ValueError("Need at least 5 samples")

    # ---------- Phase 1: quick repair of impossible accelerations ----------
    v0 = np.gradient(hs, ts)
    a0 = np.gradient(v0, ts)
    invalid = np.abs(a0) > accel_max
    hs_rep = hs.copy()
    extrap_used = False
    if filter_hs and np.any(invalid):
        valid = ~invalid
        vidx = np.flatnonzero(valid)
        if vidx.size == 0:
            c = np.polyfit(ts, hs, 1)
            hs_rep = np.polyval(c, ts)
            extrap_used = True
        else:
            f = interp1d(ts[valid], hs[valid], kind="linear",
                         fill_value="extrapolate", assume_sorted=True)
            bad = np.flatnonzero(invalid)
            hs_rep[bad] = f(ts[bad])
            extrap_used = np.any(bad < vidx[0]) or np.any(bad > vidx[-1])

    # ---------- Phase 2: noise scale estimates ----------
    dt = np.median(np.diff(ts))
    # robust per-sample altitude noise
    dh = np.diff(hs_rep)
    if dh.size >= 5:
        mad = np.median(np.abs(dh - np.median(dh)))
        sigma_h = 1.4826 * mad / np.sqrt(2.0)
    else:
        sigma_h = np.std(hs_rep) * 1e-3
    sigma_h = max(sigma_h, 1e-6)
    R_base = sigma_h**2

    # ---------- State-space model (white jerk) ----------
    # x = [h, v, a]
    def F_dt(dt):
        return np.array([[1.0, dt, 0.5*dt*dt],
                         [0.0, 1.0, dt],
                         [0.0, 0.0, 1.0]])
    def Q_dt(dt, q):
        # Q = q * G G^T, G = [dt^3/6, dt^2/2, dt]^T
        g = np.array([dt**3/6.0, dt**2/2.0, dt])
        return q * np.outer(g, g)
    H = np.array([[1.0, 0.0, 0.0]])

    # ---------- Robust (Huber) weighting for measurements ----------
    def huber_weights(resid, scale, k=1.345):
        # scale ~ sigma; k*sigma is the transition
        a = k * scale
        w = np.ones_like(resid, float)
        mask = np.abs(resid) > a
        w[mask] = (a / (np.abs(resid[mask]) + 1e-12))
        return w

    # ---------- RTS smoother given q and per-step R multipliers ----------
    def rts_smoother(q, R_mult=None):
        # allocate
        x_f = np.zeros((n, 3))
        P_f = np.zeros((n, 3, 3))
        x_p = np.zeros((n, 3))
        P_p = np.zeros((n, 3, 3))
        # init: use first two samples to set rough v; neutral a
        x = np.array([hs_rep[0], (hs_rep[1]-hs_rep[0])/(ts[1]-ts[0]+1e-12), 0.0])
        P = np.diag([sigma_h**2, (sigma_h/dt)**2, (accel_max/5.0)**2])  # fairly loose on a

        for k in range(n):
            if k == 0:
                F = np.eye(3)
                Q = np.zeros((3,3))
                x_pred, P_pred = x, P
            else:
                dt_k = ts[k]-ts[k-1]
                F = F_dt(dt_k)
                Q = Q_dt(dt_k, q)
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
            # measurement update with robust weight
            z = hs_rep[k]
            Rk = R_base * (1.0 if R_mult is None else R_mult[k])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + Rk
            K = (P_pred @ H.T) / (S + 1e-12)
            x = x_pred + (K * y).ravel()
            P = (np.eye(3) - K @ H) @ P_pred
            x_p[k], P_p[k] = x_pred, P_pred
            x_f[k], P_f[k] = x, P

        # RTS backward pass
        x_s = x_f.copy()
        P_s = P_f.copy()
        for k in range(n-2, -1, -1):
            dt_k = ts[k+1]-ts[k]
            F = F_dt(dt_k)
            C = P_f[k] @ F.T @ np.linalg.pinv(P_p[k+1] + 1e-12*np.eye(3))
            x_s[k] = x_f[k] + C @ (x_s[k+1] - x_p[k+1])
            P_s[k] = P_f[k] + C @ (P_s[k+1] - P_p[k+1]) @ C.T
        return x_s, P_s

    def compute_kalman_diagnostics(ts, hs_meas, hs_rep, h_s, v_s, a_s, extrap_used,
                                q_used=None, sigma_h=None):
        """
        Compute diagnostics for one flight record.
        Returns dict and prints a short summary.
        """
        n = len(ts)
        assert len(hs_meas) == n == len(h_s)
        # robust sigma_h if not provided
        if sigma_h is None:
            dh = np.diff(hs_rep) if len(hs_rep) > 1 else np.array([0.0])
            if dh.size >= 5:
                mad = np.median(np.abs(dh - np.median(dh)))
                sigma_h = max(1.4826 * mad / np.sqrt(2.0), 1e-6)
            else:
                sigma_h = max(np.std(hs_rep) * 1e-3, 1e-6)

        residuals = hs_meas - h_s
        rms_resid = np.sqrt(np.mean(residuals**2))
        frac_repaired = np.mean(~np.isfinite(hs_meas) | (hs_meas != hs_rep))  # or use a separate mask if you track repaired indices

        # accel percentiles
        a_abs = np.abs(a_s)
        a_p50 = np.percentile(a_abs, 50)
        a_p90 = np.percentile(a_abs, 90)
        a_p99 = np.percentile(a_abs, 99)

        # basic innovation-ish metric (approx): predicted innovation normalized by sigma_h
        innov_norm = np.abs(residuals) / max(sigma_h, 1e-12)
        innov_max = np.nanmax(innov_norm)
        innov_rms = np.sqrt(np.nanmean(innov_norm**2))

        diag = {
            "n": int(n),
            "frac_repaired": float(frac_repaired),
            "sigma_h": float(sigma_h),
            "rms_resid": float(rms_resid),
            "rms_resid_over_sigma": float(rms_resid / max(sigma_h, 1e-12)),
            "a_p50": float(a_p50),
            "a_p90": float(a_p90),
            "a_p99": float(a_p99),
            "innov_max_over_sigma": float(innov_max),
            "innov_rms_over_sigma": float(innov_rms),
            "extrap_used": bool(extrap_used),
        }
        if q_used is not None:
            diag["q_used"] = float(q_used)

        # short print summary
        print(f"[diag] n={n}, repaired_frac={diag['frac_repaired']:.3f}, "
            f"rms_resid/sigma={diag['rms_resid_over_sigma']:.2f}, "
            f"a_p99={diag['a_p99']:.1f} m/s^2, innov_max/sigma={diag['innov_max_over_sigma']:.1f}, "
            f"extrap={diag['extrap_used']}")
        return diag

    # ---------- Two small loops: robust reweighting + q adaptation ----------
    # Start with a conservative q guess (units: m^2/s^5)
    q = max((sigma_h / max(dt,1e-6))**2 * 1e-1, 1e-10)

    # Robust weighting loop (2–3 iters is enough)
    R_mult = np.ones(n)
    for _ in range(3):
        xs, _ = rts_smoother(q, R_mult)
        resid = hs_rep - xs[:, 0]
        w = huber_weights(resid, sigma_h, k=1.5)
        # convert weights to variance multipliers (downweight outliers)
        R_mult = 1.0 / np.clip(w, 1e-2, 1.0)

    # Adapt q to keep acceleration within bound without oversmoothing
    margin = 1.05
    for _ in range(10):
        xs, _ = rts_smoother(q, R_mult)
        a = xs[:, 2]
        a_p99 = np.nanpercentile(np.abs(a), 99.0)
        resid = hs_rep - xs[:, 0]
        rms = np.sqrt(np.mean(resid**2))
        # If acceleration too spiky, reduce q (smoother)
        if a_p99 > margin * accel_max:
            q *= 0.3
        # If too smooth (residuals >> noise), increase q
        elif rms > 2.5 * sigma_h:
            q *= 2.0
        else:
            break

    xs, _ = rts_smoother(q, R_mult)
    h_s, v_s, a_s = xs[:,0], xs[:,1], xs[:,2]
    diag = compute_kalman_diagnostics(ts, hs, hs_rep, h_s, v_s, a_s, extrap_used, q_used=q) # TODO: HANDLE THESE DIAGNOSTICS
    return h_s, v_s, a_s, extrap_used


def loadAltusMetrumLog(filename):
    dt = 0.01
    file_path = os.path.abspath(filename)

    flight_log = None
    try:
        fd = FlightData()
        flight_log = open(file_path, 'r')
        flight_df = pd.read_csv(flight_log, skipinitialspace=True)
        flight_df = flight_df.drop_duplicates(subset=['time'])
        ts = flight_df['time']
        hs = flight_df['height']
        hs_filt, Vs, azs, extrapolated = calcDerivedVelAccel(ts, hs, filter_hs=True)
        all_mask = [True]*len(ts)
        series_defs = [
            ('t', np.array(ts), None, 'Measured', 'Time since liftoff (s)'),
            ('Vz', np.array(Vs), all_mask, 'Calculated', 'Axial velocity (m/s)'),
            ('hraw', np.array(hs), None, 'Measured', 'Altitude above the pad (m)'),
            ('h', np.array(hs_filt), all_mask, 'Calculated', 'Altitude above the pad (m)'),
            ('az', azs, all_mask, 'Calculated', 'Axial acceleration (m/s^2)')            
        ]            
        for dest, data, interp_mask, source, desc in series_defs:
            fd[dest] = {
                'values': data,
                'interpolation_mask': interp_mask,
                'source': source,
                'description': desc
            }       
        fd['hraw'].ts = fd['t']
        fd['h'].ts = fd['t']
        fd['Vz'].ts = fd['t']
        fd['az'].ts = fd['t']        

        return fd
    except Exception as e:
        print(e)
        raise e
    finally:
        if flight_log is not None:
            flight_log.close()
    return None


def loadEggtimerLog(filename):
    dt = 0.01
    file_path = os.path.abspath(filename)

    flight_log = None
    try:
        fd = FlightData()
        flight_log = open(file_path, 'r')
        flight_df = pd.read_csv(flight_log, skipinitialspace=True)
        print(len(flight_df))
        flight_df = flight_df.drop_duplicates(subset=['time'])
        print(len(flight_df))
        ts = flight_df['time']
        hs = flight_df['height']
        hs_filt, Vs, azs, extrapolated = calcDerivedVelAccel(ts, hs, filter_hs=True)
        
        all_mask = [True]*len(ts)
        series_defs = [
            ('t', np.array(ts), None, 'Measured', 'Time since liftoff (s)'),
            ('Vz', np.array(Vs), all_mask, 'Calculated', 'Axial velocity (m/s)'),
            ('hraw', np.array(hs), None, 'Measured', 'Altitude above the pad (m)'),
            ('h', np.array(hs_filt), all_mask, 'Calculated', 'Altitude above the pad (m)'),
            ('az', azs, all_mask, 'Calculated', 'Axial acceleration (m/s^2)')            
        ]            
        for dest, data, interp_mask, source, desc in series_defs:
            fd[dest] = {
                'values': data,
                'interpolation_mask': interp_mask,
                'source': source,
                'description': desc
            }        
        return fd
    except Exception as e:
        print(e)
        raise e
    finally:
        if flight_log is not None:
            flight_log.close()
    return None


def generate_dummy_br_dfs(N, t0=-0.35, dt=0.002, M=5, pre=35, a=2.5):
    low_rate_df = pd.DataFrame()
    high_rate_df = pd.DataFrame()

    N = max(pre, N)
    a /= 9.80665

    low_ts = []
    low_vs = []
    low_hs = []

    high_ts = []
    high_azs = []

    t = t0
    for _ in range(pre):
        high_ts.append(round(t, 3))
        high_azs.append(0 if t < 0 else a)
        t += dt

    idx = 0
    while idx < (N - pre):
        low_ts.append(round(t, 3))
        low_vs.append(0 if t < 0 else (t * a))
        low_hs.append(0 if t < 0 else (t**2 * a / 2))

        high_ts.append(round(t, 3))
        high_azs.append(0 if t < 0 else a)
        t += dt
        idx += 1
        for _ in range(M-1):
            if idx == (N - pre - 1):
                break
            high_ts.append(round(t, 3))
            high_azs.append(0 if t < 0 else a)
            t += dt
            idx += 1

    low_rate_df['Flight_Time_(s)'] = low_ts
    low_rate_df['Baro_Altitude_AGL_(feet)'] = low_hs
    low_rate_df['Velocity_Up'] = low_vs
    high_rate_df['Flight_Time_(s)'] = high_ts
    high_rate_df['Accel_X'] = high_azs
    return low_rate_df, high_rate_df

def loadFeatherweightTrackerLog(filename):
    pass

def loadBlueRavenLog(summary_filename, low_rate_filename, high_rate_filename, dummy=False, dummy_samples=155):
    low_rate_path = os.path.abspath(low_rate_filename)
    high_rate_path = os.path.abspath(high_rate_filename)

    low_rate_log = None
    high_rate_log = None
    try:
        fd = FlightData()
        low_rate_df = None
        high_rate_df = None
        if dummy:
            low_rate_df, high_rate_df = generate_dummy_br_dfs(dummy_samples)
        else:
            low_rate_log = open(low_rate_path, 'r')
            low_rate_df = pd.read_csv(low_rate_log, skipinitialspace=True)
            high_rate_log = open(high_rate_path, 'r')
            high_rate_df = pd.read_csv(high_rate_log, skipinitialspace=True)
        fd.files = [low_rate_path, high_rate_path]

        map_idxs = []
        low_ts = low_rate_df['Flight_Time_(s)']
        high_ts = high_rate_df['Flight_Time_(s)']        
        low_idx, low_num = 0, len(low_ts)
        high_idx, high_num = 0, len(high_ts)
        done = False
        while not done:
            t_low, t_high = low_ts[low_idx], high_ts[high_idx]
            if (t_low == t_high):
                map_idxs.append(high_idx)
                low_idx += 1
            else:
                high_idx += 1
            done = (low_idx >= low_num) or (high_idx >= high_num)

        # print(map_idxs[:-1])

        ts = []
        interpolation_mask = []
        hs = []
        hs_kf = []
        axs = []
        ays = []
        azs = []
        world_axs = []
        world_ays = []
        world_azs = []        
        gzs = []
        gys = []
        gxs = []
        Vzs = []
        Vzs_kf = []
        Vhs = []
        Vhs_kf = []
        off_verticals = []

        low_hs = low_rate_df['Baro_Altitude_AGL_(feet)'] / 3.28
        low_v_ups = low_rate_df['Velocity_Up'] / 3.28
        low_v_cr = low_rate_df['Velocity_CR'] / 3.28
        low_v_dr = low_rate_df['Velocity_DR'] / 3.28

        axs_b = 9.80665 * high_rate_df['Accel_X']
        ays_b = 9.80665 * high_rate_df['Accel_Y']
        azs_b = 9.80665 * high_rate_df['Accel_Z']
        gxs_b = high_rate_df['Gyro_X'] / 57.3
        gys_b = high_rate_df['Gyro_Y'] / 57.3
        gzs_b = high_rate_df['Gyro_Z'] / 57.3

        qs_w = high_rate_df['Quat_1']
        qs_x = high_rate_df['Quat_2']
        qs_y = high_rate_df['Quat_3']
        qs_z = high_rate_df['Quat_4']

        ws = []
        xs = []
        ys = []
        zs = []

        v0 = 0
        q_vertical = quaternion.Quaternion(1, 0, 0, 0)

        # print('len(map_idxs) = {}, len(low_ts) = {}'.format(len(map_idxs), len(low_ts)))
        for low_idx in range(len(map_idxs)-1): # TODO: ENSURE THAT THE RANGE MAPPING/SYNCHRONIZATION IS CORRECT
            t_min, t_max = low_ts[low_idx], low_ts[low_idx+1]
            h0, h1 = low_hs[low_idx], low_hs[low_idx+1]
            v0, v1 = low_v_ups[low_idx], low_v_ups[low_idx+1]
            v_cr_0, v_cr_1 = low_v_cr[low_idx], low_v_cr[low_idx+1]
            v_dr_0, v_dr_1 = low_v_dr[low_idx], low_v_dr[low_idx+1]

            high_idx = map_idxs[low_idx]
            dt = t_max - t_min
            dh = h1 - h0
            dv = v1 - v0
            dv_cr = v_cr_1 - v_cr_0
            dv_dr = v_dr_1 - v_dr_0

            t_high = high_ts[high_idx]

            force_done = False
            while (not force_done) and (t_high < t_max):
                p = (t_high - t_min) / dt
                ts.append(t_high)
                interpolation_mask.append(t_high == t_min)
                hs.append((p * dh) + h0)
                ax, ay, az = axs_b[high_idx], ays_b[high_idx], azs_b[high_idx]
                w, x, y, z = qs_w[high_idx], qs_x[high_idx], qs_y[high_idx], qs_z[high_idx]
                ws.append(w)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                q = quaternion.Quaternion(w, x, y, z)
                as_f = quaternion.rotate_vector(q, vector3d.Vector3D(ax, ay, az))
                gxs.append(gxs_b[high_idx])
                gys.append(gys_b[high_idx])
                gzs.append(gzs_b[high_idx])
                azs.append(as_f.z)      
                ays.append(as_f.y)
                axs.append(as_f.x)          
                Vzs.append((p * dv) + v0)  
                Vcr = (p * dv_cr) + v_cr_0
                Vdr = (p * dv_dr) + v_dr_0
                Vhs.append(math.sqrt(Vcr**2 + Vdr**2))     
                high_idx += 1
                if high_idx < len(high_ts): # Avoid a sync error
                    t_high = high_ts[high_idx]
                else:
                    force_done = True
                off_verticals.append(quaternion.angle_between(q, q_vertical))
            v0 = v1

        hs_kf = signal.savgol_filter(hs, 251, 5, delta=0.002,)
        Vzs_kf = signal.savgol_filter(Vzs, 251, 5, delta=0.002)
        Vhs_kf = signal.savgol_filter(Vhs, 251, 5, delta=0.002)

        all_mask = [True]*len(Vzs_kf)
        series_defs = [
            ('t', np.array(ts), None, 'Measured', 'Time since liftoff (s)'),
            ('Vz', np.array(Vzs_kf), all_mask, 'Calculated', 'Axial velocity (m/s)'),
            ('Vzraw', np.array(Vzs), interpolation_mask, 'Measured', 'Axial velocity (m/s)'),
            ('Vh', np.array(Vhs_kf), all_mask, 'Calculated', 'Radial velocity (m/s)'),
            ('Vhraw', np.array(Vhs), interpolation_mask, 'Measured', 'Radial velocity (m/s)'),            
            ('hraw', np.array(hs), interpolation_mask, 'Measured', 'Altitude above the pad (m)'),
            ('h', np.array(hs_kf), interpolation_mask, 'Calculated', 'Altitude above the pad (m)'),
            ('ax', azs, None, 'Measured', 'Axial acceleration (m/s^2)'),
            ('ay', ays, None, 'Measured', 'Axial acceleration (m/s^2)'),
            ('az', axs, None, 'Measured', 'Axial acceleration (m/s^2)'),
            ('gx', gzs, None, 'Measured', 'Body frame angular rates (deg/s)'),
            ('gy', gys, None, 'Measured', 'Body frame angular rates (deg/s)'),
            ('gz', gxs, None, 'Measured', 'Body frame angular rates (deg/s)'),
            ('qw', ws, None, 'Measured', 'Quaternion Component W'),
            ('qx', xs, None, 'Measured', 'Quaternion Component X'),
            ('qy', ys, None, 'Measured', 'Quaternion Component Y'),
            ('qz', zs, None, 'Measured', 'Quaternion Component Z'),
            ('off_vertical', off_verticals, all_mask, 'Calculated', 'Angle from vertical (radians)')
        ]            
        for dest, data, interp_mask, source, desc in series_defs:
            fd[dest] = {
                'values': data,
                'interpolation_mask': interp_mask,
                'source': source,
                'description': desc
            }

        return fd
    except Exception as e:
        print(e)
        raise e
    finally:
        if low_rate_log is not None:
            low_rate_log.close()

        if high_rate_log is not None:
            high_rate_log.close()
    return None


def loadProjectPotatoLog(filename):
    path = os.path.abspath(filename)
    with open(path, 'r') as log:
        df = pd.read_csv(log, skipinitialspace=True)
        fd = FlightData()
        fd.files = [path]
        h = df['h'] - df['h_pad'][0]
        fd['h'] = {
            'values': h,
            'source': 'Measured',
            'description': 'Altitude above the pad (m)'
        }
        
        series_defs = [
            ('t', 't', 'Time since liftoff (s)'),
            ('Vz', 'v', 'Axial velocity (m/s)'),
            ('az', 'a', 'Axial acceleration (m/s^2)')
        ]            
        for src, dest, desc in series_defs:
            fd[dest] = {
                'values': df[src],
                'source': 'Measured',
                'description': desc
            }
        return fd
    return None


def loadRockSimExport(filename):
    path = os.path.abspath(filename)

    try:
        fd = FlightData()
        df = None
        with open(path, 'r') as file:
            lines = file.readlines()
            headers = [n.strip() for n in lines[0].split(',')]
            units = [n.strip() for n in lines[1].split(',')]
            file.seek(0)
            df = pd.read_csv(file, skipinitialspace=True, comment='#', skiprows=2, names=headers)

            if 'x-Thrust' in headers:

                series_defs = [
                    ('t', 'Time', 'Time since liftoff (s)'),
                    ('h', 'Altitude', 'Altitude above the pad (m)'),
                    ('Vz', 'Velocity', 'Axial velocity (m/s)'),
                    ('az', 'y-Acceleration', 'Axial acceleration (m/s^2)'),
                    ('aoa', 'Wind angle of attack', 'Angle of attack (rad)'),
                    ('mass', 'Mass', 'Total mass of rocket (kg)'),
                    ('Cd', 'Cd', 'Total Coefficient of Drag')
                ]            
            elif 'Alt AGL' in headers:
                raise Exception('RockSim Pro parsing is not currently implemented!')
            else:
                raise Exception('Unknown file type')                

            for dest, src, desc in series_defs:
                scale_map = {header:FlightDataUnitConverter.standard_scale(unit) for header, unit in zip(headers, units)}

                try:
                    fd[dest] = {
                        'values': scale_map[src] * df[src],
                        'source': 'Calculated',
                        'description': desc
                    }   
                except Exception as e:
                    print('Error: While adding flight data - {} to {}'.format(src, dest))
            return fd         
    except Exception as e:
        print(e)
    return None


def loadOpenRocketExport(filename):
    path = os.path.abspath(filename)

    try:
        fd = FlightData()
        df = None
        with open(path, 'r') as file:
            events = []
            headers = []
            for idx, line in enumerate(file.readlines()): # Parse header and events
                if line[0]== '#':
                    if idx == 3: # Parse the headers
                        line = line[1:]
                        headers = [n.strip() for n in line.split(',')]
                        headers = [name.replace('Â', '') for name in headers]
                        headers = [name.replace('(â€‹)', '').strip() for name in headers]
                    else:
                        if line.startswith('# Event'):
                            try:
                                txt = line[8:]
                                evt_name, middle, t_txt = txt.partition(' occurred at t=')
                                t = float(t_txt.split(' ')[0])
                                events.append( (evt_name, t) )
                            except Exception as e:
                                print('ERROR: While parsing event line \"{}\" - {}'.format(line, e))
            evt_map = {
                'IGNITION':'Ignition', 'LAUNCH':'Launch', 'LIFTOFF':'Liftoff', 'LAUNCHROD':'Cleared Guide', 
                'BURNOUT':'Burnout', 'APOGEE':'Apogee', 'EJECTION_CHARGE':'Charge', 'RECOVERY_DEVICE_DEPLOYMENT':'Deployment',
                'GROUND_HIT':'Ground Hit'}
            add_events = []
            for name, t in events:
                try:
                    name = evt_map[name]
                    add_events.append( (name, t))
                except:
                    pass
            fd.addEvents(add_events)
            file.seek(0)
            units = []
            for header in headers:
                idx_start = header.find('(')
                idx_end = header.find(')')
#                print('{} -- {}({}) - {}({})'.format(header, idx_start, type(idx_start), idx_end, type(idx_end)))
                unit = ''
                if (not idx_start == -1) and (not idx_end == -1):
                    unit = header[idx_start+1:idx_end]
                units.append(unit)

            df = pd.read_csv(file, skipinitialspace=True, comment='#', names=headers)
            
            series_defs = [
                ('t', 'Time (s)', 'Time since liftoff (s)'),
                ('h', 'Altitude (m)', 'Altitude above the pad (m)'),
                ('Vz', 'Vertical velocity (m/s)', 'Axial velocity (m/s)'),
                ('az', 'Vertical acceleration (m/s²)', 'Axial acceleration (m/s^2)'),
                ('V', 'Total velocity (m/s)', 'Total Velocity (m/s)'),
                ('aoa', 'Angle of attack (°)', 'Angle of attack (rad)'),
                ('mass', 'Mass (g)', 'Total mass of rocket (kg)'),
                ('Cd', 'Drag coefficient', 'Total Coefficient of Drag')
            ]            
            for dest, src, desc in series_defs:
                scale_map = {header:FlightDataUnitConverter.standard_scale(unit) for header, unit in zip(headers, units)}

                try:
                    fd[dest] = {
                        'values': scale_map[src] * df[src],
                        'source': 'Calculated',
                        'description': desc
                    }   
                except Exception as e:
                    print('Error: While adding flight data - {} to {}'.format(src, dest))
            return fd         
    except Exception as e:
        print(e)
    return None

def loadRASAero(filename):
    pass


def alignment_offsets(fds, event_types=['Ignition', 'Launch'], ref=None):
    def calc_t_ref(fds, ref, t_offs):
        t_ref = np.min(t_offs)
        for idx, fd in enumerate(fds):
            if fd == ref:
                return t_offs[idx]
        return t_ref

    t_offs = []
    for fd in fds:
        t_sum = 0
        count = 0
        for ev in [ev for k, ev in fd.events.items() if k in event_types]:
            count += 1
            t_sum += ev.t
        t_offs.append(t_sum / count)
    t_offs = np.array(t_offs)
    t_ref = calc_t_ref(fds, ref, t_offs) 
    t_offs -= t_ref
    return t_offs
