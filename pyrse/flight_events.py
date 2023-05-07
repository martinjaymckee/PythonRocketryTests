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
