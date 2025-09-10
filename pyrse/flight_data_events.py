import array
import math


V_liftoff = 15.0 # Minimum 15 m/s
a_liftoff = 9.81*2 #29.4 # Minimum 3G
a_burnout = -5.0 # m/s^2
t_burnout = 0.15 # s 
v_preapogee_thresh = 5.0 # m/s
v_apogee_thresh = -1.0 # m/s
dh_apogee_minimum = 1.0 # m
dh_apogee_thresh = 10.0 # m


class FlightDataLiftoffDetector:
    class Event:
        def __init__(self):
            self.detected = False
            self.t = None
            self.V = None
            self.h = None
            self.cnt = None

    def __init__(self, t_buffer=None):
        self.__t_buffer = (1.5 * V_liftoff / a_liftoff) if t_buffer is None else t_buffer
        self.__buffer_dt = []
        self.__buffer_dv = []
        self.__t_buffer_sum = 0
        self.__v_buffer_sum = 0
        self.__liftoff_detected = False
#        print('t_buffer = {}'.format(self.__t_buffer))
        self.__V = None
        self.__h = None
        self.__t_latency = None

    @property
    def detected(self):
        return self.__liftoff_detected
    
    def __call__(self, dt, az):
        event = FlightDataLiftoffDetector.Event()
        if self.__liftoff_detected:
            event.t = self.__t_latency
            event.v = self.__V
            event.h = self.__h
        else:
            buffer_dt = self.__buffer_dt
            buffer_dv = self.__buffer_dv
            t_buffer_sum = self.__t_buffer_sum
            v_buffer = self.__v_buffer_sum
            t_buffer_sum += dt
            dv = dt * az
            v_buffer += dv
            buffer_dt.append(dt)            
            buffer_dv.append(dv)
            if t_buffer_sum > self.__t_buffer:
                t_buffer_sum -= buffer_dt[0]
                v_buffer -= buffer_dv[0]
                buffer_dt = buffer_dt[1:]
                buffer_dv = buffer_dv[1:]
            
            self.__v_buffer_sum = v_buffer
            self.__t_buffer_sum = t_buffer_sum

            if (v_buffer > V_liftoff) and (az > a_liftoff):
                self.__liftoff_detected = True
                event.detected = True

                h = 0
                t = 0
                V = 0
                for dt, dv in zip(reversed(buffer_dt), reversed(buffer_dv)):
                    V += dv
                    t += dt
                    h += dv * dt
                    if V > V_liftoff:
                        break
                # print('***** Liftoff Detected! t = {}, V = {}, h = {}, cnt = {}'.format(t, V, h, len(buffer_dt)))
                
                self.__V = V
                self.__h = h
                self.__t_latency = t

                event.V = self.__V
                event.h = self.__h
                event.t = self.__t_latency
        event.cnt = len(self.__buffer_dt)
        return event


class FlightDataBurnoutDetector:
    class Event:
        def __init__(self):
            self.detected = False
    
    def __init__(self):
        self.__running = False
        self.__t_decel = None
        self.__burnout_count = 0

    def init(self):
        self.__running = False
        self.__t_decel = None
        self.__burnout_count = 0

    def start(self):
        self.__running = True

    def stop(self):
        self.__running = False

    @property
    def detected(self):
        return self.__burnout_count > 0

    @property
    def burnout_count(self):
        return self.__burnout_count

    def __call__(self, dt, az):
        result = FlightDataBurnoutDetector.Event()
        if not self.__running:
            return result

        burnout_detected = False
        t_decel = self.__t_decel

        if t_decel is None:
            if az < a_burnout:
                t_decel = 0
            else:
                t_decel = None
        else:
            if az < a_burnout:
                if t_decel < t_burnout:
                    burnout_detected = ((t_decel + dt) > t_burnout)
                t_decel += dt                    
            else:
                t_decel = None
        self.__t_decel = t_decel
        if burnout_detected:
            self.__burnout_count += 1
        result.detected = burnout_detected
        return result


class FlightDataPreapogeeDetector:
    class Event:
        def __init__(self):
            self.detected = False
            self.V = None
    
    def __init__(self):
        self.__running = False
        self.__detected = False
    
    def init(self):
        self.__running = False
        self.__detected = False

    def start(self):
        self.__running = True

    def stop(self):
        self.__running = False
            
    @property
    def detected(self):
        return self.__detected

    def __call__(self, dt, Vz):
        result = FlightDataPreapogeeDetector.Event()
        result.V = Vz
        if not self.__running:
            return result

        if self.__detected:
            return result

        if Vz < v_preapogee_thresh:
            self.__detected = True
            result.detected = True
        return result


class FlightDataApogeeDetector:
    class Event:
        def __init__(self):
            self.detected = False
            self.Vz = None
            self.h = None
    
    def __init__(self):
        self.__running = False
        self.__detected = False
        self.__h_max = None
        self.__v_apogee_thresh = -5.0 # m/s #HACK: These threshold numbers are very high due to some failures....
        self.__dh_apogee_minimum = 1.0 # m
        self.__dh_apogee_thresh = 11.0 # m

    def init(self):
        self.__running = False
        self.__detected = False
        self.__h_max = None

    def start(self):
        self.__running = True
    
    @property
    def detected(self):
        return self.__detected

    def __call__(self, dt, Vz, h):
        result = FlightDataApogeeDetector.Event()
        result.Vz = Vz
        result.h = h
        h_max = self.__h_max
        if (h_max is None) or (h > h_max):
            h_max = h
            self.__h_max = h_max

        if not self.__running:
            return result

        if self.__detected:
            return result

        if ((Vz < self.__v_apogee_thresh) and ((h_max - h) > self.__dh_apogee_minimum)) or ((h_max - h) > self.__dh_apogee_thresh):
            self.__detected = True
            result.detected = True
        return result
