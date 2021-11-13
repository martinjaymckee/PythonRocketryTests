class SaturationLimiter:
    def __init__(self, ymin, ymax):
        self.__ymin = ymin
        self.__ymax = ymax

    def valid(self):
        return self.__ymax >= self.__ymin

    def __call__(self, y, **kwargs):
        if y > self.__ymax:
            return self.__ymax, True
        elif y < self.__ymin:
            return self.__ymin, True
        return y, False


class DirectIntegrator:
    def __init__(self):
        self.__I = 0

    @property
    def I(self):
        return self.__I

    def reset(self):
        self.__I = 0

    def __call__(self, dt, err, output_pre, output_post):
        self.__I += dt * err


class ConditionalIntegrator:
    def __init__(self):
        self.__match_threshold = 0.0001
        self.__I = 0

    @property
    def I(self):
        return self.__I

    def reset(self):
        self.__I = 0

    def __call__(self, dt, err, output_pre, output_post):
        print(output_pre, output_post)
        if abs(output_pre - output_post) < self.__match_threshold:
            self.__I += dt * err


class RecursiveSmoothingFilter:
    def __init__(self, e=0.75):
        self.__e = e
        self.__x_last = None

    def __call__(self, x):
        if self.__x_last is None:
            self.__x_last = x
        self.__x_last = self.__e * self.__x_last + (1 - self.__e) * x
        return self.__x_last


class PID:
    def __init__(self, Kp=1, Ki=0, Kd=0, limiter=None, dpp_filter=None):
        self.__Kp = Kp
        self.__Ki = Ki
        self.__Kd = Kd
        self.__sp = 0
        self.__pp_last = None
        self.__integrator = DirectIntegrator()
        self.__limiter = limiter
        self.__dpp_filter = dpp_filter
        self.__pre_output = None
        self.__output = None
        self.reset()

    def __str__(self):
        return 'PID(Kp = {}, Ki = {}, Kd = {}, I = {})'.format(self.__Kp, self.__Ki, self.__Kd, self.__I_sum)

    def reset(self, limiter_kwargs={}):
        self.__integrator.reset()
        self.__pp_last = None
        self.__pre_output = 0 if self.__limiter is None else self.__limiter(0, **limiter_kwargs)[0]
        self.__output = 0 if self.__limiter is None else self.__limiter(0, **limiter_kwargs)[0]

    @property
    def Kp(self): return self.__Kp

    @Kp.setter
    def Kp(self, K):
        self.__Kp = K
        return self.__Kp

    @property
    def Ki(self): return self.__Ki

    @Ki.setter
    def Ki(self, K):
        self.__Ki = K
        return self.__Ki

    @property
    def Kd(self): return self.__Kd

    @Kd.setter
    def Kd(self, K):
        self.__Kd = K
        return self.__Kd

    @property
    def e(self): return self.__e

    @e.setter
    def e(self, _e):
        self.__e = _e
        return self.__e

    @property
    def sp(self): return self.__sp

    @sp.setter
    def sp(self, _sp):
        self.__sp = _sp
        return self.__sp

    def __call__(self, dt, pp, dpp=None, debug=False, limiter_kwargs={}, dpp_filter_kwargs={}):
        err = self.sp - pp

        # Calculate Proportional
        P_term = self.Kp * err

        # Calculate Integral
        self.__integrator(dt, err, self.__pre_output, self.__output)
        I_term = (self.Ki * self.__integrator.I)

        # Calculate Derivative
        if dpp is None:
            diff = 0 if self.__pp_last is None else (pp - self.__pp_last)
            dpp = (diff / dt) if not dt == 0 else 0
        if self.__dpp_filter is not None:
            dpp = self.__dpp_filter(dpp, **dpp_filter_kwargs)
        D_term = (self.Kd * dpp)

        self.__pre_output = P_term + I_term - D_term

        if self.__limiter is not None:
            self.__output, _ = self.__limiter(self.__pre_output, **limiter_kwargs)
        else:
            self.__output = self.__pre_output

        self.__pp_last = pp

        if debug:
            data = {
                'P_term': P_term,
                'I_term': I_term,
                'D_term': D_term,
                'out_pre': self.__pre_output,
                'out_post': self.__output,
                'limited': not (self.__pre_output == self.__output)
            }
            return self.__output, data
        return self.__output
