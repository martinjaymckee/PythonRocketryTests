# import picola

class Kinematic1DStateObserver:
    def __init__(self, R=0.5, Q_00=0.00075, Q_01=0.0075, Q_11=0.075):
        self.__X_0 = None
        self.__X_1 = None
        self.__X_pre_0 = None
        self.__X_pre_1 = None
        self.__P_00 = 2000
        self.__P_01 = 0
        self.__P_11 = 2000
        self.__R = R
        self.__Q_00 = Q_00
        self.__Q_01 = Q_01
        self.__Q_11 = Q_11

    def init(self, x=0, dx=0):
        self.__X_0 = x
        self.__X_1 = dx
        self.__P_00 = 2000
        self.__P_01 = 0
        self.__P_11 = 2000
    
    @property
    def x(self):
        return self.__X_0

    @x.setter
    def x(self, alt):
        self.__X_0 = alt
        
    @property
    def dx(self):
        return self.__X_1

    @dx.setter
    def dx(self, v):
        self.__X_1 = v
        
    # @property
    # def P(self):
    #     return picola.Matrix([[self.__P_00, self.__P_01], [self.__P_01, self.__P_11]])
    
    @property
    def converged(self):
        return (self.__P_00 < CONVERGENCE_THRESHOLD) and (self.__P_01 < CONVERGENCE_THRESHOLD) and (self.__P_11 < CONVERGENCE_THRESHOLD)
    
    def predict(self, dt, ddx):
        x = self.__X_0
        dx = self.__X_1
        dt2 = dt * dt
        self.__X_0 = x + (dt*dx) + (0.5 * dt2 * ddx)
        self.__X_1 = dx + (dt*ddx)

        P_00 = self.__P_00
        P_01 = self.__P_01
        P_11 = self.__P_11
        self.__P_00 = P_00 + (2 * dt * P_01) + (dt2 * P_11) + self.__Q_00
        self.__P_01 = P_01 + (dt * P_11) + self.__Q_01
        self.__P_11 = P_11 + self.__Q_11

    def correct(self, x):
        X_0 = self.__X_0
        X_1 = self.__X_1
        P_00 = self.__P_00
        P_01 = self.__P_01
        P_11 = self.__P_11
        y = x - X_0
        S = P_00 + self.__R
        k = 1 / S
        K_0 = k * P_00
        K_1 = k * P_01
        self.__X_0 = X_0 + (K_0 * y)
        self.__X_1 = X_1 + (K_1 * y)
        self.__P_00 = P_00 * (1 - K_0)
        self.__P_01 = P_01 * (1 - K_0)
        self.__P_11 = P_11 - (K_1 * P_01)
        return False

    def __call__(self, dt, h, az):
        self.predict(dt, az)
        return self.correct(h)

