

class GimbalPhysicalModel:
    # TODO: IMPLEMENT GIMBAL MODEL USING THE VELOCITY AND ACCELERATION LIMITS AS WELL AS
    #   THE SERVO UPDATE RATE
    def __init__(self, f_update=100, omega_max=1.0, domega_max=1.0):
        self.__alpha = 0
        self.__alpha_cmd = 0
        self.__dalpha = 0
        self.__beta = 0
        self.__beta_cmd = 0
        self.__dbeta = 0
        self.__alpha_offset = 0
        self.__beta_offset = 0
        self.__alpha_gain = 1.0
        self.__beta_gain = 1.0
        self.__omega_max = omega_max
        self.__domega_max = domega_max
        self.__f_update = f_update
        self.__t = 0

    def pos(self, t):
        self.__update()
        alpha_out = (self.__alpha_gain * self.__alpha) + self.__alpha_offset
        beta_out = (self.__beta_gain * self.__beta) + self.__beta_offset
        return alpha_out, beta_out

    def command(self, t, alpha, beta):
        self.__alpha_cmd = alpha
        self.__beta_cmd = beta
        return self.pos(t)

    def __update(self):
        self.__alpha = self.__alpha_cmd
        self.__beta = self.__beta_cmd
