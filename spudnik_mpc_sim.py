import math

#import timestamp

class SpudnikEnvironment:
    def __init__(self, g=9.80655, rho=1.225):
        self.g = g
        self.rho = rho

class SpudnikRocketProperties:
    def __init__(self, S_ref=math.pi*((0.102/2)**2), S_brake = 3 * 0.0762 * 0.0762):
        print('S_ref = {}, S_brake = {}'.format(S_ref, S_brake))
        self.alpha_min = 0
        self.alpha_max = math.radians(30)
        self.omega_max = (math.radians(60) / .19)
        self.S_ref = S_ref
        self.S_brake = S_brake
        self.Cd0 = 0.45
        self.m = 1.037


class ConstantMotor: # NOTE: THIS SHOULD ACTUALLY USE THE PYRSE CLASSES TO ALLOW SIMULATION WITH THRUST CURVES AND VARIABLE MASS....
    def __init__(self, T, t_burn):
        self.__T = T
        self.__t_burn = t_burn
    
    def thrust(self, t):
        if 0 <= t <= self.__t_burn:
            return self.__T
        return 0


class SpudnikOptimizationSimulator:
    def __init__(self, env, rocket, motor=None):
        self.__env = env
        self.__rocket = rocket
        self.__motor = ConstantMotor(80, 1.1) if motor is None else motor # This is roughly a G80
        self.__dt = 0.01
        self.__t = 0
        self.__h = 0
        self.__Vz = 0
        self.__alpha_current = 0
        self.__alpha_target = 0
        self.__omega_max = (math.radians(60) / .19)

    @property
    def alpha(self):
        return self.__alpha_current

    @alpha.setter
    def alpha(self, val):
        self.__alpha_target = val
        return self.__alpha_current

    @property
    def t(self):
        return self.__t

    @property
    def h(self):
        return self.__h

    @property
    def Vz(self):
        return self.__Vz

    def update(self):
        env = self.__env
        rocket = self.__rocket
        motor = self.__motor
        t = self.__t
        dt = self.__dt
        h = self.__h
        Vz = self.__Vz

        T = motor.thrust(t)
        m = rocket.m
        self.__update_alpha(dt)        
        Cd = self.__est_cd(self.__alpha_current)
        D = 0.5 * env.rho * (Vz * Vz) * rocket.S_ref * Cd
        az = ((T - D) / m) - env.g
        Vz += dt * az
        self.__h += Vz * dt
        self.__Vz = Vz
        self.__t = t + dt
        print('t = {}, h = {}, Vz = {}, az = {}'.format(t, h, Vz, az))


    def __update_alpha(self, dt):
        max_step = self.__omega_max * dt
        remaining = self.__alpha_target - self.__alpha_current

        if remaining > 0: # Increasing
            if remaining > max_step:
                self.__alpha_current += max_step
            else:
                self.__alpha_current = self.__alpha_target
        else: # Decreasing
            if -remaining > max_step:
                self.__alpha_current -= max_step
            else:
                self.__alpha_current = self.__alpha_target

    def __est_cd(self, alpha):
        rocket = self.__rocket
        return rocket.Cd0 + (1.2 * (rocket.S_brake / rocket.S_ref) * math.sin(alpha))


class SpudnikDataSource:
    def __init__(self):
        pass

class SpudnikHardwareDataSource(SpudnikDataSource):
    def __init__(self, hw, rocket):
        self.__hw = hw
        self.__rocket = rocket
    
    @property
    def h(self):
        return 0

    @property
    def Vz(self):
        return 0

    @property
    def az(self):
        return 0

    @property
    def alpha(self):
        return 0
    
    @alpha.setter
    def alpha(self, val):
        val = max(self.__rocket.alpha_min, min(self.__rocket.alpha_max, val))
        # TODO: DO SOMETHING WITH THE VALUE


class SpudnikSimulationDataSource(SpudnikDataSource):
    def __init__(self, sim, rocket):
        self.__sim = sim
        self.__rocket = rocket
        self.__h_sd = 0.05
        self.__Vz_sd = 0.15
        self.__az_sd = 0.01
    
    @property
    def h(self):
        return 0

    @property
    def Vz(self):
        return 0

    @property
    def az(self):
        return 0

    @property
    def alpha(self):
        return 0
    
    @alpha.setter
    def alpha(self, val):
        val = max(self.__rocket.alpha_min, min(self.__rocket.alpha_max, val))
        # TODO: DO SOMETHING WITH THE VALUE



#@micropython.native
#def est_airbrake_cd(alpha, S, Sb):
def est_airbrake_cd(alpha, state):
    return 1.2 * (state.Sb / state.S) * math.sin(alpha)

#@micropython.native
#def calc_ascent(dt, m, S, Sb, V0, Cd0, alpha):
def calc_ascent(dt, env, state):
    V = state.V0
    C = (0.5 * env.rho * state.S) / (2 * state.m)
    N = 0
    b = 0
    Cd = state.Cd0 + est_airbrake_cd(alpha, state)
    while V >= 0:
        N += 1
        V2 = (V ** 2)
        a = (C * V2 * Cd) + g
        V -= (dt * a)
        dh += (dt * V)
        t += dt
        
    return N, t, dh

#@micropython.native
#def mpc_opt_angle(h0, h_tgt, dt, m, S, Sb, V0, Cd0):
def mpc_opt_angle(dt, env, state):
    N = 6
    alpha_max = math.radians(30)
    alpha_min = 0
    
    alpha = alpha_max
    t = 0
    h = 0
    for _ in range(N):
#        t_start = timestamp.now()    
        _, t, dh = calc_ascent(dt, m, S, Sb, V0, Cd0, alpha)
#        t_end = timestamp.now()
        
        h = h0 + dh
        
        #print('N = {}, dh = {} m, t = {} s'.format(N, dh, t))
        #print('h = {} m'.format(h0 + dh))
        #print('Time = {} ms'.format((t_end - t_start)/1e3))
        
        if h > h_tgt:
            state.alpha_min = alpha
            alpha = (alpha + state.alpha_max) / 2
        else:
            state.alpha_max = alpha
            alpha = (alpha + state.alpha_min) / 2            
        #print('Alpha = {} deg'.format(math.degrees(alpha)))
    return alpha, h, t


def main():
    import matplotlib.pyplot as plt

    env = SpudnikEnvironment()
    rocket = SpudnikRocketProperties()
    simulator = SpudnikOptimizationSimulator(env, rocket)

    h = 0
    h_last = 0
    ts = []
    hs = []
    alphas = []
    while h >= h_last:
        t = simulator.t
        if t > 3.5:
            simulator.alpha = math.radians(25)
        h_last = simulator.h
        simulator.update()
        h = simulator.h
        ts.append(t)
        hs.append(h)
        alphas.append(simulator.alpha)

    fig, axs = plt.subplots(2, constrained_layout=True)
    axs[0].plot(ts, hs)
    axs[1].plot(ts, alphas)

    plt.show()

    # optimize = True

    # data_source = None

    # if optimize:
        
    # dt = 0.25
    # d = 7.62 / 100
    # m = 1.037
    # S = math.pi * ((d / 2) ** 2)
    # Sb = 3 * (5 / 100) * (3 / 100)
    # V0 = 112
    # md = 0
    # Cd0 = .5
    # h0 = 240
    # h_tgt = 2024 / 3.28
    
    # t_est_start = timestamp.now()
    # alpha, h, t = mpc_opt_angle(h0, h_tgt, dt, m, S, Sb, V0, Cd0)
    # t_est_end = timestamp.now()
    # print('alpha = {} deg, h_tgt = {} m, h = {} m'.format(math.degrees(alpha), h_tgt, h)) 
    # print('Time = {} ms'.format((t_est_end - t_est_start)/1e3))
    
    
if __name__ == '__main__':
    main()