import copy
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import engines


plt.style.use('seaborn-colorblind')


class PistonLaunchResults:
    def __init__(self):
        self.ts = []
        self.Fs = []
        self.Fs_eff = []
        self.mgs = []
        self.ns = []
        self.xs = []
        self.dxs = []
        self.ddxs = []
        self.Vs = []
        self.Ps = []
        self.dTs = []
        self.Ts = []
        self.Is_raw = []
        self.Is_imparted = []
        self.rhos = []
        self.dmleaks = []
        self.dmexs = []
        self.Ds = []
        self.MWs = []
        self.params = []
        self.t_launch = 0
        self.t_ascent = 0
        self.t_apogee = 0
        self.t_liftoff_detection = 0
        self.t_burnout_detection = 0


def simulatePistonLaunch(dt, piston, mr, engine, ax=None, ax2=None, accel_sd=50e-6):
    results = PistonLaunchResults()

    x = piston.L0
    dx = 0
    ddx = 0
    r_exit = 0.0025 # HACK: A_EXIT SHOULD BE PASSED IN OR -- ideally -- INCLUDED IN THE ENGINE OBJECT
    a_exit = math.pi * (r_exit**2)
    g0 = 9.80665
    P0 = 101325.
    T0 = 293 # K
    MWex = 34.75 / 1000 # kg/mol
    MWair = 28.97 / 1000 # kg/mol
    gamma = 1.26 # HACK: THIS COULD RANGE FROM 1.26 - 1.29
    Cv = 682
    hb = 57
    Ri = 8.3144621 # J/mol-K
    Rs = Ri / (1000*MWex) # J/mol-K
    Isp = engine.specific_impulse
    Tex = 570. # K
    mg = (MWair*piston.V0*P0) / (Ri*T0)
    rho0 = (MWair*P0)/(Ri*T0)
    Ap = math.pi * (piston.r**2)
    mtot = mr + piston.mass
    D = 0
    T = T0
    MW=MWair
    I_raw = 0
    I_imparted = 0
    Cd_rocket = .34
    A_rocket = 4.83e-4 # BT-50

    liftoff_detector = LiftoffTrigger(1.2*9.81, 2.5*9.81, 25)
    liftoff_detected = False
    burnout_detector = BurnoutTrigger(-50, -10)
    burnout_detected = False

    t = 0
    segment = 'prelaunch'
    while True:
        # F, m_motor, _, Iremaining = engine.properties(t)
        F = engine.thrust(t)
        m_motor = engine.mass(t)
        Iremaining = engine.total_impulse - engine.spent_impulse(t)
        I_raw += dt * F
        if segment == 'prelaunch' or segment == 'piston':
            dmex = F / (Isp*g0)
            V = Ap * x
            n = mg / MWex
            P = n * Ri * T / V
            F_fix = F + a_exit*(P0-P)
            rho = mg / V
            P_ratio = max(0, (P-P0) / P)
            inner = (2*g0*Rs*T)*(P_ratio+((1/(2*gamma))*(P_ratio**2)))
            vleak = math.sqrt(inner)
            vleak_eff = vleak-dx
            dmleak = piston.Cd*piston.Aleak*rho*vleak_eff
            dmg = dt * (dmex - dmleak)
            MW = ((MW*(mg-(dt*dmleak))) + (MWex*(dt*dmex)))/(mg+dmg) # HACK: THIS IS NOT DOING THE AVERAGE CORRECTLY...
            mg += dmg
        D = 0.5 * rho0 * (dx**2) * A_rocket * Cd_rocket
        ddx = 0
        if segment == 'prelaunch' or segment == 'piston':
            F_eff = (F_fix + ((P-P0)*Ap) - D - piston.Ffrict)
            ddx = ((1/mtot)*F_eff) - g0
            if ddx < 0 and x <= piston.L0:
                ddx = 0
                dx = 0
                x = piston.L0
                results.t_launch = t
            else:
                segment = 'piston'
                x = max(x, piston.L0)
                F_eff = ddx*mr
                I_imparted += (dt * ddx * mr)
            dT0 = ((-T/mg)*(dmex-dmleak))
            dT1 = ((gamma/mg)*((Tex*dmex)-(T*dmleak)))
            dT2 = ((ddx*mtot*dx)/(mg*Cv))
            dT3 = ((T-T0)*((hb*piston.A)/(mg*Cv)))
            dT = (dT0 + dT1 - dT2 - dT3)
            T += dt * dT
            if x > piston.L:
                T = math.nan
                P = math.nan
                V = math.nan
                n = math.nan
                F_fix = math.nan
                rho = math.nan
                dmleak = math.nan
                dmex = math.nan
                mg = math.nan
                segment = 'ascent'
                results.t_ascent = t
        elif segment == 'ascent':
            ddx = ((1/(mr+m_motor))*(F-D)) - g0
        else:
            pass # Error

        x += dt * dx + 0.5 * (dt**2) * ddx
        dx += dt * ddx

        if segment == 'ascent' and dx < 0:
            segment = 'descent'
            results.t_apogee = t
            break

        ddx_noise = random.gauss(0, accel_sd)

        if not liftoff_detected:
            detection, time_delta = liftoff_detector(dt, t, ddx+ddx_noise)
            if detection:
                print('Detected Liftoff at t = {} s, time_delta = {} s, V = {} m/s ({} m/s), a = {} m/s^2'.format(t, time_delta, dx, liftoff_detector.v_est, ddx))
                liftoff_detected = True
                results.t_liftoff_detection = t-time_delta

        if not burnout_detected:
            detection = burnout_detector(dt, t, ddx+ddx_noise)
            if detection:
                print('Detected burnout at t = {}, j_est = {}, a = {}'.format(t, burnout_detector.j_est, ddx))
                burnout_detected = True
                results.t_burnout_detection = t

        results.dmexs.append(dmex)
        results.Vs.append(V)
        results.ns.append(n)
        results.Ps.append(P)
        results.rhos.append(rho)
        results.dmleaks.append(dmleak)
        results.MWs.append(MW)
        results.Ds.append(D)

        results.params.append( (dT0, dT1, dT2, dT3) )
        results.dTs.append(dT)
        results.Ts.append(T)

        results.mgs.append(mg)
        results.Fs.append(F)
        results.Fs_eff.append(F_eff)
        results.ts.append(t)
        results.ddxs.append(ddx)
        results.dxs.append(dx)
        results.xs.append(x)
        results.Is_raw.append(I_raw)
        results.Is_imparted.append(I_imparted + Iremaining)
        t += dt

    if not ax is None:
        ax.plot(results.ts, results.dTs, alpha=0.33)

    if not ax2 is None:
        ax2.plot(results.ts, results.dxs, alpha=0.33)

    return results


def batchPistonLaunch(pistons, mrs, engine, dt=1e-4):
    fig, ax = plt.subplots(1, figsize=(15,12))
    ax2 = ax.twinx()
    results = []

    for piston in pistons:
        for mr in mrs:
            result = simulatePistonLaunch(dt, piston, mr, engine, ax=ax, ax2=ax2)
            results.append( (piston, mr, result) )

    fig, ax = plt.subplots(1, figsize=(15,12))
    xs = []
    ys = []
    hs = []
    for piston, mr, result in results:
        xs.append(piston.Ltravel)
        ys.append(result.Is_imparted[-1])
        hs.append(mr)
    sns.scatterplot(xs, ys, hue=hs, alpha=0.5, ax=ax)


def optimizePistonLength(ref_piston, mr, engine, N=10000):
    pass


class Piston:
    def __init__(self, L, d, m_additional=0, t=0.0003, L0=0.015, gap=0/1000, Cd=0.9, Ffrict=0):
        self.__rho = 689 # kg/m^3
        self.__t = t
        self.__L = L
        self.__L0 = L0
        assert L > L0, 'Piston total length ({} m) is not larger than its initial length ({} m)'.format(L, L0)
        self.__r = d/2
        self.__m_additional = m_additional
        self.__gap = gap
        self.__Cd = Cd
        self.__Ffrict = Ffrict
        self.__calc()

    def __str__(self):
        return 'L = {} mm, r = {} mm, m = {} g'.format(self.L*1000, self.r*1000, self.mass*1000)

    @property
    def mass(self): return self.__m

    @property
    def L(self): return self.__L

    @L.setter
    def L(self, val):
        self.__L = val
        self.__calc()
        return self.__L

    @property
    def L0(self): return self.__L0

    @L0.setter
    def L0(self, val):
        self.__L0 = val
        self.__calc()
        return self.__L0

    @property
    def Ltravel(self): return self.__L_travel

    @property
    def r(self): return self.__r

    @property
    def A(self): return self.__A

    @property
    def Aleak(self): return self.__Aleak

    @property
    def gap(self): return self.__gap

    @property
    def Cd(self): return self.__Cd

    @property
    def V0(self): return self.__V0

    @property
    def Ffrict(self): return self.__Ffrict

    def __calc(self):
        self.__V = 2*math.pi*self.__r*self.__L*self.__t # HACK: THIS IS AN APPROXIMATION
        self.__m = self.__V * self.__rho + self.__m_additional
        self.__A = math.pi * ((self.__r**2) + 2*self.__r*self.__L)
        self.__Aleak = 2*math.pi * self.__r * self.__gap # HACK: THIS IS AN APPROXIMATION
        self.__L_travel = self.__L - self.__L0
        self.__V0 = math.pi*(self.r**2)*self.__L0


def makePistonsFromRef(ref_piston, Ls=None, L0s=None, N=30):
    Ls = np.linspace(Ls[0], Ls[1], N) if not Ls is None else [None]*N
    L0s = np.linspace(L0s[0], L0s[1], N) if not L0s is None else [None]*N
    pistons = []
    for L, L0 in zip(Ls, L0s):
        piston = copy.deepcopy(ref_piston)
        if not L is None:
            piston.L = L
        if not L0 is None:
            piston.L0 = L0
        pistons.append(piston)
    return pistons


class LiftoffTrigger:
    def __init__(self, a_summing, a_trigger, v_trigger, t_buffer = 1.5, N=25):
        self.__a_summing = a_summing
        self.__a_trigger = a_trigger
        self.__v_trigger = v_trigger
        self.__t_buffer = t_buffer
        self.__dt_load = float(t_buffer) / N
        self.__on_pad = True
        self.__v_buffer = []
        self.__v_est = 0
        self.__is_summing = True
        self.__t_trigger = 0
        self.__N = N
        self.__t_last = None
        self.__a_last = 0

    @property
    def v_est(self): return self.__v_est

    def __call__(self, dt, t, a):
        if self.__on_pad:
            if (self.__t_last is None) or ((t - self.__t_last) >= self.__dt_load):
                self.__t_last = t
                # NOTE: USE ACCELERATION AVERAGE TO CALCULATE CHANGE IN VELOCITY
                dv = self.__dt_load*((a + self.__a_last) / 2)
                self.__a_last = a
                self.__v_est += dv
                self.__v_buffer.append(dv)
                if len(self.__v_buffer) == self.__N:
                    self.__v_est -= self.__v_buffer[0]
                    self.__v_buffer = self.__v_buffer[-self.__N:]
                if a > self.__a_summing:
                    if not self.__is_summing:
                        self.__t_trigger = t
                        self.__is_summing = True
                else:
                    self.__is_summing = False

                if a > self.__a_trigger:
                    if self.__v_est > self.__v_trigger:
                        self.__on_pad = False
                        return True, (t - self.__t_trigger)
        return False, None


class BurnoutTrigger:
    def __init__(self, j_trigger, a_trigger):
        self.__j_trigger = j_trigger
        self.__a_trigger = a_trigger
        self.__a_last = None
        self.__j_est = 0

    @property
    def j_est(self): return self.__j_est

    def __call__(self, dt, t, a):
        burnout = False
        if not self.__a_last is None:
            self.__j_est = (a - self.__a_last) / dt
            burnout = (self.__j_est < self.__j_trigger) and (a < self.__a_trigger)
        self.__a_last = a
        return burnout


if __name__ == '__main__':
    import os
    import os.path

    dt = 1e-3
    engine_file = 'Estes_A3.eng'
    engine_directory = 'Engines'
    rocket_masses = np.linspace(.035, .075, 5)
    piston_ref = Piston(0.85, 0.0138, m_additional=0.015, L0 = 0.05, Ffrict=0.5)
    engine = engines.Engine.RASP(os.path.join(engine_directory, engine_file))
    t_delay = 4
    print('Prop Weight = {} g'.format(1000*engine.propellent_mass))
    pistons = makePistonsFromRef(piston_ref, L0s=(0.01, 0.25), N=30)
    #batchPistonLaunch(pistons, rocket_masses, engine, dt=dt)

    result = simulatePistonLaunch(dt, piston_ref, 0.0223, engine)

    v_descent = 4.77
    fig, axs = plt.subplots(2, figsize=(18,12), sharex=True)
    #fig.tight_layout()

    idx = 0
    t_deployment = 0
    for idx, t in enumerate(result.ts):
        if t > engine.burn_time + t_delay:
            t_deployment = t
            break

    axs[0].plot(result.ts, result.xs, c='c', alpha=0.33)
    axs[0].axvline(result.t_launch, c='m', alpha=0.2)
    axs[0].axvline(result.t_liftoff_detection, c='r', alpha=0.8)
    axs[0].axvline(result.t_ascent, c='m', alpha=0.2)
    axs[0].axvline(result.t_burnout_detection, c='c', alpha=0.8)
    axs[0].axvline(engine.burn_time, c='k', alpha=0.8)
    axs[0].axvline(t_deployment, c='k', alpha=0.8)
    ax2 = axs[0].twinx()
    ax2.plot(result.ts, result.dxs, c='g', alpha=0.33)
    axs[1].plot(result.ts, result.Fs, c='c', alpha=0.33)
    axs[1].axvline(result.t_burnout_detection, c='c', alpha=0.8)
    axs[1].axvline(t_deployment, c='k', alpha=0.8)
    ax3 = axs[1].twinx()
    ax3.plot(result.ts, result.ddxs, c='g', alpha=0.33)
    # Fs = np.array(result.Fs)
    # Fs_eff = np.array(result.Fs_eff)
    # Fs_additional = Fs_eff - Fs
    # axs[2].plot(result.ts, result.Fs, c='c', alpha=0.33)
    # axs[2].plot(result.ts, result.Fs_eff, c='g', alpha=0.33)
    # ax4 = axs[2].twinx()
    # ax4.plot(result.ts, Fs_additional, c='m', alpha=0.33)
    x_apogee = np.max(result.xs)
    P_max = np.nanmax(result.Ps)
    print('Maximum Altitude = {}'.format(x_apogee))
    print('Maximum Velocity = {}'.format(np.max(result.dxs)))
    print('Optimum Delay = {}'.format(result.t_apogee - engine.burn_time))
    print('Time to Apogee (Optimum) = {}'.format(result.t_apogee))
    print('Maximum Pressure = {} Pa, {} Bar'.format(P_max, P_max/100000))

    v_deployment = result.dxs[idx]
    x_deployment = result.xs[idx]
    print('Velocity at deployment = {}'.format(v_deployment))
    print('Altitude at deployment = {}'.format(x_deployment))
    t_descent = x_deployment / v_descent
    t_total = t_deployment + t_descent
    print('Descent time (from deployment) = {}'.format(t_descent))
    print('Total Flight Time = {}'.format(t_total))
    print('Liftoff Time Estimate = {} s'.format(result.t_liftoff_detection))
    print('Burnout Time Estimate = {} s'.format(result.t_burnout_detection))
    print('Actual Motor Burnout = {} s'.format(engine.burn_time))
