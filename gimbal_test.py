import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import test_gimbal_linearization as gimbal
import pyrse.gimbal_pid as gimbal_pid
import pyrse.gimbal_planner as gimbal_planner
import pyrse.rocket_components as rc


# NOTE: THERE ARE A NUMBER OF THINGS ABOUT THIS TEST THAT DON'T SEEM TO MATCH
#   THE CURRENT USE OF THE PYRSE PACKAGE AND SHOULD, POTENTIALLY, BE RETHOUGHT
class TestRotationModel:
    @classmethod
    def __generate_random_params(cls, f_base, terms=5, amplitude_base=1):
        def initParams():
            return (
                amplitude_base,
                2 * math.pi * f_base * random.uniform(0.75, 1.25),
                random.uniform(0, 2*math.pi)
            )

        def nextParams(a0, b0, c0):
            return (
                a0*random.uniform(0.05, 0.15),
                b0*random.uniform(1.5, 3),
                random.uniform(0, 2*math.pi)
            )

        alpha_params = None
        for _ in range(terms):
            if alpha_params is None:
                params = initParams()
                alpha_params = []
            else:
                params = nextParams(*params)
            alpha_params.append(params)

        beta_params = None
        for _ in range(terms):
            if beta_params is None:
                params = initParams()
                beta_params = []
            else:
                params = nextParams(*params)
            beta_params.append(params)
        return alpha_params, beta_params

    @classmethod
    def Constrained(cls, theta_max=None, omega_max=None, domega_max=None, f_base=0.5, terms=9, test_samples=250):
        def accelerationRescaleParams(params):
            if len(params) > 0:
                t_cycle = (2*math.pi) / params[0][1]
                ts = np.linspace(0, t_cycle, test_samples)
                x_max = 0
                dx_max = 0
                ddx_max = 0
                for t in ts:
                    x = 0
                    dx = 0
                    ddx = 0
                    for a, b, c in params:
                        p = b*t + c
                        x += a * math.sin(p)
                        dx += a * b * math.cos(p)
                        ddx += -a * (b**2) * math.sin(p)
                    x = abs(x)
                    dx = abs(dx)
                    ddx = abs(ddx)
                    if x > x_max:
                        x_max = x
                    if dx > dx_max:
                        dx_max = dx
                    if ddx > ddx_max:
                        ddx_max = ddx
                amplitude_scale = 1
                if theta_max is not None:
                    amplitude_scale = min(theta_max / x_max, amplitude_scale)
                if omega_max is not None:
                    amplitude_scale = min(omega_max / dx_max, amplitude_scale)
                if domega_max is not None:
                    amplitude_scale = min(domega_max / ddx_max, amplitude_scale)
                new_params = []
                for a, b, c in params:
                    new_params.append((amplitude_scale*a, b, c))
                return new_params
            return params
        alpha_params, beta_params = cls.__generate_random_params(f_base, terms)
        alpha_params = accelerationRescaleParams(alpha_params)
        beta_params = accelerationRescaleParams(beta_params)
        return cls(alpha_params, beta_params)

    def __init__(self, alpha_params=[], beta_params=[]):
        self.__alpha_params = alpha_params
        self.__beta_params = beta_params

    def __call__(self, t):
        alpha_values = self.__calc_values(t, self.__alpha_params)
        beta_values = self.__calc_values(t, self.__beta_params)
        return alpha_values, beta_values

    def sample(self, t0, t1, fs):
        N = int((t1-t0) * fs) + 1
        ts = np.linspace(t0, t1, N)
        alphas = []
        dalphas = []
        ddalphas = []
        betas = []
        dbetas = []
        ddbetas = []
        for t in ts:
            alpha_values, beta_values = self.__call__(t)
            alphas.append(alpha_values[0])
            dalphas.append(alpha_values[1])
            ddalphas.append(alpha_values[2])
            betas.append(beta_values[0])
            dbetas.append(beta_values[1])
            ddbetas.append(beta_values[2])
        alphas = np.array(alphas)
        betas = np.array(betas)
        dalphas = np.array(dalphas)
        dbetas = np.array(dbetas)
        ddalphas = np.array(ddalphas)
        ddbetas = np.array(ddbetas)
        return ts, (alphas, betas), (dalphas, dbetas), (ddalphas, ddbetas)

    def __calc_values(self, t, params):
        x, dx, ddx = 0, 0, 0
        for a, b, c in params:
            p = b*t + c
            x += a * math.sin(p)
            dx += a * b * math.cos(p)
            ddx += -a * (b**2) * math.sin(p)
        return (x, dx, ddx)


class ConstantEngineModel:
    def __init__(self, T=1):
        self.__T = T

    @property
    def average_thrust(self):
        return self.__T

    def thrust(self, t):
        return self.__T


class TestRocketConfig:
    def __init__(self, p_F, structure, engine_model):
        self.__p_F = p_F
        self.__structure = structure
        self.engine_model = engine_model

    def mass(self, t):
        return self.__structure.mass(t)

    def cg(self, t):
        return self.__structure.cg(t)

    @property
    def p_F(self):
        return self.__p_F

    def mmoi(self, t):
        return self.__structure.mmoi(t)


def GeneratePIDs(a_limit):
    def makePID():
        pid = gimbal_pid.AngleControlPID(output_constraints=(-a_limit, a_limit))
        pid.theta_pid.Kp = 50
        pid.theta_pid.Kd = 1.5
        pid.theta_pid.e = 0.99
        pid.omega_pid.Kp = 100
        pid.omega_pid.kd = 5
        pid.omega_pid.e = 0.9
        return pid
    alpha_pid = makePID()
    beta_pid = makePID()
    return alpha_pid, beta_pid


def GenerateFeedforwards(dt, a_limit, theta_limit=math.pi/12, t_min=None):
    t_min = dt if t_min is None else t_min
    alpha_ff = gimbal_planner.AngleControlFeedforward(dt, theta_limit, a_limit, t_min=t_min)
    beta_ff = gimbal_planner.AngleControlFeedforward(dt, theta_limit, a_limit, t_min=t_min)
    return alpha_ff, beta_ff


def runPIDTest(t, fs, rotation_model, pids, p_sd=0, v_sd=0, a_sd=0, t_settling=2):
    ts, ps_ref, vs_ref, as_ref = rotation_model.sample(-t_settling, t, fs)
    dt = 1 / fs
    as_calc = [[], []]
    for idx, t in enumerate(ts):
        for ch in (0, 1):
            p_ref = ps_ref[ch][idx]
            v_ref = vs_ref[ch][idx]
            a_ref = as_ref[ch][idx]
            pids[ch].sp = p_ref
            pids[ch].v_sp = v_ref
            p_est = random.gauss(p_ref, p_sd)
            v_est = random.gauss(v_ref, v_sd)
            a_est = random.gauss(a_ref, a_sd)
            a_calc = pids[ch](dt, p_est, v_est, a_est)
            as_calc[ch].append(a_calc)

    print(pids[0])
    print(pids[1])
    as_calc = [np.array(as_calc[0]), np.array(as_calc[1])]
    fig, axs = plt.subplots(2, figsize=(15, 12))
    axs[0].set_title('Alpha')
    axs[0].plot(ts, as_calc[0], c='b', alpha=0.5)
    axs[0].plot(ts, as_ref[0], c='b', alpha=0.25)
    ax0 = axs[0].twinx()
    ax0.plot(ts, as_ref[0]-as_calc[0], c='g', alpha=0.5)
    axs[1].set_title('Beta')
    axs[1].plot(ts, as_calc[1], c='b', alpha=0.5)
    axs[1].plot(ts, as_ref[1], c='b', alpha=0.25)
    ax1 = axs[1].twinx()
    ax1.plot(ts, as_ref[1]-as_calc[1], c='g', alpha=0.5)
    fig.tight_layout()


def runGimbalTest(t, fs, rotation_model, gimbal_model, rocket_config, ctrls, linearized=True, domega_sd=0, noise_seed=None, t_settling=2.0):
    if noise_seed is not None:
        random.seed(noise_seed)

    dt = 1.0 / fs
    ts, p_ref, v_ref, a_ref = rotation_model.sample(-t_settling, t, fs)
    p_F = rocket_config.p_F
    alpha_cmds = []
    beta_cmds = []
    alpha = p_ref[0][0]
    beta = p_ref[1][0]
    dalpha = v_ref[0][0]
    dbeta = v_ref[1][0]
    ddalpha = a_ref[0][0]
    ddbeta = a_ref[1][0]
    alphas = []
    betas = []
    dalphas = []
    dbetas = []
    ddalphas = []
    ddbetas = []
    alpha_ctrl, beta_ctrl = ctrls
    N_settling = 0
    for idx, t in enumerate(ts):
        settling = t < 0
        p_cg = rocket_config.cg(t)
        alpha_ctrl.sp = p_ref[0][idx]
        alpha_ctrl.v_sp = v_ref[0][idx]
        ddalpha_tgt = alpha_ctrl(dt, alpha, dalpha, ddalpha)
        beta_ctrl.sp = p_ref[1][idx]
        beta_ctrl.v_sp = v_ref[1][idx]
        ddbeta_tgt = beta_ctrl(dt, beta, dbeta, ddbeta)
        domega = np.array([0, ddalpha_tgt, ddbeta_tgt])
        T = rocket_config.engine_model.thrust(t)
        MMOI = np.diag(rocket_config.mmoi(t))
        alpha_current, beta_current = gimbal_model.pos(t)
        alpha_set, beta_set = gimbal.gimbal_angles(domega, p_cg, p_F, T, alpha_current, beta_current, MMOI, linearized=linearized, N=5)
        alpha_current, beta_current = gimbal_model.command(t, alpha_set, beta_set)
        domega = gimbal.angular_acceleration(p_cg, p_F, T, alpha_current, beta_current, MMOI)
        ddalpha = domega[1]
        ddbeta = domega[2]
        ddalpha_est = random.gauss(ddalpha, domega_sd)
        ddbeta_est = random.gauss(ddbeta, domega_sd)
        alpha += (dt * dalpha + ((dt**2) * ddalpha_est / 2))
        beta += (dt * dbeta + ((dt**2) * ddbeta_est / 2))
        dalpha += dt * ddalpha_est
        dbeta += dt * ddbeta_est
        if not settling:
            ddalphas.append(ddalpha)
            ddbetas.append(ddbeta)
            alphas.append(alpha)
            betas.append(beta)
            dalphas.append(dalpha)
            dbetas.append(dbeta)
            alpha_cmds.append(alpha_set)
            beta_cmds.append(beta_set)
        else:
            N_settling += 1
        gimbal_model.command(t, alpha_set, beta_set)
    alpha_cmds = np.array(alpha_cmds)
    beta_cmds = np.array(beta_cmds)

    p_ref = (p_ref[0][N_settling:], p_ref[1][N_settling:])
    v_ref = (v_ref[0][N_settling:], v_ref[1][N_settling:])
    a_ref = (a_ref[0][N_settling:], a_ref[1][N_settling:])

    p_calc = (np.array(alphas), np.array(betas))
    v_calc = (np.array(dalphas), np.array(dbetas))
    a_calc = (np.array(ddalphas), np.array(ddbetas))
    return ts[N_settling:], (p_ref, v_ref, a_ref), (p_calc, v_calc, a_calc), (alpha_cmds, beta_cmds)


def plotGimbalTest(title, fs, ts, refs, calcs, cmds, domega_limit=None):
    (p_ref, v_ref, a_ref) = refs
    (p_calc, v_calc, a_calc) = calcs
    (alpha_cmds, beta_cmds) = cmds
    fig, axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    fig.suptitle(title)

    axs[0].plot(ts, p_ref[0], c='b', alpha=0.25)
    axs[0].plot(ts, p_ref[1], c='g', alpha=0.25)
    axs[1].plot(ts, v_ref[0], c='b', alpha=0.25)
    axs[1].plot(ts, v_ref[1], c='g', alpha=0.25)
    axs[2].plot(ts, a_ref[0], c='b', alpha=0.25)
    axs[2].plot(ts, a_ref[1], c='g', alpha=0.25)
    # axs[3].plot(ts, alpha_cmds, c='b')
    # axs[3].plot(ts, beta_cmds, c='g')

    axs[0].plot(ts, p_calc[0], c='b', alpha=0.5)
    axs[0].plot(ts, p_calc[1], c='g', alpha=0.5)
    axs[1].plot(ts, v_calc[0], c='b', alpha=0.5)
    axs[1].plot(ts, v_calc[1], c='g', alpha=0.5)
    axs[2].plot(ts, a_calc[0], c='b', alpha=0.5)
    axs[2].plot(ts, a_calc[1], c='g', alpha=0.5)

    if domega_limit is not None:
        ylim = axs[2].get_ylim()
        y_max = max(*ylim)
        y_min = min(*ylim)
        if y_max > domega_limit:
            axs[2].axhspan(domega_limit, y_max, facecolor='r', alpha=0.15)
        if y_min < -domega_limit:
            axs[2].axhspan(y_min, -domega_limit, facecolor='r', alpha=0.15)
    fig.tight_layout()

    # fig, axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    # fig.suptitle(title + ' -- Errors')
    # # axs[0].psd(p_ref[0] - p_calc[0], fs, c='b', alpha=0.5)
    # # axs[0].psd(p_ref[1] - p_calc[1], fs, c='g', alpha=0.5)
    # # axs[1].psd(v_ref[0] - v_calc[0], fs, c='b', alpha=0.5)
    # # axs[1].psd(v_ref[1] - v_calc[1], fs, c='g', alpha=0.5)
    # # axs[2].psd(a_ref[0] - a_calc[0], fs, c='b', alpha=0.5)
    # # axs[2].psd(a_ref[1] - a_calc[1], fs, c='g', alpha=0.5)
    # sns.regplot(ts, p_ref[0] - p_calc[0], ax=axs[0])
    # sns.regplot(ts, p_ref[1] - p_calc[1], ax=axs[0])
    # sns.regplot(ts, v_ref[0] - v_calc[0], ax=axs[1])
    # sns.regplot(ts, v_ref[1] - v_calc[1], ax=axs[1])
    # sns.regplot(ts, a_ref[0] - a_calc[0], ax=axs[2])
    # sns.regplot(ts, a_ref[1] - a_calc[1], ax=axs[2])
    #
    # fig.tight_layout()


if __name__ == '__main__':
    import pyrse.gimbal_model as gimbal_model
    fs = 75
    t_max = 30
    f_base = 0.25
    domega_sd = 0.5
    gimbal_max_angle = math.radians(12)

    dt = 1 / fs
    gimbal_model = gimbal_model.GimbalPhysicalModel()
    engine_model = ConstantEngineModel(T=2.3)
    p_F = np.array([0.8, 0, 0])
    rocket_structure = rc.TubeComponent(0.85, 0.0405, 0.0415, None, 0.047)
    nosecone = rc.Component(pos=np.array([-0.1, 0, 0]), mass=0.025)
    flight_computer = rc.Component(pos=np.array([0.5, 0, 0]), mass=0.020)
    gimbal_component = rc.Component(pos=p_F, mass=0.05)
    rocket_structure.add(nosecone)
    rocket_structure.add(flight_computer)
    rocket_structure.add(gimbal_component)
    rocket_config = TestRocketConfig(p_F, rocket_structure, engine_model)

    p_cg = rocket_config.cg(0)
    MOI = np.diag(rocket_config.mmoi(0))
    a_limit = abs(0.95 * (gimbal.angular_acceleration(p_cg, p_F, engine_model.average_thrust, gimbal_max_angle, 0, MOI)[2]))
    print('Acceleration Limit = {} rad/s^2'.format(a_limit))

    random.seed(12345)  # Always produce the same rotational system... This is good for debugging
    # rotation_model = TestRotationModel.AccelerationLimited(0.9 * a_limit, f_base=0.025)
    rotation_model = TestRotationModel.Constrained(theta_max=math.pi/6, domega_max=a_limit, f_base=f_base)

    ts, p_ref, v_ref, a_ref = rotation_model.sample(0, t_max, fs)

    # fig, axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    # for p, v, a in zip(p_ref, v_ref, a_ref):
    #     axs[0].plot(ts, p)
    #     axs[1].plot(ts, v)
    #     axs[2].plot(ts, a)

    pids = GeneratePIDs(a_limit)
    ts, reference, results, angles = runGimbalTest(t_max, fs, rotation_model, gimbal_model, rocket_config, pids, domega_sd=domega_sd, noise_seed=12345)
    plotGimbalTest('PID Tracking', fs, ts, reference, results, angles, domega_limit=a_limit)

    ffs = GenerateFeedforwards(dt, a_limit, t_min=0.5)
    ts, reference, results, angles = runGimbalTest(t_max, fs, rotation_model, gimbal_model, rocket_config, ffs, domega_sd=domega_sd, noise_seed=12345)
    plotGimbalTest('Feedforward Tracking', fs, ts, reference, results, angles, domega_limit=a_limit)

    # pids = GeneratePIDs(a_limit)
    # runPIDTest(t_max, fs, rotation_model, pids)

    plt.show()
