import math
import random

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import seaborn as sns


def angleBetweenQs(q1, q2):
    q2i = q2.inverse()
    qm = q1 * q2i
    theta = 2 * math.asin(math.sqrt(qm.x**2 + qm.y**2 + qm.z**2))
    return theta


class RotationModel:
    @classmethod
    def __generate_random_params(cls, f_base, terms=5, amplitude_base=1, zero_init=False):
        if isinstance(f_base, list) or isinstance(f_base, tuple):
            f_base = tuple(f_base)
        else:
            f_base = (f_base, f_base, f_base)

        def initParams(base_frequency):
            return (
                amplitude_base,
                2 * math.pi * base_frequency * random.uniform(0.75, 1.25),
                0 if zero_init else random.uniform(0, 2*math.pi)
            )

        def nextParams(a0, b0, c0):
            return (
                a0*random.uniform(0.075, 0.3),
                b0*random.uniform(1.1, 2.75),
                0 if zero_init else random.uniform(0, 2*math.pi)
            )

        alpha_params = None
        for _ in range(terms):
            if alpha_params is None:
                params = initParams(f_base[0])
                alpha_params = []
            else:
                params = nextParams(*params)
            alpha_params.append(params)

        beta_params = None
        for _ in range(terms):
            if beta_params is None:
                params = initParams(f_base[1])
                beta_params = []
            else:
                params = nextParams(*params)
            beta_params.append(params)

        gamma_params = None
        for _ in range(terms):
            if gamma_params is None:
                params = initParams(f_base[2])
                gamma_params = []
            else:
                params = nextParams(*params)
            gamma_params.append(params)
        return alpha_params, beta_params, gamma_params

    @classmethod
    def Constrained(cls, theta_max=None, omega_max=None, domega_max=None, f_base=0.5, terms=9, test_samples=250, zero_init=True):
        if theta_max is None:
            theta_max = None
        elif isinstance(theta_max, list) or isinstance(theta_max, tuple):
            theta_max = tuple(theta_max)
        else:
            theta_max = (theta_max, theta_max, theta_max)

        if omega_max is None:
            omega_max = None
        elif isinstance(omega_max, list) or isinstance(omega_max, tuple):
            omega_max = tuple(omega_max)
        else:
            omega_max = (omega_max, omega_max, omega_max)

        if domega_max is None:
            domega_max = None
        elif isinstance(domega_max, list) or isinstance(domega_max, tuple):
            domega_max = tuple(domega_max)
        else:
            domega_max = (domega_max, domega_max, domega_max)

        def accelerationRescaleParams(params, idx):
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
                    amplitude_scale = min(theta_max[idx] / x_max, amplitude_scale)
                if omega_max is not None:
                    amplitude_scale = min(omega_max[idx] / dx_max, amplitude_scale)
                if domega_max is not None:
                    amplitude_scale = min(domega_max[idx] / ddx_max, amplitude_scale)
                new_params = []
                for a, b, c in params:
                    new_params.append((amplitude_scale*a, b, c))
                return new_params
            return params
        alpha_params, beta_params, gamma_params = cls.__generate_random_params(f_base, terms, zero_init)
        alpha_params = accelerationRescaleParams(alpha_params, 0)
        beta_params = accelerationRescaleParams(beta_params, 1)
        gamma_params = accelerationRescaleParams(gamma_params, 2)
        return cls(alpha_params, beta_params, gamma_params)

    def __init__(self, alpha_params=[], beta_params=[], gamma_params=[]):
        self.__alpha_params = alpha_params
        self.__beta_params = beta_params
        self.__gamma_params = gamma_params

    def __call__(self, t):
        alpha_values = self.__calc_values(t, self.__alpha_params)
        beta_values = self.__calc_values(t, self.__beta_params)
        gamma_values = self.__calc_values(t, self.__gamma_params)
        return alpha_values, beta_values, gamma_values

    def sample(self, t0, t1, fs, q0=None):
        q0 = np.quaternion(1, 0, 0, 0) if q0 is None else q0
        N = int((t1-t0) * fs) + 1
        ts = np.linspace(t0, t1, N)
        alphas = []
        dalphas = []
        ddalphas = []
        betas = []
        dbetas = []
        ddbetas = []
        gammas = []
        dgammas = []
        ddgammas = []
        for t in ts:
            alpha_values, beta_values, gamma_values = self.__call__(t)
            alphas.append(alpha_values[0])
            dalphas.append(alpha_values[1])
            ddalphas.append(alpha_values[2])
            betas.append(beta_values[0])
            dbetas.append(beta_values[1])
            ddbetas.append(beta_values[2])
            gammas.append(gamma_values[0])
            dgammas.append(gamma_values[1])
            ddgammas.append(gamma_values[2])
        alphas = np.array(alphas)
        betas = np.array(betas)
        gammas = np.array(gammas)
        dalphas = np.array(dalphas)
        dbetas = np.array(dbetas)
        dgammas = np.array(dgammas)
        ddalphas = np.array(ddalphas)
        ddbetas = np.array(ddbetas)
        ddgammas = np.array(ddgammas)
        qws = [q0]  # TODO: INTEGRATE THE SYSTEM TO GET THE WORLD ORIENTATION BASED ON AN INITIAL ORIENTATION
        q = q0
        dt = 1/fs
        for dalpha, dbeta, dgamma in zip(dalphas[:-1], dbetas[:-1], dgammas[:-1]):
            q = q + 0.5 * dt * q * np.quaternion(dalpha, dbeta, dgamma)
            qws.append(q)
        qws = np.array(qws)
        return ts, qws, (alphas, betas, gammas), (dalphas, dbetas, dgammas), (ddalphas, ddbetas, ddgammas)

    def __calc_values(self, t, params):
        x, dx, ddx = 0, 0, 0
        for a, b, c in params:
            p = b*t + c
            x += a * math.sin(p)
            dx += a * b * math.cos(p)
            ddx += -a * (b**2) * math.sin(p)
        return (x, dx, ddx)


class RotationOrientationIntegrator:
    def __init__(self, oversampling=None, dt_max=1e-1):
        self.__oversampling = oversampling
        self.__dt_max = dt_max

    def __call__(self, dt, omegas, domegas=None, q0=None):
        min_oversampling = int(dt / self.__dt_max)
        oversampling = min_oversampling if self.__oversampling is None else max(min_oversampling, self.__oversampling)
        dt_interp = dt / oversampling

        def interpolatedOmegas(omega_0, omega_1, domega_0, domega_1):
            ts_est = np.linspace(0, dt, oversampling)
            a = (-2/dt)*(((omega_1 - omega_0)/dt**2) - ((domega_1 + domega_0)/(2*dt)))
            b = (domega_1 - domega_0 - (3 * a * dt**2)) / (2*dt)
            omegas_interp = (a * np.power(ts_est, 3)) + (b * np.square(ts_est)) + (domega_0*ts_est) + omega_0
            # domegas_est = (3*a*np.square(ts_est)) + (2*b*ts_est) + domega_0
            return omegas_interp

        q0 = np.quaternion(1, 0, 0, 0) if q0 is None else q0
        q = q0
        qws = [q0]
        omegas = np.array(omegas)
        if domegas is None:
            domegas = (omegas[:-1] - omegas[1:]) / dt
        else:
            domegas = np.array(domegas)
        for idx in range(len(omegas[0])-1):
            alpha_0, alpha_1 = omegas[0][idx], omegas[0][idx+1]
            dalpha_0, dalpha_1 = domegas[0][idx], domegas[0][idx+1]
            beta_0, beta_1 = omegas[1][idx], omegas[1][idx+1]
            dbeta_0, dbeta_1 = domegas[1][idx], domegas[1][idx+1]
            gamma_0, gamma_1 = omegas[2][idx], omegas[2][idx+1]
            dgamma_0, dgamma_1 = domegas[2][idx], domegas[2][idx+1]
            alphas = interpolatedOmegas(alpha_0, alpha_1, dalpha_0, dalpha_1)
            betas = interpolatedOmegas(beta_0, beta_1, dbeta_0, dbeta_1)
            gammas = interpolatedOmegas(gamma_0, gamma_1, dgamma_0, dgamma_1)
            for alpha, beta, gamma in zip(alphas, betas, gammas):
                q = q + 0.5 * dt_interp * q * np.quaternion(0, alpha, beta, gamma)
            qws.append(q)
        return np.array(qws)


def plotRotationModel(title, fs, t0, t1, rotation_model, oversampling=None, offset_fig=None, offset_ax=None):
    ts, qws, p_ref, v_ref, a_ref = rotation_model.sample(t0, t1, fs)
    fig, axs = plt.subplots(3, figsize=(15, 12), sharex=True)
    #fig.suptitle(title)
    # N = len(ts)
    # v_ref = [[0]*N, [0]*N, [0]*N]
    # a_ref = [[0]*N, [0]*N, [0]*N]
    dt = 1.0 / fs
    interpolator = RotationOrientationIntegrator(oversampling=oversampling)
    qws = interpolator(dt, v_ref, a_ref)

    axs[0].plot(ts, p_ref[0], c='b', alpha=0.5, label=r'$\alpha_{c}$')
    axs[0].plot(ts, p_ref[1], c='g', alpha=0.5, label=r'$\beta_{c}$')
    axs[0].plot(ts, p_ref[2], c='k', alpha=0.5, label=r'$\gamma_{c}$')
    axs[0].set_title('Angular Position')
    axs[0].set_ylabel('$rad$')
    axs[1].plot(ts, v_ref[0], c='b', alpha=0.5, label=r'$\dot{\alpha}_{c}$')
    axs[1].plot(ts, v_ref[1], c='g', alpha=0.5, label=r'$\dot{\beta}_{c}$')
    axs[1].plot(ts, v_ref[2], c='k', alpha=0.5, label=r'$\dot{\gamma}_{c}$')
    axs[1].set_title('Angular Velocity')
    axs[1].set_ylabel('$rad-s^{-1}$')
    axs[2].plot(ts, a_ref[0], c='b', alpha=0.5, label=r'$\ddot{\alpha}_{c}$')
    axs[2].plot(ts, a_ref[1], c='g', alpha=0.5, label=r'$\ddot{\beta}_{c}$')
    axs[2].plot(ts, a_ref[2], c='k', alpha=0.5, label=r'$\ddot{\gamma}_{c}$')
    axs[2].set_title('Angular Acceleration')
    axs[2].set_xlabel('Time ($s$)')
    axs[2].set_ylabel('$rad-s^{-2}$')
    for ax in axs:
        ax.legend()
    fig.tight_layout()

    q0 = np.quaternion(1, 0, 0, 0)
    if offset_ax is None:
        offset_fig, offset_ax = plt.subplots(1, figsize=(18, 10))
    errs = [angleBetweenQs(q0, q) for q in qws]
    sns.lineplot(x=ts, y=errs, ax=offset_ax)
    offset_ax.set_title('Angle From Initial Orientation')
    # offset_fig.tight_layout()
    return ts, qws


def generateParamVector(p_min, p_max, N=3):
    vs = []
    for _ in range(N):
        vs.append(random.uniform(p_min, p_max))
    return tuple(vs)


if __name__ == '__main__':
    random.seed(1234)  # Always produce the same rotational system

    fs = 50
    t_max = 60
    f_base = generateParamVector(0.025, 0.075)
    theta_max = generateParamVector(math.pi/20, math.pi/12)
    v_limit = generateParamVector(math.radians(1000), math.radians(2000))  # rad/s
    terms = 9

    fig, ax = plt.subplots(1, figsize=(18, 10))

    random.seed(1234)  # Always produce the same rotational system
    rotation_model = RotationModel.Constrained(theta_max=theta_max, omega_max=v_limit, f_base=f_base, terms=terms)
    ts, qws_0 = plotRotationModel('Clean Rotation Model', fs, 0, t_max, rotation_model, oversampling=20, offset_fig=fig, offset_ax=ax)

    random.seed(1234)  # Always produce the same rotational system
    rotation_model = RotationModel.Constrained(theta_max=theta_max, omega_max=v_limit, f_base=f_base, terms=terms)
    ts, qws_1 = plotRotationModel('Clean Rotation Model', fs, 0, t_max, rotation_model, oversampling=2, offset_fig=fig, offset_ax=ax)

    ax1 = ax.twinx()
    errs = [angleBetweenQs(q0, q1) for q0, q1 in zip(qws_0, qws_1)]
    sns.lineplot(x=ts, y=errs, ax=ax1)
    fig.tight_layout()

    plt.show()
