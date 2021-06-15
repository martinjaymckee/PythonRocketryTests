import cmath
import math
import random


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import progress.bar
import seaborn as sns


class PlannerBinarySearch:
    def __init__(self, t_update_limit, N=15, domega_tolerance=5e-2, t_init=0.2):
        self.__t_update_limit = t_update_limit
        self.__N = N
        self.__domega_tolerance = domega_tolerance
        self.__steps = 0
        self.__t_min = t_update_limit
        self.__t_max = None
        self.__scale = 1.25
        self.__t = t_init

    @property
    def t(self): return self.__t

    def clearSteps(self):
        self.__steps = 0
        self.__t_min = self.__t_update_limit
        self.__t_max = None

    def __call__(self, dt, domega_max, domega_limits, scale=None):
        t_start = self.__t
        scale = self.__scale if scale is None else scale
        done = False
        if domega_max > (domega_limits + self.__domega_tolerance):
            if self.__t_max is None:
                self.__t_min = max(self.__t, self.__t_update_limit)
                self.__t *= scale
            else:
                self.__t_min = self.__t
                if ((self.__t_max - self.__t) / 2) <= self.__t_update_limit:
                    done = True
                self.__t = (self.__t + self.__t_max) / 2
        elif domega_max < (domega_limits - self.__domega_tolerance):
            if self.__t <= self.__t_update_limit:
                self.__t = self.__t_update_limit
                done = True
            else:
                self.__t_max = self.__t
                if ((self.__t - self.__t_min) / 2) <= self.__t_update_limit:
                    done = True
                self.__t = (self.__t + self.__t_min) / 2
        else:
            done = True
        self.__steps += 1
        done = done or (self.__steps >= self.__N)
        t = self.__t
        if done:
            self.__t = max(0, self.__t - dt)
        if self.__t == 0:
            assert False, print('Time is zero!!!!, t_start = {}, domega_max = {}, domega_limits = {}, t_min = {}, t_max = {}'.format(t_start, domega_max, domega_limits, self.__t_min, self.__t_max))
        return done, t


class GimbalPlanner:
    def __init__(self, dt, t_init=0.4, t_min=1.0, t_max=10.0):
        self.__dt = dt
        self.__t_max = t_max
        self.__time_search = PlannerBinarySearch(t_min, t_init=t_init)

    # TODO: PASS IN CURRENT TIME AND TARGET TIME TO CALCULATE MINIMUM TRANSITION TIME AND CONSTRAIN RESULTS TO THAT.
    #       FIRST, IF T_MINIMUM IS > 0, CALCULATE DOMEGA_MAX AND IF IT'S VALID, SET T = T_MINIMUM
    #       IF T_MINIMUM DOES NOT WORK, INITIALIZE THE SEARCH WITH  T_MIN = T_MINIMUM AND T=T_MINIMUM, THEN SEARCH.
    def __call__(self, theta_cmd, domega_limits, measurements, debug=False, scale=None):
        theta_est, omega_est, domega_est = measurements

        def solve(t):
            b = None
            A = None
            try:
                b = np.array([
                    theta_cmd - (domega_est*(t**2)/2) - (omega_est*t) - theta_est,
                    -(domega_est*t) - omega_est,
                    -domega_est
                ])
                A = np.array([
                    [((t**5)/20), ((t**4)/12), ((t**3)/6)],
                    [((t**4)/4), ((t**3)/3), ((t**2)/2)],
                    [(t**3), (t**2), t]
                ])
                x = np.linalg.solve(A, b)
                return x[0], x[1], x[2]
            except Exception as e:
                # print('t = {}'.format(t))
                # print('b = {}'.format(b))
                # print('A = {}'.format(A))
                # assert False, str(e)
                raise e

        def predict(t, A, B, C):
            theta = (A*(t**5)/20) + (B*(t**4)/12) + (C*(t**3)/6) + (domega_est*(t**2)/2) + (omega_est*t) + theta_est
            omega = (A*(t**4)/4) + (B*(t**3)/3) + (C*(t**2)/2) + (domega_est*t) + omega_est
            domega = (A*(t**3)) + (B*(t**2)) + (C*t) + domega_est
            return theta, omega, domega

        def domegaMax(t, A, B, C):
            det = cmath.sqrt((B**2) - (3*A*C))
            tms = [(-B + det) / (3*A), (-B - det) / (3*A), t]
            domega_max = None
            for tm in tms:
                if 0 < tm <= t:
                    _, _, domega = predict(tm, A, B, C)
                    if (domega_max is None) or (abs(domega) > domega_max):
                        domega_max = abs(domega)
            # if domega_max is None:
            #     _, _, domega0 = predict(0, A, B, C)
            #     _, _, domega1 = predict(t, A, B, C)
            #     domega_max = max(domega0, domega1)
            return domega_max

        A, B, C = None, None, None
        ##############################################################
        if debug:
            N = 1
            cmap = plt.get_cmap('viridis')
            fig, axs = plt.subplots(3, figsize=(15, 12),  sharex=True)
            fig.tight_layout()
            axs[0].axhline(theta_cmd, c='k', alpha=0.25)
            axs[1].axhline(0, c='k', alpha=0.25)
            axs[2].axhline(0, c='k', alpha=0.25)
            axs[0].axvline(0, c='k', alpha=0.25)
            axs[1].axvline(0, c='k', alpha=0.25)
            axs[2].axvline(0, c='k', alpha=0.25)
            axs[0].axvline(self.__dt, c='m', alpha=0.25)
            axs[1].axvline(self.__dt, c='m', alpha=0.25)
            axs[2].axvline(self.__dt, c='m', alpha=0.25)
            domega_display_max = None
        #############################################################
        self.__time_search.clearSteps()
        done = False
        t = self.__time_search.t
        M = 0
        while not done:
            # t = max(t, self.__dt)  # NOTE THIS IS A HACK TO AVOID A SINGULAR MATRIX IN SOLVE...
            M += 1
            A, B, C = solve(t)
            domega_max = domegaMax(t, A, B, C)
            #######################################################
            if debug:
                print('domega_max = {}'.format(domega_max))
                ts = np.linspace(0, t, 20)
                thetas = []
                omegas = []
                domegas = []
                for t_x in ts:
                    theta, omega, domega = predict(t_x, A, B, C)
                    thetas.append(theta)
                    omegas.append(omega)
                    domegas.append(domega)
                ratio = min(domega_max, domega_limits) / max(domega_max, domega_limits)
                color = cmap(1-ratio)
                alpha = math.sqrt(ratio)
                axs[0].plot(ts, thetas, c=color, alpha=alpha, label='Step {}: t = {:0.3f}s'.format(N, t))
                axs[0].axvline(t, c='k', alpha=0.05)
                #axs[0].annotate('{}'.format(N), (t, (1-alpha)*theta_cmd))
                axs[1].plot(ts, omegas, c=color, alpha=alpha)
                axs[1].axvline(t, c='k', alpha=0.05)
                axs[1].annotate('{}'.format(N), (t, 0.9*max(*omegas)))
                axs[2].plot(ts, domegas, c=color, alpha=alpha)
                axs[2].axvline(t, c='k', alpha=0.05)
                if domega_display_max is None or max(*domegas) > domega_display_max:
                    domega_display_max = max(*domegas)
                #axs[2].annotate('{}'.format(N), (t, 0.3*domega_display_max))
                N += 1
            ######################################################
            done, t = self.__time_search(self.__dt, domega_max, domega_limits, scale=scale)

        ###############################################################
        if debug:
            axs[0].legend()
            plt.show()
            domega_err = domega_max - domega_limits
            print('Acceleration Error = {} ({} %)'.format(domega_err, 100 * (domega_err / domega_limits)))
            theta_cmd, omega_cmd, domega_cmd = predict(t, A, B, C)
            print('Final: t = {}, theta = {}, omega = {}, domega = {}'.format(t, theta_cmd, omega_cmd, domega_cmd))
        #############################################################
        theta_cmd, omega_cmd, domega_cmd = predict(self.__dt, A, B, C)
        return t, M, theta_cmd, omega_cmd, domega_cmd


class GimbalPlannerController:
    def __init__(self, theta_offset, domega_max, scale_min=1.15, scale_max=1.5):
        self.__theta_offset = theta_offset
        self.__domega_max = domega_max
        self.__domega_min = 0.25 * domega_max
        self.__scale_min = scale_min
        self.__scale_max = scale_max
        self.__theta_cmd_last = 0
        self.__t_correct = math.sqrt(theta_offset / domega_max)
        self.__omega_max = domega_max * self.__t_correct
        self.__A = (theta_offset**2) / (self.__omega_max**2)

    def __call__(self, theta_cmd, theta_est, omega_cmd, omega_est, domega_est):
        theta_cmd_err = abs(self.__theta_cmd_last - theta_cmd)
        self.__theta_cmd_last = theta_cmd
        scale = self.__scale_min if theta_cmd_err < (self.__theta_offset / 5) else self.__scale_max
        theta_err = abs(theta_cmd - theta_est)
        omega_err = abs(omega_cmd - omega_est)
        err_max = (self.__theta_offset**2)
        domega_range = self.__domega_max - self.__domega_min
        err = (theta_err**2) + (self.__A * (omega_err**2))
        domega = self.__domega_min + (domega_range * (min(err_max, err) / err_max))
        return scale, self.__domega_max if domega < domega_est else domega


class GimbalAxisModel:
    def __init__(self, p_i, v_i=0, a_i=0):
        self.__p = p_i
        self.__v = v_i
        self.__a = a_i

    def __call__(self, dt, a_cmd):
        a = a_cmd
        self.__a = a_cmd
        v_last = self.__v
        self.__v += (dt * a)
        v = v_last
        self.__p += (dt * v + (((dt**2)*a)/2))
        return self.__p, self.__v, self.__a


class AngleControlFeedforward:
    def __init__(self, dt, theta_limit, domega_limit, t_min=0.5):
        self.__planner = GimbalPlanner(dt, t_min=t_min)
        self.__controller = GimbalPlannerController(theta_limit, domega_limit)
        self.__goal = [0, 0, 0]

    @property
    def sp(self): return self.__goal[0]

    @sp.setter
    def sp(self, _sp):
        self.__goal[0] = _sp
        return self.sp

    @property
    def v_sp(self): return self.__goal[1]

    @v_sp.setter
    def v_sp(self, _sp):
        self.__goal[1] = _sp
        return self.v_sp

    @property
    def controller(self):
        return self.__controller

    @property
    def planner(self):
        return self.__planner

    def __call__(self, dt, theta, omega, domega):
        state = (theta, omega, domega)
        scale, a_lim = self.__controller(self.__goal[0], state[0], self.__goal[1], state[1], state[2])
        _, _, _, _, domega_cmd = self.__planner(self.__goal[0], a_lim, state, scale=scale)
        return domega_cmd


def runTrackingTest(dt, p_f, p_i, v_i, a_i, a_limit, p_err=2.5e-2, v_err=5e-2, a_sd=0, v_sd=0, p_sd=0, v_f=0, t_hold=0.5, fig=None, axs=None, doPlot=False):
    if doPlot:
        print('Tracking Test:')
        print('\tTarget: theta = {}, omega = {}, domega = {}'.format(p_f, v_f, 0))
    planner = GimbalPlanner(dt)
    axis_model = GimbalAxisModel(p_i, v_i, a_i)
    gimbal_controller = GimbalPlannerController(math.pi/12, a_limit)
    p, v, a = p_i, v_i, a_i
    done = False
    ts = []
    t_preds = []
    pos = []
    vels = []
    accels = []
    accel_ests = []
    a_lims = []
    scales = []
    iterations = []
    t = 0
    t_hold_start = None
    while not done:
        scale, a_lim = gimbal_controller(p_f, p, v_f, v, a)
        t_pred, M, p_cmd, v_cmd, a_cmd = planner(p_f, a_lim, (p, v, a), scale=scale)
        a_est = random.gauss(a_cmd, a_sd)
        p, v, a = axis_model(dt, a_est)
        t += dt
        ts.append(t)
        pos.append(p)
        vels.append(v)
        accels.append(a_cmd)
        accel_ests.append(a_est)
        t_preds.append(t_pred)
        scales.append(scale)
        a_lims.append(a_lim)
        iterations.append(M)
        if (t_hold_start is None) and (abs(p_f - p) <= p_err) and (abs(v_f - v) < v_err):
            t_hold_start = t
        done = (t_hold_start is not None) and ((t - t_hold_start) >= t_hold)

    data = {
        'ts': ts,
        'pos': pos,
        'vels': vels,
        'accels': accels,
        'accel_ests': accel_ests,
        't_preds': t_preds,
        'scales': scales,
        'a_lims': a_lims,
        'iterations': iterations
    }

    if doPlot:
        if fig is None:
            fig, axs = plt.subplots(5, figsize=(15, 12), sharex=True)
            # fig.tight_layout()
            fig.suptitle('Tracking (dt = {} $s$, a_lim = {} $rad/s^2$, a_sd = {} $rad/s^2$)'.format(dt, a_limit, a_sd), fontsize=16)
            axs[0].axhline(p_f, c='k', alpha=0.25)
            axs[0].set_title('Position ($rad$)')
            axs[1].axhline(v_f, c='k', alpha=0.25)
            axs[1].set_title('Angular Velocity ($rad/s$)')
            axs[2].axhline(0, c='k', alpha=0.25)
            axs[2].set_title('Angular Acceleration ($rad/s^{2}$)')
        axs[0].plot(ts, pos, c='c', alpha=0.15)
        axs[0].axvline(t_hold_start, c='k', alpha=0.25)
        axs[1].plot(ts, vels, c='c', alpha=0.15)
        axs[1].axvline(t_hold_start, c='k', alpha=0.25)
        axs[2].plot(ts, accels, c='c', alpha=0.15)
        axs[2].plot(ts, accel_ests, c='y', alpha=0.2)
        axs[2].axvline(t_hold_start, c='k', alpha=0.25)
        axs[3].plot(ts, a_lims, c='c', alpha=0.15)
        axs[3].axvline(t_hold_start, c='k', alpha=0.25)
        ax3 = axs[3].twinx()
        ax3.plot(ts, scales, c='y', alpha=0.15)
        axs[4].plot(ts, t_preds, c='c', alpha=0.15)
        axs[4].axvline(t_hold_start, c='k', alpha=0.25)
        ax4 = axs[4].twinx()
        ax4.plot(ts, iterations, c= 'y', alpha=0.15)
        print('\tTracking Complete at t = {} s'.format(t))
        print('\tFinal: theta = {}, omega = {}, domega = {}'.format(pos[-1], vels[-1], accels[-1]))
        print('\tErrors: theta = {}, omega = {}, domega = {}'.format(p_f-pos[-1], v_f-vels[-1], accels[-1]))
        print()
    return fig, axs, t_hold_start, data


def runNoiseSensitivityTest(dt, p_f, p_i, v_i, a_i, a_limit, a_sd_range=None, N=75, M=55, debug=False):
    bar = progress.bar.ShadyBar('Processing', max=M, suffix='%(percent)d%%')
    a_sd_range = (0, 0.2*a_limit) if a_sd_range is None else a_sd_range
    tt_means = []
    tt_sds = []
    ttss = []
    as_sd = np.linspace(*a_sd_range, M)
    test_a_sds = []
    failures = []
    for a_sd in as_sd:
        try:
            bar.next()
            tts = []
            datas = []
            failure_count = 0
            for _ in range(N):
                try:
                    _, _, tt, data = runTrackingTest(dt, p_f, p_i, v_i, a_i, a_limit, a_sd=a_sd)
                    tts.append(tt)
                    datas.append(data)
                except Exception:
                    if debug:
                        print('Tracking Test Failed with a_sd = {}'.format(a_sd))
                    failure_count += 1
            tt_means.append(np.mean(tts))
            tt_sds.append(np.std(tts))
            ttss.append(tts)
            test_a_sds.append(a_sd)
            failures.append(failure_count)
        except Exception:
            pass
    bar.finish()
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(2, figsize=(6.5, 4), sharex=True)
    sns.regplot(test_a_sds, tt_means, order=2, ax=axs[0])
    axs[0].set_title('Tracking Time (mean)')
    axs[0].set_ylabel('Mean Tracking Time ($s$)', fontsize=9)
    sns.regplot(test_a_sds, tt_sds, order=2, ax=axs[1])
    axs[1].set_title('Tracking Time (s.d.)')
    axs[1].set_ylabel('Tracking Time S.D. ($s$)', fontsize=9)
    axs[1].set_xlabel('Acceleration Noise S.D. ($rad-s^{-2}$)', fontsize=9)
    fig.tight_layout()

    fig, ax = plt.subplots(1, figsize=(15, 12))
    sns.regplot(test_a_sds, failures, ax=ax)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    dt = 1/250.0
    planner = GimbalPlanner(dt)

    p_f = 1.31
    p_i = 0
    v_i = -1.71
    a_i = -3
    a_limit = 10

    # NOTE: IT MAY MAKE SENSE TO USE THE COMMANDED ACCELERATION RATHER THAN THE MEASURED ACCELERATION TO
    #   CALCULATE IN THE PLANNER AS THAT WILL ENSURE THE MOST CONTINUITY IN THE COMMANDS
    #   ALTERNATIVELY, IF THERE IS ENOUGH FILTERING IN THE STATE OBSERVER, THE FILTERED VALUE MAY
    #   GIVE MORE ACCURACY.  THIS IS ONLY TRUE IF THE DISTRUBANCES ARE REJECTED.  THIS IS ANOTHER
    #   PALCE THAT SOMETHING LIKE A TRIMMING PI CONTROLLER WOULD BE HANDY.
    # _, _, theta_cmd, omega_cmd, domega_cmd = planner(p_f, a_limit, (p_i, v_i, a_i), debug=True)
    # print('Command: omega = {}, domega = {}'.format(omega_cmd, domega_cmd))
    # _, _, theta_cmd, omega_cmd, domega_cmd = planner(p_f, a_limit, (theta_cmd, omega_cmd, domega_cmd), debug=True)
    # print('Command: omega = {}, domega = {}'.format(omega_cmd, domega_cmd))
    # _, _, theta_cmd, omega_cmd, domega_cmd = planner(p_f, a_limit, (theta_cmd, omega_cmd, domega_cmd), debug=True)
    # print('Command: omega = {}, domega = {}'.format(omega_cmd, domega_cmd))

    # fig, axs = None, None
    # tts = []
    # datas = []
    # for _ in range(30):
    #     fig, axs, tt, data = runTrackingTest(dt, p_f, p_i, v_i, a_i, a_limit, a_sd=0.5, fig=fig, axs=axs)
    #     tts.append(tt)
    #     datas.append(data)
    # print('\nTracking Time: Mean = {}, S.D. = {}'.format(np.mean(tts), np.std(tts)))
    # sns.displot(tts)

    # plt.show()
    runNoiseSensitivityTest(dt, p_f, p_i, v_i, a_i, a_limit)
