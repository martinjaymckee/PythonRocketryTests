
import math
import random
import sys

import filterpy as fp
import filterpy.kalman as fpk
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
import numpy.polynomial
import quaternion
import seaborn as sns

import pyrse.rotation_model as rotation_model
import gyro_model

#
# These conversion routines are using intrensic tait-bryan angles of z-y'-x'' order
#
def taitBryanToQ(yaw, pitch, roll):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    w = (cr*cp*cy) + (sr*sp*sy)
    x = (sr*cp*cy) - (cr*sp*sy)
    y = (cr*sp*cy) + (sr*cp*sy)
    z = (cr*cp*sy) - (sr*sp*cy)
    return np.quaternion(w, x, y, z)

def taitBryanToQ_Linearized_1st(yaw, pitch, roll):
    sr, cr = roll / 2, 1
    sp, cp = pitch / 2, 1
    sy, cy = yaw / 2, 1
    w = (cr*cp*cy) + (sr*sp*sy)
    x = (sr*cp*cy) - (cr*sp*sy)
    y = (cr*sp*cy) + (sr*cp*sy)
    z = (cr*cp*sy) - (sr*sp*cy)
    return np.quaternion(w, x, y, z)

def qToTaitBryan(q):
    roll = math.atan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x**2 + q.y**2))
    C = 2*(q.w*q.y - q.z*q.x)
    pitch = math.copysign(C) * math.pi / 2 if abs(C) > 1 else math.asin(C)
    yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
    return yaw, pitch, roll


def generateRotationReference(fs, t_max, model):
    ts, qws, thetas, omegas, domegas = model.sample(0, t_max, fs)
    return ts, qws, thetas, omegas, domegas


def generateTransformedThetas(ts, thetas, q0=None):
    q0 = np.quaternion(1, 0, 0, 0) if q0 is None else q0
    qs = []
    thetas_transformed = [[], [], []]
    for idx in range(len(ts)):
        qr = taitBryanToQ(thetas[0][idx], thetas[1][idx], thetas[2][idx])
        qs.append(qr)
        yaw, pitch, roll = qToTaitBryan(qr*q0)
        thetas_transformed[0].append(yaw)
        thetas_transformed[1].append(pitch)
        thetas_transformed[2].append(roll)
    return thetas_transformed

def gyroSignals(ts, omegas, gyro):
    return gyro.signal(ts, gyro)


def generateParamVector(p_min, p_max, N=3):
    vs = []
    for _ in range(N):
        vs.append(random.uniform(p_min, p_max))
    return tuple(vs)


class EulerQuaternionEstimator1:
    def __init__(self, q_mag_error_thresh=1e-6):
        self.__q = np.quaternion(1, 0, 0, 0)
        self.__q_mag_error_thresh = q_mag_error_thresh
        self.__norm_count = 0

    @property
    def name(self): return 'Euler Quaternion Estimator v1'

    @property
    def q(self): return self.__q

    @q.setter
    def q(self, _q):
        assert isinstance(_q, np.quaternion), 'Attempting to assign a {} as a quaternion.'.format(_q.__class__.__name__)
        self.__q = _q

    @property
    def angles(self): return qToTaitBryan(self.__q)

    def initOrientation(self, angles):
        self.__q = taitBryanToQ(*angles)

    @property
    def norm_count(self): return self.__norm_count

    def __call__(self, dt, omegas):
        dq = 0.5 * self.__q * np.quaternion(0, omegas[0], omegas[1], omegas[2])
        self.__q += (dt * dq)
        if abs(self.__q.norm() - 1.0) > self.__q_mag_error_thresh:
            self.__q = self.__q.normalized()
            self.__norm_count += 1
        return qToTaitBryan(self.__q)


def getQApproxFunc(impl):
    return {
        '1st Order': taitBryanToQ_Linearized_1st,
        'Exact': taitBryanToQ
    }[impl];


class EulerQuaternionEstimator2:
    def __init__(self, q_approx_impl='Exact', q_mag_error_thresh=1e-6, approx_norm=False):
        self.__q = np.quaternion(1, 0, 0, 0)
        self.__q_approx_func = getQApproxFunc(q_approx_impl)
        self.__q_approx_impl = q_approx_impl
        self.__q_mag_error_thresh = q_mag_error_thresh
        self.__norm_count = 0
        self.__approx_norm = approx_norm

    @property
    def name(self): return 'Euler Quaternion Estimator v2 ({})'.format(self.__q_approx_impl)

    @property
    def q(self): return self.__q

    @q.setter
    def q(self, _q):
        assert isinstance(_q, np.quaternion), 'Attempting to assign a {} as a quaternion.'.format(_q.__class__.__name__)
        self.__q = _q

    @property
    def angles(self): return qToTaitBryan(self.__q)

    def initOrientation(self, angles):
        self.__q = taitBryanToQ(*angles)

    @property
    def norm_count(self): return self.__norm_count

    def __call__(self, dt, omegas):
        self.__q = self.__q * self.__q_approx_func(dt*omegas[2], dt*omegas[1], dt*omegas[0])
        if self.__approx_norm:
            scale = 2/(1+(self.__q.norm()**2))
            self.__q *= scale
        else:
            if abs(self.__q.norm() - 1.0) > self.__q_mag_error_thresh:
                self.__q = self.__q.normalized()  # NOTE: THIS IS NOT NEEDED FOR FOR SHORTER SIMULATIONS WITH THE EXACT CONVERSION
                self.__norm_count += 1
        return qToTaitBryan(self.__q)


class EulerTaitBryanEstimator:
    def __init__(self):
        self.__angles = np.array([0, 0, 0])

    def initOrientation(self, angles):
        self.__angles = np.array(angles)

    @property
    def name(self): return 'Euler Tait-Bryan Estimator'

    @property
    def angles(self): return list(self.__angles)

    @property
    def q(self):
        return taitBryanToQ(*self.__angles)

    @q.setter
    def q(self, _q):
        self.__angles = np.array(qToTaitBryan(_q))
        return _q

    @property
    def norm_count(self): return 0.0

    def __call__(self, dt, omegas):
        self.__angles += dt * omegas
        return list(self.__angles)


def calcAngleError(ref, sig):
    return ref-sig
    errs = []
    for r, s in zip(ref, sig):
        e = r - s
        if e > math.pi:
            errs.append(e-math.pi)
        elif e < -math.pi:
            errs.append(e+math.pi)
        else:
            errs.append(e)
    return errs


def normalizeAngles(angles):
    norm_angles = []
    for a in angles:
        if a > math.pi:
            norm_angles.append(a-math.pi)
        elif a < -math.pi:
            norm_angles.append(a+math.pi)
        else:
            norm_angles.append(a)
    return norm_angles


def measurementColor(ax, idx):
    return {
        0: 'b',
        1: 'g',
        2: 'c',
        3: 'm',
        4: 'r'
    }[idx]


def referenceColor(ax, idx):
    return 'k'


def runRotationTest(fs, estimators, gyro, model=None, model_cls=None, model_kws={}, t_max=120, estimator_test=False, axs=None, t_cal=5, range_tests=None):
    t_maxs = [t_max]
    if range_tests is not None:
        t_maxs = np.linspace(t_max/range_tests, t_max, range_tests)

    if model is None:
        model_cls = rotation_model.RotationModel.Constrained if model_cls is None else model_cls
        if 'theta_max' not in model_kws:
            model_kws['theta_max'] = generateParamVector(math.pi/24, math.pi/12)
        if 'omega_max' not in model_kws:
            model_kws['omega_max'] = generateParamVector(math.radians(125), math.radians(250))
        if 'f_base' not in model_kws:
            model_kws['f_base'] = generateParamVector(0.025, 0.075)
        if 'terms' not in model_kws:
            model_kws['terms'] = 9
        model = model_cls(**model_kws)

    # Calculate the rotation reference
    ts, qws, thetas_ref, omegas_ref, domegas_ref = generateRotationReference(fs, t_max, model)

    # NOTE: This is a very basic calibration.  It may be possible to do better with improved modeling
    gyro.calBias(fs, N=t_cal*fs)

    # Calculate the gyro measurements
    _, omegas_gyro, biases_gyro, noises_gyro = gyro.signal(fs, ts, omegas_ref)

    # Integrate the gyro measurements to track the orientation
    #   Initialize the estimator orientation
    dt = 1 / fs

    qs_estimated = []
    for estimator in estimators:
        estimator.q = np.quaternion(1, 0, 0, 0)
        # estimator.initOrientation(tait_bryan_init)
        qs = []
        omegas_yaw, omegas_pitch, omegas_roll = omegas_ref if estimator_test else omegas_gyro
        for yaw, pitch, roll in zip(omegas_yaw[:-1], omegas_pitch[:-1], omegas_roll[:-1]):
            estimator(dt, np.array([yaw, pitch, roll]))
            qs.append(estimator.q)
        qs_estimated.append(qs)

    # Analyze the estimation and plot the results
    errors_estimated = []
    for estimator, qs_est in zip(estimators, qs_estimated):
        errs = [rotation_model.angleBetweenQs(qw, q) for qw, q in zip(qws, qs_est)]
        # errs = np.array(thetas_ref) - np.array(thetas_est)  # TODO: THIS SHOULD HANDLE WRAPPING...
        for t in t_maxs:
            idx = np.argmax(t < ts)
            idx = len(ts) if idx == 0 else idx
            errs_slice = errs[:idx]
            q_mags = [q.norm() for q in qs_est[:idx]]
            stats = (
                np.mean(errs_slice),
                np.std(errs_slice),
                np.max(errs_slice),
                np.mean(q_mags),
                np.std(q_mags)
            )
            errors_estimated.append( (estimator.name, t, stats) )

    return errors_estimated


def runRotationErrorGrowthTest(fs, estimators, gyro, model_cls=None, model_kws={}, t_max=120, estimator_test=False, axs=None, t_cal=5, run_tests=10, range_tests=5, critical_angle=math.pi/18):
    collected_errors = []

    print('Simulating Orientation Tracking:')
    for idx in range(run_tests):
        print('\r\tTest {:3d} of {:3d}'.format(idx+1, run_tests), end='')
        sys.stdout.flush()
        run_errors = runRotationTest(fs, estimators, gyro, t_max=t_max, model_cls=model_cls, model_kws=model_kws, estimator_test=estimator_test, range_tests=range_tests)
        collected_errors += run_errors
    print('\n')
    print('Collating Errors....')
    ids = []
    t_maxs = []
    offset_means = []
    offset_sds = []
    offset_max_error = []
    q_mags = []

    for name, t, stats in collected_errors:
        ids.append(name)
        t_maxs.append(t)
        offset_means.append(stats[0])
        offset_sds.append(stats[1])
        offset_max_error.append(stats[2])
        q_mags.append(stats[3])
    ids = np.array(ids)
    t_maxs = np.array(t_maxs)
    offset_means = np.array(offset_means)
    offset_sds = np.array(offset_sds)
    offset_max_error = np.array(offset_max_error)
    q_mags = np.array(q_mags)
    print('Plotting results....')
    fig, axs = plt.subplots(3, figsize=(18, 10), sharex=True)
    axs[0].set_title('Max Offset')
    axs[1].set_title('Offset SDS')
    axs[2].set_title('Quaternion Magnitude')
    axs[2].axhline(1, c='k', alpha=0.25)

    def plotRegExtrema(ax, ts, errs, c=None, alpha=0.5):
        extrema = {}
        for t, err in zip(ts, errs):
            if t in extrema:
                val = extrema[t]
                if err > val:
                    extrema[t] = err
            else:
                extrema[t] = err
        xs = np.array(list(extrema.keys()))
        ys = np.array(list(extrema.values()))
        ax.scatter(xs, ys, c=c, alpha=alpha)
        return xs, ys

    for name in [estimator.name for estimator in estimators]:
        select = ids == name
        offset_errors = offset_max_error[select]
        offset_sd = offset_sds[select]
        ts = t_maxs[select]
        mags = q_mags[select]
        g = sns.regplot(x=ts, y=offset_errors, x_bins=range_tests, ci=100, ax=axs[0], label=name)
        c = g.get_lines()[-1].get_color()
        xs, ys = plotRegExtrema(axs[0], ts, offset_errors, c=c)
        p_max = np.polynomial.Polynomial.fit(xs, ys, deg=1)
        print('Maximum Error Trend for {} = {}'.format(name, str(p_max)))
        print('\tTimes -> ts = {}'.format(xs))
        print('\tErrors -> ys = {}'.format(ys))
        sns.regplot(x=ts, y=offset_sd, x_bins=range_tests, ci=100, ax=axs[1], label=name)
        xs, ys = plotRegExtrema(axs[1], ts, offset_sd, c=c)
        p_sd = np.polynomial.Polynomial.fit(xs, ys, deg=1)
        print('Standard Deviation Trend for {} = {}'.format(name, str(p_sd)))
        sns.regplot(x=ts, y=mags, x_bins=range_tests, ci=100, ax=axs[2], label=name)
        plotRegExtrema(axs[2], ts, mags, c=c)
    for ax in axs:
        ax.legend()
    axs[0].axhline(critical_angle, c='r', alpha=0.5)
    axs[2].set_xlim(np.min(t_maxs) - t_max/20, np.max(t_maxs) + t_max/20)
    fig.tight_layout()

    fig, axs = plt.subplots(3, figsize=(18, 10), sharex=True)
    axs[0].set_title('Max Offset')
    axs[1].set_title('Offset Standard Deviation')
    axs[2].set_title('Quaternion Magnitudes')
    axs[2].axhline(1, c='k', alpha=0.25)
    sns.boxplot(x=t_maxs, y=offset_max_error, hue=ids, ax=axs[0])
    sns.boxplot(x=t_maxs, y=offset_sds, hue=ids, ax=axs[1])
    sns.boxplot(x=t_maxs, y=q_mags, hue=ids, ax=axs[2])
    axs[0].axhline(critical_angle, c='r', alpha=0.5)
    fig.tight_layout()
    #
    # fig, axs = plt.subplots(2, figsize=(18, 10), sharex=True)
    # axs[0].set_title('Max Offset')
    # axs[1].set_title('Offset Standard Deviation')
    # sns.swarmplot(x=t_maxs, y=offset_max_error, hue=ids, dodge=True, ax=axs[0])
    # sns.swarmplot(x=t_maxs, y=offset_sds, hue=ids, dodge=True, ax=axs[1])
    # axs[0].axhline(math.pi/18, c='r', alpha=0.5)
    # fig.tight_layout()
    #
    # fig, axs = plt.subplots(3, figsize=(18, 10), sharex=True)
    # axs[0].set_title('Max Offset')
    # axs[1].set_title('Offset Standard Deviation')
    # axs[2].set_title('Quaternion Magnitudes')
    # axs[2].axhline(1, c='k', alpha=0.25)
    # split = len(estimators) == 2
    # sns.violinplot(x=t_maxs, y=offset_max_error, hue=ids, split=split, ax=axs[0])
    # sns.violinplot(x=t_maxs, y=offset_sds, hue=ids, split=split, ax=axs[1])
    # sns.violinplot(x=t_maxs, y=q_mags, hue=ids, split=split, ax=axs[2])
    # axs[0].axhline(critical_angle, c='r', alpha=0.5)
    # fig.tight_layout()
    plt.show()


def runEstimatorQuaternionTrackingTest(fs, estimators, gyro, model=None, model_cls=None, model_kws={}, estimator_test=False, t_max=120, axs=None, t_cal=5):
    if model is None:
        model_cls = rotation_model.RotationModel.Constrained if model_cls is None else model_cls
        if 'theta_max' not in model_kws:
            model_kws['theta_max'] = generateParamVector(math.pi/24, math.pi/12)
        if 'omega_max' not in model_kws:
            model_kws['omega_max'] = generateParamVector(math.radians(125), math.radians(250))
        if 'f_base' not in model_kws:
            model_kws['f_base'] = generateParamVector(0.025, 0.075)
        if 'terms' not in model_kws:
            model_kws['terms'] = 9
        model = model_cls(**model_kws)

    # Calculate the rotation reference
    ts, qws, thetas_ref, omegas_ref, domegas_ref = generateRotationReference(fs, t_max, model)

    # NOTE: This is a very basic calibration.  It may be possible to do better with improved modeling
    gyro.calBias(fs, N=t_cal*fs)

    # Calculate the gyro measurements
    _, omegas_gyro, biases_gyro, noises_gyro = gyro.signal(fs, ts, omegas_ref)

    # Integrate the gyro measurements to track the orientation
    #   Initialize the estimator orientation
    dt = 1 / fs

    qs_estimated = []
    for estimator in estimators:
        estimator.q = np.quaternion(1, 0, 0, 0)
        omegas = omegas_ref if estimator_test else omegas_gyro
        omegas_yaw, omegas_pitch, omegas_roll = omegas
        qs = [estimator.q]
        for yaw, pitch, roll in zip(omegas_yaw[:-1], omegas_pitch[:-1], omegas_roll[:-1]):
            estimator(dt, np.array([yaw, pitch, roll]))
            qs.append(estimator.q)
        qs_estimated.append( qs )

    # Analyze the estimation and plot the results
    fig, axs = plt.subplots(4, figsize=(18, 12), sharex=True)
    axs[0].set_title('Error W')
    axs[1].set_title('Error X')
    axs[2].set_title('Error Y')
    axs[3].set_title('Error Z')
    axs_alt = [ax.twinx() for ax in axs]
    ws = np.array([q.w for q in qws])
    xs = np.array([q.x for q in qws])
    ys = np.array([q.y for q in qws])
    zs = np.array([q.z for q in qws])
    for estimator, qs_est in zip(estimators, qs_estimated):
        ws_sig = np.array([q.w for q in qs_est])
        xs_sig = np.array([q.x for q in qs_est])
        ys_sig = np.array([q.y for q in qs_est])
        zs_sig = np.array([q.z for q in qs_est])

        errs = np.array(qws) - np.array(qs_est)
        ws_errs = np.array([q.w for q in errs])
        xs_errs = np.array([q.x for q in errs])
        ys_errs = np.array([q.y for q in errs])
        zs_errs = np.array([q.z for q in errs])

        axs[0].plot(ts[::fs], ws[::fs], '+k')
        g = axs[0].plot(ts, ws_sig, label=estimator.name)
        c = g[-1].get_color()
        axs_alt[0].plot(ts, ws_errs, c=c, linestyle='dashed', alpha=0.5)
        axs_alt[0].axhline(0, c=c, alpha=0.25)

        axs[1].plot(ts[::fs], xs[::fs], '+k')
        g = axs[1].plot(ts, xs_sig, label=estimator.name)
        c = g[-1].get_color()
        axs_alt[1].plot(ts, xs_errs, c=c, linestyle='dashed', alpha=0.5)
        axs_alt[1].axhline(0, c=c, alpha=0.25)

        axs[2].plot(ts[::fs], ys[::fs], '+k')
        g = axs[2].plot(ts, ys_sig, label=estimator.name)
        c = g[-1].get_color()
        axs_alt[2].plot(ts, ys_errs, c=c, linestyle='dashed', alpha=0.5)
        axs_alt[2].axhline(0, c=c, alpha=0.25)

        axs[3].plot(ts[::fs], zs[::fs], '+k')
        g = axs[3].plot(ts, zs_sig, label=estimator.name)
        c = g[-1].get_color()
        axs_alt[3].plot(ts, zs_errs, c=c, linestyle='dashed', alpha=0.5)
        axs_alt[3].axhline(0, c=c, alpha=0.25)

    for ax in axs:
        ax.legend()
    fig.tight_layout()

    fig, axs = plt.subplots(3, figsize=(18,12), sharex=True)
    for idx, (ref, sig) in enumerate(zip(omegas_ref, omegas_gyro)):
        axs[idx].plot(ts, ref[:len(ts)], c='k', alpha=0.8)
        axs[idx].plot(ts, sig[:len(ts)], c='c', alpha=0.33)
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    plt.style.use('seaborn-colorblind')
    # random.seed(12345)
    # EVEN WITH THE REFERENCE SIGNAL THE TRACKING IS CRAPPY.  NEED TO MAKE SURE THE ESTIMATOR IS WORKING PROPERLY
    range_tests = 12
    run_tests = 30

    critical_angle = math.acos(0.99)
    estimator_test = False
    fs = 50
    t_max = 120
    t_cal = 5
    bw_gyro = max(50, min(150, fs/2.5))

    print('Gyro Bandwidth = {} Hz'.format(bw_gyro))
    # estimators = [EulerQuaternionEstimator1(), EulerQuaternionEstimator2(), EulerQuaternionEstimator2(q_approx_impl='1st Order', q_mag_error_thresh=1e-3)]
    estimators = [EulerQuaternionEstimator2(q_approx_impl='1st Order', approx_norm=True)]
    # estimators = [EulerQuaternionEstimator1(), EulerQuaternionEstimator2()]
    # estimators = [EulerTaitBryanEstimator(), EulerQuaternionEstimator1()]

    # NOTE: The max_value is fine, but the bias slope should be increased
    # gyro = gyro_model.GyroModel('MPU6050', lp_freq=bw_gyro, max_bias=math.radians(20), bias_stability=math.radians(0.09), sd_noise=math.radians(0.008*math.sqrt(bw_gyro)))
    gyro = gyro_model.GyroModel('BMX160', lp_freq=bw_gyro, max_bias=math.radians(3), bias_stability=math.radians(0.025), sd_noise=math.radians(0.007*math.sqrt(bw_gyro)))
    # gyro = gyro_model.GyroModel('ICM-20602', lp_freq=bw_gyro, max_bias=math.radians(1), bias_stability=math.radians(0.025), sd_noise=math.radians(0.004*math.sqrt(bw_gyro)))
    # gyro = gyro_model.GyroModel('ICM-42688', lp_freq=bw_gyro, max_bias=math.radians(0.5), bias_stability=math.radians(0.025), sd_noise=math.radians(0.0028*math.sqrt(bw_gyro)))

    # runEstimatorQuaternionTrackingTest(fs, estimators, gyro, t_max=t_max, t_cal=t_cal, estimator_test=estimator_test)
    runRotationErrorGrowthTest(fs, estimators, gyro, t_max=t_max, estimator_test=estimator_test, t_cal=t_cal, run_tests=run_tests, range_tests=range_tests, critical_angle=critical_angle)

    for estimator in estimators:
        print('Estimator, {}, required {} normalizations'.format(estimator.name, estimator.norm_count))
