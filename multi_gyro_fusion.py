import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.random


def generateCleanSignal(t_max, N=1000):
    B = random.uniform(0.025, 0.06)
    A0 = random.uniform(3, 7)
    B0 = random.uniform(B, 4*B)
    A1 = random.uniform(1.9, 3.5)
    B1 = random.uniform(5*B, 11*B)
    A2 = random.uniform(2.4, 5.75)
    B2 = random.uniform(17*B, 29*B)
    A3 = random.uniform(0.49, 1.1)
    B3 = random.uniform(47*B, 98*B)
    ts = np.linspace(0, t_max, N)
    domegas = A0*((np.cos(2*math.pi*B*ts) + 1.5) / 2)*np.sin(2*math.pi*B0*ts)
    domegas += A1*np.sin(2*math.pi*B1*ts+A0)
    domegas += A2*np.sin(2*math.pi*B2*ts+A1)
    domegas += A3*np.sin(2*math.pi*B3*ts+A2)
    return ts, domegas


def generateRandomWalk(initial, sd, max_deviation, N):
    domegas = [initial]
    for _ in range(int(N-1)):
        update = random.gauss(0, sd)
        mult = -1 if ((update * domegas[-1]) > (max_deviation * random.uniform(0, 1))) else 1
        domegas.append((domegas[-1] + (mult*update)))
    return np.array(domegas)


def generateCalibrationSignal(t_max, sd=None, max_deviation=None, N=1000):
    sd = 2.5e-5 if sd is None else sd
    max_deviation = 5e-4 if max_deviation is None else max_deviation
    ts = np.linspace(0, t_max, N)
    return ts, generateRandomWalk(0, sd, max_deviation, N)


def generateBiasSignal(fs, N, last_bias=0, max_deviation=None, drift=None):
    max_deviation = math.radians(3) if max_deviation is None else max_deviation
    drift = (max_deviation / 3600) if drift is None else drift
    rate = drift / fs
    return generateRandomWalk(last_bias, rate, max_deviation, N)


def generateNoiseSignal(ts, sd=None):
    sd = math.radians(0.07) if sd is None else sd
    return np.random.normal(0, sd, (len(ts),))


class GyroModel:
    def __init__(self, max_bias=None, bias_stability=None, sd_noise=None):
        self.__max_bias = math.radians(3) if max_bias is None else max_bias
        self.__bias_offset = random.gauss(0, self.__max_bias)
        self.__max_stability_slope = self.__max_bias / 3600 if bias_stability is None else bias_stability
        self.__sd_noise = math.radians(0.07) if sd_noise is None else sd_noise
        self.__bias_offset_est = 0
        self.__last_bias = 0

    def calBias(self, fs, ts, domegas, N=1000):
        self.__bias_offset_est = 0
        domegas, _, _ = self.signal(fs, ts, domegas)
        N = min(len(domegas), N)
        self.__bias_offset_est = np.mean(domegas[:N])
        return self.__bias_offset_est, np.std(domegas[:N])

    def signal(self, fs, ts, domegas):
        biases = generateBiasSignal(fs, len(ts), self.__last_bias, self.__max_bias, self.__max_stability_slope)
        noise = generateNoiseSignal(ts, self.__sd_noise)
        self.__last_bias = biases[-1]
        return (domegas + biases + noise + self.__bias_offset - self.__bias_offset_est), biases, noise


class SimpleFusionFilter:
    def __init__(self, w=0.75):
        self.__w = w
        self.__x = None

    def __call__(self, x0, x1):
        if self.__x is None:
            self.__x = (x0+x1)/2
        else:
            self.__x = (self.__w * self.__x) + ((1-self.__w) / 2) * (x0 + x1)
        return self.__x


class FusionAlphaBetaFilter:
    @classmethod
    def Optimal(cls, dt, var_process, var_m0, var_m1):
        sd_process = math.sqrt(var_process)
        sd_m0 = math.sqrt(var_m0)
        sd_m1 = math.sqrt(var_m1)
        mean_var_m = math.sqrt(var_m0 + var_m1)
        L0 = sd_process * dt / sd_m0
        L1 = sd_process * dt / sd_m1
        r0 = (4 + L0 - math.sqrt((8*L0) + (L0**2))) / 4
        r1 = (4 + L1 - math.sqrt((8*L1) + (L1**2))) / 4
        alpha0 = (1 - (r0**2)) * (sd_m0 / mean_var_m)
        alpha1 = (1 - (r1**2)) * (sd_m1 / mean_var_m)
        beta0 = (2*(2-alpha0))-(4*math.sqrt(1-alpha0))
        beta1 = (2*(2-alpha1))-(4*math.sqrt(1-alpha1))
        assert 0 < alpha0 < 1, 'Invalid Alpha0 value of {}'.format(beta0)
        assert 0 < alpha1 < 1, 'Invalid Alpha1 value of {}'.format(beta1)
        assert 0 < beta0 < 1, 'Invalid Beta0 value of {}'.format(beta0)
        assert 0 < beta1 < 1, 'Invalid Beta1 value of {}'.format(beta1)
        assert 0 < (4 - 2*alpha0 - beta0), 'Invalid Alpha0/Beta0 combination of {} and {}'.format(alpha0, beta0)
        assert 0 < (4 - 2*alpha1 - beta1), 'Invalid Alpha1/Beta1 combination of {} and {}'.format(alpha1, beta1)
        return FusionAlphaBetaFilter(dt, alpha0, alpha1, beta0, beta1)

    def __init__(self, dt, alpha0, alpha1, beta0, beta1):
        print('Alpha0 = {:0.4f}, Beta0 = {:0.4f}, Alpha1 = {:0.4f}, Beta1 = {:0.4f}'.format(alpha0, beta0, alpha1, beta1))
        self.__dt = dt
        self.__alpha0 = alpha0
        self.__alpha1 = alpha1
        self.__beta0 = beta0
        self.__beta1 = beta1
        self.__x = None
        self.__dx = None

    def __call__(self, x0, x1):
        if self.__x is None:
            w_sum = self.__alpha0 + self.__alpha1
            self.__x = ((self.__alpha0*x0) + (self.__alpha1*x1)) / w_sum
            self.__dx = 0
            return self.__x
        xp = self.__x + (self.__dt * self.__dx)
        r0 = x0 - xp
        r1 = x1 - xp
        self.__x += ((self.__alpha0 * r0) + (self.__alpha1 * r1))
        self.__dx += (((self.__beta0 * r0)/self.__dt) + ((self.__beta1 * r1)/self.__dt))
        return xp


def runGyroTest(fs, gyro, t_cal, ts, domegas):
    ts_cal, domegas_cal = generateCalibrationSignal(t_cal, N=int(t_cal*fs))
    gyro_bias_est, gyro_sd_est = gyro.calBias(fs, ts_cal, domegas_cal)
    domegas_est, gyro_biases, _ = gyro.signal(fs, ts, domegas)
    return domegas_est, gyro_biases, gyro_bias_est, gyro_sd_est


def est_delay(dt, ref, tgt):
    tgt = (tgt - np.mean(tgt)) / np.std(tgt)
    ref = (ref - np.mean(ref)) / np.std(ref)
    corr = np.correlate(ref, tgt, 'full')
    idx = np.argmax(corr)
    return dt * (len(corr)/2.0 - idx)


def calculateError(dt, ts, domegas, domegas_est):
    errs = domegas-domegas_est
    return errs, np.mean(errs), np.std(errs), est_delay(dt, domegas, domegas_est)


def filterGyroResults(filt, domegas_a, domegas_b):
    domegas_filt = []
    for a, b in zip(domegas_a, domegas_b):
        domega = filt(a, b)
        domegas_filt.append(domega)
    return np.array(domegas_filt)


def filterOptimization(fs, t_run, gyro0, gyro1, filter_gen, param_range=(.01, 1000), N=35, cal_gen=None, signal_gen=None):
    cal_gen = generateCalibrationSignal if cal_gen is None else cal_gen
    signal_gen = generateCleanSignal if signal_gen is None else signal_gen

    #
    # Calibrate Gyroscopes
    #
    ts_cal, domegas_cal = cal_gen(1, N=fs)
    gyro0.calBias(fs, ts_cal, domegas_cal)
    gyro1.calBias(fs, ts_cal, domegas_cal)

    #
    # Generate Test Signals
    #
    signals = []
    for _ in range(N):
        signals.append(signal_gen(t_run, t_run*fs))

    #
    # Define Function to Optimize
    #
    def filter_error(param):
        filt = filter_gen(param)
        filtered_sds = []
        for ts, domegas in signals:
            gyro0_domegas, _, _ = gyro0.signal(fs, ts, domegas)
            gyro1_domegas, _, _ = gyro1.signal(fs, ts, domegas)
            filtered_domegas = filterGyroResults(filt, gyro0_domegas, gyro1_domegas)
            filtered_sd = np.std(domegas - filtered_domegas)
            filtered_sds.append(filtered_sd)
        mean_sd = np.mean(filtered_sds)  # SHOULD THE ARITHMETIC MEAN BE USED???
        # sd_of_sds = np.std(filtered_sds)
        # print('With Parameter = {:0.3f}, Error( Mean = {:0.2E}, S.D. = {:0.2E} )'.format(param, mean_sd, sd_of_sds))
        return mean_sd

    params = np.geomspace(*param_range, 100)
    errs = []
    for param in params:
        err = filter_error(param)
        errs.append(err)
    errs = np.array(errs)
    idx = np.argmin(errs)
    print('Minimal Error ({:0.3e}) at Param = {:0.2e}'.format(errs[idx], params[idx]))
    plt.scatter(params, errs, alpha=0.25)
    plt.show()
    # TODO: IMPLEMENT THE OPTIMIZATION HERE....


if __name__ == '__main__':
    import seaborn as sns
    plt.style.use('seaborn-colorblind')

    fs = 3200
    f_servo = 250

    bw_gyro = fs / 5
    t_run = 10
    t_cal = 1

    dt = 1 / fs

    bmx160 = GyroModel(max_bias=math.radians(3), sd_noise=math.radians(0.007*bw_gyro))
    lsm6dsm = GyroModel(max_bias=math.radians(2), sd_noise=math.radians(0.0038*bw_gyro))

#    def filter_gen(param):
#        return FusionAlphaBetaFilter.Optimal(dt, max(param, 1e-6), math.radians(0.007*bw_gyro)**2, math.radians(0.0038*bw_gyro)**2)
#    filterOptimization(fs, t_run, bmx160, lsm6dsm, filter_gen)

    fab_filt = FusionAlphaBetaFilter.Optimal(dt, 10, math.radians(0.007*bw_gyro)**2, math.radians(0.0038*bw_gyro)**2)
    simple_filt = SimpleFusionFilter(w=0.85)
    #
    ts, domegas = generateCleanSignal(t_run, N=int(t_run*fs))
    ts_cal, domegas_cal = generateCalibrationSignal(t_cal, N=int(t_cal*fs))
    #
    #
    bmx160_domegas, bmx160_biases, bmx160_bias_est, bmx160_sd_est = runGyroTest(fs, bmx160, t_cal, ts, domegas)
    lsm6dsm_domegas, lsm6dsm_biases, lsm6dsm_bias_est, lsm6dsm_sd_est = runGyroTest(fs, lsm6dsm, t_cal, ts, domegas)
    fab_filtered_domegas = filterGyroResults(fab_filt, bmx160_domegas, lsm6dsm_domegas)
    simple_filtered_domegas = filterGyroResults(simple_filt, bmx160_domegas, lsm6dsm_domegas)
    
    bmx160_errors, bmx160_mean_error, bmx160_error_sd, bmx160_delay = calculateError(dt, ts, domegas, bmx160_domegas)
    lsm6dsm_errors, lsm6dsm_mean_error, lsm6dsm_error_sd, lsm6dsm_delay = calculateError(dt, ts, domegas, lsm6dsm_domegas)
    fab_filtered_errors, fab_filtered_mean_error, fab_filtered_error_sd, fab_filtered_delay = calculateError(dt, ts, domegas, fab_filtered_domegas)
    simple_filtered_errors, simple_filtered_mean_error, simple_filtered_error_sd, simple_filtered_delay = calculateError(dt, ts, domegas, simple_filtered_domegas)
    
    print('BMX160 Delay = {:0.2f} ms'.format(1000*bmx160_delay))
    print('LSM6DSM Delay = {:0.2f} ms'.format(1000*lsm6dsm_delay))
    print('Fusion Alpha-Beta Filtered Delay = {:0.2f} ms'.format(1000*fab_filtered_delay))
    print('Simple Fusion Filtered Delay = {:0.2f} ms'.format(1000*simple_filtered_delay))
    print('Servo Update Period = {:0.2f} ms'.format(1000*(1.0 / f_servo)))
    
    print('\n')
    print('BMX160 S.D. = {:0.3f} rad/s'.format(bmx160_error_sd))
    print('LSM6DSM S.D. = {:0.3f} rad/s'.format(lsm6dsm_error_sd))
    print('Fusion Alpha-Beta Filter S.D. = {:0.3f} rad/s'.format(fab_filtered_error_sd))
    print('Simple Fusion Filter S.D. = {:0.3f} rad/s'.format(simple_filtered_error_sd))
    #
    fig, axs = plt.subplots(3, figsize=(15, 12))
    sns.lineplot(ts, bmx160_domegas, ax=axs[0], lw=0.15)
    sns.lineplot(ts, lsm6dsm_domegas, ax=axs[0], lw=0.15)
    sns.lineplot(ts, fab_filtered_domegas, ax=axs[0], lw=1)
    sns.lineplot(ts, simple_filtered_domegas, ax=axs[0], lw=1)
    # axs[0].plot(ts, domegas, c='k', alpha=0.7, lw=0.5)
    #
    sns.lineplot(ts, bmx160_errors, ax=axs[1], lw=0.15)
    sns.lineplot(ts, lsm6dsm_errors, ax=axs[1], lw=0.15)
    sns.lineplot(ts, fab_filtered_errors, ax=axs[1], lw=0.15)
    sns.lineplot(ts, simple_filtered_errors, ax=axs[1], lw=0.15)
    #
    sns.distplot(bmx160_errors, ax=axs[2])
    sns.distplot(lsm6dsm_errors, ax=axs[2])
    sns.distplot(fab_filtered_errors, ax=axs[2])
    sns.distplot(simple_filtered_errors, ax=axs[2])
    #
    plt.show()
