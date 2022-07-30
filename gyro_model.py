import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy
import scipy.signal



class GyroModel:
    def __init__(self, name, lp_freq=None, hp_freq=None, max_bias=None, bias_stability=None, sd_noise=None, axes=3):
        self.__name = name
        self.__axes = axes
        self.__lp_freq = lp_freq
        self.__hp_freq = hp_freq
        self.__max_bias = math.radians(3) if max_bias is None else max_bias
        self.__bias_offset = [random.gauss(0, self.__max_bias/5) for _ in range(axes)]
        self.__max_stability_slope = (self.__max_bias if bias_stability is None else bias_stability)
        self.__sd_noise = math.radians(0.07) if sd_noise is None else sd_noise
        self.__bias_offset_est = [0] * axes
        self.__last_bias = [0] * axes

    def calBias(self, fs, ts=None, domegas=None, N=1000):
        if domegas is None:
            domegas = [np.array([0] * N)] * self.__axes
        if ts is None:
            t_start = -(N/fs)
            ts = np.linspace(t_start, 0, N)
        self.__bias_offset_est = [0] * self.__axes
        _, domegas, _, _ = self.signal(fs, ts, domegas)
        N = min(len(domegas), N)
        sds = []
        for idx, domega in enumerate(domegas):
            self.__bias_offset_est[idx] = np.mean(domega[:N])
            sds.append(np.std(domega[:N]))
        return self.__bias_offset_est, sds

    def signal(self, fs, ts, domegas):
        biases = []
        noises = []
        domegas_return = []
        N = len(ts)
        for idx, domega in enumerate(domegas):
            bias = self.__generate_bias_signal(fs, N, idx)
            noise = self.__generate_noise_signal(N)
            self.__last_bias[idx] = bias[-1]
            biases.append(bias)
            noises.append(noise)
            vals = domega + bias + noise + self.__bias_offset[idx] - self.__bias_offset_est[idx]
            domegas_return.append(vals)
        return ts, domegas_return, biases, noises

    def __generate_random_walk(self, initial, sd, N):
        domegas = [initial]
        for _ in range(int(N-1)):
            update = random.gauss(0, sd)
            mult = -1 if ((update * domegas[-1]) > (self.__max_bias * random.uniform(0, 1))) else 1
            domegas.append((domegas[-1] + (mult*update)))
        return np.array(domegas)

    def __generate_bias_signal(self, fs, N, idx):
        rate = self.__max_stability_slope / fs
        return self.__generate_random_walk(self.__last_bias[idx], rate, N)

    def __generate_noise_signal(self, N):
        return np.random.normal(0, self.__sd_noise, (N,))


if __name__ == '__main__':
    import seaborn as sns
    plt.style.use('seaborn-colorblind')

    test_samples = 150
    fs = 250
    t_max = 10
    t_cal=5

    bw_gyro = fs / 5
    N = int(t_max * fs)
    ts = np.linspace(0, t_max, N)
    domegas = [np.array([0] * N)] * 3

    b, a = scipy.signal.butter(4, 100, 'low', analog=True)
    w, h = scipy.signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()

    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('10 Hz and 20 Hz sinusoids')
    ax1.axis([0, 1, -2, 2])

    sos = scipy.signal.butter(10, 15, 'low', fs=1000, output='sos')
    print(sos)
    filtered = scipy.signal.sosfilt(sos, sig)
    ax2.plot(t, filtered)
    ax2.set_title('After 15 Hz high-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show()

    lsm6dsm = GyroModel('LSM6DSM', max_bias=math.radians(2), sd_noise=math.radians(0.0038*bw_gyro))
    fig, axs = plt.subplots(2, 3, figsize=(15,12))
    
    x_errors = []
    y_errors = []
    z_errors = []
    
    for _ in range(test_samples):
        gyro = GyroModel('BMX160', max_bias=math.radians(3), sd_noise=math.radians(0.007*bw_gyro))
        gyro.calBias(fs, N=t_cal*fs)
        ts, gyro_domegas, gyro_biases, gyro_noises = gyro.signal(fs, ts, domegas)
        x_error = domegas[0] - gyro_domegas[0]
        y_error = domegas[1] - gyro_domegas[1]
        z_error = domegas[2] - gyro_domegas[2]
        x_errors.append(np.mean(x_error))
        y_errors.append(np.mean(y_error))
        z_errors.append(np.mean(z_error))
        sns.lineplot(x=ts, y=x_error, alpha=0.1, ax=axs[0][0])
        sns.lineplot(x=ts, y=y_error, alpha=0.1, ax=axs[0][1])
        sns.lineplot(x=ts, y=z_error, alpha=0.1, ax=axs[0][2])
    sns.distplot(np.array(x_errors), rug=True, hist=False, ax=axs[1][0])
    x_mean = np.mean(x_errors)
    x_sd = np.std(x_errors)
    axs[1][0].axvline(x_mean, c='k', alpha=0.33)
    axs[1][0].axvline(x_mean - 2*x_sd, c='g', alpha=0.33)
    axs[1][0].axvline(x_mean + 2*x_sd, c='g', alpha=0.33)
    axs[1][0].set_title('X-axis errors (mean={:0.2g}, s.d.={:0.2g})'.format(x_mean, x_sd))
    
    sns.distplot(np.array(y_errors), rug=True, hist=False, ax=axs[1][1])
    y_mean = np.mean(y_errors)
    y_sd = np.std(y_errors)
    axs[1][1].axvline(y_mean, c='k', alpha=0.33)
    axs[1][1].axvline(y_mean - 2*y_sd, c='g', alpha=0.33)
    axs[1][1].axvline(y_mean + 2*y_sd, c='g', alpha=0.33)
    axs[1][1].set_title('Y-axis errors (mean={:0.2g}, s.d.={:0.2g})'.format(y_mean, y_sd))
    
    sns.distplot(np.array(z_errors), rug=True, hist=False, ax=axs[1][2])
    z_mean = np.mean(z_errors)
    z_sd = np.std(z_errors)
    axs[1][2].axvline(z_mean, c='k', alpha=0.33)
    axs[1][2].axvline(z_mean - 2*z_sd, c='g', alpha=0.33)
    axs[1][2].axvline(z_mean + 2*z_sd, c='g', alpha=0.33)
    axs[1][2].set_title('Z-axis errors (mean={:0.2g}, s.d.={:0.2g})'.format(z_mean, z_sd))
    
    fig.tight_layout()
    plt.show()
