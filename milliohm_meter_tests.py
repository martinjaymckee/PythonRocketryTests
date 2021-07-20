import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import adc_models
import electronics.discretes as discretes
from normal_rvs import NRV


def calcMinimumCounts(accuracy, oversampling=5):
    return oversampling * (100 / accuracy)


def calcErrorLimitedResistance(Rb, bits=10, N_min=None, Re=0):
    N_min = (2**bits) / 10 if N_min is None else N_min
    return ((2**bits) / N_min) * (Rb - ((N_min/(2**bits))*(Rb+Re)))


# def calcResistorMeasurementAccuracy(bits, Vin, Vref, Rs, Rb, Re=0):
#     return 100 / math.floor(((2**bits) * (Vin * Rb) / (Vref * (Rb + Rs + Re))))


def johnsonNoise(R, BW, T=300):
    kb = 1e6 * 1.380649e-23  # Boltzmann's constant
    return math.sqrt(4 * kb * T * R * BW)


def calcResistorMeasurementAccuracy(Vin, Vref, Rs, Rb, adc_b, adc_s, Re=0, Rb_accuracy=0.01, BW=50):
    def volts(adc, bits):
        return Vref * adc / 2**bits

    def adcCounts(gain, V, bits):
        def clipADC(counts):
            if counts > 2**bits:
                print('V actual = {} v, V estimated = {}'.format(gain * V, volts(counts, bits)))
            return min(counts, 2**bits)
        minimum = clipADC(math.floor(gain * (2**bits) * V / Vref))
        maximum = clipADC(minimum + 1)
        best = clipADC(int((gain * (2**bits) * V / Vref) + 0.5))
        return minimum, best, maximum

    I_test = Vin / (Rb + Rs + Re)
    Vb = Rb * I_test
    Vs = Rs * I_test
    bits_b, bits_s = adc_b.enob, adc_s.enob
    adc_Rb_min, adc_Rb_best, adc_Rb_max = adcCounts(adc_b.gain, Vb, bits_b)
    adc_Rs_min, adc_Rs_best, adc_Rs_max = adcCounts(adc_s.gain, Vs, bits_s)
    Vb_min, Vb_best, Vb_max = volts(adc_Rb_min, bits_b), volts(adc_Rb_best, bits_b), volts(adc_Rb_max, bits_b)
    Vs_min, Vs_best, Vs_max = volts(adc_Rs_min, bits_s), volts(adc_Rs_best, bits_s), volts(adc_Rs_max, bits_s)
    Rs_est_min = Vs_min * Rb / Vb_max
    Rs_est_max = Vs_max * Rb / Vb_min
    offset = 0 if Rs_est_min == 0 else (100 * (Rs_est_max - Rs_est_min) / Rs_est_min)
    offset += Rb_accuracy
    Vb_noise = johnsonNoise(Rb, adc_b.odr)
    Vs_noise = johnsonNoise(Rs, adc_s.odr)
    eff_Vb_noise = Vb_noise**2 + adc_b.noise_rms**2
    eff_Vs_noise = Vs_noise**2 + adc_s.noise_rms**2
    Vb_rv = NRV(Vb_best, variance=eff_Vb_noise)
    Vs_rv = NRV(Vs_best, variance=eff_Vs_noise)
    Rs_noise = (Vs_rv * Rb) / Vb_rv
    noise = 100 * Rs_noise.standard_deviation / Rs
    return offset, noise


def runAD7177Test(Rs_range, Vin, Vref, odr=5, N=1e4, target_accuracy_percent=0.015, critical_accuracy_percent=0.1, noise_limit_percent=0.005, NPLC=10):
    N_min_samples = 1
    ch_b = adc_models.AD7177Channel(Vref)
    ch_s = adc_models.AD7177Channel(Vref)
    ch_b.odr = odr
    ch_s.odr = odr

    Rss = np.linspace(Rs_range[0], Rs_range[1], int(N))
    dRs = []
    noises = []
    idx_start = None
    idx_end = None
    for idx, Rs in enumerate(Rss):
        dR, noise = calcResistorMeasurementAccuracy(Vin, Vref, Rs, Rb, ch_b, ch_s, Re=1.5)
        dRs.append(dR)
        noises.append(noise)
        if idx_start is None:
            if dR < critical_accuracy_percent:
                idx_start = idx
        else:
            if (idx_end is None) and (dR >= critical_accuracy_percent):
                idx_end = idx
    idx_start = 0 if idx_start is None else idx_start
    idx_end = -1 if idx_end is None else idx_end
    fig, axs = plt.subplots(2, figsize=(16, 9))
    N_per_plc = int(max(1, ch_b.odr / 60))
    N_samples = max(N_min_samples, NPLC * N_per_plc)
    NPLC = int(math.ceil((N_samples + 2) * 60 / ch_b.odr))  # Note: two more samples to set up the delta offset correction

    axs[0].set_title('Offset Error (% of measuement) -- ENOB = {:0.1f}'.format(ch_b.noise_free_bits))
    axs[1].set_title('Calculation Noise (% of measurement) -- ODR = {:0.2f} Hz, NPLC = {}, Samples = {}'.format(ch_b.odr, NPLC, N_samples))
    sns.lineplot(x=Rss[idx_start:idx_end], y=dRs[idx_start:idx_end], ax=axs[0])
    axs[0].axhspan(0, target_accuracy_percent, fc='g', alpha=0.1)
    ymax = min(critical_accuracy_percent, axs[0].get_ylim()[1])
    if np.max(dRs) >= critical_accuracy_percent:
        ytext = ymax - (ymax - target_accuracy_percent) / 10
        axs[0].annotate(
            'Critical Offset at $Rs = {:0.3g} \\Omega$'.format(Rss[idx_start]),
            xy=(Rss[idx_start], critical_accuracy_percent),
            xytext=(Rss[min(int(3*N/4), idx_start + int(N/30))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )
    error_idx = np.argmax(np.array(dRs) < target_accuracy_percent)
    error_idx -= 1
    if error_idx >= 0:
        Rs_target_offset = Rss[error_idx]
        ytext = target_accuracy_percent + (ymax - target_accuracy_percent) / 10
        axs[0].annotate(
            'Offset Limit at $Rs = {:0.3g} \\Omega$'.format(Rs_target_offset),
            xy=(Rs_target_offset, dRs[error_idx]),
            xytext=(Rss[min(int(3*N/4), error_idx + int(N/30))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )

    noises = np.array(noises) / math.sqrt(N_samples)
    error_idx = np.argmax(np.array(noises) < noise_limit_percent)
    Rs_critical_noise = Rss[error_idx]
    sns.lineplot(x=Rss[idx_start:idx_end], y=noises[idx_start:idx_end], ax=axs[1])
    axs[1].axhspan(0, noise_limit_percent, fc='g', alpha=0.1)
    if noises[idx_start] >= noise_limit_percent:
        ytext = noise_limit_percent + (np.max(noises[idx_start:idx_end]) - noise_limit_percent) / 10
        axs[1].annotate(
            'Noise Limit at $Rs = {:0.3g} \\Omega$'.format(Rs_critical_noise),
            xy=(Rs_critical_noise, noise_limit_percent),
            xytext=(Rss[min(int(3*N/4), error_idx + int(N/30))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    Vref = 5
    Rs_range = (1e-4, 1)
    # ad7177_ch0 = adc_models.AD7177Channel(Vref)
    # ad7177_ch1 = adc_models.AD7177Channel(Vref)
    # ad7177_ch0.odr = 200
    # ad7177_ch1.odr = 200
    # print('noise_bits = {}'.format(ad7177_ch0.noise_bits))
    # print('noise_free_bits = {}'.format(ad7177_ch0.noise_free_bits))
    # print('enob = {}'.format(ad7177_ch0.enob))

    Vin = 5
    Rb = 100
    target_accuracy_percent = 0.0125
    critical_accuracy_percent = 0.05
    # N_min = calcMinimumCounts(target_accuracy_percent)
    # print('N_min = {} counts'.format(int(N_min)))
    # Rs_max = calcErrorLimitedResistance(Rb, bits=ad7177_ch0.enob, N_min=N_min, Re=10)
    # print('Rs_max = {} ohms'.format(Rs_max))

    runAD7177Test(Rs_range, Vin, Vref, odr=16.67, target_accuracy_percent=target_accuracy_percent, critical_accuracy_percent=critical_accuracy_percent)
