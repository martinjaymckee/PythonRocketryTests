import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import adc_models
import electronics.discretes as discretes
import normal_rvs


def calcMinimumCounts(accuracy, oversampling=5):
    return oversampling * (100 / accuracy)


def calcErrorLimitedResistance(Rb, bits=10, N_min=None, Re=0):
    N_min = (2**bits) / 10 if N_min is None else N_min
    return ((2**bits) / N_min) * (Rb - ((N_min/(2**bits))*(Rb+Re)))


def johnsonNoise(R, BW, T=300):
    kb = 1e6 * 1.380649e-23  # Boltzmann's constant
    return math.sqrt(4 * kb * T * R * BW)


def oversamplingBits(N):
    assert N > 0, 'Error: Unable to calculculate oversampling bits with negative number of samples ({})'.format(N)
    return math.log(N) / math.log(4)


def calcResistorMeasurementAccuracy(Vin, Rs, Rb, adc_b, adc_s, Re=0, Rb_accuracy=0.01, BW=50, N=1, oversampling=False):
    def volts(adc, bits):
        """
        This is something of a hack and should be removed with the functionality moving
        to the ADCChannel base class.  Additionally, the adcCounts function should become
        the proper implementation of the __call__ method for an ADCClass
        """
        return adc_b.Vfs[1] * adc / 2**bits

    def adcCounts(gain, V, bits):
        def clipADC(counts):
            return min(counts, 2**bits)
        minimum = clipADC(math.floor(gain * (2**bits) * V / adc_b.Vref.mean))
        maximum = clipADC(minimum + 1)
        best = clipADC(int((gain * (2**bits) * V / adc_b.Vref.mean) + 0.5))
        return minimum, best, maximum

    def Itest_est(Vb):
        return Vb / Rb

    def Rs_est(gain, Vs, Itest):
        return Vs / (gain * Itest)

    I_test = Vin / (Rb + Rs + Re)
    Vb = Rb * I_test
    Vs = Rs * I_test
    bits_os = 0 if not oversampling else oversamplingBits(N)
    bits_b, bits_s = adc_b.enob + bits_os, adc_s.enob + bits_os
    adc_Rb_min, adc_Rb_best, adc_Rb_max = adcCounts(adc_b.gain, Vb, bits_b)
    adc_Rs_min, adc_Rs_best, adc_Rs_max = adcCounts(adc_s.gain, Vs, bits_s)
    Vb_min, Vb_best, Vb_max = volts(adc_Rb_min, bits_b), volts(adc_Rb_best, bits_b), volts(adc_Rb_max, bits_b)
    Vs_min, Vs_best, Vs_max = volts(adc_Rs_min, bits_s), volts(adc_Rs_best, bits_s), volts(adc_Rs_max, bits_s)
    Rs_est_min = adc_b.gain * Vs_min * Rb / (adc_s.gain * Vb_max)
    Rs_est_max = adc_b.gain * Vs_max * Rb / (adc_s.gain * Vb_min)
    Rs_est_best = adc_b.gain * Vs_best * Rb / (adc_s.gain * Vb_best)
    # print('adc_Rb_best = {}, adc_Rs_best = {}'.format(adc_Rb_best, adc_Rs_best))
    # print('Vs_best = {}, Rb = {}, Vb_best = {}'.format(Vs_best, Rb, Vb_best))
    # print('Rs_est_best = {}'.format(Rs_est_best))
    offset = 0 if Rs_est_min == 0 else (100 * (Rs_est_max - Rs_est_min) / Rs_est_min)/2
    offset += Rb_accuracy
    calc_offset = 100 * abs(Rs - Rs_est_best) / Rs
    offset = max(offset, calc_offset)

    # print(Rs, Rs_est_min, Rs_est_best, Rs_est_max, offset, calc_offset)
    Vb_noise = johnsonNoise(Rb, adc_b.odr)
    Vs_noise = johnsonNoise(Rs, adc_s.odr)
    eff_Vb_noise = Vb_noise**2 + adc_b.noise_rms**2
    eff_Vs_noise = Vs_noise**2 + adc_s.noise_rms**2
    Vb_rv = normal_rvs.NRV(Vb_best, variance=eff_Vb_noise)
    Vs_rv = normal_rvs.NRV(Vs_best, variance=eff_Vs_noise)
    # print('Vs_min = {}, Vs_best = {}, Vs_max = {}'.format(Vs_min, Vs_best, Vs_max))
    # print('Vs_Rv = {}, Rb = {}, Vb_rv = {}'.format(Vs_rv, Rb, Vb_rv))
    Rs_noise = (Vs_rv * Rb) / Vb_rv
    noise = 100 * Rs_noise.standard_deviation / Rs
    params = {}
    params['Vb_best'] = Vb_best
    params['Vs_best'] = Vs_best
    params['Rs_est'] = Rs_est_best
    return offset, noise, params


def plotADCVoltages(fig, axs, Rss, Vbs, Vss, Vref, model=None):
    if fig is None:
        fig, axs = plt.subplots(2, figsize=(16, 9))
        axs[0].set_title('Vb Measurements')
        axs[1].set_title('Vs Measurements')
    sns.lineplot(x=Rss, y=Vbs, label=model, ax=axs[0])
    sns.lineplot(x=Rss, y=Vss, label=model, ax=axs[1])
    axs[0].set_ylim(0, normal_rvs.mean(Vref))
    axs[1].set_ylim(0, normal_rvs.mean(Vref))
    return fig, axs


def plotResistanceErrorTest(title, Rss, dRs, noises, idx_start, idx_end, Vbs, Vss, N_samples, noise_limit_percent=0.01):
    N = len(Rss)
    fig, axs = plt.subplots(2, figsize=(16, 9))
    fig.suptitle(title)
    axs[0].set_title('Offset Error (% of measuement)')
    axs[1].set_title('Calculation Noise (% of measurement)')

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
    return fig, axs


def sampleAndNPLCEstimation(odr, NPLC, simultaneous=False, N_min_samples=5, N_additional_samples=4, f_line=60, cycle_accuracy=0.025):
    def minNPLCs():
        samplesPerNPLC = odr / f_line
        # print('Samples per NPLC = {}'.format(samplesPerNPLC))
        NPLC_min = 1
        while (1 - int(NPLC_min * samplesPerNPLC) / (NPLC_min * samplesPerNPLC)) > cycle_accuracy:
            NPLC_min += 1
        return NPLC_min, int(NPLC_min * samplesPerNPLC)
    NPLC_min, N_samples_min = minNPLCs()
    N_per_plc = int(max(1, odr / f_line))
    N_samples = max(N_min_samples, NPLC * N_per_plc)
    N_samples_total = N_samples + N_additional_samples
    if not simultaneous:
        N_samples_total *= 2
    ratio = math.ceil(N_samples_total) / N_samples_min
    NPLC = int(ratio * NPLC_min)
    N_samples = (ratio * N_samples_min)
    if not simultaneous:
        N_samples /= 2
    N_samples -= N_additional_samples
    return int(N_samples), NPLC


# TODO: CREATE MORE FUNCTIONS TO MAKE THIS EASIER TO BUILD DIFFERENT TESTS
# TODO: CALCULATE UNDERFLOW ERROR WHEN THE SAMPLES ARE BELOW THE ENOB BITS....
def runAD7177Test(Rs_range, Vin, Vref, odr=5, N=1e3, target_accuracy_percent=0.015, critical_accuracy_percent=0.1, noise_limit_percent=0.01, NPLC=1, **kwargs):
    ch_b = adc_models.AD7177Channel(Vref)
    ch_s = adc_models.AD7177Channel(Vref)
    ch_b.odr = odr
    ch_s.odr = odr
    print('AD7177 ODR = {} sps'.format(ch_b.odr))
    N_samples, NPLC = sampleAndNPLCEstimation(ch_b.odr, NPLC, simultaneous=ch_b.simultaneous, N_min_samples=5)
    print('\tNPLC = {}, Samples = {} per channel'.format(NPLC, N_samples))
    print('\tSampling Time = {} s'.format(NPLC/60))

    Rss = np.linspace(Rs_range[0], Rs_range[1], int(N))
    dRs = []
    noises = []
    idx_start = None
    idx_end = None
    Vbs = []
    Vss = []
    for idx, Rs in enumerate(Rss):
        dR, noise, params = calcResistorMeasurementAccuracy(Vin, Rs, Rb, ch_b, ch_s, Re=1.5, N=N_samples)
        dRs.append(dR)
        Vbs.append(params['Vb_best'])
        Vss.append(params['Vs_best'])
        noises.append(noise)
        if idx_start is None:
            if dR < critical_accuracy_percent:
                idx_start = idx
        else:
            if (idx_end is None) and (dR >= critical_accuracy_percent):
                idx_end = idx
    idx_start = 0 if idx_start is None else idx_start
    idx_end = -1 if idx_end is None else idx_end

    title = 'AD7177 Milli-Ohm Meter (${:0.3f} \\Omega$ to ${:0.3f} \\Omega$)'.format(*Rs_range)
    fig, axs = plotResistanceErrorTest(title, Rss, dRs, noises, idx_start, idx_end, Vbs, Vss, N_samples, noise_limit_percent)

    if 'plot_adc_voltages' in kwargs:
        v_fig, v_axs = kwargs['plot_adc_voltages']
        plotADCVoltages(v_fig, v_axs, Rss, Vbs, Vss, Vref, model='AD7177')
    data = {
        'Rss': Rss,
        'dRs': dRs,
        'noise': noise,
        'Vbs': Vbs,
        'Vss': Vss
    }
    return fig, axs, data


def runADS1283Test(Rs_range, Vin, Vref, odr=5, N=1e3, target_accuracy_percent=0.015, critical_accuracy_percent=0.1, noise_limit_percent=0.01, NPLC=1, **kwargs):
    Re = 1.5

    def getBestGain(ch, Rtest, Itest, N_samples):
        Vtest = Rtest * Itest
        best_gain = None
        for gain in ch.gains:
            ch.gain = gain
            if Vtest > ch.Vfs[1]: break  # Exit if the measurement is not within range
            Vres = ch.Vres
            Vnoise = ch.noise_rms / math.sqrt(N_samples)

            if (best_gain is None) or (Vnoise < Vres):
                best_gain = gain
        return best_gain

    ch_b = adc_models.ADS1283Channel(Vref)
    ch_s = adc_models.ADS1283Channel(Vref)
    ch_b.odr = odr
    ch_s.odr = odr
    print('ADS1283 ODR = {} sps'.format(ch_b.odr))
    N_samples, NPLC = sampleAndNPLCEstimation(ch_b.odr, NPLC, simultaneous=ch_b.simultaneous, N_min_samples=5)
    print('\tNPLC = {}, Samples = {} per channel'.format(NPLC, N_samples))
    print('\tSampling Time = {} s'.format(NPLC/60))

    Rss = np.linspace(Rs_range[0], Rs_range[1], int(N))
    dRs = []
    noises = []
    idx_start = None
    idx_end = None
    Vbs = []
    Vss = []
    gains_b = []
    gains_s = []
    for idx, Rs in enumerate(Rss):
        Itest = Vin / (Rb + Rs + Re)
        gain_b = getBestGain(ch_b, Rb, Itest, N_samples)
        ch_b.gain = gain_b
        gains_b.append(gain_b)
        gain_s = getBestGain(ch_s, Rs, Itest, N_samples)
        ch_s.gain = gain_s
        gains_s.append(gain_s)
        dR, noise, params = calcResistorMeasurementAccuracy(Vin, Rs, Rb, ch_b, ch_s, Re=Re, N=N_samples)
        dRs.append(dR)
        Vbs.append(params['Vb_best'])
        Vss.append(params['Vs_best'])
        noises.append(noise)
        if idx_start is None:
            if dR < critical_accuracy_percent:
                idx_start = idx
        else:
            if (idx_end is None) and (dR >= critical_accuracy_percent):
                idx_end = idx
    idx_start = 0 if idx_start is None else idx_start
    idx_end = -1 if idx_end is None else idx_end

    title = 'ADS1283 Milli-Ohm Meter (${:0.3f} \\Omega$ to ${:0.3f} \\Omega$)'.format(*Rs_range)
    fig, axs = plotResistanceErrorTest(title, Rss, dRs, noises, idx_start, idx_end, Vbs, Vss, N_samples, noise_limit_percent)

    if 'plot_adc_voltages' in kwargs:
        v_fig, v_axs = kwargs['plot_adc_voltages']
        plotADCVoltages(v_fig, v_axs, Rss, Vbs, Vss, Vref, model='ADS1283')

    gain_fig, gain_ax = plt.subplots(1, figsize=(16, 9))
    gain_fig.suptitle('Optimal gain settings for ADS1283')
    sns.lineplot(x=Rss, y=gains_b, label='Rb gain', ax=gain_ax)
    sns.lineplot(x=Rss, y=gains_s, label='Rs gain', ax=gain_ax)
    gain_fig.tight_layout()

    data = {
        'Rss': Rss,
        'dRs': dRs,
        'noise': noise,
        'Vbs': Vbs,
        'Vss': Vss
    }
    return fig, axs, data


if __name__ == '__main__':
    Vref = normal_rvs.NRV(5, 2.85e-6/6)  # TODO: MAKE THIS A REFERENCE OBJECT WITH NOISE DENSITY SO THAT NOISE CAN BE CALCULATED
    Vin = 5
    # Rs_range = (0.01, 15)
    Rs_range = (0.01, 1e4)
    N = 1e3
    Rb = 100
    noise_limit_percent = 0.01
    target_accuracy_percent = 0.015
    critical_accuracy_percent = 0.1
    NPLC = 10
    odr = 5
    voltages_fig, voltages_axs = plt.subplots(2, figsize=(16, 9))
    voltages_axs[0].set_title('Vb Measurements')
    voltages_axs[1].set_title('Vs Measurements')
    # N_min = calcMinimumCounts(target_accuracy_percent)
    # print('N_min = {} counts'.format(int(N_min)))
    # Rs_max = calcErrorLimitedResistance(Rb, bits=ad7177_ch0.enob, N_min=N_min, Re=10)
    # print('Rs_max = {} ohms'.format(Rs_max))

    runAD7177Test(
        Rs_range,
        Vin,
        Vref,
        odr=odr,
        N=N,
        target_accuracy_percent=target_accuracy_percent,
        critical_accuracy_percent=critical_accuracy_percent,
        noise_limit_percent=noise_limit_percent,
        NPLC=NPLC,
        plot_adc_voltages=(voltages_fig, voltages_axs)
    )

    Vin = 2.5  # This needs to be changed due to the scaling of the ADS1283
    runADS1283Test(
        Rs_range,
        Vin,
        Vref,
        odr=odr,
        N=N,
        target_accuracy_percent=target_accuracy_percent,
        critical_accuracy_percent=critical_accuracy_percent,
        noise_limit_percent=noise_limit_percent,
        NPLC=NPLC,
        plot_adc_voltages=(voltages_fig, voltages_axs)
    )

    # plt.show()
