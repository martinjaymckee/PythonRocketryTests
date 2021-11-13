import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import adc_models
# import electronics.discretes as discretes
import normal_rvs


class MilliohmMeterConstraints:
    def __init__(self, target_accuracy=0.015, critical_accuracy=0.1, noise_limit=0.01, **kwargs):
        self.__target_accuracy = target_accuracy
        self.__critical_accuracy = critical_accuracy
        self.__noise_limit = noise_limit
        if not len(kwargs) == 0:
            classname = self.__class__.__name__
            keys = '{}'.format(', '.join(list(kwargs.keys())))
            assert False, 'Error: Unused keyword arguments in {} -- {}'.format(classname, keys)

    @property
    def target_accuracy(self):
        return self.__target_accuracy

    @property
    def critical_accuracy(self):
        return self.__critical_accuracy

    @property
    def noise_limit(self):
        return self.__noise_limit


def johnsonNoise(R, BW, T=300):
    kb = 1e6 * 1.380649e-23  # Boltzmann's constant
    return math.sqrt(4 * kb * T * R * BW)


def calcResistorMeasurementAccuracy(Vin, Rs, Rb, adc_b, adc_s, Re=0, Rb_accuracy=0.01, BW=None, N=1, oversampling=False):
    BW_b = adc_b.odr / 2 if BW is None else BW
    BW_s = adc_s.odr / 2 if BW is None else BW

    I_test = Vin / (Rb + Rs + Re)

    effective_samples = 1 if not oversampling else N
    Vb = normal_rvs.NRV(Rb * I_test, johnsonNoise(Rb, BW_b))
    Vs = normal_rvs.NRV(Rs * I_test, johnsonNoise(Rs, BW_s))

    Vb_rv, data_b = adc_b(Vb, return_type='voltage_rti', full=True, samples=effective_samples)
    Vs_rv, data_s = adc_s(Vs, return_type='voltage_rti', full=True, samples=effective_samples)
    # print('Vb = {}, Vb_rv = {}'.format(Vb, Vb_rv))
    Rs_est_min = data_s['v_rti_min'] * Rb / data_b['v_rti_max']
    Rs_est_max = data_s['v_rti_max'] * Rb / data_b['v_rti_min']
    Rs_est_best = Vs_rv * Rb / Vb_rv

    offset = 0 if Rs_est_min == 0 else normal_rvs.mean(50 * (Rs_est_max - Rs_est_min) / Rs_est_min)
    offset += Rb_accuracy

    noise = 100 * normal_rvs.standard_deviation(Rs_est_best) / Rs
    params = {}
    params['Vb_best'] = normal_rvs.mean(Vb_rv)
    params['Vs_best'] = normal_rvs.mean(Vs_rv)
    params['Vb_adc'] = normal_rvs.mean(data_b['v_adc'])
    params['Vs_adc'] = normal_rvs.mean(data_s['v_adc'])
    params['Rs_est'] = Rs_est_best
    return offset, noise, params


def plotADCVoltages(fig, axs, Rss, Vbs, Vss, Itests, Vref, model=None):
    Itest_max = 0
    if fig is None:
        fig, axs = plt.subplots(2, figsize=(16, 9))
        axs[0].set_title('Vb Measurements (v)')
        axs[1].set_title('Vs Measurements (v)')
        axs[2].set_title('Itest (mA)')
    else:
        _, Itest_max = axs[2].get_ylim()
    sns.lineplot(x=Rss, y=Vbs, label=model, ax=axs[0])
    sns.lineplot(x=Rss, y=Vss, label=model, ax=axs[1])
    sns.lineplot(x=Rss, y=1000*Itests, label=model, ax=axs[2])
    axs[0].set_ylim(0, normal_rvs.mean(Vref))
    axs[1].set_ylim(0, normal_rvs.mean(Vref))
    # axs[2].set_ylim(0, max(Itest_max, np.max(Itests)))
    return fig, axs


def plotResistanceErrorTest(title, Rss, dRs, noises, idx_start, idx_end, Vbs, Vss, N_samples, offset_min, constraints):
    N = len(Rss)
    fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True)
    fig.suptitle(title)
    axs[0].set_title('Offset Error (% of measuement)')
    axs[1].set_title('Calculation Noise (% of measurement)')

    sns.lineplot(x=Rss[idx_start:idx_end], y=dRs[idx_start:idx_end], ax=axs[0])
    axs[0].axhspan(0, constraints.target_accuracy, fc='g', alpha=0.1)
    _, y_max = axs[0].get_ylim()
    axs[0].set_ylim(offset_min, min(y_max, constraints.critical_accuracy))
    ymax = min(constraints.critical_accuracy, axs[0].get_ylim()[1])
    if np.max(dRs) >= constraints.critical_accuracy:
        ytext = ymax - (ymax - constraints.target_accuracy) / 10
        axs[0].annotate(
            'Critical Offset at $Rs = {:0.3g} \\Omega$'.format(Rss[idx_start]),
            xy=(Rss[idx_start], constraints.critical_accuracy),
            xytext=(Rss[min(int(3*N/4), idx_start + int(N/45))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )
    error_idx = np.argmax(np.array(dRs) < constraints.target_accuracy)
    error_idx -= 1
    if error_idx >= 0:
        Rs_target_offset = Rss[error_idx]
        ytext = constraints.target_accuracy + (ymax - constraints.target_accuracy) / 10
        axs[0].annotate(
            'Offset Limit at $Rs = {:0.3g} \\Omega$'.format(Rs_target_offset),
            xy=(Rs_target_offset, dRs[error_idx]),
            xytext=(Rss[min(int(3*N/4), error_idx + int(N/45))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )

    noises = np.array(noises) / math.sqrt(N_samples)
    error_idx = np.argmax(np.array(noises) < constraints.noise_limit)
    Rs_critical_noise = Rss[error_idx]
    sns.lineplot(x=Rss[idx_start:idx_end], y=noises[idx_start:idx_end], ax=axs[1])
    axs[1].axhspan(0, constraints.noise_limit, fc='g', alpha=0.1)
    axs[1].set_ylim(0, min(np.max(noises[idx_start:idx_end]), constraints.noise_limit))
    if noises[idx_start] >= constraints.noise_limit:
        ytext = constraints.noise_limit + (np.max(noises[idx_start:idx_end]) - constraints.noise_limit) / 10
        axs[1].annotate(
            'Noise Limit at $Rs = {:0.3g} \\Omega$'.format(Rs_critical_noise),
            xy=(Rs_critical_noise, constraints.noise_limit),
            xytext=(Rss[min(int(3*N/4), error_idx + int(N/45))], ytext),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3')
        )

    fig.tight_layout()
    return fig, axs


def sampleAndNPLCEstimation(odr, NPLC, simultaneous=False, N_min_samples=5, N_additional_samples=4, f_line=60, cycle_accuracy=0.025):
    def minNPLCs():
        samplesPerNPLC = odr / f_line
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


# TODO: CALCULATE UNDERFLOW ERROR WHEN THE SAMPLES ARE BELOW THE ENOB BITS....


def runMilliohmMeterTest(Rs_range, Vin, Vref, ADCType, odr=5, N=1e3, NPLC=1, oversampling=False, constraints=None, **kwargs):
    Re = 100
    constraints = MilliohmMeterConstraints() if constraints is None else constraints

    def getBestGain(ch, Rtest, Itest, N_samples):
        Vtest = Rtest * Itest
        best_gain = None
        for gain in ch.gains:
            ch.gain = gain
            if Vtest >= ch.Vfs[1]: break  # Exit if the measurement is not within range
            Vres = ch.Vres
            Vnoise = ch.noise_rms / math.sqrt(N_samples)
            if (best_gain is None) or (Vnoise < constraints.noise_limit):  #Vres):
                best_gain = gain
        return best_gain

    ch_b = ADCType(Vref)
    ch_s = ADCType(Vref)
    ch_b.odr = odr
    ch_s.odr = odr
    print('{} ODR = {} sps'.format(ch_b.name, ch_b.odr))
    N_samples, NPLC = sampleAndNPLCEstimation(ch_b.odr, NPLC, simultaneous=ch_b.simultaneous, N_min_samples=5)
    print('\tNPLC = {}, Samples = {} per channel'.format(NPLC, N_samples))
    print('\tSampling Time = {} s'.format(NPLC/60))

    R_scale = pow((Rs_range[1]/Rs_range[0]), (1/N))
    Rss = np.array([Rs_range[0] * pow(R_scale, idx) for idx in range(int(N))])  # np.linspace(Rs_range[0], Rs_range[1], int(N))
    dRs = []
    noises = []
    idx_start = None
    idx_end = None
    Vbs = []
    Vss = []
    gains_b = []
    gains_s = []
    Itests = []
    for idx, Rs in enumerate(Rss):
        Itest = Vin / (Rb + Rs + Re)
        Itests.append(Itest)
        if len(ch_b.gains) > 1:
            gain_b = getBestGain(ch_b, Rb, Itest, N_samples)
            ch_b.gain = gain_b
            gains_b.append(gain_b)
            gain_s = getBestGain(ch_s, Rs, Itest, N_samples)
            ch_s.gain = gain_s
            gains_s.append(gain_s)
        dR, noise, params = calcResistorMeasurementAccuracy(Vin, Rs, Rb, ch_b, ch_s, Re=Re, N=N_samples, oversampling=oversampling)
        dRs.append(dR)
        Vbs.append(params['Vb_adc'])
        Vss.append(params['Vs_adc'])
        noises.append(noise)
        if idx_start is None:
            if dR < constraints.critical_accuracy:
                idx_start = idx
        else:
            if (idx_end is None) and (dR >= constraints.critical_accuracy):
                idx_end = idx
    idx_start = 0 if idx_start is None else idx_start
    idx_end = -1 if idx_end is None else idx_end

    title = '{} Milli-Ohm Meter (${:0.3f} \\Omega$ to ${:0.3f} \\Omega$) -- ODR = {} Hz, , Rb = ${} \\Omega$'.format(ch_b.name, *Rs_range, ch_b.odr, Rb)
    # TODO: PASS IN MINIMUM OFFSET VALUE
    fig, axs = plotResistanceErrorTest(title, Rss, dRs, noises, idx_start, idx_end, Vbs, Vss, N_samples, 0.01, constraints)

    adc_model = '{} (odr = {} Hz, Rb = ${} \\Omega$)'.format(ch_b.name, ch_b.odr, Rb)
    if 'plot_adc_voltages' in kwargs:
        v_fig, v_axs = kwargs['plot_adc_voltages']
        plotADCVoltages(v_fig, v_axs, Rss, Vbs, Vss, np.array(Itests), Vref, model=adc_model)

    if len(ch_b.gains) > 1:
        gain_fig, gain_ax = plt.subplots(1, figsize=(16, 9))
        gain_fig.suptitle('Optimal gain settings for {}'.format(adc_model))
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


def runAD7177Test(Rs_range, Vin, Vref, odr=5, N=1e3, NPLC=1, oversampling=False, constraints=None, **kwargs):
    return runMilliohmMeterTest(Rs_range, Vin, Vref, adc_models.AD7177Channel, odr, N, NPLC, oversampling, constraints, **kwargs)


def runADS1283Test(Rs_range, Vin, Vref, odr=5, N=1e3, NPLC=1, oversampling=False, constraints=None, **kwargs):
    return runMilliohmMeterTest(Rs_range, Vin, Vref, adc_models.ADS1283Channel, odr, N, NPLC, oversampling, constraints, **kwargs)


if __name__ == '__main__':
    # TODO: READ THE TI WHITE PAPER ON PRECISION ADC NOISE ANALYSIS AND IMPLEMENT WHATEVER I FIND THERE
    Vref = normal_rvs.NRV(5, 2.85e-6/6)
    print('Vref = {}'.format(Vref))
    Vin = 5
    oversampling = True
    # Rs_range = (1e-5, 1)
    Rs_range = (1e-4, 15)
    # Rs_range = (1e-4, 1e4)
    N = 500
    Rb = 200  # TODO: CHECK IF THIS RESISTOR IS EVEN AVAILABLE AT THE HIGH ACCURACY AND STABILITY THAT IS REQUIRED....

    constraints = MilliohmMeterConstraints(
        noise_limit=0.01,
        target_accuracy=0.015,
        critical_accuracy=0.1
    )

    NPLC = 10
    odr = 200
    voltages_fig, voltages_axs = plt.subplots(3, figsize=(16, 9))
    voltages_axs[0].set_title('Vb_adc Measurements (v)')
    voltages_axs[1].set_title('Vs_adc Measurements (v)')
    voltages_axs[2].set_title('Itests (mA)')

    # runAD7177Test(
    #     Rs_range,
    #     Vin,
    #     Vref,
    #     odr=odr,
    #     N=N,
    #     oversampling=oversampling,
    #     constraints=constraints,
    #     NPLC=NPLC,
    #     plot_adc_voltages=(voltages_fig, voltages_axs)
    # )

    Vin = 2.5  # This needs to be changed due to the scaling of the ADS1283
    Rb = 100   # This needs to be changed to keep the maximum test current consistent
    NPLC = 3
    odr = 250
    runADS1283Test(
        Rs_range,
        Vin,
        Vref,
        odr=odr,
        N=N,
        oversampling=oversampling,
        constraints=constraints,
        NPLC=NPLC,
        plot_adc_voltages=(voltages_fig, voltages_axs)
    )

    NPLC = 3
    odr = 1000
    runADS1283Test(
        Rs_range,
        Vin,
        Vref,
        odr=odr,
        N=N,
        oversampling=oversampling,
        constraints=constraints,
        NPLC=NPLC,
        plot_adc_voltages=(voltages_fig, voltages_axs)
    )

    plt.show()

    # Vin = 2.5
    # odr = 250
    # Rs = 0.1
    # Rb = 100
    # ch_b = adc_models.ADS1283Channel(Vref)
    # ch_s = adc_models.ADS1283Channel(Vref)
    # ch_b.odr = odr
    # ch_s.odr = odr
    # ch_s.gain = 64
    # N = 100
    # dR, noise, data = calcResistorMeasurementAccuracy(Vin, Rs, Rb, ch_b, ch_s, Re=1.5, BW=odr/2, N=N, oversampling=False)
    # print('dR = {:0.6f} %, noise = {:0.4f} %'.format(dR, noise))
    # # print('data = {}'.format(data))
    # for key, value in data.items():
    #     print('\t{}: {}'.format(key, value))
    #
    # Vin = 2.5
    # odr = 250
    # Rs = 0.1
    # Rb = 100
    # ch_b = adc_models.ADS1283Channel(Vref)
    # ch_s = adc_models.ADS1283Channel(Vref)
    # ch_b.odr = odr
    # ch_s.odr = odr
    # ch_s.gain = 8
    # N = 100
    # dR, noise, data = calcResistorMeasurementAccuracy(Vin, Rs, Rb, ch_b, ch_s, Re=1.5, BW=odr/2, N=N, oversampling=False)
    # print('dR = {:0.6f} %, noise = {:0.4f} %'.format(dR, noise))
    # # print('data = {}'.format(data))
    # for key, value in data.items():
    #     print('\t{}: {}'.format(key, value))
