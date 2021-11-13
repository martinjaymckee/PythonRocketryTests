#
# Standard Library
#
import os
import os.path
import random


#
# Import 3rd Party Libraries
#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#
# Import application Libraries
#
import adc_noise_test


def load_data(path):
    ts, vs = [], []
    with open(path, 'r') as file:
        for line in file:
            if line.find(',') > 0:
                try:
                    tokens = [t.strip() for t in line.split(',')]
                    t = int(tokens[0].strip())
                    v = int(tokens[1].strip())
                    ts.append(t)
                    vs.append(v)
                except Exception as e:
                    print(e)
    return np.array(ts), np.array(vs)


def reprocess_data(input_path, output_path):
    with open(input_path, 'r') as infile:
        parser = adc_noise_test.ADCNoiseParser()
        samples = []
        for c in infile.read():
            new_samples = parser.parse(c.encode('ascii'))
            samples += new_samples
        print('Number of samples = {}'.format(len(samples)))
        ts = np.array([t for t, _ in samples])
        vs = np.array([v for _, v in samples])
        adc_noise_test.save_data(output_path, ts, vs)
        return np.array(samples)


def read_adc_noise_data(directory, prefix='adc_noise_test_data', ext='.csv'):
    data = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and (os.path.splitext(file)[1] == ext):
            path = os.path.join(directory, file)
            ts, vs = load_data(path)
            vs_mean = np.mean(vs)
            vs_sd = np.std(vs)
            data.append((path, ts, vs, vs_mean, vs_sd))
    return data


def volts(xs, Vref, adc_max):
    return Vref * (xs / adc_max)


def noiseBitsApproximation(data_directory, Vref, adc_max):
    data = read_adc_noise_data(data_directory)
    adcs = []
    sds = []
    for datum in data:
        _, _, _, mean, sd = datum
        adcs.append(mean)
        sds.append(sd)
    adcs = np.array(adcs)
    sds = np.array(sds)
    vs = volts(adcs, Vref, adc_max)
    coeff = np.polyfit(vs, sds, 1)
    return coeff


def estimateNoiseBits(vin, Vref, bits=12, data_directory='data'):
    m, b = noiseBitsApproximation(data_directory, Vref, adc_max)
    return m*vin + b


def simulateADCReadings(vin, Vref, N=16, bits=12, sps=1000, t0=None, adc_sd=None, data_directory='data'):
    t = random.uniform(0, 2**32-1) if t0 is None else t0

    def advanceTimestamp(t, dt):
        t = int(t + dt) & 0xffffffff
        return t

    adc_max = 2**bits
    adc_mean = adc_max * (vin / Vref)
    if adc_sd is None:
        adc_sd = estimateNoiseBits(vin, Vref, bits=bits, data_directory=data_directory)
    dt = 1e6 / sps
    ts, vs = [], []
    for _ in range(N):
        t = advanceTimestamp(t, dt)
        v = int(random.gauss(adc_mean, adc_sd) + 0.5)
        ts.append(t)
        vs.append(v)
    ts = np.array(ts)
    vs = np.array(vs)
    return ts, vs


def plotADCNoiseSamples(ts, vs, Vref, ax, color=None, bits=12, reference=False):
    fill = False
    alpha = 0.05
    if reference:
        fill = True
        alpha = 0.15
    sns.kdeplot(vs, color=color, alpha=alpha, fill=fill, bw_adjust=pow(len(vs), 1/5), lw=0.75, ax=ax)
    c = 'k'
    try:
        c = ax.get_lines()[-1].get_c()
    except Exception as e:
        pass
    ax.axvline(np.mean(vs), alpha=0.25, c=c)
    vs = volts(vs, Vref, 2**bits)
    v_mean = np.mean(vs)
    v_sd = np.std(vs)
    return v_mean, v_sd


def plotSimulationError(Vref, reference_path, num_samples=64, bits=12, simulated_sd=False, num_tests=5):
    ts, vs = load_data(reference_path)
    fig, axs = plt.subplots(2, figsize=(16, 9), constrained_layout=True, gridspec_kw={'height_ratios': [1, 3]})
    title = 'Measurement Error Distribution (with {} samples)'.format(num_samples)
    if simulated_sd:
        title += '\n-- Estimated S.D. --'
    fig.suptitle(title)
    axs[0].set_xticks([], minor=[])
    axs[0].set_ylabel('ADC Reading (lsb)')
    axs[1].set_xlabel('ADC Reading (lsb)')
    axs[1].set_ylabel('Reading Density')

    v_mean, v_sd = plotADCNoiseSamples(ts, vs, Vref, color='k', ax=axs[1], reference=True)

    adc_sd = estimateNoiseBits(v_mean, Vref, bits=bits, data_directory='data') if simulated_sd else np.std(vs)

    errs_est = []
    vs_est = []
    for idx in range(num_tests):
        ts, vs = simulateADCReadings(v_mean, Vref, N=num_samples, bits=bits, adc_sd=adc_sd)
        axs[0].plot(vs, alpha=0.5, lw=0.75)
        new_v_mean, new_v_sd = plotADCNoiseSamples(ts, vs, Vref, ax=axs[1])
        vs_est.append(new_v_mean)
        errs_est.append(v_mean - new_v_mean)
    errs_est = np.array(errs_est)
    err_max = 100 * np.max(np.abs(errs_est)) / v_mean
    err_std = 100 * np.std(errs_est) / v_mean
    msg = 'Sample Errors\nS.D. = {:0.4g} %\nMax = {:0.4g} %'.format(err_std, err_max)
    x_min, x_max = axs[1].get_xlim()
    y_min, y_max = axs[1].get_ylim()
    x_range = x_max - x_min
    x = x_min + 0.1 * x_range
    y = 0.67 * y_max
    axs[1].text(x, y, msg)
    axs[0].axhline(adc_max * v_mean / Vref, alpha=0.5, c='k')


if __name__ == '__main__':
    Vref = 3.3
    bits = 12
    num_samples = 1024
    data_directory = 'data'
    data = read_adc_noise_data(data_directory)

    adcs = []
    sds = []
    adc_max = 2**bits

    reference_path = os.path.join(data_directory, 'adc_noise_test_data_11.csv')
    plotSimulationError(Vref, reference_path, num_samples=num_samples, bits=bits, num_tests=35)
    # for datum in data:
    #     _, _, _, mean, sd = datum
    #     adcs.append(mean)
    #     sds.append(sd)
    # adcs = np.array(adcs)
    # sds = np.array(sds)
    # fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
    # vs = volts(adcs, Vref, adc_max)
    # sns.regplot(vs, sds, ax=ax)
    # coeff = np.polyfit(vs, sds, 1)
    # print('Noise bits (s.d.) = {:0.3f} * vin + {:0.4f}'.format(*coeff))
    #
    # adcs_est = np.linspace(0, adc_max, 20)
    # vs_est = volts(adcs_est, Vref, adc_max)
    # sds_est = np.array([estimateNoiseBits(v, Vref, bits=bits, data_directory=data_directory) for v in vs_est])
    # sns.regplot(vs_est, sds_est,  color='g', ax=ax)


    plt.show()
