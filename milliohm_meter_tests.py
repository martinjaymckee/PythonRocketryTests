import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import electronics.discretes as discretes


class ADCChannel:
    def __init__(self, V_ref, bits=12, noise_free_bits=10, Av_err=0.01, Voffset=None):
        self.__V_ref = V_ref
        self.__bits = bits
        self.__noise_free_bits = noise_free_bits
        self.__resolution = 2**bits
        self.__gain = random.gauss(1, Av_err/3)
        self.__noise_bits = bits - noise_free_bits

    @property
    def bits(self):
        return self.__bits

    @property
    def noise_free_bits(self):
        return self.__noise_free_bits

    @property
    def noise_bits(self):
        return self.__noise_bits

    def __call__(self, V):
        V = min(self.__V_ref, max(0, V))
        adc_noise = random.gauss(0, 2**self.__noise_bits/3)
        return int(((self.__resolution-1) * V / self.__V_ref) + adc_noise)


def calcMinimumCounts(accuracy, oversampling=5):
    return oversampling * (100 / accuracy)


def calcErrorLimitedResistance(Rb, bits=10, N_min=None, Re=0):
    N_min = (2**bits) / 10 if N_min is None else N_min
    return ((2**bits) / N_min) * (Rb - ((N_min/(2**bits))*(Rb+Re)))


# def calcResistorMeasurementAccuracy(bits, Vin, Vref, Rs, Rb, Re=0):
#     return 100 / math.floor(((2**bits) * (Vin * Rb) / (Vref * (Rb + Rs + Re))))


def johnsonNoise(R, BW, T=300):
    kb = 1.380649e-23  # Boltzmann's constant
    return math.sqrt(4 * kb * T * R * BW)


def calcResistorMeasurementAccuracy(bits, Vin, Vref, Rs, Rb, Re=0, Rb_accuracy=0.01, BW=50):
    I_test = Vin / (Rb + Rs + Re)
    Vb = Rb * I_test
    Vs = Rs * I_test
    adc_Rb_min = math.floor((2**bits) * Vb / Vref)
    adc_Rs_min = math.floor((2**bits) * Vs / Vref)
    adc_Rb_max = adc_Rb_min + 1
    adc_Rs_max = adc_Rs_min + 1
    Vb_adc_min, Vb_adc_max = Vref * adc_Rb_min / 2**bits, Vref * adc_Rb_max / 2**bits
    Vs_adc_min, Vs_adc_max = Vref * adc_Rs_min / 2**bits, Vref * adc_Rs_max / 2**bits
    Rs_est_min = Vs_adc_min * Rb / Vb_adc_max
    Rs_est_max = Vs_adc_max * Rb / Vb_adc_min
    # print('Rs = {:0.4f} ohms, Vb = {:0.4f} v, Itest = {:0.4f} mA'.format(Rs, Vb, 1000*I_test))
    # print('adc_Rb_min = {}'.format(adc_Rb_min))
    offset = 0 if Rs_est_min == 0 else (100 * (Rs_est_max - Rs_est_min) / Rs_est_min)
    offset += Rb_accuracy
    noise = johnsonNoise(Rb+Rs+Re, BW)
    # print('Johnson Noise = {:0.4f} nV'.format(1e9 * noise))
    noise = 100 * noise / Vs
    return offset, noise


if __name__ == '__main__':
    Vref = 2.5
    Rs_range = (1e-2, 1e6)
    noise_free_bits = 26
    ad7177_ch0 = ADCChannel(Vref, bits=32, noise_free_bits=noise_free_bits, Av_err=0.1e-6)
    ad7177_ch1 = ADCChannel(Vref, bits=32, noise_free_bits=noise_free_bits, Av_err=0.1e-6)

    Vin = 2.5
    Rb = 10
    target_accuracy_percent = 0.015
    N_min = calcMinimumCounts(target_accuracy_percent)
    Rs_max = calcErrorLimitedResistance(Rb, bits=ad7177_ch0.noise_free_bits, N_min=N_min, Re=10)
    print('Rs_max = {} ohms'.format(Rs_max))

    Rss = np.linspace(Rs_range[0], Rs_range[1], 100000)
    bits = ad7177_ch0.noise_free_bits
    dRs = []
    uppers = []
    lowers = []
    for Rs in Rss:
        dR, noise = calcResistorMeasurementAccuracy(bits, Vin, Vref, Rs, Rb, Re=1.5)
        dRs.append(dR)
        uppers.append(dR + noise)
        lowers.append(dR - noise)
    fig, ax = plt.subplots(1, figsize=(16, 9))
    sns.lineplot(x=Rss, y=dRs, ax=ax)
    ax.axhline(target_accuracy_percent, c='m', alpha=0.25)
    ax.plot(Rss, lowers, c='tab:blue', alpha=0.1)
    ax.plot(Rss, uppers, c='tab:blue', alpha=0.1)
    ax.fill_between(Rss, lowers, uppers, alpha=0.2)
    # print('dR = {:0.3f} %'.format(dR_percent))
    fig.tight_layout()
    plt.show()
