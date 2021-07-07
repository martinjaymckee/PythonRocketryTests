import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import electronics.discretes as discretes


def parRs(Rs):
    sum_Rinv = 0
    for R in Rs:
        sum_Rinv += 1.0 / R
    return 1.0 / sum_Rinv


class PyroChannel:
    def __init__(self, Vcc, R1, R2, R3, R4, R5, Av_R, Av_I, Rarm=None, Rshunt=None, Rfire=None, R_err=0.01, Av_err=0.01, Voffset=150e-6):
        f_BW = 350e3
        dV_noise_density = 40e-9  # V / sqrt(Hz)
        self.__Vnoise_rti = dV_noise_density * math.sqrt(f_BW)
        self.__Vcc = Vcc
        self.__R1 = R1
        self.__R2 = R2
        self.__R3 = R3
        self.__R4 = R4
        self.__R5 = R5
        self.__Rarm = random.uniform(6e-3, 15e-3) if Rarm is None else Rarm  # Based on DMP3007
        self.__Rshunt = 10e-3 if Rshunt is None else Rshunt
        self.__Rfire = random.uniform(2.5e-3, 4e-3) if Rfire is None else Rfire  # Based on DMN22M5
        self.__Av_R = Av_R * random.gauss(1, Av_err/3)
        self.__Voffset_R = random.gauss(0, Voffset/3)
        self.__Av_I = Av_I * random.gauss(1, Av_err/3)
        self.__Voffset_I = random.gauss(0, Voffset/3)
        print('Av_R = {:0.1f}, Voffset_R = {:0.1f} uV'.format(self.__Av_R, 1e6*self.__Voffset_R))
        print('Av_I = {:0.1f}, Voffset_I = {:0.1f} uV'.format(self.__Av_I, 1e6*self.__Voffset_I))
        self.__arm_closed = False
        self.__shunt_closed = False
        self.__fire_closed = False

    def __call__(self, Vin, Rign=None, Rsource=None):
        Rsource = 0 if Rsource is None else Rsource
        Rign = 1e9 if Rign is None else Rign  # Rign defaults to 1Gohm
        Vbat_sense = Vin * (self.__R2 / (self.__R1 + self.__R2))
        R12 = parRs([self.__Rarm, self.__R3]) if self.__arm_closed else self.__R3
        R23 = parRs([self.__Rshunt, Rign]) if self.__shunt_closed else Rign
        R4R5 = self.__R4 + self.__R5
        R34 = parRs([self.__Rfire, R4R5]) if self.__fire_closed else R4R5
        V2 = Vin * (R23 + R34) / (R12 + R23 + R34)
        V3 = Vin * R34 / (R12 + R23 + R34)
        Vcont_sense = V3 * (self.__R5 / R4R5)
        Rsense = self.__Av_R * (V2 - V3 + self.__Voffset_R + random.gauss(0, self.__Vnoise_rti))
        Isense = self.__Av_I * (V3 + self.__Voffset_I + random.gauss(0, self.__Vnoise_rti))
        R_eff = parRs([self.__R1+self.__R2, (R12+R23+R34)])
        Iin = Vin / R_eff
        # print('V2 = {:0.2f} v, V3 = {:0.2f} v'.format(V2, V3))
        # print('R12 = {:0.2f} ohms, R23 = {:0.2f} ohms, R34 = {:0.2f} ohms'.format(R12, R23, R34))
        return Iin, Vbat_sense, Vcont_sense, self.__constrain_amp(Rsense), self.__constrain_amp(Isense)

    def __constrain_amp(self, V):
        Vmin = 5e-3  # INA180
        Vmax_offset = 2e-2
        if V < Vmin:
            return Vmin
        return min(self.__Vcc-Vmax_offset, V)


class ADCChannel:
    def __init__(self, V_ref, bits=12, Av_err=0.01, Voffset=None):
        self.__V_ref = V_ref
        self.__resolution = 2**bits
        self.__gain = random.gauss(1, Av_err/3)
        self.__noise_bits = 4

    def __call__(self, V):
        V = min(self.__V_ref, max(0, V))
        adc_noise = random.gauss(0, 2**self.__noise_bits/3)
        return int(((self.__resolution-1) * V / self.__V_ref) + adc_noise)


class ResistanceEstimator:
    @classmethod
    def Default(cls):
        C = 1
        b = 0
        return ResistanceEstimator(C, b)

    def __init__(self, C, b, bits=12):
        self.__C = C
        self.__b = b
        self.__bits = bits

    def __call__(self, adc, Vref, Vin, Rcont, Av):
        Vadc = self.__C*V_ref*adc / 2**self.__bits + self.__b
        Rign = (Rcont * Vadc) / (Av * Vin - Vadc)
        return Rign


def calcMeasurementLimits(V_max, V_ref, Av_I, Av_R, Rfire, Icont_max=2.5e-3, bits=12):
    Rfire_min, Rfire_max = Rfire
    R3 = V_max / Icont_max
    Imax = V_ref / (Av_I * Rfire_max)
    Rmax = (V_ref * R3) / (Av_R*V_max - V_ref)
    samples = 2**bits
    Ires = Imax / samples
    Rres = Rmax / samples
    return Imax, Ires, Rmax, Rres, R3


def resistiveDivider(ratio, Rtot=10e3):
    R2 = ratio * Rtot
    R1 = (R2 * (1-ratio)) / ratio
    return R1, R2


def calcADCDivider(Vref, Vin_max, Rtot, Zin=100e3, f_bw=100, e=0.001):
    R2 = (Rtot*Vref*Zin) / (Vin_max*Zin - Vref*Rtot)
    R1 = ((R2 * Zin) / (R2 + Zin)) * ((Vin_max / Vref) - 1)
    R2 = (1-e) * R2
    R1 = (1+e) * R1
    R2_final = discretes.Resistor.closest(R2)
    R1_final = discretes.Resistor.closest(R1) if float(R2_final) <= R2 else discretes.Resistor.ceil(R1)
    Reff = 1 / ((1/R1_final) + (1/R2_final) + (1/Zin))
    C = 1 / (2 * math.pi * Reff * f_bw)
    C = discretes.Capacitor.ceil(C)
    params = {}
    Rlower = (float(R2_final)*Zin) / (float(R2_final) + Zin)
    params['Rtot'] = float(R1_final) + Rlower
    params['Reff'] = Reff
    params['ratio'] = Rlower / params['Rtot']
    params['f_bw'] = 1 / (2 * math.pi * Reff * float(C))
    return R1_final, R2_final, C, params


if __name__ == '__main__':
    V_max = 12.6
    Vcc = 3.3
    V_ref = 3.3
    Av = 50.0
    Rfire = (2.5e-3, 4e-3)
    Icont_max = 3e-3
    Imax, Ires, Rmax, Rres, R3 = calcMeasurementLimits(V_max, V_ref, Av, Av, Rfire, Icont_max=Icont_max)
    print('Imax = {:0.2f} A (resolution {:0.3f} mA)'.format(Imax, 1000*Ires))
    print('Rmax = {:0.2f} ohms (resolution {:0.3f} ohms)'.format(Rmax, Rres))
    print('R3 = {} ohms'.format(R3))

    R1, R2 = resistiveDivider(V_ref/V_max, 10e3)
    R4, R5 = resistiveDivider(V_ref/V_max, R3)
    print('R1 = {:0.1f} ohms, R2 = {:0.1f}'.format(R1, R2))
    print('R4 = {:0.1f} ohms, R5 = {:0.1f}'.format(R4, R5))

    pyro = PyroChannel(Vcc, R1, R2, R3, R4, R5, Av, Av)
    Rsense_adc = ADCChannel(V_ref)
    estimator_R = ResistanceEstimator.Default()

    N = 32
    Rign = 1
    Rs = []
    Rcont = R3 + R4 + R5
    for _ in range(1000):
        adc_sum = 0
        for _ in range(N):
            I, Vbat_sense, Vcont_sense, Rsense, Isense = pyro(V_max, Rign)
            adc = Rsense_adc(Rsense)
            adc_sum += adc
        Rs.append(estimator_R(adc_sum/N, V_ref, V_max, Rcont, Av))

    fig, ax = plt.subplots(1, figsize=(15, 10))
    sns.distplot(Rs, ax=ax)
    Rs = np.array(Rs)
    R_mean = np.mean(Rs)
    R_sd = np.std(Rs)
    ax.axvline(R_mean)
    print('R = {:0.2f} +/- {:0.3f} ohms'.format(R_mean, 3*R_sd))
    fig.tight_layout()
    # plt.show()

    R1, R2, C, params = calcADCDivider(V_ref, V_max, 10e3)
    print('R1 = {:0.2f} ohms, R2 = {:0.2f} ohms, C = {:0.2f} uF'.format(R1, R2, 1e6*C))
    print(params)
    print('ratio * V_max = {:0.2f} v'.format(V_max * params['ratio']))
