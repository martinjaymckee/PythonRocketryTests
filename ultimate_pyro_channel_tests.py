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


def calcADCDivider(Vref, Vin_max, Rtot=None, Zin=100e3, f_bw=50, e=0.001):
    # TODO: NEED TO MAKE THIS WORK WITH A VARIABLE ZIN.  FIRST, CALCULATE THE DIVIDER FOR THE
    #   TOTAL RANGE OF ZIN VALUES, THEN CHOOSE DIVIDER VALUES THAT ALLOW FOR calibration
    #   OF ANY ERRORS.  FINALLY, ESTIMATE THE EFFECTS OF RESISTOR ERRORS, ZIN CHANGES, ETC.
    # TODO: CALCULATE DIVIDER OUTPUT RATE OF CHANGE FUNCTIONS
    #   dVadc/dR1
    #   dVadc/dR2
    #   dVadc/dZin
    # TODO: ESTIMATE ADC READING ERROR BASED ON ZIN RANGE AND RESISTOR ERRORS
    Rtot = Zin / 10 if Rtot is None else Rtot
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
    params['error'] = 0  # Estimated error range in percent
    return R1_final, R2_final, C, params


def calcResistanceSenseErrors(Rign_range, Vref, Vin_range, R3, R45, N_channels=4, p_r=1e-3, p_amp=1e-2, sd_adc=0, sd_amp=2.4e-5, N_samples=1):
    Avs = [20, 50, 100, 200]  # INA180 Gains
    # Avs = [25, 50, 100, 200, 500]  # INA186 Gains
    Avs = sorted(Avs)
    adc_inl = 2  # lsb
    adc_dnl = 3  # lsb
    adc_noise = 7   # lsb
    adc_bits = 12

    def calcItest(Vin, R3, Rign_a, R45_a, Rign_b, R45_b):
        Itot = Vin / (R3 + (parRs([(Rign_a + R45_a), ((Rign_b + R45_b) / N_channels)])))
        Itest = (Vin - (Itot * R3)) / (Rign_a + R45_a)
        print('Itot = {:0.3f} mA, Itest = {:0.3f} mA, V2 = {:0.3f} v'.format(1000*Itot, 1000*Itest, (Vin - (Itot * R3))))
        return Itest, Itot

    Vin_min, Vin_max = Vin_range
    Rign_min, Rign_max = Rign_range

    def calcItestError(Vin, Rign):
        # Calculate Itest_max
        R3_est = (1-p_r) * R3
        R45_a_est = (1+p_r) * R45
        Rign_b = Rign_max
        R45_b_est = (1+p_r) * R45
        Itest_max, _ = calcItest(Vin_max, R3_est, Rign, R45_a_est, Rign_b, R45_b_est)

        # Calculate Itest_min
        R3_est = (1+p_r) * R3
        R45_a_est = (1+p_r) * R45
        Rign_b = Rign_min
        R45_b_est = (1-p_r) * R45
        Itest_min, _ = calcItest(Vin_max, R3_est, Rign, R45_a_est, Rign_b, R45_b_est)

        Itest_err = abs((Itest_max - Itest_min) / Itest_min)
        print('Itest range = {:0.5f} mA to {:0.5f} mA'.format(1000*Itest_min, 1000*Itest_max))
        print('Itest_err = {:0.4f} %'.format(100 * Itest_err))
        return Itest_err

    Itest_err_Rmin = calcItestError(Vin_max, Rign_min)
    Itest_err_Rmax = calcItestError(Vin_max, Rign_max)

    Itest, _ = calcItest(Vin_min, R3, Rign_min, R45, Rign_min, R45)
    Av_r_optimal = Vref / ((1+p_amp) * Itest * Rign_max)
    print('Av_r_optimal = {:0.2f}'.format(Av_r_optimal))
    Av_r = Avs[0]
    for Av in Avs:
        if Av_r_optimal >= Av:
            Av_r = Av
        else:
            break
    print('Av_r = {:0.2f}'.format(Av_r))
    adc_Rmin = math.ceil((2**adc_bits) * ((1-p_amp) * Av_r * Itest * Rign_min) / Vref)
    print('adc_Rmin = {} counts'.format(adc_Rmin))
    adc_nl_err_Rmin = abs((adc_inl + adc_dnl + 1/2) / adc_Rmin)  # ADC non-linearity and quantization
    print('adc_nl_err_Rmin = {:0.4f} %'.format(100 * adc_nl_err_Rmin))
    adc_Rmax = math.ceil((2**adc_bits) * ((1-p_amp) * Av_r * Itest * Rign_max) / Vref)
    adc_nl_err_Rmax = abs((adc_inl + adc_dnl + 1/2) / adc_Rmax)  # ADC non-linearity and quantization
    print('adc_nl_err_Rmax = {:0.4f} %'.format(100 * adc_nl_err_Rmax))

    sd_adc = adc_noise / (3*(2**adc_bits))
    SE_noise_err = abs((sd_adc + ((1+p_amp)*Av_r*sd_amp)) / math.sqrt(N_samples))
    print('SE_noise_err = {:0.4f} %'.format(100 * SE_noise_err))

    adc_err_Rmin = Itest_err_Rmin + SE_noise_err + adc_nl_err_Rmin
    adc_err_Rmax = Itest_err_Rmax + SE_noise_err + adc_nl_err_Rmax
    print('adc_err = ({:0.4f} %, {:0.4f} %)'.format(100 * adc_err_Rmin, 100 * adc_err_Rmax))


def calcOptimalResistanceSenseCurrent(Rmin, Voffset=500e-6, sense_scale=1.0):
    return (sense_scale * Voffset) / Rmin


def calcResistanceSenseParameters(Rmin, Vref, Isafe=0.3, safety_factor=50, min_samples=25, bits=12, Voffset=150e-6, sense_scale=2.1):
    """
        Rmin - the minimum resistor value that needs to be accurate (>min_samples)
        Isafe - the maximum no-fire current of the possible initiators to be used
        safety_factor - the maximum amount to divide the no-fire current by for resistance testing
        min_samples - a minimum number of samples... greater than the ADC noise Floor
        bits - the resolution of the ADC
    """
    Rres = Rmin / min_samples
    Rmax = 2**bits * Rres
    Itest_max = Isafe / safety_factor
    Itest_optimal = calcOptimalResistanceSenseCurrent(Rmin, Voffset, sense_scale)
    print('Itest_optimal = {:0.3f} mA'.format(1000 * Itest_optimal))
    Itest = min(Itest_max, Itest_optimal)
    Rsense_fs = Rmax * Itest

    params = {}
    params['fs_ratio'] = Rsense_fs / Voffset
    params['min_ratio'] = (Rmin * Itest) / Voffset
    params['gain'] = Vref / (Rmax * Itest)
    print(params)
    return Rmax, Rres, Itest_max, Itest, Rsense_fs, params


if __name__ == '__main__':
    V_max = 12.6
    Vcc = 3.3
    V_ref = 3.3
    Av_i = 50.0
    Av_r = 50.0
    Rfire = (2.5e-3, 4e-3)
    Icont_max = 1.5e-3
    Imax, Ires, Rmax, Rres, R3 = calcMeasurementLimits(V_max, V_ref, Av_i, Av_r, Rfire, Icont_max=Icont_max)
    print('Imax = {:0.2f} A (resolution {:0.3f} mA)'.format(Imax, 1000*Ires))
    print('Rmax = {:0.2f} ohms (resolution {:0.3f} ohms)'.format(Rmax, Rres))
    print('R3 = {} ohms'.format(R3))

    R1, R2 = resistiveDivider(V_ref/V_max, 10e3)
    R4, R5 = resistiveDivider(V_ref/V_max, R3)
    print('R1 = {:0.1f} ohms, R2 = {:0.1f}'.format(R1, R2))
    print('R4 = {:0.1f} ohms, R5 = {:0.1f}'.format(R4, R5))

    pyro = PyroChannel(Vcc, R1, R2, R3, R4, R5, Av_r, Av_i)
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
        Rs.append(estimator_R(adc_sum/N, V_ref, V_max, Rcont, Av_r))

    fig, ax = plt.subplots(1, figsize=(15, 10))
    sns.distplot(Rs, ax=ax)
    Rs = np.array(Rs)
    R_mean = np.mean(Rs)
    R_sd = np.std(Rs)
    ax.axvline(R_mean)
    print('R = {:0.2f} +/- {:0.3f} ohms'.format(R_mean, 3*R_sd))
    fig.tight_layout()
    # plt.show()

    # Note: Here the minimum possible Vref is being used, assuming a +/- 1% voltage regulator.
    # Note: Calculating with the highest Zin possible will result in an underestimation of the
    #   voltage and - as such = provide headroom for calibration.
    #
    Rtot = 25e3
    Vref_min = 0.99 * V_ref
    print('Calc ADC Divider with Zin = 150k:')
    R1, R2, C, params = calcADCDivider(Vref_min, V_max, Rtot=Rtot, Zin=150e3)
    print('R1 = {:0.2f} ohms, R2 = {:0.2f} ohms, C = {:0.2f} uF'.format(R1, R2, 1e6*C))
    print(params)
    print('ratio * V_max = {:0.2f} v'.format(V_max * params['ratio']))

    print('\nCalc ADC Divider with Zin = 100k:')
    R1, R2, C, params = calcADCDivider(Vref_min, V_max, Rtot=Rtot)
    print('R1 = {:0.2f} ohms, R2 = {:0.2f} ohms, C = {:0.2f} uF'.format(R1, R2, 1e6*C))
    print(params)
    print('ratio * V_max = {:0.2f} v'.format(V_max * params['ratio']))

    print('\nCalc ADC Divider with Zin = 50k:')
    R1, R2, C, params = calcADCDivider(Vref_min, V_max, Rtot=Rtot, Zin=50e3)
    print('R1 = {:0.2f} ohms, R2 = {:0.2f} ohms, C = {:0.2f} uF'.format(R1, R2, 1e6*C))
    print(params)
    print('ratio * V_max = {:0.2f} v'.format(V_max * params['ratio']))

    # TODO: THIS SHOULD PROBABLY BE USED TO CALCULATE THE BEST RANGE OF TEST CURRENT AND MAXIMUM RESISTANCE
    #   MEASURED.  THIS SHOULD ENSURE THAT THE MAXIMUM RESISTANCE IS AT LEAST AS HIGH AS THE LARGEST
    #   INITIATOR LIKELY TO BE MEASURED.  ALSO, SOME ANALYSIS OF THE INFORMATION INHERENT IN THE ADC SIGNALS
    #   NEEDS TO BE DONE SO THAT THE ACCURACY GOALS ARE CAPABLE OF BEING HIT.
    print()
    Rmax, Rres, Isense_max, Itest, Rsense_fs, params = calcResistanceSenseParameters(0.1, 3.3)
    print('Rmax = {:0.2f} ohms, Rres = {:0.2f} mohms'.format(Rmax, 1000*Rres))
    print('Isense_max = {:0.3f} mA, Itest = {:0.3f} mA, Rsense_fs = {:0.3f} v'.format(1000*Isense_max, 1000*Itest, Rsense_fs))
    print(params)

    print()
    Itest_optimal = calcOptimalResistanceSenseCurrent(0.1)
    print('Itest_optimal = {:0.2f} mA'.format(1000 * Itest_optimal))

    R45 = R3 = 1050
    Rign_range = (0.4, 5)
    print('R3 = {} ohms'.format(R3))
    calcResistanceSenseErrors(Rign_range, 3.3, (4.2, 12.6), R3, R45)
