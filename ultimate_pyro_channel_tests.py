import math
import random


def parRs(Rs):
    sum_Rinv = 0
    for R in Rs:
        sum_Rinv += 1.0 / R
    return 1.0 / sum_Rinv


class PyroChannel:
    def __init__(self, R1, R2, R3, R4, R5, Av_R, Av_I, Rarm=None, Rshunt=None, Rfire=None, R_err=0.01, Av_err=0.01, Voffset=150e-6):
        f_BW = 350e3
        dV_noise_density = 40e-9  # V / sqrt(Hz)
        self.__Vnoise_rms_rti = dV_noise_density * math.sqrt(f_BW)
        self.__R1 = R1
        self.__R2 = R2
        self.__R3 = R3
        self.__R4 = R4
        self.__R5 = R5
        self.__Rarm = random.uniform(6e-3, 15e-3) if Rarm is None else Rarm  # Based on DMP3007
        self.__Rshunt = random.uniform(6e-3, 15e-3) if Rarm is None else Rarm  # Based on DMP3007
        self.__Rfire = random.uniform(2.5e-3, 4e-3) if Rarm is None else Rfire  # Based on DMN22M5
        self.__Av_R = Av_R
        self.__Voffset_R = random.gauss(0, Voffset/3)
        self.__Av_I = Av_I
        self.__Voffset_I = random.gauss(0, Voffset/3)
        self.__arm_closed = False
        self.__shunt_closed = True
        self.__fire_closed = False

    def __call__(self, Vin, Rign=None, Rsource=None):
        Rsource = 0 if Rsource is None else Rsource
        Rign = 1e9 if Rign is None else Rign  # Rign defaults to 1Gohm
        Vbat_sense = Vin * (self.__R2 / (self.__R1 + self.__R2))
        R12 = parRs([self.__Rarm, self.__R3]) if self.__arm_closed else self.__R3
        R23 = parRs([self.__Rshunt, Rign]) if self.__shunt_closed else Rign
        R4R5 = self.__R4 + self.__R5
        R34 = parRs([self.__Rfire, R4R5]) if self.__shunt_closed else R4R5
        V2 = Vin * (R23 + R23) / (R12 * R23 * R34)
        V3 = Vin * R34 / (R12 + R23 + R34)
        Vcont_sense = V3 * (self.__R5 / R4R5)
        Rsense = self.__Av_R * (V2 - V3 + self.__Voffset_R + random.gauss(0, self.__Vnoise_rms_rti))
        Isense = self.__Av_I * (V3 + self.__Voffset_I + random.gauss(0, self.__Vnoise_rms_rti))
        return Vbat_sense, Vcont_sense, Rsense, Isense


def calcMeasurementLimits(V_max, V_ref, Av_I, Av_R, Rfire, Icont_max=2.5e-3, bits=10):
    Rfire_min, Rfire_max = Rfire
    R3 = V_max / Icont_max
    Imax = V_ref / (Av_I * Rfire_max)
    Rmax = (V_ref * R3) / (Av_R*V_max - V_ref)
    samples = 2**bits
    Ires = Imax / samples
    Rres = Rmax / samples
    return Imax, Ires, Rmax, Rres, R3


if __name__ == '__main__':
    V_max = 12.6
    V_ref = 3.3
    Av = 50.0
    Rfire = (2.5e-3, 4e-3)
    Icont_max = 3e-3
    Imax, Ires, Rmax, Rres, R3 = calcMeasurementLimits(V_max, V_ref, Av, Av, Rfire, Icont_max=Icont_max)
    print('Imax = {:0.2f} A (resolution {:0.3f} mA)'.format(Imax, 1000*Ires))
    print('Rmax = {:0.2f} ohms (resolution {:0.3f} ohms)'.format(Rmax, Rres))
    print('R3 = {} ohms'.format(R3))
