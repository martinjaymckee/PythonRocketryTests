import math
import random

import numpy as np

import normal_rvs


# TODO: ADD DEFAULT RETURN_TYPE TO ADC CHANNEL TO ALLOW SHORTER FUNCTION CALLS
#   THIS COULD ALSO SIMPLY BE DEFAULT KEYWORD ARGUMENTS

def ppFromRMS(rms):
    return math.sqrt(2) * rms


def rmsFromPP(pp):
    return pp / math.sqrt(2)


def noiseFromSNR(Vref, SNR):
    return Vref / (2**((SNR - 1.76) / 6.02))


def noiseFromENOB(Vref, enob):
    return Vref / 2**enob


def enobFromSNR(SNR):
    return (SNR - 1.76) / 6.02


def snrFromENOB(enob):
    return (6.02 * enob) + 1.76


class ADCChannel(object):
    def __init__(self, name, V_ref, bits=10, Av_err=0.01, is_signed=False, Voffset=None, odrs=[100], gains=[1]):
        self.__name = name
        self.__V_ref = normal_rvs.NRV.Construct(V_ref)
        self.__bits = bits
        self.__resolution = 2**bits
        self.__is_signed = is_signed
        self.__gain = random.gauss(1, Av_err/3)
        self.__odrs = odrs
        self.__odr_idx = 0
        self.__gains = gains
        self.__gain_idx = 0

    def __str__(self):
        gain = self.gain
        if gain > 1:
            return '{} (Vref = {} v, odr = {} sps, gain = {})'.format(
                        self.name,
                        normal_rvs.mean(self.Vref),
                        self.odr,
                        self.gain
                    )
        return '{} (Vref = {} v, odr = {} sps)'.format(
                    self.name,
                    normal_rvs.mean(self.Vref),
                    self.odr
                )

    @property
    def name(self):
        return self.__name

    @property
    def Vref(self):
        return self.__V_ref

    @property
    def Vfs(self):
        Vmag = self.Vref / self.gain
        if self.signed:
            return (-Vmag, Vmag)
        return (normal_rvs.NRV(0), Vmag)

    @property
    def Vadc(self):
        if self.signed:
            return (-self.Vref, self.Vref)
        return (normal_rvs.NRV(0), self.Vref)

    @property
    def Vres(self):
        Vmin, Vmax = self.Vfs
        return (Vmax - Vmin) / (2**self.enob)

    @property
    def signed(self):
        return self.__is_signed

    @property
    def simultaneous(self):
        return False

    @property
    def bits(self):
        return self.__bits

    @property
    def noise_free_bits(self):
        if (self.noise_pp == 0) and (self.Vref.varaince == 0):
            return self.bits
        Vnoise = math.sqrt(self.noise_pp**2 + self.Vref.variance)
        return math.log(self.Vref / Vnoise) / math.log(2)

    @property
    def noise_bits(self):
        return self.bits - self.noise_free_bits

    @property
    def enob(self):
        if (self.noise_rms == 0) and (self.Vref.varaince == 0):
            return self.__bits
        Vnoise = math.sqrt(self.noise_rms**2 + self.Vref.variance)
        return math.log(self.Vref / Vnoise) / math.log(2)

    @property
    def odr(self):
        return self.__odrs[self.__odr_idx]

    @odr.setter
    def odr(self, val):
        self.__odr_idx = self.__find_nearest_index(val, self.__odrs)
        return self.odr

    # TODO: ADD SOME FORM OF BANDWIDTH ADJUST TO THE ADC???

    @property
    def gains(self):
        return self.__gains[:]

    @property
    def gain(self):
        return self.__gains[self.__gain_idx]

    @gain.setter
    def gain(self, val):
        self.__gain_idx = self.__find_nearest_index(val, self.__gains)
        return self.gain

    @property
    def noise_rms(self):
        return 2.5e-3

    @property
    def noise_pp(self):
        return 6 * self.noise_rms

    def quantitization_error(self, V):
        Vres = self.Vres
        return 100 * Vres / V

    def __call__(self, V, return_type='counts', full=False, samples=1):
        """
        Eventually this needs to return a normal random variable with the mean being the best
        adc count value and a variance based on the adc noise.  Then, simply converting the
        return value to an int will give a valid sample.
        """
        data = {}
        adc_noise = normal_rvs.NRV.Noise(sd=self.noise_rms)
        V = self.__clip_input(self.gain*V + adc_noise)
        V = normal_rvs.oversample(V, samples)
        adc_intermediate = self.__convert_adc(V)
        return_value = None
        if return_type == 'counts':
            return_value = normal_rvs.NRV(
                self.__clip_counts(int(adc_intermediate.mean + 0.5)),
                int(adc_intermediate.standard_deviation),
                dtype=int,
                limits = (-2**(self.bits-1), 2**(self.bits-1) - 1) if self.signed else (0, 2**self.bits)
            )
        elif return_type == 'voltage_adc':
            return_value = self.__calc_v_adc(adc_intermediate)
        elif return_type == 'voltage_rti':
            return_value = self.__calc_v_rti(adc_intermediate)
        if full:
            data['adc_min'] = normal_rvs.NRV(int(math.floor(adc_intermediate.mean)), int(adc_intermediate.standard_deviation))
            bits_os = math.log(samples) / math.log(4)
            data['adc_min_step'] = 2**(self.bits - self.noise_free_bits - bits_os)
            data['adc_max'] = normal_rvs.NRV(int(math.ceil(adc_intermediate.mean + data['adc_min_step'])), int(adc_intermediate.standard_deviation))
            data['v_adc_min'] = self.__calc_v_adc(data['adc_min'])
            data['v_adc'] = self.__calc_v_adc(adc_intermediate)
            data['v_adc_max'] = self.__calc_v_adc(data['adc_max'])
            data['v_rti_min'] = self.__calc_v_rti(data['adc_min'])
            data['v_rti'] = self.__calc_v_rti(adc_intermediate)
            data['v_rti_max'] = self.__calc_v_rti(data['adc_max'])
            return return_value, data
        return return_value

    def __clip_input(self, V):
        Vmin, Vmax = self.Vadc
        Vmean = V.mean
        if V.mean < Vmin.mean:
            Vmean = Vmin.mean
        elif V.mean > Vmax.mean:
            Vmean = Vmax.mean
        return normal_rvs.NRV(Vmean, V.standard_deviation)

    def __find_nearest_index(self, val, options):
        return (np.abs(np.asarray(options) - val)).argmin()

    def __clip_counts(self, counts):
        """This is currently assuming that the value is unsigned.  It needs to be changed to
        work with differential input ADCs.
        """
        return min(counts, 2**self.bits)

    def __convert_adc(self, V):
        if self.signed:
            pass  # TODO: IMPLEMENT THIS FOR SIGNED CONVERSION
        return (float(self.__resolution - 1) * V) / self.Vadc[1]

    def __calc_v_adc(self, counts):
        v = (normal_rvs.mean(self.Vadc[1]) * counts) / (self.__resolution - 1)
        vmin, vmax = self.Vfs
        v.limits = (self.gain*vmin, self.gain*vmax)
        return v

    def __calc_v_rti(self, counts):
        v = (normal_rvs.mean(self.Vadc[1]) * counts) / (self.gain * (self.__resolution - 1))
        v.limits = self.Vfs
        return v

class AD7177Channel(ADCChannel):
    odrs = [5, 10, 16.66, 20, 49.96, 59.92, 100, 200, 397.5, 500, 1000, 2500, 5000, 10000]

    def __init__(self, V_ref):
        super().__init__('AD7177', V_ref, bits=32, is_signed=True, Av_err=0.01, odrs=AD7177Channel.odrs)

    # Note: the noise is currently based on using the Sinc5 + Sinc1 filter and input buffers
    @property
    def noise_rms(self):
        noise = {
            5: 0.07e-6, 10: 0.1e-6, 16.66: 0.13e-6, 20: 0.14e-6, 49.96: 0.2e-6,
            59.92: 0.23e-6, 100: 0.32e-6, 200: 0.43e-6, 397.5: 0.6e-6,
            500: 0.68e-6, 1000: 0.92e-6, 2500: 1.5e-6, 5000: 2.1e-6,
            10000: 3e-6
        }
        return noise[self.odr]

    @property
    def noise_pp(self):
        noise = {
            5: 0.32e-6, 10: 0.47e-6, 16.66: 0.66e-6, 20: 0.75e-6, 49.96: 1e-6,
            59.92: 1.2e-6, 100: 1.7e-6, 200: 2.2e-6, 397.5: 3.7e-6,
            500: 3.9e-6, 1000: 5.7e-6, 2500: 10e-6, 5000: 16e-6,
            10000: 23e-6
        }
        return noise[self.odr]


class ADS1283Channel(ADCChannel):
    odrs = [250, 500, 1000, 2000, 4000]
    gains = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, V_ref):
        super().__init__('ADS1283', V_ref, bits=32, is_signed=True, Av_err=0.01, odrs=ADS1283Channel.odrs, gains=ADS1283Channel.gains)

    @property
    def Vfs(self):
        Vmag = self.Vref / (2 * self.gain)
        return (-Vmag, Vmag)

    @property
    def Vadc(self):
        Vmag = self.Vref / 2
        if self.signed:
            return (-Vmag, Vmag)
        return (normal_rvs.NRV(0), Vmag)

    # Note: the noise is currently based on the chop bit being enabled
    @property
    def noise_rms(self):
        noise = {
            # 250 sps
            (250, 1): 0.59, (250, 2): 0.6, (250, 4): 0.64, (250, 8): 0.8,
            (250, 16): 1.12, (250, 32): 1.92, (250, 64): 3.84,
            # 500 sps
            (500, 1): 0.84, (500, 2): 0.86, (500, 4): 0.92, (500, 8): 1.12,
            (500, 16): 1.44, (500, 32): 2.88, (500, 64): 5.12,
            # 1000 sps
            (1000, 1): 1.19, (1000, 2): 1.2, (1000, 4): 1.28, (1000, 8): 1.6,
            (1000, 16): 2.08, (1000, 32): 3.84, (1000, 64): 7.04,
            # 2000 sps
            (2000, 1): 1.68, (2000, 2): 1.72, (2000, 4): 1.84, (2000, 8): 2.24,
            (2000, 16): 2.88, (2000, 32): 5.44, (2000, 64): 10.24,
            # 4000 sps
            (4000, 1): 2.4, (4000, 2): 2.44, (4000, 4): 2.64, (4000, 8): 3.2,
            (4000, 16): 4.16, (4000, 32): 8, (4000, 64): 14.72
        }
        return noise[(self.odr, self.gain)] * 1e-6


class LPCChannel(ADCChannel):
    odrs = [100000]

    def __init__(self, V_ref):
        super().__init__('LPCADC', V_ref, bits=12, odrs=LPCChannel.odrs)

    @property
    def noise_rms(self):
        return 1.5 * self.Vref * (1/2**self.bits)


if __name__ == '__main__':
    ch = AD7177Channel(2.5)
    ch.odr = 500

    noise = noiseFromSNR(5, 154)
    print('Given a Vref of 5v and SNR of 154 dB, the noise level is {:0.3g} uV'.format(1e6*noise))

    noise = noiseFromENOB(5, 26.7)
    print('Given a Vref of 5v and ENOB of 26.7, the noise level is {:0.3g} uV'.format(1e6*noise))

    ch = ADS1283Channel(2.5)
    ch.odr = 2000
    ch.gain = 32

    print('{} noise_rms = {}, noise_pp = {}'.format(ch, ch.noise_rms, ch.noise_pp))

    print(ch)
    ch.gain = 1
    print(ch)

    ch = ADCChannel('Test', 2.5)
    v, data = ch(1.25, return_type='voltage_rti', full=True)
