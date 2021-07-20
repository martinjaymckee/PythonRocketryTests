import math
import random

import numpy as np


class ADCChannel(object):
    def __init__(self, V_ref, bits=12, Av_err=0.01, Voffset=None, odrs=[100], gains=[1, 2, 4, 8, 16, 32, 64]):
        self.__V_ref = V_ref
        self.__bits = bits
        self.__resolution = 2**bits
        self.__gain = random.gauss(1, Av_err/3)
        self.__odrs = odrs
        self.__odr_idx = 0
        self.__gains = gains
        self.__gain_idx = 0

    @property
    def Vref(self):
        return self.__V_ref

    @property
    def bits(self):
        return self.__bits

    @property
    def noise_free_bits(self):
        if self.noise_pp == 0:
            return self.bits
        return math.log(self.Vref / self.noise_pp) / math.log(2)

    @property
    def noise_bits(self):
        return self.bits - self.noise_free_bits

    @property
    def enob(self):
        if self.noise_rms == 0:
            return self.__bits
        return math.log(self.Vref / self.noise_rms) / math.log(2)

    @property
    def odr(self):
        return self.__odrs[self.__odr_idx]

    @odr.setter
    def odr(self, val):
        self.__odr_idx = self.__find_nearest_index(val, self.__odrs)
        return self.odr

    # TODO: ADD SOME FORM OF BANDWIDTH ADJUST TO THE ADC???

    @property
    def gain(self):
        return self.__gains[self.__gain_idx]

    @gain.setter
    def gain(self, val):
        self.__gain_idx = self.__find_nearest_index(val, self.__gains)
        return self.gain

    @property
    def noise_rms(self):
        # Note: this should be overloaded based on odr, gain, etc...
        return 0.07e-6

    @property
    def noise_pp(self):
        # Note: this should be overloaded based on odr, gain, etc...
        return 0.32e-6

    def __call__(self, V):
        V = min(self.__V_ref, max(0, V))
        adc_noise = random.gauss(0, 2**self.__noise_bits/3)
        return int(((self.__resolution-1) * V / self.__V_ref) + adc_noise)

    def __find_nearest_index(self, val, options):
        return (np.abs(np.asarray(options) - val)).argmin()


class AD7177Channel(ADCChannel):
    odrs = [5, 10, 16.66, 20, 49.96, 59.92, 100, 200, 397.5, 500, 1000, 2500, 5000, 10000]

    def __init__(self, V_ref):
        super().__init__(V_ref, bits=32, Av_err=0.01, odrs=AD7177Channel.odrs)

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

if __name__ == '__main__':
    ch = AD7177Channel(2.5)
    ch.odr = 500
