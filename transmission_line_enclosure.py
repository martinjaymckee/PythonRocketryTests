import math


class TheilSmallCharacteristics:
    def __init__(self, Qts, Fs, Sd):
        self.Qts = Qts
        self.Fs = Fs
        self.Sd = Sd


class TransmissionLineEnclosure:
    def __init__(self, driver_ts):
        self.__driver_ts = driver_ts

    def __call__(self, c_air=343):
        L = c_air / self.__driver_ts.Fs / 4
        S = self.__driver_ts.Sd
        return L, S


if __name__ == '__main__':
    dma45_params = TheilSmallCharacteristics(Qts=0.59, Fs=150.7, Sd=3.86)
    dma80_params = TheilSmallCharacteristics(Qts=0.47, Fs=97.6, Sd=31.2)
    tl = TransmissionLineEnclosure(dma80_params)
    L, S = tl()
    d = 2 * math.sqrt(S / math.pi)
    print('L = {:0.2f} m, S = {:0.2f} cm^2 (d = {:0.2f} cm)'.format(L, S, d))
