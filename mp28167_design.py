import numpy as np
import matplotlib.pyplot as plt


def calcInductor(Vout, Vin, Imax, fsw=500e3, dI_scalar=0.3):
    dIL = dI_scalar * Imax
    if Vin > Vout:
        return (Vout / (fsw * dIL)) * (1 - (Vout / Vin))
    return (Vin * (Vout - Vin)) / (Vout * fsw * dIL)


def calcAdjustableLT3086Resistors(Vdac, Vouts, Vref=0.4, Iset=50e-6, Rb=None, doplot=False):
    Iset += 0 if Rb is None else (Vref / Rb)
    Vout_min, Vout_max = Vouts
    Vout_range = Vout_max - Vout_min
    Ra = (Vout_range * (1 - (Vref / Vdac))) / Iset
    Rdac = (Ra * Vdac) / Vout_range
    print('Vouts = {}'.format(Vouts))
    if doplot:
        fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
        fig.suptitle('Calculated Output Voltages')
        Vdacs = np.linspace(0, Vdac, 10)
        Vouts = Ra * (Iset + (Vref / Ra) - ((Vdacs - Vref) / Rdac))
        Ias = (Vouts - Vref) / Ra
        Idacs = (Vdacs - Vref) / Rdac
        if Rb is None:
            axs[0].set_title('$Ra = {:0.2f} k\Omega$, $Rb = DNP$, $Rdac = {:0.2f} k\Omega$'.format(Ra/1000, Rdac/1000))
        else:
            axs[0].set_title('$Ra = {:0.2f} k\Omega$, $Rb = {:0.2f} k\Omega$, $Rdac = {:0.2f} k\Omega$'.format(Ra/1000, Rb/1000, Rdac/1000))
        axs[0].plot(Vdacs, Vouts)
        axs[1].plot(Vdacs, 1e6*Ias)
        axs[2].plot(Vdacs, 1e6*Idacs)
        axs[0].axhline(Vout_max, alpha=0.25, c='k')
        axs[0].axhline(Vout_min, alpha=0.25, c='k')
        axs[0].set_xlabel('Vdac (volts)')
        axs[0].set_ylabel('Vout (volts)')
        axs[1].set_ylabel('Ia (micro-amps)')
        axs[2].set_ylabel('Idac (micro-amps)')
    return Ra, Rb, Rdac


# ADS1260 Design
class ADS126x:
    __gains = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self, AVdd, AVss=0, gain=1):
        self.__AVdd = AVdd
        self.__AVss = AVss
        self.gain = gain

    @property
    def Vref(self):
        return 2.5

    @property
    def gain(self):
        return self.__gain

    @gain.setter
    def gain(self, val):
        if val in ADS126x.__gains:
            self.__gain = val
        return self.gain

    @property
    def vin_diff(self):
        v = self.Vref / self.gain
        return (-v, v)

    @property
    def vin_abs(self): # TODO: FIGURE OUT HOW TO MAKE THIS CORRECT
        if self.gain == 1:
            return (self.__AVss - 0.1, self.__AVdd + 0.1)
        _, Vin = self.vin_diff
        Voff = (Vin * (self.gain - 1) / 2.0)
        print('Voffset = {}'.format(Voff))
        Vin_p = self.__AVdd - 0.3 - Voff
        Vin_n = self.__AVss + 0.3 + Voff
        return (Vin_n, Vin_p)



if __name__ == '__main__':

    Vin = 5
    Vout = (1, 9)
    Imax = 0.5
    fsw = 750e3

    # fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    # Ls = []
    # Vs = []
    # for V in np.linspace(*Vout, 100):
    #     L = calcInductor(V, Vin, Imax, fsw)
    #     Vs.append(V)
    #     Ls.append(L)
    # Ls = 1e6 * np.array(Ls)
    # axs[0].plot(Vs, Ls)

    calcAdjustableLT3086Resistors(2.5, (0.4, 9), doplot=True)

    # plt.show()

    adc = ADS126x(2.5, -2.5)
    adc.gain = 128
    print('ADC gain = {}'.format(adc.gain))
    print('ADC differential input voltage = {}'.format(adc.vin_diff))
    print('ADC absolute input voltage = {}'.format(adc.vin_abs))
