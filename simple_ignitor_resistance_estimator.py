import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Vbus = 8.4
    Vbus = 6
    # R3 = 16.2e3
    # R4 = 10e3
    R3 = 3.3e3
    R4 = 2e3
    Vref = 3.3
    oversampling_bits = 4
    adc_bits = 12
    adc_max = 2**(adc_bits + oversampling_bits)
    alpha, gamma = Vref/adc_max, 0

    def calc_adc(R):
        return ((Vbus * R4) / (alpha * (R3 + R4 + R))) - (gamma / alpha)

    def calc_dadc(R):
        return -(Vbus * R4) / (alpha * (R3 + R4 + R)**2)

    def calc_dR(R):
        return -1 / calc_dadc(R)

    Rs = np.linspace(0.5, 4.5)
    dadcs = np.array([calc_dadc(R) for R in Rs])
    adcs = np.array([calc_adc(R) for R in Rs])
    dRs = np.array([calc_dR(R) for R in Rs])
    fig, axs = plt.subplots(3, figsize=(16, 9), constrained_layout=True)

    axs[0].plot(Rs, adcs)
    axs[1].plot(Rs, dadcs)
    axs[2].plot(Rs, dRs)

    print('Static Current = {} mA'.format(1000 * Vbus / (R3 + R4)))
    print('ADC samples = {}'.format(4**oversampling_bits))
    plt.show()
