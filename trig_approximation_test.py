import math


import matplotlib.pyplot as plt
import numpy as np
import numpy.random


def reference_values(angles):
    sines = np.sin(angles)
    cosines = np.cos(angles)
    return sines, cosines


def approximate_values(angles, order, N=None):
    N = int(len(angles) / 2 if N is None else N)
    order = int(max(1, order))
    test_angles = np.sort(np.random.choice(angles, N))

    sines = np.zeros(N)
    cosines = np.zeros(N)

    doAdd = True
    for n in range(0, order+1):
        term = ((test_angles**n) / math.factorial(n))
        if (n % 2) == 0:  # Even order... update cosine estimates
            if doAdd:
                cosines += term
            else:
                cosines -= term
        else:  # Odd order... update sine estimates
            if doAdd:
                sines += term
            else:
                sines -= term
            doAdd = not doAdd
    return test_angles, sines, cosines


def naive_approximation(angles):
    sines = angles
    cosines = np.ones(len(angles))
    return sines, cosines


def better_approximation(angles):
    sines = angles
    cosines = 1.0 - ((angles**2) / 2)
    return sines, cosines


def good_approximation(angles):
    sines = angles - ((angles**3) / 6)
    cosines = 1.0 - ((angles**2) / 2)
    return sines, cosines


def great_approximation(angles):
    sines = angles - ((angles**3) / 6)
    cosines = 1.0 - ((angles**2) / 2) + ((angles**4) / math.factorial(4))
    return sines, cosines


def awesome_approximation(angles):
    sines = angles - ((angles**3) / 6) + ((angles**5) / math.factorial(5))
    cosines = 1.0 - ((angles**2) / 2) + ((angles**4) / math.factorial(4))
    return sines, cosines


def format_order(order):
    if order == 1:
        return '$1^{st}$'
    elif order == 2:
        return '$2^{nd}$'
    elif order == 3:
        return '$3^{rd}$'
    return '${}^{{th}}$'.format(order)


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"

    gimbal_max_angle = math.radians(15)
    max_angle = 3 * gimbal_max_angle
    N = 100
    sample_size = 0.8 * N
    max_order = 5
    angles = np.linspace(0, max_angle, N)

    fig, axs = plt.subplots(3, figsize=(6.5, 6), sharex=True)
    for ax in axs:
        ax.set_yscale('log')
        ax.set_ylabel('Error (%)', fontsize=9)
    axs[2].set_xlabel('Angle ($rad$)', fontsize=9)
    axs[0].set_title('Sin Error')
    axs[1].set_title('Cos Error')
    axs[2].set_title('Magnitude Error')

    for order in range(1, max_order+1):
        test_angles, sines, cosines = approximate_values(angles, order, N=sample_size)
        ref_sines, ref_cosines = reference_values(test_angles)
        sine_errors = np.abs(100 * (sines - ref_sines) / ref_sines)
        cosine_errors = np.abs(100 * (cosines - ref_cosines) / ref_cosines)
        magnitude_errors = np.abs(100 * (((sines**2) + (cosines**2)) - 1))
        axs[0].scatter(test_angles, sine_errors, s=15, alpha=0.5, label='{} Order'.format(format_order(order)))
        axs[1].scatter(test_angles, cosine_errors, s=15, alpha=0.5)
        axs[2].scatter(test_angles, magnitude_errors, s=15, alpha=0.5)


    # ref_sines, ref_cosines = reference_values(angles)
    # naive_sines, naive_cosines = naive_approximation(angles)
    # better_sines, better_cosines = better_approximation(angles)
    # good_sines, good_cosines = good_approximation(angles)
    # great_sines, great_cosines = great_approximation(angles)
    # awesome_sines, awesome_cosines = awesome_approximation(angles)
    #
    # axs[0].plot(angles, np.abs(100 * (naive_sines - ref_sines) / ref_sines), c='b', alpha=0.5, label='First-Order Approximation')
    # axs[1].plot(angles, np.abs(100 * (naive_cosines - ref_cosines) / ref_cosines), c='b', alpha=0.5)
    # axs[0].plot(angles, np.abs(100 * (better_sines - ref_sines) / ref_sines), c='c', alpha=0.5, label='Second-Order Approximation')
    # axs[1].plot(angles, np.abs(100 * (better_cosines - ref_cosines) / ref_cosines), c='c', alpha=0.5)
    # axs[0].plot(angles, np.abs(100 * (good_sines - ref_sines) / ref_sines), c='g', alpha=0.5, label='Third-Order Approximation')
    # axs[1].plot(angles, np.abs(100 * (good_cosines - ref_cosines) / ref_cosines), c='g', alpha=0.5)
    # axs[0].plot(angles, np.abs(100 * (great_sines - ref_sines) / ref_sines), c='m', alpha=0.5, label='Fourth-Order Approximation')
    # axs[1].plot(angles, np.abs(100 * (great_cosines - ref_cosines) / ref_cosines), c='m', alpha=0.5)
    # axs[0].plot(angles, np.abs(100 * (awesome_sines - ref_sines) / ref_sines), c='y', alpha=0.5, label='Fifth-Order Approximation')
    # axs[1].plot(angles, np.abs(100 * (awesome_cosines - ref_cosines) / ref_cosines), c='y', alpha=0.5)
    #
    # naive_magnitude_error = 100 * (((naive_sines**2) + (naive_cosines**2)) - 1)
    # better_magnitude_error = 100 * (((better_sines**2) + (better_cosines**2)) - 1)
    # good_magnitude_error = 100 * (((good_sines**2) + (good_cosines**2)) - 1)
    # great_magnitude_error = 100 * (((great_sines**2) + (great_cosines**2)) - 1)
    # awesome_magnitude_error = 100 * (((awesome_sines**2) + (awesome_cosines**2)) - 1)
    #
    # axs[2].plot(angles, naive_magnitude_error, c='b', alpha=0.5)
    # axs[2].plot(angles, better_magnitude_error, c='c', alpha=0.5)
    # axs[2].plot(angles, good_magnitude_error, c='g', alpha=0.5)
    # axs[2].plot(angles, great_magnitude_error, c='m', alpha=0.5)
    # axs[2].plot(angles, awesome_magnitude_error, c='y', alpha=0.5)

    # axs[0].axhline(1, c='k', alpha=0.25)
    axs[0].axhline(0.1, c='k', alpha=0.25)
    axs[0].axvline(gimbal_max_angle, c='k', alpha=0.25)
    # axs[1].axhline(1, c='k', alpha=0.25)
    axs[1].axhline(0.1, c='k', alpha=0.25)
    axs[1].axvline(gimbal_max_angle, c='k', alpha=0.25)
    # axs[2].axhline(1, c='k', alpha=0.25)
    axs[2].axhline(0.1, c='k', alpha=0.25)
    axs[2].axvline(gimbal_max_angle, c='k', alpha=0.25)

    # ylims = axs[1].get_ylim()
    # axs[0].set_ylim(*ylims)

    axs[0].legend(prop={'size': 9})

    fig.tight_layout()
    plt.show()
