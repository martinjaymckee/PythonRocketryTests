import math


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


def residual_sin(variables, thetas):
    C0, C1, C2, C3 = variables
    C0 = 1
    s = (C0*thetas) - (C1*(thetas**3))
    return np.abs(100 * (s - np.sin(thetas)) / np.sin(thetas))


def residual_cos(variables, thetas):
    C0, C1, C2, C3 = variables
    c = C2 - (C3*(thetas**2))
    return np.abs(100 * (c - np.cos(thetas)) / np.cos(thetas))


if __name__ == '__main__':
    theta_max = 0.5
    thetas = np.linspace(-theta_max, theta_max, 250)
    std_variables = [1.0, 1/6, 1.0, 1/2]

    out = scipy.optimize.leastsq(residual_sin, std_variables, args=(thetas, ))
    variables = out[0]
    out = scipy.optimize.leastsq(residual_cos, variables, args=(thetas, ))
    variables = out[0]
    print('Optimal C0 = {}, C1 = {}, C2 = {}, C3 = {}'.format(*variables))

    fig, axs = plt.subplots(2, figsize=(6.5, 4), sharex=True)
    for ax in axs:
        ax.axhline(0.1, c='k', alpha=0.25)
        ax.set_yscale('log')
    fig.suptitle('Least Squares Optimized $3^{{rd}}$ Order Approximation')
    fig.text(0.22, 0.89, 'C0 = {:0.6g}, C1 = {:0.6g}, C2 = {:0.6g}, C3 = {:0.6g}'.format(*variables))
    axs[0].set_title('Sine Absolute Error')
    axs[1].set_title('Cosine Absolute Error')
    axs[1].set_xlabel('Angle (rad)', fontsize=9)

    test_thetas = np.linspace(0, theta_max, 50)
    standard_sin_errors = np.abs(residual_sin(std_variables, test_thetas))
    optimized_sin_errors = np.abs(residual_sin(variables, test_thetas))
    mean_standard_sin_errors = np.nanmean(standard_sin_errors)
    mean_optimized_sin_errors = np.nanmean(optimized_sin_errors)
    p = axs[0].plot(test_thetas, standard_sin_errors, alpha=0.5, label='Standard ({:0.3g} %)'.format(mean_standard_sin_errors))
    axs[0].axhline(mean_standard_sin_errors, c=p[0].get_color(), alpha=0.1)
    p = axs[0].plot(test_thetas, optimized_sin_errors, alpha=0.5, label='Optimized ({:0.3g} %)'.format(mean_optimized_sin_errors))
    axs[0].axhline(mean_optimized_sin_errors, c=p[0].get_color(), alpha=0.1)
    standard_cos_errors = np.abs(residual_cos(std_variables, test_thetas))
    optimized_cos_errors = np.abs(residual_cos(variables, test_thetas))
    mean_standard_cos_errors = np.nanmean(standard_cos_errors)
    mean_optimized_cos_errors = np.nanmean(optimized_cos_errors)

    p = axs[1].plot(test_thetas, standard_cos_errors, alpha=0.5, label='Standard ({:0.3g} %)'.format(mean_standard_cos_errors))
    axs[1].axhline(mean_standard_cos_errors, c=p[0].get_color(), alpha=0.1)
    p = axs[1].plot(test_thetas, optimized_cos_errors, alpha=0.5, label='Optimized ({:0.3g} %)'.format(mean_optimized_cos_errors))
    axs[1].axhline(mean_optimized_cos_errors, c=p[0].get_color(), alpha=0.1)

    for ax in axs:
        ax.legend(prop={'size': 9})

    fig.tight_layout()

    plt.show()
