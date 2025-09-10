import os
import os.path

import ambiance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

def reference_air_density(h0=0):
    return ambiance.Atmosphere(h0).density[0]


def create_pandas_flightdata(flightdata, h0=0):
    rho_0 = reference_air_density(h0)
    engine_names = []
    total_impulse = []
    base_altitude = []
    peak_altitude_agl = []
    base_air_density = []
    corrected_altitude_agl = []

    for name, impulse, base, peak in flightdata:
        if peak is not None:
            engine_names.append(name)
            total_impulse.append(impulse)
            base_altitude.append(base)
            peak_altitude_agl.append(peak)
            rho = ambiance.Atmosphere(base).density[0]
            base_air_density.append(rho)
            corrected_altitude_agl.append(peak * rho / rho_0)

    return pd.DataFrame({
        'Engine': engine_names,
        'Total Impulse (Ns)': total_impulse,
        'Base Altitude (m)': base_altitude,
        'Peak AGL (m)': peak_altitude_agl,
        'Base Air Density (kg/m^3)': base_air_density,
        'Corrected AGL (m)': corrected_altitude_agl
    })


if __name__ == '__main__':
    output_dir = r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images'

    flightdata = [
        ['Estes E16', 33.4, 1663, 24.7], # SCORE Club Launch -- Motor Eject
        ['Aerotech F67W', 61.1, 1663, 44.5], # SCORE Club Launch -- Motor Eject
        ['Aerotech G74W', 82.8, 2336, 106.1], # NSL West 2023 -- Motor Eject
        ['Aerotech G74W', 82.8, 1663, 86.9], # SCORE Club Launch -- Electronic Eject
        ['Aerotech G75J', 135.6, 1663, 103], # ULA Intern Launch -- Electronic Eject
        ['Aerotech H128W', 172.9, 2336, None], # NSL West 2024 -- Electronic Eject
        ['Aerotech H165R', 165.0, 1663, None] # NARAM 2025 -- Electronic Eject
    ]

    h_ref = np.max(np.array([x[2] for x in flightdata]))

    pandas_flightdata = create_pandas_flightdata(flightdata, h_ref)
    fig, ax = plt.subplots(1, layout='constrained', figsize=(8, 7))
    fig.suptitle('Altitude vs Total Impulse')
    ax.set_title('Altitude corrected for Alamosa air density', fontdict={'size': 8})

    res = stats.linregress(pandas_flightdata['Total Impulse (Ns)'], pandas_flightdata['Corrected AGL (m)'])

    sns.regplot(data = pandas_flightdata, x='Total Impulse (Ns)', y='Corrected AGL (m)', ax=ax)
    sns.scatterplot(data = pandas_flightdata, x='Total Impulse (Ns)', y='Corrected AGL (m)', color='r', ax=ax)  
    ax.text(0.1, 0.8, '$h_{{agl}} = {:.3f} I_{{total}} + {:.3f}$'.format(res.slope, res.intercept), transform=ax.transAxes)

    # print(res)
    print(pandas_flightdata['Corrected AGL (m)'])

    fig.savefig(os.path.join(output_dir, 'part_I_altitude_vs_impulse'), dpi=300)

    plt.show()
