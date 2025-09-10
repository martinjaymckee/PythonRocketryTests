import math
import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyrse.flight_data as flight_data

if __name__ == '__main__':
    file_path = r'D:\Workspace\Rockets\HPR\Antares Explorer XL\Datalog\2025-08-20-serial-9797-flight-0001.csv'

    data = flight_data.loadAltusMetrumLog(file_path)
    print(data)

    fig, axs = plt.subplots(3, layout='constrained', sharex=True)

    axs[0].plot(data['t'], data['az'])
    axs[1].plot(data['t'], data['Vz'])
    axs[2].plot(data['t'], data['h'])
    # axs[2].plot(data['t'], data['hraw'])


    ts_ref = np.linspace(np.min(data['t'].values), np.max(data['t'].values), 35)
    azs = [data['az'].at(t) for t in ts_ref]
    Vzs = [data['Vz'].at(t) for t in ts_ref]
    hs = [data['h'].at(t) for t in ts_ref]    
    axs[0].scatter(ts_ref, azs, c='b', alpha=0.25)
    axs[1].scatter(ts_ref, Vzs, c='b', alpha=0.25)
    axs[2].scatter(ts_ref, hs, c='b', alpha=0.25)

    
    plt.show()