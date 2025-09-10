import math
import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyrse.flight_data as flight_data

if __name__ == '__main__':
    file_path = r'D:\Workspace\Rockets\HPR\Antares Explorer XL\Datalog\dtl5.csv'

    data = flight_data.loadEggtimerLog(file_path)

    fig, axs = plt.subplots(3, layout='constrained', sharex=True)
    
    axs[0].plot(data['t'], data['az'])
    axs[1].plot(data['t'], data['Vz'])
    axs[2].plot(data['t'], data['h'])
    axs[2].plot(data['t'], data['hraw'])
    
    plt.show()