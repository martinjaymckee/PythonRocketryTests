# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 08:52:50 2022

@author: marti
"""

import os
import os.path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def validate(key, val, minimum, maximum):
    if (val > maximum) or (val < minimum):
        raise(ValueError('{} = {}'.format(key, val)))
    
    
def getHistRange(vals, sd=None, N=6):
    mean = vals.mean()
    sd = vals.std() if sd is None else sd
    return (mean-(N*sd), mean+(N*sd))


def loadIMUTestdata(directory, filename):
    path = os.path.join(directory, filename)

    data = {
        'ax' : [],
        'ay' : [],
        'az' : [],
        'gx' : [],
        'gy' : [],
        'gz' : [],
    }
    
    with open(path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            
            if not line == '':
                tokens = [t.strip() for t in line.split(',')]
                if len(tokens) == 6:
                    try:
                        values = [float(t) for t in tokens]
                        validate('ax', values[0], 8.8, 10.8)
                        validate('ay', values[1], -1.5, 1.5)
                        validate('az', values[2], -1.5, 1.5)
                        validate('gx', values[3], -0.25, 0.25)                        
                        validate('gy', values[4], -0.25, 0.25)
                        validate('gz', values[5], -0.25, 0.25)
                        data['ax'].append(values[0])
                        data['ay'].append(values[1])
                        data['az'].append(values[2])
                        data['gx'].append(values[3])
                        data['gy'].append(values[4])
                        data['gz'].append(values[5])                        
                    except Exception as e:
                        print(e)   
    return pd.DataFrame(data)


def main():
    directory = 'GPS Logger GUI/testdata'
    filename = 'teraterm7.log'

    df = loadIMUTestdata(directory, filename)
    print(len(df))
    
    dt = 1 / 40
    ts = dt * np.array(range(len(df)))
    sigmas = 3
    
#    df.plot()
    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    axs[0].set_title("Acceleration Relative Bias")
    axs[1].set_title("Angular Rate Relative Bias")    
    scatter_kws = {'alpha':0.005, 'edgecolors':None, 's':15}
    
    sns.regplot(x=ts, y=df['ax'] - df['ax'].mean(), color='b', order=2, ax=axs[0], scatter_kws=scatter_kws)
    sns.regplot(x=ts, y=df['ay'] - df['ay'].mean(), color='c', order=2, ax=axs[0], scatter_kws=scatter_kws)
    sns.regplot(x=ts, y=df['az'] - df['az'].mean(), color='g', order=2, ax=axs[0], scatter_kws=scatter_kws)
    sd_a = max(df['ax'].std(), df['ay'].std(), df['az'].std())
    axs[0].set_ylim(-sigmas * sd_a, sigmas * sd_a)
    
    sns.regplot(x=ts, y=df['gx'] - df['gx'].mean(), color='b', order=2, ax=axs[1], scatter_kws=scatter_kws)
    sns.regplot(x=ts, y=df['gy'] - df['gy'].mean(), color='c', order=2, ax=axs[1], scatter_kws=scatter_kws)
    sns.regplot(x=ts, y=df['gz'] - df['gz'].mean(), color='g', order=2, ax=axs[1], scatter_kws=scatter_kws)
    sd_g = max(df['gx'].std(), df['gy'].std(), df['gz'].std())
    axs[1].set_ylim(-sigmas * sd_g, sigmas * sd_g)
    
    # fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    # axs[0].set_title("Accelerations")
    # axs[1].set_title("Angular Rates")
    
    # axs[0].plot(df['ax'] - df['ax'].mean(), c='b', alpha=0.35, label='ax')
    # axs[0].plot(df['ay'] - df['ay'].mean(), c='c', alpha=0.35, label='ay')
    # axs[0].plot(df['az'] - df['az'].mean(), c='g', alpha=0.35, label='az')
    
    # axs[1].plot(df['gx'] - df['gx'].mean(), c='b', alpha=0.35, label='gx')
    # axs[1].plot(df['gy'] - df['gy'].mean(), c='c', alpha=0.35, label='gy')
    # axs[1].plot(df['gz'] - df['gz'].mean(), c='g', alpha=0.35, label='gz')
    
    num_bins = 21
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    axs[0][0].hist(df['ax'], color='b', range=getHistRange(df['ax'], sd=sd_a, N=sigmas), bins=num_bins)
    axs[0][1].hist(df['ay'], color='c', range=getHistRange(df['ay'], sd=sd_a, N=sigmas), bins=num_bins)
    axs[0][2].hist(df['az'], color='g', range=getHistRange(df['az'], sd=sd_a, N=sigmas), bins=num_bins)

    axs[1][0].hist(df['gx'], color='b', range=getHistRange(df['gx'], sd=sd_g, N=sigmas), bins=num_bins)
    axs[1][1].hist(df['gy'], color='c', range=getHistRange(df['gy'], sd=sd_g, N=sigmas), bins=num_bins)
    axs[1][2].hist(df['gz'], color='g', range=getHistRange(df['gz'], sd=sd_g, N=sigmas), bins=num_bins)
    
    for row, sensor in enumerate(["a", "g"]):
        for column, axis in enumerate(["x", "y", "z"]):
            key = sensor + axis
            mean = df[key].mean()
            sd = df[key].std()
            axs[row][column].set_title('{} ($\mu$ = {:0.4G}, $\sigma$ = {:0.4G})'.format(key, mean, sd))
            axs[row][column].axvline(mean, c='k')

    plt.show()

    
if __name__ == '__main__':
    main()
    