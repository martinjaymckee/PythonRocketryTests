#
# Standard Library
#
import os
import os.path
import random


#
# Import 3rd Party Libraries
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(path):
    columns = ['Point', 'Sample', 'Timestamp', "Position", "Reference", "ADC"]
    return pd.read_csv(path, sep=',', names=columns, dtype=int, comment='#')


def plotADCvsPosition(df, plot_ref=False, plot_test=True):
    fig, axs = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
    scatter_kws = {'alpha': 0.15}
    df = df.groupby('Point').mean()
    if plot_ref:
        sns.regplot(x='Position', y='Reference', data=df, ax=axs[0], scatter_kws=scatter_kws)
        coeff = np.polyfit(df['Position'], df['Reference'], deg=1)
        adc_est = (coeff[0] * df['Position']) + coeff[1]
        sns.lineplot(x=df['Position'], y=(df['Reference'] - adc_est), alpha=0.2, ax=axs[1])
    if plot_test:
        sns.regplot(x='Position', y='ADC', data=df, ax=axs[0], scatter_kws=scatter_kws)
        coeff = np.polyfit(df['Position'], df['ADC'], deg=1)
        adc_est = (coeff[0] * df['Position']) + coeff[1]
        sns.lineplot(x=df['Position'], y=(df['ADC'] - adc_est), alpha=0.2, ax=axs[1])
    return fig, axs


def plotSampleRingdown(df, plot_pos=None, segments=10):
    import more_itertools

    def getSampleSegments(data):
        N = int(len(data) / segments)
        index_chunks = more_itertools.chunked(data.index, N)
        chunks = [df.iloc[slice_index] for slice_index in index_chunks]
        return list(range(len(chunks))), chunks

    fig, axs = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
    plot_pos = df['Position'][0] if plot_pos is None else plot_pos
    df_sub = df[df['Position'] == plot_pos]
    groups = df_sub.groupby('Point')
    means = []
    sds = []
    for idx, (key, data) in enumerate(groups):
        sns.lineplot(x='Sample', y='ADC', data=data, ax=axs[0], alpha=0.1)
        means.append(data['ADC'].mean())
        sds.append(data['ADC'].std())
        idxs, chunks = getSampleSegments(data)
        segment_noise = [chunk['ADC'].std() for chunk in chunks]
        sns.lineplot(x=idxs, y=segment_noise, ax=axs[1])
    mean = np.mean(means)
    sd = np.max(sds)
    axs[0].set_ylim(mean - 7*sd, mean + 7*sd)
    return fig, ax


def plotADCNoise(df, plot_ref=False, plot_test=True):
    fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
    scatter_kws = {'alpha': 0.15}
    groups = df.groupby('Point')
    pos, sds_ref, sds_test = [], [], []
    for point, data in groups:
        pos.append(data['Position'].mean())
        sds_ref.append(data['ADC'].std())
        sds_test.append(data['Reference'].std())
    if plot_ref:
        sns.regplot(x=pos, y=sds_ref, scatter_kws=scatter_kws)
    if plot_test:
        sns.regplot(x=pos, y=sds_test, scatter_kws=scatter_kws)
    # sns.regplot(x=sds_ref, y=sds_test, scatter_kws=scatter_kws)
    return fig, ax


if __name__ == '__main__':
    directory = 'data'
    filename = 'adc_test_samples_0.csv'
    path = os.path.join(directory, filename)
    df = load_data(path)

    fig, ax = plotADCvsPosition(df)
    fig, ax = plotSampleRingdown(df)
    fig, ax = plotADCNoise(df)

    plt.show()
