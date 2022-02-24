import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def series_stats(data, W=8):
    mean = data.mean()
    sd = data.std()
    return mean, sd, (mean-W*sd, mean+W*sd)


def graph_series(title, data, W=6, clip=True):
    mean, sd, limits = series_stats(data, W=W)
    min = np.min(data)
    max = np.max(data)
    fig, ax = plt.subplots(1, figsize=(15,12))
    ax.plot(data)
    if clip:
        ax.set_ylim(*limits)
    ax.set_title("{} (mean = {:0.5G}, sd = {:0.5G}, min = {:0.5G}, max = {:0.5G})".format(title, mean, sd, min, max))
    return ax


def validity_map(data, limits, validity=None):
    valid = []
    for idx, value in enumerate(data):
        check = value > limits[0] and value < limits[1]
        if validity is not None:
            check = check and validity[idx]
        valid.append(check)
    return np.array(valid)


if __name__ == '__main__':
    import random

    W = 1.5
    data = pd.read_csv("data/gps_stationary_2_16_2022_3.csv")
    # data = pd.read_csv("data/gps_stationary_2_20_2022.csv")

    idx_start = 500
    idxs = data.iloc[idx_start:, 0]
    dts = data.iloc[idx_start:, 1]
    sats = data.iloc[idx_start:, 2]
    lats = data.iloc[idx_start:, 3]
    lons = data.iloc[idx_start:, 4]
    alts = data.iloc[idx_start:, 5]

    # dts_plot = graph_series("Time Deltas", dts)
    lats_plot = graph_series("Latitudes", lats)
    lons_plot = graph_series("Longitudes", lons)
    alts_plot = graph_series("Altitudes", alts)
    # sats_plot = graph_series("Satellites", sats)

    lat_mean, lat_sd, lat_limits = series_stats(lats, W=W)
    lon_mean, lon_sd, lon_limits = series_stats(lons, W=W)
    alt_mean, alt_sd, alt_limits = series_stats(alts, W=W)

    validity = validity_map(lats, lat_limits)
    validity = validity_map(lons, lon_limits, validity)
    validity = validity_map(alts, alt_limits, validity)

    lats = lats.loc[validity]
    lons = lons.loc[validity]

    m_per_deg = 40.075e6 / 360.0
    deg_per_m = 360.0 / 40.075e6
    print('lat sd = {} m, lon sd = {} m, alt sd = {}'.format(lat_sd*m_per_deg, lon_sd*m_per_deg, alt_sd))

    # sns.jointplot(x=lats, y=lons, kind='kde', joint_kws={"fill": True})

    # lats_err = m_per_deg * (lats - lat_mean)
    # lons_err = m_per_deg * (lons - lon_mean)
    # lat_lon_errs = np.sqrt(lats_err**2 + lons_err**2)
    # lat_lon_err_plot = graph_series("Lat/Lon Errors", lat_lon_errs)

    # sns.jointplot(x=sats, y=lat_lon_errs, kind='kde', joint_kws={"fill": True})
    # alts_err = alts - alt_mean
    # alts_err_plot = graph_series("Altitude Errors", alts_err)
    # sns.jointplot(x=sats, y=alts_err, kind='kde', joint_kws={"fill": True})

    # sns.jointplot(x=lat_lon_errs, y=alts_err, kind='kde', joint_kws={"fill": True})

    lats_diffs = np.diff(lats)
    lons_diffs = np.diff(lons)
    lats_diffs_plot = graph_series("Latitude/Longitude Sequential Differences", lats_diffs, W=20)
    lats_diffs_plot.plot(lons_diffs)

    steps = []
    d = 1.75e-6
    sigma = lats_diffs.std()
    p = (sigma**2) / (d**2)
    offset = 0
    offsets = []
    for _ in range(len(lats_diffs)):
        p_step = random.uniform(0, 1)
        # print('p_step = {}'.format(p_step))
        if p_step <= p:
            # print('\tDo step')
            p_dir = random.uniform(0, 1)
            if p_dir < 0.5:
                steps.append(d)
                offset += d
            else:
                steps.append(-d)
                offset -= d
        else:
            steps.append(0)
        offsets.append(offset)
    steps = np.array(steps)
    offsets = np.array(offsets)
    print('sigma = {}, p = {}, sigma_est = {}'.format(sigma, p, steps.std()))

    lats_diffs_plot.plot(steps, alpha=0.25)
    random_walk_plot = graph_series("Random Walk Error", offsets)

    alts_diffs = np.diff(alts)
    alts_diffs_plot = graph_series("Altitude Sequential Differences", alts_diffs, W=20)

    plt.show()
