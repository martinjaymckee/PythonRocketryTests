import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def series_stats(data, W=6):
    mean = data.mean()
    sd = data.std()
    return mean, sd, (mean-W*sd, mean+W*sd)


def graph_series(title, data, W=6, clip=True):
    mean, sd, limits = series_stats(data, W=W)
    fig, ax = plt.subplots(1, figsize=(15,12))
    ax.plot(data)
    if clip:
        ax.set_ylim(*limits)
    ax.set_title("{} (mean = {:0.5G}, sd = {:0.5G})".format(title, mean, sd))
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
    W = 2
    data = pd.read_csv("../../../../Arduino/GPS_Error_Test/gps_log_2_23_2021_1")
    # data = pd.read_csv("../../../../Arduino/GPS_Error_Test/gps_log_2_14_2021_1")
    # data = pd.read_csv("../../../../Arduino/GPS_Error_Test/gps_log_2_13_2021_1")
    lats = data.iloc[:, 0]
    lons = data.iloc[:, 1]
    # alts = data.iloc[:, 2]
    sats = data.iloc[:, 4]
    # hdops = data.iloc[:, 5]
    lats_plot = graph_series("Latitudes", lats)
    lons_plot = graph_series("Longitudes", lons)
#    alts_plot = graph_series("Altitudes", alts)
    sats_plot = graph_series("Satellites", sats)
#    hdops_plot = graph_series("HDOPs", hdops, clip=False)

    lat_mean, lat_sd, lat_limits = series_stats(lats, W=W)
    lon_mean, lon_sd, lon_limits = series_stats(lons, W=W)
    # alt_mean, alt_sd, alt_limits = series_stats(alts, W=W)

    validity = validity_map(lats, lat_limits)
    validity = validity_map(lons, lon_limits, validity)
    # validity = validity_map(alts, alt_limits, validity)

    lats = lats.loc[validity]
    lons = lons.loc[validity]

    m_per_deg = 40.075e6 / 360.0
    print('lat sd = {} m, lon sd = {} m'.format(lat_sd*m_per_deg, lon_sd*m_per_deg))

    # sns.jointplot(x=lats, y=lons, kind='hex')
    sns.jointplot(x=lats, y=lons, kind='kde', joint_kws={"fill": True})
    plt.show()
