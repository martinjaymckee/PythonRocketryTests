import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

import coordinates


def getIntersectionPlane(llh, distance=1000):
    # NOTE: AS IMPLEMENTED, THIS WILL FAIL AT THE POLES, AND INACCURACY INCREASES AS THE POLES ARE REACHED
    lat, lon, h = llh
    m_to_deg = 360 / 40.075e6
    dlon = distance * m_to_deg / math.cos(math.radians(lat))  # Toward East
    dlat = (distance * m_to_deg) * (1 if lat < 0 else -1)  # Toward Equator
    p0 = coordinates.LLHToECEF(llh)
    p1 = coordinates.LLHToECEF((lat, lon + dlon, h))
    p2 = coordinates.LLHToECEF((lat + dlat, lon, h))

    print('p0 = {}, p1 = {}, p2 = {}'.format(p0, p1, p2))

    return None


class TrackerBeaconRateConfig:
    def __init__(self, t_max=5, f_max=4, l_max=25):
        self.__t_max = t_max
        self.__t_min = 1 / f_max
        self.__f_max = f_max
        self.__l_max = l_max

    @property
    def t_max(self):
        return self.__t_max

    @property
    def f_max(self):
        return self.__f_max

    @property
    def l_max(self):
        return self.__l_max

    def isValidBeacon(self, dt, d):
        if dt >= self.__t_max: return True  # Send at minimum rate
        if d >= self.__l_max:
            if dt >= self.__t_min:
                return True  # Send on motion... limited by maximum rate
        return False


class LocalizationBeacon:
    def __init__(self, t=None, pos=None, vel=None):
        self.__t = t
        self.__pos = pos
        self.__vel = vel

    @property
    def valid(self):
        return (self.__t is not None) and (self.__pos is not None) and (self.__vel is not None)

    @property
    def t(self):
        return self.__t

    @property
    def pos(self):
        return self.__pos

    @property
    def vel(self):
        return self.__vel


def plotBeacons(beacons):
    fig = plt.figure(constrained_layout=True)
    ax = plt.axes(projection='3d')

    ts = []
    xs = []
    ys = []
    zs = []
    us = []
    vs = []
    ws = []
    for beacon in beacons:
        ts.append(beacon.t)
        xs.append(beacon.pos[0])
        ys.append(beacon.pos[1])
        zs.append(beacon.pos[2])
        us.append(beacon.vel[0])
        vs.append(beacon.vel[1])
        ws.append(beacon.vel[2])
    ax.quiver(xs, ys, zs, us, vs, ws, alpha=0.5, length=0.5)
    ax.scatter3D(xs, ys, zs, c=ts, cmap='viridis', lw=0)


def estimatedApogeeTime(beacons):
    def radius(xyz):
        return math.sqrt(np.sum(np.square(xyz)))
    if len(beacons) == 0:
        return 0
    if len(beacons) == 1:
        return beacons[0].t
    t = beacons[0].t
    r_last = radius(beacons[0].pos)
    for beacon in beacons[1:]:
        t = beacon.t
        r = radius(beacon.pos)
        if r < r_last:
            break
        r_last = r
    return t


if __name__ == '__main__':
    import gps_error_model
    import openrocket_api

    class OpenrocketGPSTrackerSource:
        def __init__(self, filename, beacon_rate_kwargs={}, gps_err_kwargs={}):
            self.__filename = filename
            self.__parser = openrocket_api.OpenRocketReader(filename)
            self.__beacon_rate_config = TrackerBeaconRateConfig(**beacon_rate_kwargs)
            self.__error_model = gps_error_model.GPSErrorModel(**gps_err_kwargs)
            self.__localization_beacons = self.__extract_tracker_beacons()

        @property
        def localization_beacons(self):
            return self.__localization_beacons

        def __extract_tracker_beacons(self):
            localization_beacons = []
            ts = self.__parser.ts
            llhs = self.__parser.pos_llh
            t_last = ts[0]
            xyz_last = np.array(coordinates.LLHToECEF(self.__error_model(*llhs[0])))
            localization_beacons.append(LocalizationBeacon(t_last, xyz_last, np.array([0, 0, 0])))
            for t, llh in zip(ts, llhs):
                xyz = np.array(coordinates.LLHToECEF(self.__error_model.offset(*llh)))
                dt = t - t_last
                d = self.__distance(xyz_last, xyz)
                if self.__beacon_rate_config.isValidBeacon(dt, d):
                    vel = self.__velocity(dt, xyz_last, xyz)
                    localization_beacons.append(LocalizationBeacon(t, xyz, vel))
                    t_last = t
                    xyz_last = xyz
                    self.__error_model.update()
            return localization_beacons

        def __distance(self, xyz_a, xyz_b):
            return math.sqrt(np.sum(np.square(xyz_a - xyz_b)))

        def __velocity(self, dt, xyz_a, xyz_b):
            return (xyz_b - xyz_a) / dt

    # filename = '../HPR/66mm_L2_1.csv'
    filename = '../LPR/Black_Brant_VB_Mule_Wind_80.csv'

    beacon_rate_kwargs = {'t_max': 5, 'l_max': 5, 'f_max': 4}
    tracker_source = OpenrocketGPSTrackerSource(filename, beacon_rate_kwargs=beacon_rate_kwargs)
    beacons = tracker_source.localization_beacons
    print('Number of Beacons = {}'.format(len(beacons)))
    plotBeacons(beacons)

    print("Apogee Time = {}".format(estimatedApogeeTime(beacons)))
    llh = coordinates.ECEFToLLH(beacons[0].pos)
    print('LLH = {}'.format(llh))
    getIntersectionPlane(llh)
    plt.show()
    # fig, axs = plt.subplots(3, figsize=(16, 9), sharex=True)
    # fig.suptitle('Position - LLH')
    # sns.lineplot(x='t', y='latitude', data=parser.data, ax=axs[0], label='Latitude')
    # sns.lineplot(x='t', y='longitude', data=parser.data, ax=axs[1], label='Longitude')
    # sns.lineplot(x='t', y='h', data=parser.data, ax=axs[2], label='Height')
    # for ax in axs:
    #     ax.legend()
    # fig.tight_layout()
    # fig.canvas.manager.window.showMaximized()
    #
    # fig, axs = plt.subplots(4, figsize=(16, 9), sharex=True)
    # fig.suptitle('Position - ECEF')
    # ecefs = parser.pos_ecef
    # xs = np.array([ecef[0] for ecef in ecefs])
    # ys = np.array([ecef[1] for ecef in ecefs])
    # zs = np.array([ecef[2] for ecef in ecefs])
    # rs = np.sqrt(xs**2 + ys**2 + zs**2)
    # sns.lineplot(x=parser.data['t'], y=xs, ax=axs[0], label='X')
    # sns.lineplot(x=parser.data['t'], y=ys, ax=axs[1], label='Y')
    # sns.lineplot(x=parser.data['t'], y=zs, ax=axs[2], label='Z')
    # sns.lineplot(x=parser.data['t'], y=rs, ax=axs[3], label='R')
    # for ax in axs:
    #     ax.legend()
    # fig.tight_layout()
    # fig.canvas.manager.window.showMaximized()
    #
    # plt.show()
