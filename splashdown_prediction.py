import itertools
import math

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

import coordinates
import geometry


def getIntersectionPlane(llh, distance=100):
    # NOTE: AS IMPLEMENTED, THIS WILL FAIL AT THE POLES, AND INACCURACY INCREASES AS THE POLES ARE REACHED
    lat, lon, h = llh
    m_to_deg = 360 / 40.075e6
    dlon = distance * m_to_deg / math.cos(math.radians(lat))  # Toward East
    dlat = (distance * m_to_deg) * (1 if lat < 0 else -1)  # Toward Equator
    p0 = coordinates.LLHToECEF(llh)
    p1 = coordinates.LLHToECEF((lat, lon + dlon, h))
    p2 = coordinates.LLHToECEF((lat + dlat, lon, h))
    return geometry.Plane3D.FromPoints(np.array(p0), np.array(p1), np.array(p2))


class SplashdownBuffer:
    def __init__(self, N=10):
        self.__N = N
        self.__buffer = []

    @property
    def beacons(self):
        return self.__buffer

    @property
    def full(self):
        return len(self.__buffer) == self.__N

    def __len__(self):
        return len(self.__buffer)

    def add(self, beacon):
        self.__buffer.append(beacon)
        if len(self.__buffer) > self.__N:
            self.__buffer = self.__buffer[-self.__N:]


class SplashdownStatistics:
    def __init__(self, mean, covariance):
        self.__mean = mean
        self.__covariance = covariance

    @property
    def mean(self):
        return self.__mean

    @property
    def covariance(self):
        return self.__covariance


def localizationCombinations(n, k):
    N = n-1
    return math.factorial(N) / (math.factorial(k) * (math.factorial(N-k)))


class SplashdownStatisticsProcessor:
    def __init__(self, buffer=None, N=6, M=3, history_size=None, sim_kws={}, f_ref=1, track_reference_plane=True):
        self.__buffer = SplashdownBuffer(N=N-1) if buffer is None else buffer
        self.__M = M
        self.__reference_plane = None
        self.__reference_height = 0
        self.__track_reference_plane = track_reference_plane
        self.__history = []
        self.__history_size = int(5 * localizationCombinations(N, M) if history_size is None else history_size)
        t_ref = 1 / f_ref
        self.__tau_localization = -math.log(0.5) / (N * t_ref)
        combs_ref = localizationCombinations(N, M)
        self.__tau_history = -math.log(0.15) / ((self.__history_size / combs_ref) * t_ref)
        print('History Size = {}'.format(self.__history_size))
        print('Tau Localization = {}'.format(self.__tau_localization))
        print('Tau History = {}'.format(self.__tau_history))
        self.__simulate = False
        if ('mean' in sim_kws) and ('cov' in sim_kws):
            self.__sim_mean = sim_kws['mean']
            self.__sim_cov = sim_kws['cov']
            self.__simulate = True
        self.ts = []
        self.lat_means = []
        self.lat_covs = []
        self.lon_means = []
        self.lon_covs = []
        self.history_buffer = []

    @property
    def splashdown_plane(self):
        return self.__reference_plane

    @property
    def history(self):
        return self.__history[:]

    def set_splashdown_plane(self, llh):
        self.__reference_plane = getIntersectionPlane(llh)
        self.__reference_height = llh[2]

    def update(self, new_beacon):
        if self.__simulate:
            rng = np.random.default_rng()
            for llh in rng.multivariate_normal(self.__sim_mean, self.__sim_cov, size=self.__history_size):
                self.__history.append(llh)
        else:
            if len(self.__buffer) > 0:
                self.__update_reference_plane(beacon)
                t_now = new_beacon.t
                weighted_beacons = []
                for b in self.__buffer.beacons:
                    age = t_now - b.t
                    w = math.exp(-self.__tau_localization * age)
                    weighted_beacons.append((b, age, w))
                size = min(self.__M, len(self.__buffer))
                for past_beacons in itertools.combinations(weighted_beacons, size):
                    # pos, vel = beacon.pos, beacon.vel
                    pos, vel = self.__process_beacons_pos_vel(t_now, past_beacons, new_beacon)
                    line = geometry.Line3D.FromPointVector(pos, vel)
                    ecef = geometry.LinePlaneIntersect3D(line, self.__reference_plane)
                    llh = coordinates.ECEFToLLH(ecef)
                    self.__history.append((llh[0], llh[1], llh[2], t_now))
            else:
                line = geometry.Line3D.FromPointVector(new_beacon.pos, new_beacon.vel)
                ecef = geometry.LinePlaneIntersect3D(line, self.__reference_plane)
                llh = coordinates.ECEFToLLH(ecef)
                self.__history.append((llh[0], llh[1], llh[2], new_beacon.t))
            self.__buffer.add(new_beacon)

        if len(self.__history) > self.__history_size:
            self.__history = self.__history[-self.__history_size:]

        if len(self.__history) > 0:
            lats = []
            lons = []
            ws = []
            for lat, lon, _, t in self.__history:
                lats.append(lat)
                lons.append(lon)
                ws.append(math.exp(-self.__tau_history * t))
            lats = np.array(lats)
            lons = np.array(lons)
            self.ts.append(beacon.t)
            lat_mean = np.average(lats, weights=ws)
            self.lat_means.append(lat_mean)
            self.lat_covs.append(np.cov(lats, aweights=ws))
            lon_mean = np.average(lons, weights=ws)
            self.lon_means.append(lon_mean)
            self.lon_covs.append(np.cov(lons, aweights=ws))
            self.history_buffer.append((beacon.t, self.__history[:]))

            if False:
                g = sns.JointGrid(x=lats, y=lons)
                g.plot_joint(sns.kdeplot, fill=True, alpha=0.5, legend=False)
                g.plot_marginals(sns.rugplot, height=0.1, alpha=0.5)
                g.ax_joint.set_aspect('equal')
                # TODO: CALCULATE SPLASHDOWN STATISTICS
        return None

    def __update_reference_plane(self, beacon):
        if self.__reference_plane is None:
            llh = coordinates.ECEFToLLH(beacon.pos)
            self.set_splashdown_plane((llh[0], llh[1], 0)) # THIS NEEDS TO BE BASED ON THE GROUND POSITION AND, AS SUCH, PROCESSING EITHER NEEDS TO START PREFLIGHT OR USE THE GROUNDSTATION AS REFERENCE INNITIALLY
        elif self.__track_reference_plane and (len(self.__history) > 0):
            lats = []
            lons = []
            ws = []
            for lat, lon, _, t in self.__history:
                lats.append(lat)
                lons.append(lon)
                ws.append(math.exp(-self.__tau_history * t))
            lats = np.array(lats)
            lons = np.array(lons)
            lat_mean = np.average(lats, weights=ws)
            lon_mean = np.average(lons, weights=ws)
            self.set_splashdown_plane((lat_mean, lon_mean, self.__reference_height))

    def __process_beacons_pos_vel(self, t_now, past_beacons, beacon):
        def f_vec(x):
            return '[{:0.3G}, {:0.3G}, {:0.3G}]'.format(x[0], x[1], x[2])
        p = np.array(beacon.pos)  # Note: need to copy these values
        v = np.array(beacon.vel)
        w_sum = 1
        # print('\tInitial p = {}, v = {}'.format(f_vec(p), f_vec(v)))
        for b, age, w in past_beacons:
            w_sum += w
            v_b = b.vel
            p_b = b.pos + age*v_b
            p += w * p_b
            v += w * v_b
            # print('\t\tage = {:02f}, p_est = {}, v_est = {}, w = {:0.3G}'.format(age, f_vec(p_b), f_vec(v_b), w))
        p = p / w_sum
        v = v / w_sum
        # print('\t\tEstimated p = {}, v = {}'.format(f_vec(p), f_vec(v)))
        return p, v


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
    return fig, ax


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
        def __init__(self, filename, beacon_rate_kwargs={}, gps_err_kwargs={}, noError=False):
            self.__filename = filename
            self.__parser = openrocket_api.OpenRocketReader(filename)
            self.__beacon_rate_config = TrackerBeaconRateConfig(**beacon_rate_kwargs)
            self.__error_model = None if noError else gps_error_model.GPSErrorModel(**gps_err_kwargs)
            self.__localization_beacons = self.__extract_tracker_beacons()

        @property
        def localization_beacons(self):
            return self.__localization_beacons

        def __extract_tracker_beacons(self):
            localization_beacons = []
            ts = self.__parser.ts
            llhs = self.__parser.pos_llh
            t_last = ts[0]
            xyz_last = None
            if self.__error_model is None:
                xyz_last = np.array(coordinates.LLHToECEF(llhs[0]))
            else:
                xyz_last = np.array(coordinates.LLHToECEF(self.__error_model(*llhs[0])))
            localization_beacons.append(LocalizationBeacon(t_last, xyz_last, np.array([0, 0, 0])))
            for t, llh in zip(ts, llhs):
                xyz = None
                if self.__error_model is None:
                    xyz = np.array(coordinates.LLHToECEF(llh))
                else:
                    xyz = np.array(coordinates.LLHToECEF(self.__error_model.offset(*llh)))
                dt = t - t_last
                d = self.__distance(xyz_last, xyz)
                if self.__beacon_rate_config.isValidBeacon(dt, d):
                    vel = self.__velocity(dt, xyz_last, xyz)
                    localization_beacons.append(LocalizationBeacon(t, xyz, vel))
                    t_last = t
                    xyz_last = xyz
                    if self.__error_model is not None:
                        self.__error_model.update()
            return localization_beacons

        def __distance(self, xyz_a, xyz_b):
            return math.sqrt(np.sum(np.square(xyz_a - xyz_b)))

        def __velocity(self, dt, xyz_a, xyz_b):
            return (xyz_b - xyz_a) / dt

    deg_to_m = 40.075e6 / 360
    # filename = '../HPR/66mm_L2_1.csv'
    #filename = '../LPR/Black_Brant_VB_Mule_Wind_80.csv'
    filename = '../LPR/Black_Brant_VB_Mule_High_Altitude.csv'

    beacon_rate_kwargs = {'t_max': 5, 'l_max': 7.5, 'f_max': 4}
    tracker_source = OpenrocketGPSTrackerSource(filename, beacon_rate_kwargs=beacon_rate_kwargs)
    beacons = tracker_source.localization_beacons
    t_apogee = estimatedApogeeTime(beacons)
    print("Apogee Time = {}".format(t_apogee))

    mean = np.array([38, -104, 0])
    cov = np.array([[1e-3, 1e-3, 0], [1e-3, 3.5e-3, 0], [0, 0, 1e-3]])

    # splashdown_processor = SplashdownStatisticsProcessor(sim_kws={'mean': mean, 'cov': cov})
    splashdown_processor = SplashdownStatisticsProcessor()
    splashdown_processor.set_splashdown_plane(coordinates.ECEFToLLH(beacons[-1].pos))
    for beacon in beacons:
        if beacon.t > t_apogee:  # TODO: THIS IS NOT THE RIGHT WAY TO DO THIS... BUT THE SPLASHDOWN SHOULD ONLY BE ESTIMATED DURING DESCENT
            # print('\n*** Run Update ***')
            splashdown_processor.update(beacon)

    # Plot Processing Effectiveness
    lat_ref, lon_ref, _ = coordinates.ECEFToLLH(beacons[-1].pos)
    t_end = beacons[-1].t
    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    fig.suptitle('Splashdown Testing')
    axs[0].set_title('Latitude')
    lat_errs = deg_to_m * np.abs(np.array(splashdown_processor.lat_means) - lat_ref)
    axs[0].plot(splashdown_processor.ts, lat_errs, marker='o', c='r', alpha=0.5, label='$abs(Offset)$')
    axs[0].set_ylabel('Latitude Estimation Error (m)')
    # ax0 = axs[0].twinx()
    axs[0].plot(splashdown_processor.ts, 3 * deg_to_m * np.sqrt(splashdown_processor.lat_covs), c='c', alpha=0.5, label='$3*sd$')
    # ax0.set_ylabel('Latitude S.D. (m)')
    axs[1].set_title('Longitude')
    lon_errs = deg_to_m * np.abs(np.array(splashdown_processor.lon_means) - lon_ref)
    axs[1].plot(splashdown_processor.ts, lon_errs, marker='o', c='r', alpha=0.5, label='$abs(Offset)$')
    axs[1].set_ylabel('Longitude Estimation Error (m)')
    # ax1 = axs[1].twinx()
    axs[1].plot(splashdown_processor.ts, 3 * deg_to_m * np.sqrt(splashdown_processor.lon_covs), c='c', alpha=0.5, label='$3*sd$')
    # ax1.set_ylabel('Longitude S.D. (m)')
    for ax in axs:
        ax.axvline(t_end, c='y')
        ax.axvline(t_apogee, c='y')
        ax.legend()

    # Plot History Points
    fig, ax = plt.subplots(1, constrained_layout=True)
    ax.set_aspect('equal')

    ts = []
    lats = []
    lons = []
    for t, history in splashdown_processor.history_buffer:
        for lat, lon, _, _ in history:
            ts.append(t)
            lats.append(deg_to_m * (lat-lat_ref))
            lons.append(deg_to_m * (lon-lon_ref))
    sns.scatterplot(x=lats, y=lons, c=ts, alpha=0.5, s=5, cmap='viridis', ax=ax, ec=None)

    # Plot Beacons and Estimates
    # print('Number of Beacons = {}'.format(len(beacons)))
    # fig, ax = plotBeacons(beacons)
    # base_size = 150
    # ref_plane = splashdown_processor.splashdown_plane
    # xlim = (ref_plane.origin[0] - base_size, ref_plane.origin[0] + base_size)
    # ylim = (ref_plane.origin[1] - base_size, ref_plane.origin[1] + base_size)
    # geometry.plotPlane3D(ref_plane, xlim, ylim, ax, lw=0.5, color='k')
    # print('number of estimate points = {}'.format(len(splashdown_processor.history)))
    # for ecef in [coordinates.LLHToECEF((lat, lon, h)) for lat, lon, h, _ in splashdown_processor.history]:
    #     geometry.plotPoint3D(ecef, ax, alpha=0.15, color='m')
    # geometry.plotPoint3D(beacons[-1].pos, ax, alpha=0.15, color='k')

    plt.show()

    # n = 9
    # k = 6
    # print('Localization Combinations (n = {}, k = {}) = {}'.format(n, k, localizationCombinations(n, k)))
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
