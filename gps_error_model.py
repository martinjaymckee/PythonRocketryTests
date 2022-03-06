import random


class GPSErrorModel:
    def __init__(self, lat_params=None, lon_params=None, alt_params=None, force_noisy=False, enable_white_noise=True):
        m_to_deg = 360 / 40.075e6
        self.__lat_d_step = 1.85e-6 if lat_params is None else lat_params[0]
        self.__lat_sigma_step = 1.5e-7 if lat_params is None else lat_params[1]
        self.__lat_sigma_abs = 2.5 * m_to_deg if lat_params is None else lat_params[2]
        self.__lat_p = (self.__lat_sigma_step**2) / (self.__lat_d_step**2)
        self.__lat_offset = random.gauss(0, self.__lat_sigma_abs)

        self.__lon_d_step = 1.85e-6 if lon_params is None else lon_params[0]
        self.__lon_sigma_step = 1.5e-7 if lon_params is None else lon_params[1]
        self.__lon_sigma_abs = 2.5 * m_to_deg if lon_params is None else lon_params[2]
        self.__lon_p = (self.__lon_sigma_step**2) / (self.__lon_d_step**2)
        self.__lon_offset = random.gauss(0, self.__lon_sigma_abs)

        self.__alt_d_step = 0.1 if alt_params is None else alt_params[0]
        self.__alt_sigma_step = 0.0175 if alt_params is None else alt_params[1]
        self.__alt_sigma_abs = 6 if alt_params is None else alt_params[2]
        self.__alt_p = (self.__alt_sigma_step**2) / (self.__alt_d_step**2)
        self.__alt_offset = random.gauss(0, self.__alt_sigma_abs)

        self.__force_noisy = force_noisy
        self.__enable_white_noise = enable_white_noise

    def __call__(self, lat, lon, alt, doupdate=True):
        def doStep(offset, d, sigma_abs, p):
            if self.__force_noisy:
                p = min(1, 3 * p)
                d = 10 * d
            p_step = random.uniform(0, 1)
            if p_step <= p:
                p_dir = random.uniform(0, 1)
                p_thresh = (offset / (10 * sigma_abs)) + 0.5
                if p_dir > p_thresh:
                    offset += d
                else:
                    offset -= d
            return offset
        if doupdate:
            self.__lat_offset = doStep(self.__lat_offset, self.__lat_d_step, self.__lat_sigma_abs, self.__lat_p)
            self.__lon_offset = doStep(self.__lon_offset, self.__lon_d_step, self.__lon_sigma_abs, self.__lon_p)
            self.__alt_offset = doStep(self.__alt_offset, self.__alt_d_step, self.__alt_sigma_abs, self.__alt_p)
        lat_noise = random.gauss(0, self.__lat_sigma_step) if self.__enable_white_noise else 0
        lon_noise = random.gauss(0, self.__lon_sigma_step) if self.__enable_white_noise else 0
        alt_noise = random.gauss(0, self.__alt_sigma_step) if self.__enable_white_noise else 0
        lat += (self.__lat_offset + lat_noise)
        lon += (self.__lon_offset + lon_noise)
        alt += (self.__alt_offset + alt_noise)
        return lat, lon, alt

    def offset(self, lat, lon, alt):
        return self(lat, lon, alt, doupdate=False)

    def update(self):
        self(0, 0, 0, doupdate=True)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    N = 5000
    M = 5

    fig, axs = plt.subplots(2, constrained_layout=True)

    for _ in range(M):
        error_model = GPSErrorModel()
        lats = []
        lons = []
        alts = []
        for _ in range(N):
            lat, lon, alt = error_model(0, 0, 0)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)
        lats = np.array(lats)
        lons = np.array(lons)
        alts = np.array(alts)
        axs[0].scatter(lats, lons)
        axs[1].plot(alts)
    axs[0].set_aspect('equal')
    plt.show()
