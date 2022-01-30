import ambiance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quaternion
import seaborn as sns

import rotation_model


# TODO: ADD THE ABILITY TO READ VALUES FROM THIS BY RESAMPLING (AT ARBITRARY FREQUENCY) TO A BASE TYPE

class OpenRocketReader:
    def __init__(self, filename):
        self.__filename = filename
        self.__columns = [
                't', 'h', 'v_v', 'a_v', 'v', 'a', 'l_east', 'l_north', 'l_lat', 'gamma_lat',
                'v_lat', 'a_lat', 'latitude', 'longitude', 'g', 'aoa', 'omega_r',
                'omega_p', 'omega_y', 'mass', 'mass_prop', 'mmoi_long', 'mmoi_rot', 'x_cp',
                'x_cg', 'stability_margin', 'M', 'RN', 'Thrust', 'Drag', 'Cd', 'Cd_a',
                'Cd_f', 'Cd_p', 'Cd_b', 'Cn', 'Cm_p', 'Cm_y', 'Cf_side', 'Cm_r', 'Cf_r', 'Cdamp_r',
                'Cdamp_p', 'a_c', 'l_ref', 'S_ref', 'theta_vert', 'theta_lat', 'v_wind', 'T_air',
                'P_air', 'v_sound', 'dt', 't_tot'
            ]

        self.__data = pd.read_csv(filename, comment='#', names=self.__columns)

        self.__ts = np.array(self.__data['t'])
        self.__omega_roll = np.radians(np.array(self.__data['omega_r']))
        self.__omega_pitch = np.radians(np.array(self.__data['omega_p']))
        self.__omega_yaw = np.radians(np.array(self.__data['omega_y']))
        self.__omegas = np.array([self.__omega_roll, self.__omega_pitch, self.__omega_yaw])

        self.__thetas = np.array([[], []])
        # dt = 0.1
        # TODO: INITIALIZE THE ORIENTATION, PROBABLY USING THETA_VERT AND THETA_LAT
        q_integrator = rotation_model.RotationOrientationIntegrator(oversampling=None, dt_max=1e-1)
        qs = q_integrator(self.__omegas, ts=self.__ts)
        self.__qs = np.array(qs)

        # Calculate rocket accelerations using a_v(t), a_lat(t), and qw(t)

        as_v = self.__data['a_v']
        as_lat = self.__data['a_lat']
        accels = [self.__extract_accels(a_v, a_lat, qs) for a_v, a_lat, qs in zip(as_v, as_lat, qs)]
        self.__accels = np.array(accels)

        self.__hs = np.array(self.__data['h'])
        self.__Ps = ambiance.Atmosphere(self.__hs).pressure

    @property
    def data(self):
        return self.__data

    @property
    def ts(self):
        return self.__ts

    @property
    def qs(self):
        return self.__qs

    @property
    def thetas(self):
        return self.__thetas

    @property
    def omegas(self):
        return self.__omegas

    @property
    def accels(self):
        return self.__accels

    @property
    def hs(self):
        # h_asl = ambiance.Atmosphere.from_pressure(133.322*self.__data['P_air'][0]).h
        # print('h_asl = {} m'.format(h_asl))
        return self.__hs  # + h_asl  # TODO: MAKE THIS ABOVE SEA LEVEL....?

    @property
    def Ps(self):
        return 133.322 * np.array(self.__data['P_air'])  # return as Pascals

    @property
    def Ts(self):  # TODO: FIGURE OUT WHAT THIS IS ACTUALLY DOING?
        return np.array(self.__data['T_air'])

    def __extract_accels(self, a_v, a_lat, q):
        # print('Input: a_v = {}, a_lat = {}, q = {}'.format(a_v, a_lat, q))
        a_v = q * np.quaternion(0, 0, 0, a_v) * q.conjugate()
        a_lat = q * np.quaternion(0, a_lat, 0, 0) * q.conjugate()
        a_tot = (a_v + a_lat)
        # print('Output: a_v = {}, a_lat = {}, a_tot = {}'.format(a_v, a_lat, a_tot))
        return a_tot.x, a_tot.y, a_tot.z


if __name__ == '__main__':
    filename = '18mm_Dual_Deploy_Sustainer.csv'
    filename = '../LPR/Nartrek/Black_Brant_VB_Mule_D9.csv'

    parser = OpenRocketReader(filename)
    #
    # fig, axs = plt.subplots(3, figsize=(16, 9), sharex=True)
    # sns.lineplot(x='t', y='omega_r', data=parser.data, ax=axs[0])
    # sns.lineplot(x='t', y='omega_p', data=parser.data, ax=axs[1])
    # sns.lineplot(x='t', y='omega_y', data=parser.data, ax=axs[2])
    # # axs[0].set_xlim(0, 8)
    # fig.tight_layout()
    # fig.canvas.manager.window.showMaximized()
    #
    # fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True)
    # sns.lineplot(x='t', y='theta_vert', data=parser.data, ax=axs[0])
    # sns.lineplot(x='t', y='theta_lat', data=parser.data, ax=axs[1])
    # # axs[0].set_xlim(0, 8)
    # fig.tight_layout()
    # fig.canvas.manager.window.showMaximized()
    #
    # fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True)
    # sns.lineplot(x=parser.ts, y=parser.hs, ax=axs[0])
    # sns.lineplot(x=parser.ts, y=parser.Ps, ax=axs[1])
    # fig.tight_layout()
    # fig.canvas.manager.window.showMaximized()
    #
    fig, axs = plt.subplots(3, figsize=(15, 10), sharex=True)
    a = parser.data['a']
    a_v = parser.data['a_v']
    a_lat = parser.data['a_lat']
    a_est = np.sqrt(np.square(a_v) + np.square(a_lat))
    # sns.lineplot('t', 'a', data=data, ax=axs[0])
    sns.lineplot('t', 'a_v', data=parser.data, ax=axs[0])
    sns.lineplot('t', 'a_lat', data=parser.data, ax=axs[0])
    # sns.lineplot('t', 'a_c', data=data, ax=axs[0])
    sns.lineplot(x=parser.data['t'], y=a-a_est, ax=axs[1])  # NOTE: a = magnitude of the vector created by a_v and a_lat
    # axs[0].set_xlim(0, 8)
    # axs[0].set_ylim(-25, 50)
    as_x = [x for x, _, _ in parser.accels]
    as_y = [y for _, y, _ in parser.accels]
    as_z = [z for _, _, z in parser.accels]
    sns.lineplot(x=parser.data['t'], y=as_x, ax=axs[2], label='ax')
    sns.lineplot(x=parser.data['t'], y=as_y, ax=axs[2], label='ay')
    sns.lineplot(x=parser.data['t'], y=as_z, ax=axs[2], label='az')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.tight_layout()
    fig.canvas.manager.window.showMaximized()

    plt.show()
    # omegas = parser.omegas
    # print(omegas)
    # print(omegas.shape)
