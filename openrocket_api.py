import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    filename = '18mm_Dual_Deploy_Sustainer.csv'
    filename = '../LPR/Nartrek/Black_Brant_VB_Mule_D9.csv'

    columns = [
                't', 'h', 'v_v', 'a_v', 'v', 'a', 'l_east', 'l_north', 'l_lat', 'gamma_lat',
                'v_lat', 'a_lat', 'latitude', 'longitude', 'g', 'aoa', 'omega_r',
                'omega_p', 'omega_y', 'mass', 'mass_prop', 'mmoi_long', 'mmoi_rot', 'x_cp',
                'x_cg', 'stability_margin', 'M', 'RN', 'Thrust', 'Drag', 'Cd', 'Cd_a',
                'Cd_f', 'Cd_p', 'Cd_b', 'Cn', 'Cm_p', 'Cm_y', 'Cf_side', 'Cm_r', 'Cf_r', 'Cdamp_r',
                'Cdamp_p', 'a_c', 'l_ref', 'S_ref', 'theta_vert', 'theta_lat', 'v_wind', 'T_air',
                'P_air', 'v_sound', 'dt', 't_tot'
            ]

    data = pd.read_csv(filename, comment='#', names=columns)

    fig, axs = plt.subplots(3, figsize=(15, 10), sharex=True)
    sns.lineplot('t', 'omega_r', data=data, ax=axs[0])
    sns.lineplot('t', 'omega_p', data=data, ax=axs[1])
    sns.lineplot('t', 'omega_y', data=data, ax=axs[2])
    # axs[0].set_xlim(0, 8)
    fig.tight_layout()

    fig, axs = plt.subplots(2, figsize=(15, 10), sharex=True)
    sns.lineplot('t', 'theta_vert', data=data, ax=axs[0])
    sns.lineplot('t', 'theta_lat', data=data, ax=axs[1])
    # axs[0].set_xlim(0, 8)
    fig.tight_layout()

    fig, axs = plt.subplots(2, figsize=(15, 10), sharex=True)
    a = data['a']
    a_v = data['a_v']
    a_lat = data['a_lat']
    a_est = np.sqrt(np.square(a_v) + np.square(a_lat))
    # sns.lineplot('t', 'a', data=data, ax=axs[0])
    sns.lineplot('t', 'a_v', data=data, ax=axs[0])
    sns.lineplot('t', 'a_lat', data=data, ax=axs[0])
    # sns.lineplot('t', 'a_c', data=data, ax=axs[0])
    sns.lineplot(x=data['t'], y=a-a_est, ax=axs[1])  # NOTE: a = magnitude of the vector created by a_v and a_lat
    # axs[0].set_xlim(0, 8)
    # axs[0].set_ylim(-25, 50)
    fig.tight_layout()

    plt.show()
