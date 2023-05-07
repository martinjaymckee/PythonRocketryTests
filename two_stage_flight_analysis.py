import os
import os.path


import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt

import filterpy
import filterpy.common
import filterpy.kalman

def loadEggtimerFlightData(path):
    data = pd.read_csv(path)
    return data


class AltVelFilter:
    def __init__(self, alt_sd, vel_sd=None):
        self.__alt_sd = alt_sd
#        self.__vel_sd = alt_sd ** 2 if vel_sd is None else vel_sd
#        self.__accel_sd = self.__vel_sd ** 2
        self.__t_last = None
        self.__filt = filterpy.kalman.KalmanFilter(dim_x=3, dim_z=1)
        self.__filt.x = np.array([0, 0, 0])
        self.__filt.H = np.array([[1, 0, 0]])
        self.__filt.R = self.__alt_sd
        self.__filt.P *= 1000
        
    def __call__(self, t, h):
        if self.__t_last is not None:
            z = np.array([h])
            dt = t - self.__t_last
            self.__filt.F = np.array([
                [1, dt, (dt*dt) / 2],
                [0, 1, dt],
                [0, 0, 1]
            ]) 
            self.__filt.Q = filterpy.common.Q_discrete_white_noise(dim=3, dt=dt, var=0.13)
            # print(self.__filt.F)
            # print(self.__filt.Q)
            # print(self.__filt.P)
            # print(z)
            # print(self.__filt.S)
            self.__filt.predict()
            self.__filt.update(z)
        self.__t_last = t
        return self.__filt.x[0], self.__filt.x[1]


if __name__ == '__main__':
    do_plot = True
    t_pre = 5
    filename = 'two_blue_flight_1.csv'
    data = loadEggtimerFlightData(os.path.join('data', filename))
    print(data)
    
    err = data['Alt'] - data['FAlt']
    alt_noise = np.std(err) / 3 # NOTE: THE DIRECT ESTIMATE SEEMS TOO LARGE, SO THE DIVISION MAKES IT MORE REASONABLE
    print('Noise = {} ft.'.format(alt_noise))

    ts = data['T']
    alts = data['Alt']
    falts = data['FAlt']
    vels = data['Veloc']
    fvels = data['FVeloc']
    
    dt = ts[1] - ts[0] # THIS MAKES THE ASSUMPTION OF CONSTANT SAMPLING RATE
    N_pre = int((t_pre - dt) / dt)
    
    ts_pre = np.linspace(-(t_pre-dt), -dt, N_pre)   
    alts_pre = np.random.normal(0, alt_noise, N_pre)
    vels_pre = np.zeros(N_pre)
    
    ts_all = np.array(list(ts_pre) + list(ts))
    alts_all = np.array(list(alts_pre) + list(alts))
    vels_all = np.array(list(vels_pre) + list(vels))
    
    # RUN KALMAN FILTER AND REGENERATE ALTS_KF AND VELS_KF
    # INITIALIZE ALT AND VEL TO 0
    kf = AltVelFilter(alt_noise)
    alts_kf = []
    vels_kf = []
    for t, h in zip(ts_all, alts_all):
        alt, vel = kf(t, h)
        alts_kf.append(alt)
        vels_kf.append(vel)
    alts_kf = np.array(alts_kf)
    vels_kf = np.array(vels_kf)
    
    if do_plot:
        fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
        axs[0].set_title('Altitude')
        axs[1].set_title('Velocity')
        axs[0].axhline(0, c='k', alpha=0.2)
        axs[1].axhline(0, c='k', alpha=0.2)
        
        axs[0].plot(ts_pre, alts_pre, c='y', label='Predicted')
        axs[0].plot(ts, alts, c='g', alpha=0.25, label='Raw')
        axs[0].plot(ts, falts, c='b', alpha=0.5, label='Filtered')
        axs[0].plot(ts_all, alts_kf, c='m', alpha=0.5, label='Kalman Filtered')
    #    ax0 = axs[0].twinx()
    #    ax0.plot(data['T'], err)
        
        axs[1].plot(ts, vels, c='g', alpha=0.25, label='Raw')
        axs[1].plot(ts, fvels, c='b', alpha=0.5, label='Filtered')    
        axs[1].plot(ts_all, vels_kf, c='m', alpha=0.5, label='Kalman Filtered')
        
        vels_sd = np.std(vels)
        axs[1].set_ylim(-vels_sd, vels_sd)
        
        for ax in axs:
            ax.legend()
        
        for t, lda, apogee, n_o, drogue, main in zip(ts, data['LDA'], data['Apogee'], data['N-O'], data['Drogue'], data['Main']):
            if not lda == 0:
                print('t = {}, LDA = {}'.format(t, lda))            
                for ax in axs:
                    ax.axvline(t, c='k', alpha=0.6)
    
            if not apogee == 0:
                print('t = {}, Apogee = {}'.format(t, apogee))
                for ax in axs:
                    ax.axvline(t, c='k', alpha=0.6)
                   
            if not n_o == 0:
                print('t = {}, Nose Over = {}'.format(t, n_o))
                for ax in axs:
                    ax.axvline(t, c='k', alpha=0.6)
                    
            if not drogue == 0:
                print('t = {}, Drogue = {}'.format(t, drogue))
                for ax in axs:
                    ax.axvline(t, c='r', alpha=0.6)
                    
            if not main == 0:
                print('t = {}, Main = {}'.format(t, main))
                for ax in axs:
                    ax.axvline(t, c='r', alpha=0.6)
                    
        plt.show()