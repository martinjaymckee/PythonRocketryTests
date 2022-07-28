# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:36:42 2022

@author: marti
"""

import random

import numpy as np
import matplotlib.pyplot as plt


class BarometricDataGenerator:
    def __init__(self):
        self.__g0 = 9.80665
        self.__P0 = 101325
        self.__T0 = 288.15
        self.__h0 = 0
        self.__L0 = -0.0065
        
        
    @property
    def P0(self): return self.__P0
    
    @property
    def T0(self): return self.__T0
    
    def pressure(self, h, P0=None, T0=None):
        if P0 is not None:
            self.__P0 = P0
        if T0 is not None:
            self.__T0 = T0
        g0 = self.__g0
        R = 8.3144598
        h0 = self.__h0
        P0 = self.__P0
        T0 = self.__T0
        M = 0.0289884
        L0 = self.__L0
        return P0 * pow(((T0 + (h - h0)*L0)/T0), (-g0*M)/(R*L0))

    def temperature(self, h, P0=None, T0=None):
        if P0 is not None:
            self.__P0 = P0
        if T0 is not None:
            self.__T0 = T0
        h0 = self.__h0
        T0 = self.__T0
        L0 = self.__L0
        return T0 + (h-h0)*L0
    
    def height(self, P, P0=None, T0=None):
        if P0 is not None:
            self.__P0 = P0
        if T0 is not None:
            self.__T0 = T0
        g0 = self.__g0
        R = 8.3144598
        h0 = self.__h0
        P0 = self.__P0
        T0 = self.__T0
        M = 0.0289884
        L0 = self.__L0        
        return ((T0 * (pow(P/P0, (-R*L0)/(g0*M)))) - T0 + (h0*L0)) / L0
    
    
class BaroNoiseModel:
    def __init__(self, P_err=330, P_sd=1.2, T_err=1.5, T_sd=0.01):
        self.__P_err = random.gauss(0, P_err/5)
        self.__P_sd = P_sd
        self.__T_err = random.gauss(0, T_err/5)
        self.__T_sd = T_sd
        
    def __call__(self, P, T):
        # NOTE: THIS IS NOT HANDLING THE TEMPERATURE OFFSET ERRORS AT ALL AND
        #   THAT IS SOMETHING THAT WOULD BE VERY USEFUL TO DO, HOWEVER, IT IS
        #   OUTSIDE THE SCOPE OF THE PARAMETER ESTIMATION PROBLEM.
        dP = random.gauss(self.__P_err, self.__P_sd)
        dT = random.gauss(self.__T_err, self.__T_sd)
        return P+dP, T+dT
        

class GPSNoiseModel:
    def __init__(self, h_sd=0.67):
        self.__h_sd = h_sd
        
    def __call__(self, h):
        dh = random.gauss(0, self.__h_sd)
        return h+dh
    

if __name__ == '__main__':
    num_points = 25
    num_samples = 5
    gen = BarometricDataGenerator()
    P0 = gen.P0
    T0 = gen.T0
    # print('P0 = {} Pa, T0 = {} K'.format(P0, T0))
    
    h_min, h_max = 0, 11000
    # print('at h = {}, P = {}'.format(h, gen.pressure(h)))
    hs = []
    Ps_ref = []
    Ts_ref = []
    hs_measured = []
    Ps_measured = []
    Ts_measured = []
    hs_calc = []
    
    baro_noise = BaroNoiseModel()
    gps_noise = GPSNoiseModel()
    
    for h in np.linspace(h_min, h_max, num_points):
        P = gen.pressure(h)
        T = gen.temperature(h)

        for _ in range(num_samples):
            hs.append(h)
            Ps_ref.append(P)          
            Ts_ref.append(T)
            h_measured = gps_noise(h)
            P_measured, T_measured = baro_noise(P, T)
            h_calc = gen.height(P_measured)
            hs_measured.append(h_measured)
            Ps_measured.append(P_measured)
            Ts_measured.append(T_measured)
            hs_calc.append(h_calc)
    Ps_ref = np.array(Ps_ref)
    Ts_ref = np.array(Ts_ref)
    hs_measured = np.array(hs_measured)
    Ps_measured = np.array(Ps_measured)
    Ts_measured = np.array(Ts_measured)
    hs_calc = np.array(hs_calc)
                
    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
    axs[0].plot(hs, Ps_ref, c='k')
    axs[0].scatter(hs, Ps_measured, c='g')
    # ax0 = axs[0].twinx()
    # ax0.scatter(hs, Ps_measured - Ps_ref, c='r', alpha=0.5)
    axs[1].plot(hs, Ts_ref, c='k')
    axs[1].scatter(hs, Ts_measured, c='g')
    # ax1 = axs[1].twinx()
    # ax1.scatter(hs, Ts_measured - Ts_ref, c='r', alpha=0.5)    
    axs[2].scatter(hs, hs_measured-hs, c='r', alpha=0.5)
    axs[2].scatter(hs, hs_calc-hs, c='m', alpha=0.5)
        
#    plt.show()