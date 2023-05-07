# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:22:00 2022

@author: marti
"""

import math


def attenuate_by_db(noise, db):
    return pow(10, ((db / 20) + math.log10(noise)))


def total_noise(*noises):
    sum_squared = 0
    for n in noises:
        sum_squared += (n * n)
    return math.sqrt(sum_squared)


def johnsonNoise(R, T=None):
    kb = 1.380649e-23
    if T is None:
        T = 300        
    return math.sqrt(4 * kb * T * R)


def sallen_key_noise(f, R, e_n, i_n):
    e_in = e_n * math.sqrt(f)
    e_R = johnsonNoise(2 * R)
    i_in = 2 * R * i_n * math.sqrt(f)
    noises = [e_in, e_in, e_R, i_in]
    print(noises)
    return total_noise(*noises)
    
    
if __name__ == '__main__':
    adc_base_noise = 4.8e-6
    adc_avdd_psrr = -75
    adc_iovdd_psrr = -100
    ldo_noise = 30e-6
    ldo_psrr = -45
    noise_reg = 1.9e-6
    noise_5v_buck = 100e-3
    noise_avdd = total_noise( ldo_noise, attenuate_by_db(noise_5v_buck, ldo_psrr) )
    noise_iovdd = total_noise( ldo_noise, attenuate_by_db(noise_5v_buck, ldo_psrr) )
    noise_avdd_eff = attenuate_by_db(noise_avdd, adc_avdd_psrr)
    noise_iovdd_eff = attenuate_by_db(noise_iovdd, adc_iovdd_psrr)    
    noise_divider = johnsonNoise(1e3)
    noise_v_filt = sallen_key_noise(10e3, 909, 5.8e-9, 0.8e-9) # ADA4522
    #noise_v_filt = sallen_key_noise(10e3, 909, 50e-9, 0.1e-9) # MAX44248
    #noise_v_filt = sallen_key_noise(10e3, 909, 16e-9, 0.01e-12) # ADA4530 -- NOTE: MASSIVELY EXPENSIVE...
    #noise_v_filt = sallen_key_noise(10e3, 909, 5.7e-9, 165e-12) # OPA2182

    noises = [adc_base_noise, noise_reg, noise_avdd_eff, noise_iovdd_eff, noise_v_filt]
    
    print(noises)
    
    print('Total Noise = {} uV'.format(1e6 * total_noise(*noises)))
    
 #   print(attenuate_by_db(100e-3, -66) * 1e6)
    