# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:40:50 2022

@author: marti
"""

import math


def maxResistanceMeasurable(Vs, Rl, Av, Vref=2.5):
    Vmin, Vmax = Vs
    return (Vref * Rl) / ((Av * Vmax) - Vref)


def resResistanceMeasurable(Vs, Rl, Vos):
    Vmin, Vmax = Vs
    return (Vos * Rl) / (Vmin * (1 - (Vos / Vmin)))


def effResolutionAtResistance(R, Vs, Rl, Vos):
    res = resResistanceMeasurable(Vs, Rl, Vos)
    return math.log(R / res) / math.log(2)


def currentMax(Vs, Rl):
    Vmin, Vmax = Vs
    return Vmax / Rl


def resistanceTestProperties(Rs, Vs, Rl, Av, Vos, Vref=2.5, Itest_max = 0.010, adc_bits=12):
    Rmax = maxResistanceMeasurable(Vs, Rl, Av, Vref)
    Rmin = resResistanceMeasurable(Vs, Rl, Vos)
    
    min_bits = effResolutionAtResistance(min(Rs), Vs, Rl, Vos)
    max_bits = effResolutionAtResistance(Rmax, Vs, Rl, Vos)
    Imax = currentMax(Vs, Rl)
    
    print('Vs = ({:0.2g} v, {:0.2g} v), Rl = {:0.2g} ohms'.format(*Vs, Rl))
    print('\tAv = {:0.2g}, Vos = {:0.2g} v'.format(Av, Vos))
    print('\tRtest = ({:0.2g} ohms, {:0.2g} ohms), effective resolution = {:0.1f} bits, Imax = {:0.2g} A'.format(Rmin, Rmax, min_bits, Imax))
    if Imax >= Itest_max:
        print('\tERROR: Maximum test current ({:0.2g} A) exceeds limit ({:0.2g} A)'.format(Imax, Itest_max))
    if Rmax <= max(Rs):
        print('\tERROR: Maximum test resistance ({:0.2g} ohms) is less than requested value of {} ohms'.format(Rmax, max(Rs)))
    print('min_bits = {:0.1f}, max_bits = {:0.1f}'.format(min_bits, max_bits))
    if adc_bits >= max_bits:
        print('\tNote: Resolution limited by input offset ({:0.1f} bits vs {:0.1f} bits)'.format(max_bits, adc_bits))
    return (Rmin, Rmax), (min_bits, max_bits), Imax    
    

if __name__ == '__main__':
    Vs = (3.7, 16.8)
    Rs = (0.75, 20)
    Av = 25
    Vos = 15e-6
    Vref = 2.5
    Rls = (10e3, 1.8e3)
    adc_bits = 12

    for Rl in Rls:
        resistanceTestProperties(Rs, Vs, Rl, Av, Vos, Vref=Vref, adc_bits=adc_bits)
        print()
        
        