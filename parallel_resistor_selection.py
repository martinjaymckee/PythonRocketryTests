# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:21:11 2022

@author: marti
"""

import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def parallel(Rs):
    Rsum = 0
    for R in Rs:
        Rsum += 1 / R
    return 1 / Rsum


#__Rs = np.array(
#[0.4, 0.5, 1.0, 1.2, 1.25, 1.5, 1.65, 1.8, 1.95, 2.0, 2.2, 2.5, 2.7, 3.0, 3.3, 3.9,
#4.0, 4.7, 5.0, 5.6, 6.8, 7.5, 8.2, 10, 12, 15, 18, 20, 22, 25, 27, 30, 33, 39,
#40, 47, 50, 56, 68, 75, 82, 100, 120, 150, 180, 200, 220, 250, 270, 300, 330, 390 ]
#)

__Rs = np.array(
[0.4, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.7, 3.0, 3.3, 3.9,
4.0, 4.7, 5.0, 5.6, 6.8, 7.5, 8.2, 10, 12, 15, 18, 20, 22, 25, 27, 30, 33, 39,
40, 47, 50, 56, 68, 75, 82, 100, 120, 150, 180, 200, 220, 250, 270, 300, 330, 390,
400, 470, 500, 560, 680, 750, 820, 1000, 1200, 1500 ]
)

# __Rs = np.array( # Under 2.5 ohm calculated as duplicate parallel values
# [1.0, 1.1, 1.25, 1.35, 1.5, 1.65, 1.8, 1.95, 2.0, 2.2, 2.35, 2.5, 2.7, 2.8, 3.0, 3.3, 3.4, 3.9,
# 4.0, 4.7, 5.0, 5.6, 6.8, 7.5, 8.2, 10, 12, 15, 18, 20, 22, 25, 27, 30, 33, 39,
# 40, 47, 50, 56, 68, 75, 82, 100, 120, 150, 180, 200, 220, 250, 270, 300, 330, 390,
# 400, 470, 500, 560, 680, 750, 820, 1000, 1200, 1500 ]
# )

def nearestResistor(R):
    index = np.absolute(__Rs-R).argmin()
    return __Rs[index]


def createRandomResistorSet(Rmin, Rmax, num):
    Rs = []
    R = nearestResistor(random.uniform(Rmin, 2 * Rmin))
    Rs.append(R)
    for _ in range(num-1):
        R = nearestResistor(random.uniform(R, (R + Rmax)/2))
        Rs.append(R)
    return np.array(Rs)


def createResistorSet(Rmin, num, scale=2, jitter_sd=0.2):
    def create(R):
        Rs = []
        for _ in range(num):
            R_next = random.gauss(R, R*jitter_sd)
            Rs.append(nearestResistor(R_next))
            R = R * scale
        return np.array(Rs)        
    R0 = Rmin * (1 + scale) / scale
    Rs = create(R0)
    while parallel(Rs) < Rmin:
        R0 = R0 * 1.01
        Rs = create(R0)
    return Rs


def parallelRs(Rs):
    combs = []
    Rs_par = []
    for L in range(len(Rs) + 1):
        for subset in itertools.combinations(Rs, L):
            if len(subset) > 0:
                combs.append( subset )
                Rs_par.append( parallel(subset) )    
    return combs, Rs_par


def sensingSet(Rs, spacing, mode='relative'):
    combs, Rs_par = parallelRs(Rs)
    Rs_par, combs = zip(*sorted(zip(Rs_par, combs)))    
    Rs = np.array(Rs_par)
    Rmin = np.min(Rs_par)
    Rmax = np.max(Rs_par)
    
    sensing_combs = []
    sensing_Rs = []
    if mode == 'relative':
        R_last = None
        for comb, R in zip(combs, Rs_par):        
            if (R_last is None) or ((100*(R - R_last)/R_last) >= spacing):
                R_last = R
                sensing_combs.append(comb)
                sensing_Rs.append(R)
    elif mode == 'absolute':
        percent = 100 * Rs / Rmax
        P = None
        for comb, R, Pi in zip(combs, Rs_par, percent):        
            if (P is None) or (Pi >= (P + spacing)):
                P = Pi
                sensing_combs.append(comb)
                sensing_Rs.append(R)        
    return sensing_combs, sensing_Rs
        

def uniformity(Rs):
    dRmin, dRmax, R_last = None, None, None
    for R in Rs:
        if R_last is not None:
            dR = abs(math.log(R) - math.log(R_last))
            dRmin = dR if (dRmin is None) or (dR < dRmin) else dRmin
            dRmax = dR if (dRmax is None) or (dR > dRmax) else dRmax            
        R_last = R
    return dRmin / dRmax


def fitness(Rs, N, R_range, samples_scale=1, eff_scale=0.25, uniformity_scale=0.75, range_scale=4, debug=False):
    Rmax = np.max(Rs)
    Rmin = np.min(Rs)
    eff_term = (len(Rs) / (2**N))**eff_scale
    uniformity_term = uniformity(Rs)**uniformity_scale
    samples_term = len(Rs)**samples_scale
    range_max_term = (min(Rmax, max(R_range)) / max(Rmax, max(R_range)))**range_scale
    range_min_term = (min(Rmin, min(R_range)) / max(Rmin, min(R_range)))**range_scale    
    if debug:
        print('uniformity = {}'.format(uniformity_term))
        print('efficiency = {}'.format(eff_term))
        print('sample = {}'.format(samples_term))
        print('Rmin = {}, Rmax = {}'.format(Rmin, Rmax))
        print('range min = {}'.format(range_min_term))  
        print('range max = {}'.format(range_max_term))          
    return uniformity_term * eff_term * samples_term * range_min_term * range_max_term


def combToIndex(comb, Rs_base):
    Rs_base = np.array(Rs_base)
    new_comb = []
    for val in comb:
        idx = np.argmax(Rs_base == val)
        new_comb.append(idx)
    return tuple(new_comb)


def printSystemOption(sim, Vmax):
    score, N, scale, Rs_base, Rs, combs = sim
    print('Score: {}'.format(score))
    print('N = {}, scale = {}'.format(N, scale))
    print('Resistors: {}'.format(', '.join(['{:0.2f} ({:.0f} W)'.format(R, math.ceil(Vmax**2/R)) for R in Rs_base])))
    print('Sample Points({}):'.format(len(Rs)))
    for R, comb in zip(Rs, combs):
        print('\t{} -> {}'.format(R, list(combToIndex(comb, Rs_base))))
    return


if __name__ == '__main__':
    Vmax = 15
    Nmin = 5
    Nmax = 5
    Rmin = 1.8
    Rmax = 750
    spacing = 15
    scale_range = 3
    scale_samples = 500
    random_samples = 5000
    best_num = 1
    sims = []

    N = 6
    for _ in range(random_samples):
        scale = (Rmax / Rmin) ** (1 / (N-1))
        Rs_base = []
        Rref = random.uniform(Rmin, 2*Rmin)
        for _ in range(N):
            R1 = random.gauss(Rref, Rref/3)
            Rs_base.append(nearestResistor(R1))
            Rref *= scale
        Rs_base = np.array(Rs_base)            
        combs, Rs = sensingSet(Rs_base, spacing)
        score = fitness(Rs, N, (Rmin, Rmax))
        sims.append( (score, N, None, Rs_base, Rs, combs))
    
    # for _ in range(random_samples):
    #     Rs_base = createRandomResistorSet(Rmin, Rmax, N)
    #     combs, Rs = sensingSet(Rs_base, spacing)
    #     score = fitness(Rs, N, (Rmin, Rmax))
    #     sims.append( (score, N, None, Rs_base, Rs, combs))
        

  #   for N in range(Nmin, Nmax+1):
  #       scale_min = pow(Rmax/Rmin, 1/(2*N))
  #       scale_max = scale_min * scale_range        
  # #      print('N = {}, scale_min = {}, scale_max = {}'.format(N, scale_min, scale_max))
 
  #       for scale in np.linspace(scale_min, scale_max, scale_samples):
  #           for _ in range(random_samples):
  #               Rs_base = createResistorSet(Rmin, N, scale)
  #               combs, Rs = sensingSet(Rs_base, spacing)
  #               score = fitness(Rs, N, (Rmin, Rmax))
  #               sims.append( (score, N, scale, Rs_base, Rs, combs) ) 
                
    sorted_sims = sorted(sims, key = lambda x: x[0], reverse=True)
    figs, ax = plt.subplots(1, constrained_layout=True)
    for idx in range(best_num):
        sim = sorted_sims[idx]
        printSystemOption(sim, Vmax)
        score, N, scale, Rs_base, Rs, combs = sim
        fitness(Rs, N, (Rmin, Rmax), debug=True)
        xs = np.linspace(0, 1, len(Rs))
        #plt.plot(xs, Rs)
        plt.hist(np.log(Rs), bins=75, alpha=1)
        print()
    plt.show()
