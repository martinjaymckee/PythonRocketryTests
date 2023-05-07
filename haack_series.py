# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:33:48 2022

@author: marti
"""
import math

import matplotlib.pyplot as plt
import numpy as np


def calc_haack(x, L, r, C=0):
    theta = math.acos(1 - ((2*x) / L))
    y = (r / math.sqrt(math.pi)) * math.sqrt(theta - (math.sin(2 * theta) / 2) + (C * (math.sin(theta) ** 3)) )
    return y

def main(d, Cs, Ratios, N=50):
    r = d/2
    
    fig, axs = plt.subplots(len(Cs), len(Ratios), constrained_layout=True, sharex=True, sharey=True)
    
    for column, ratio in enumerate(Ratios):
        L = d * ratio
        xs = np.linspace(0, L, N)
        
        for row, C in enumerate(Cs):
            ys = np.array([calc_haack(x, L, r, C) for x in xs])
            axs[column][row].fill_between(xs, ys, -ys)
        #    ax.plot(xs, ys, c='g')
        #    ax.plot(xs, -ys, c='g')
            axs[column][row].set_aspect('equal')
            axs[column][row].axis('off')
#            axs[column][row].get_xaxis().set_visible(False)
#            axs[column][row].get_yaxis().set_visible(False)
    plt.show()
    

if __name__ == '__main__':
    d = 41.6
    ratio = 3
    N = 7
    C = 0.66
    l = d * ratio
    r = d / 2
 #   xs = np.linspace(0, l, N)
 #   print('Total Length = {} mm'.format(l))
    
 #   for x in xs:
 #       y = calc_haack(x, l, r, C)
 #       print('\tx = {}, y = {}'.format(x, y))
        
    Cs = [0, 0.3333, 0.6666]
    Ratios = [3.5, 4.25, 5]
    main(d, Cs, Ratios)