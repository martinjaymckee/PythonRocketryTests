import argparse
import math
import numbers
import os
import os.path
import random
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import seaborn as sns


import pyrse.continuous_ga
import pyrse.engines
import pyrse.environment
import pyrse.numpy_utils
import pyrse.pad
import pyrse.rocket_components
import pyrse.simulator
import pyrse.simulator_analysis
import pyrse.triggers
import pyrse.utils

import sim_config_parser

plt.style.use('dark_background')

class SimulationPlot:
    def __init__(self):
        pass


class SimpleRocketModel(pyrse.rocket_components.Component):
    def __init__(self, cd, frontal_area, empty_mass, engine, Re_max=5e6):
        pyrse.rocket_components.Component.__init__(self, calc_aero=True)
        self.cd = cd
        self.__frontal_area = frontal_area
        self.__L_ref = 2 * math.sqrt(frontal_area / math.pi)        
        self.__empty_mass = empty_mass
        self.__engine = engine
        self.add(engine)
        self.__Re_max = Re_max
        self.__maximum_Re_seen = None
        
    @property
    def cd(self):
        return self.__cd
    
    @cd.setter
    def cd(self, _cd):
        self.__cd = _cd
        if isinstance(_cd, numbers.Number):
            self.__cd_is_constant = True
        else:
            self.__cd_is_constant = False # TODO: PROCESS PIECEWISE LINEAR CD
        return self.__cd
    
    @property
    def Re_max(self):
        return self.__Re_max
    
    def calc_mass(self, t0):
        return self.__empty_mass

    def get_maximum_Re(self):
        return self.__maximum_Re_seen
    
    def calc_cd(self, v, env):
        v_mag = pyrse.numpy_utils.magnitude(v.ecef)        
        Re = v_mag * self.__L_ref / env.kinematic_viscosity # Calculate Reynolds Number
        if (self.__maximum_Re_seen is None) or (Re > self.__maximum_Re_seen):
            self.__maximum_Re_seen = Re        
        if self.__cd_is_constant:
            return self.__cd
        return self.__cd(Re)


    @property
    def frontal_area(self):
        return self.__frontal_area


def create_model(m, S, eng, cd):
    return SimpleRocketModel(cd, S, m, eng)


def plot_engines(cfg, N=7, M=100, with_reference=False):
    eng_value = cfg['motor']    
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs[0].set_ylabel('Thrust (N)')
    axs[1].set_ylabel('Mass (g)')
    ax1 = axs[1].twinx()
    ax1.set_ylabel('Spent Impulse (Ns) - Dotted')

    engs = []
    for idx in range(N):
        engs.append(eng_value(idx, N))
    axs[0].set_title('{} {}'.format(engs[0].manufacturer, engs[0].model))
    for eng in engs:
        eng.start(0)
        t_burnout = eng.burn_time
        ref_ts, ref_Ts = eng.reference_thrust_curve
        ts = np.linspace(0, t_burnout, M)
        Ts = [eng.thrust(t) for t in ts]
        ms = [eng.mass(t) for t in ts]
        spent = [eng.spent_impulse(t) for t in ts]

        axs[0].plot(ts, Ts, alpha=0.5)
        axs[1].plot(ts, ms, alpha=0.5)
        ax1.plot(ts, spent, linestyle='dotted', alpha=0.5)
        if with_reference: 
            axs[0].scatter(ref_ts, ref_Ts,marker='x')
    plt.show()


def main():
    directory = r"D:\Workspace\Rockets\PythonRocketryTests\Simulation Configurations"
    filename = "test_sim.cfg"
    path = os.path.join(directory, filename)
    path = os.path.abspath(path)

    cfg_parser = sim_config_parser.SimulationConfigParser()

    sim_cfg = cfg_parser(path)

    plots = None
    plot_engines(sim_cfg, with_reference=True)


if __name__ == '__main__':
    main()