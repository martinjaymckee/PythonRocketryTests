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
import numpy.random as nprand
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

#plt.style.use('dark_background')

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
    
    @property
    def Lref(self):
        return self.__L_ref

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
        return self.__cd.transform(Re)


    @property
    def frontal_area(self):
        return self.__frontal_area


def create_model(m, S, eng, cd):
    return SimpleRocketModel(cd, S, m, eng)


def plot_kinematics(N, run_data):
    fig, axs = plt.subplots(3, constrained_layout=True, sharex=True)
    axs[0].set_title('Acceleration, Velocity, and Altitude')
    axs[0].set_ylabel('Acceleration (m/s^2)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[2].set_ylabel('Altitude (m)')
    axs[2].set_xlabel('Time (s)')

    palette = sns.color_palette('husl', N, desat=0.7)
    for idx, (params, results, sim_status) in enumerate(run_data):
        color = palette[idx]
        sns.lineplot(x=results.t, y=results.a_z, ax=axs[0], color=color, alpha=1)
        sns.lineplot(x=results.t, y=results.v, ax=axs[1], color=color, alpha=1)
        sns.lineplot(x=results.t, y=results.h, ax=axs[2], color=color, alpha=1)                


def plot_forces(N, run_data):
    fig, axs = plt.subplots(2, constrained_layout=True)
    axs[0].set_title('Thrust vs Drag vs Weight and T/W Ratio')
    axs[0].set_ylabel('Force (N)')
    ax0 = axs[0].twinx()
    ax0.set_ylabel('Weight (N)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('T/W Ratio')

    palette = sns.color_palette('husl', 4, desat=0.7)
#    palette = sns.diverging_palette(145, 300, s=100, n=N)
    for idx, (params, results, sim_status) in enumerate(run_data):
        weights = 9.80665 * np.array(results.m)
        Ts = np.array(results.T)
        TWs = Ts / weights
        sns.lineplot(x=results.t, y=results.T, ax=axs[0], color=palette[0], alpha=1)
        sns.lineplot(x=results.t, y=results.D, ax=axs[0], color=palette[1], alpha=1)
        sns.lineplot(x=results.t, y=weights, ax=ax0, color=palette[2], alpha=1)
        sns.lineplot(x=results.t, y=TWs, ax=axs[1], color=palette[3], alpha=1)


def plot_flights(sim_run_data, flight_run_data, flight_noise={}):
    fig, axs = plt.subplots(2, constrained_layout=True, sharex=True)
    axs[0].set_title('Accelerations and Altitude')
    axs[0].set_ylabel('Accelerations (m/s^2)')
    axs[1].set_ylabel('Altitude (m)')
    axs[1].set_xlabel('Time (s)')

    for idx, (params, results, sim_status) in enumerate(sim_run_data):
        sns.lineplot(x=results.t, y=results.a_z, ax=axs[0], color='b', alpha=0.25)
        sns.lineplot(x=results.t, y=results.h, ax=axs[1], color='b', alpha=0.25)                

    baro_noise = 0 if 'baro_sd' not in flight_noise else flight_noise['baro_sd']
    accel_noise = 0 if 'accel_sd' not in flight_noise else flight_noise['accel_sd']

    for idx, (params, results, sim_status) in enumerate(flight_run_data):
        da = nprand.normal(scale=accel_noise, size=len(results.a))
        dh = nprand.normal(scale=baro_noise, size=len(results.h))

        sns.lineplot(x=results.t, y=results.a_z + da, ax=axs[0], color='r', alpha=0.75)
        sns.lineplot(x=results.t, y=results.h + dh, ax=axs[1], color='r', alpha=0.75)                


def run_simulations(cfg, plots, pos = None):
    pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5) if pos is None else pos
    pad = pyrse.pad.LaunchPad(pos)
    env = pyrse.environment.Environment(pad.pos)

    print(cfg)

    N = cfg['sims']()
 #   viridisN = mpl.colormaps['viridis'].resampled(N)
    #plt.set_cmap(viridisN)

    dt_value = cfg['dt']
    mass_value = cfg['mass']
    eng_value = cfg['motor']
    ref_area_value = cfg['ref_area']
    cd_value = cfg['cd']

    dts = []
    masses = []
    engs = []
    ref_areas = []
    cds = []

    for idx in range(N):
        dts.append(dt_value(idx, N))
        masses.append(mass_value(idx, N))
        engs.append(eng_value(idx, N))
        ref_areas.append(ref_area_value(idx, N))
        print(cd_value)
        cds.append(cd_value(idx, N))

    sim_stats = []
    run_data = []
    extract_values = ['t', 'dt', 'm:mass', 'T:magnitude(forces.T)', 'D:magnitude(forces.D)', 'a_z:accel.z', 'a:magnitude(accel)', 'v:magnitude(vel)', 'h:alt(pos)', 'Re']
    log_extractor = pyrse.simulator_analysis.SimulationLogExtractor(extract_values)

    for idx, (dt, m, eng, ref_area, cd) in enumerate(zip(dts, masses, engs, ref_areas, cds)):
        start_time = time.time()
        print('#{}, dt = {:0.3f} ms, m = {:0.1f} g, S = {:0.2f} cm_2, cd = {}'.format(idx, 1000 * dt, 1000 * m, (100**2) * ref_area, cd), end = '')
        model = create_model(m, ref_area, eng, cd)
        triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(30)]    
        sim = pyrse.simulator.Simulation1D(env, pad, model, dt_min=dt, triggers=triggers)    
        sim_status = pyrse.simulator.RunSimulation(sim)
        results = log_extractor(sim)[0]
        params = (dt, m, eng, ref_area, cd)
        run_data.append((params, results, sim_status))
        end_time = time.time()
        total_time = end_time - start_time
        h_max = np.max(results.h)
        print(' -- in {:0.1f} s  -- Maximum Altitude = {:0.1f} m, Maximum Reynolds Number = {:0.1f}'.format(total_time, h_max, model.Re_max))

#    plot_kinematics(N, run_data)
#    plot_forces(N, run_data)
    return run_data


def main():
    directory = r"D:\Workspace\Rockets\PythonRocketryTests\Simulation Configurations"
    sim_filename = "anomoly_detection_sim.cfg"
    flight_filename = "anomoly_detection_flights.cfg"

    sim_path = os.path.join(directory, sim_filename)
    sim_path = os.path.abspath(sim_path)

    flight_path = os.path.join(directory, flight_filename)
    flight_path = os.path.abspath(flight_path)

    cfg_parser = sim_config_parser.SimulationConfigParser()

    sim_cfg = cfg_parser(sim_path)
    flight_cfg = cfg_parser(flight_path)

    plots = None
    sim_run_data = run_simulations(sim_cfg, plots)
    flight_run_data = run_simulations(flight_cfg, plots)
    plot_flights(sim_run_data, flight_run_data, flight_noise={'baro_sd': 1.5, 'accel_sd': 0.25})

    plt.show()


if __name__ == '__main__':
    main()