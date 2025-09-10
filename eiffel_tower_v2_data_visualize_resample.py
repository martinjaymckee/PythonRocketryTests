import os
import os.path
import time

# import ambiance
import matplotlib.pyplot as plt
# import numpy as np
# import scipy.signal as signal
# import seaborn as sns

import pyrse.engines as engines
from pyrse.flight_data import loadBlueRavenLog
from pyrse.flight_data_blender import BlenderResampler
# import pyrse.flight_data_plotting as fdp


def load_flight_data_list(filename):
    flight_data_list = None
    global_vars = {}
    local_vars = {}
    with open(filename) as src:
        code = compile(src.read(), filename, 'exec')
        exec(code, global_vars, local_vars)
        flight_data_list = local_vars['flight_data_list']
    return flight_data_list


if __name__ == '__main__':
    output_directory = r'D:\Workspace\Rockets\HPR\Eiffel Tower v2\Blender Visualize Data'
    flight_data_list = load_flight_data_list("eiffel_tower_data_list.dat")
    #print(flight_data_list)
    fps = 240
    t_pre = 6.0

    resampler = BlenderResampler()

    for flight in flight_data_list:
        t_start = time.time_ns()    
        br_flight_log = loadBlueRavenLog(flight['summary_path'], flight['low_rate_path'], flight['high_rate_path'])
        t_new = time.time_ns() - t_start
        print('{} Parse Time = {:0.2f} ms'.format(flight['name'], t_new / 1e6))

        if br_flight_log is not None:
            ref_eng = engines.Engine.RSE(flight['engine_path'])            
            output_path = os.path.join(output_directory, 'blender_resample_{name}_{fps}fps.csv'.format(name=flight['name'], fps=fps))
            success, results = resampler.export_csv(output_path, br_flight_log, ref_eng, fps, t_pre, flight['delay'])

            # results = resampler.resample(br_flight_log, ref_eng, fps, t_pre, flight['delay'])
            ts = results['t']
            fig, axs = plt.subplots(2, layout='constrained')
            axs[0].plot(ts, results['dy'], 'b')
            ax_0 = axs[0].twinx()
            ax_0.plot(ts, results['dx'], 'g')
            axs[1].plot(ts, results['burn'], 'r')
            axs[1].plot(ts, results['smoke'], 'c')

    plt.show()
