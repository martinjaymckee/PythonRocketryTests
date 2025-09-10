import numpy as np

from pyrse.flight_data import FlightData, loadBlueRavenLog, loadOpenRocketExport

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # low_df, high_df = generate_dummy_br_dfs(145, t0=-0.35, pre=35)
    # print(low_df)
    # print(high_df)

    t_start = time.time_ns()    
    or_flight_log = loadOpenRocketExport(r"D:\Workspace\Rockets\MPR\Eye Sore\Simulation Data\flight_1_openrocket.csv")
    br_flight_log = loadBlueRavenLog("D:\Workspace\Rockets\MPR\Eye Sore\Flight Data\Eye Sore_summary_06-01-2024_09_51_22_.csv", "D:\Workspace\Rockets\MPR\Eye Sore\Flight Data\low_rate_06-01-2024_13_45_09.csv", "D:\Workspace\Rockets\MPR\Eye Sore\Flight Data\high_rate_06-01-2024_13_45_10.csv")#, dummy=True, dummy_samples=2000)
#    br_flight_log = loadBlueRavenLog(r"D:\Workspace\Rockets\HPR\3in Nike-Hercules\flight_logs\Nike-Hercules_07-20-2024-summary.csv", r"D:\Workspace\Rockets\HPR\3in Nike-Hercules\flight_logs\Nike-Hercules_07-20-2024_low-rate.csv", r"D:\Workspace\Rockets\HPR\3in Nike-Hercules\flight_logs\Nike-Hercules_07-20-2024_high-rate.csv")#, dummy=True, dummy_samples=2000)

    print(or_flight_log)

    for name, event in or_flight_log.events.items():
        print(event)
    t_new = time.time_ns() - t_start
    
    print('Parse Time = {:0.2f} ms'.format(t_new / 1e6))

    if True:
        if br_flight_log is not None:
            print(br_flight_log.summary)

            fig, axs = plt.subplots(3, constrained_layout=True, sharex=True)
            fig.suptitle('Flight Summary Data')
            ts = br_flight_log['t'].values
            hs_raw = br_flight_log['hraw'].values
            hs = br_flight_log['h'].values
            Vzs = br_flight_log['Vz'].values
            axs_b = br_flight_log['ax'].values        
            mask = br_flight_log['hraw'].interpolation_mask 
            cs = ['g' if measured else 'r' for measured in mask]
            axs[0].plot(ts, hs_raw, c='k', alpha=0.33)        
            axs[0].axhline(0, c='c', alpha=0.5)
            axs[0].plot(ts, hs, c='g')
            axs[1].plot(ts, Vzs, c='g')  
            axs[1].axhline(0, c='c', alpha=0.5) 
            axs[2].plot(ts, axs_b, c='g')
            axs[2].axhline(0, c='c', alpha=0.5)       
            ax0 = axs[0].twinx()
            ax0.plot(ts, hs_raw-hs, c='r', alpha=0.5)
            print('altitude noise = {} m, maximum altitude error = {} m'.format((hs_raw-hs).std(), np.abs((hs_raw-hs)).max()))
    #        axs[0].scatter(ts, hs, c=cs, s=5)
    #        axs[1].scatter(ts, Vzs, c=cs, s=5)

            fig_a, axs = plt.subplots(3, constrained_layout=True, sharex=True)
            fig_a.suptitle('Acceleration Values')
            ts = br_flight_log['t'].values
            axs_b = br_flight_log['ax'].values
            ays_b = br_flight_log['ay'].values
            azs_b = br_flight_log['az'].values
            axs[0].plot(ts, axs_b, c='g')
            axs[1].plot(ts, ays_b, c='g')
            axs[2].plot(ts, azs_b, c='g')

            fig_g, axs = plt.subplots(3, constrained_layout=True, sharex=True)
            fig_g.suptitle('Gyro Rates')
            ts = br_flight_log['t'].values
            gxs_b = br_flight_log['gx'].values
            gys_b = br_flight_log['gy'].values
            gzs_b = br_flight_log['gz'].values
            axs[0].plot(ts, gxs_b, c='g')
            axs[1].plot(ts, gys_b, c='g')
            axs[2].plot(ts, gzs_b, c='g')

            fig_q, ax = plt.subplots(1, constrained_layout=True, sharex=True)
            fig_q.suptitle('Quaternion Components')
            ts = br_flight_log['t'].values
            qs_w = br_flight_log['qw'].values
            qs_x = br_flight_log['qx'].values
            qs_y = br_flight_log['qy'].values
            qs_z = br_flight_log['qz'].values

            ax.plot(ts, qs_w, label='z')
            ax.plot(ts, qs_x, label='z')
            ax.plot(ts, qs_y, label='z')
            ax.plot(ts, qs_z, label='z')
            ax.legend()

            fig_off_vertical, ax = plt.subplots(1, constrained_layout=True, sharex=True)
            fig_off_vertical.suptitle('Angle from Vertical')
            ts = br_flight_log['t'].values
            alphas = br_flight_log['off_vertical'].values
            ax.plot(ts, 57.3 * alphas, c='g')
            ax.axhline(0, c='c', alpha=0.5)

            plt.show()

#     fd = FlightData()
#     N = 10
#     for name in ['h', 'Vz', 'az']:
#         vals = [0]*N
#         fd[name] = vals

#     print(fd)
#     fd['dict_series'] = {'values':[1]*N, 'source':'Calculated', 'description':'A basic dict series test'}
#     fd['invalid_dict_series'] = {'values':[42]*N, 'errors':[2]*(N+1), 'description':'A dict series with data lengths that don\'t match'}

#     for series in fd.values():
#         print(series.valid, series)

#     pp_flight_log = loadProjectPotatoLog('./flight_logs/2024_05_18_test_flight_log_4.csv')
#     print(pp_flight_log.summary)
