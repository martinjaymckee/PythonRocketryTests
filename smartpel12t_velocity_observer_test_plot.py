import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parseTestFiles(path):
    columns = ['test', 'N', 't', 'dt', 'ref_pos', 'ref_vel', 'meas_pos', 'meas_dt', 'est_pos', 'est_vel', 't_calc']
    tests = {}

    def add_test(idx, observer_name, df):
        if idx in tests:
            tests[idx].append( (observer_name, df) )
        else:
            tests[idx] = [ (observer_name, df) ]

    for file in os.listdir(path):
        head, ext = os.path.splitext(file)
        if ext == ".csv":
            observer_name = head.split('_')[-1]
            src = os.path.join(path, file)
            df = pd.read_csv(src, names=columns, comment='#', skip_blank_lines=True)
            for group in df.groupby('test'):
                add_test(group[0], observer_name, group[1])
    return tests


def plot_tests(idx, tests):
    fig, axs = plt.subplots(3, figsize=(16,9), sharex=True)
    fig.suptitle('Test {} Results'.format(idx))
    axs[0].set_title('Position Error (steps)')
    ax0_twin = axs[0].twinx()
    axs[1].set_title('Velocity Error (steps-$s^{-1}$)')
    ax1_twin = axs[1].twinx()
    axs[2].set_title(r'Processing Time ($\mu{}s$)')
    for (observer_name, df) in tests:
        pos_err = df['ref_pos']-df['est_pos']
        sns.lineplot(x=df['t'], y=pos_err, alpha=0.5, ax=axs[0], label=observer_name)
        sns.lineplot(x=df['t'], y=df['ref_pos'], alpha=0.1, color='k', ax=ax0_twin, label=None)
        sns.lineplot(x=df['t'], y=df['est_pos'], alpha=0.1, color='g', ax=ax0_twin, label=None)
        vel_err = df['ref_vel']-df['est_vel']
        sns.lineplot(x=df['t'], y=vel_err, alpha=0.5, ax=axs[1], label=observer_name)
        sns.lineplot(x=df['t'], y=df['ref_vel'], alpha=0.1, color='k', ax=ax1_twin, label=None)
        sns.lineplot(x=df['t'], y=df['est_vel'], alpha=0.1, color='g', ax=ax1_twin, label=None)
        sns.lineplot(x=df['t'], y=1e6*df['t_calc'], alpha=0.5, ax=axs[2], label=observer_name)
    fig.tight_layout()


if __name__ == '__main__':
    path = r'C:\Users\marti\Documents\Workspace\Kicad\BoardAWeekProjects\StagingTimer\BasicStagingTimer\LPC845_BasicStagingTimer_Workspace\SmartPEL12_Velocity_Estimator_Test'
    parsed_tests = parseTestFiles(path)

    for idx, tests in parsed_tests.items():
        plot_tests(idx, tests)

    plt.show()
