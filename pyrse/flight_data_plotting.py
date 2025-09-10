import collections.abc as abc 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pyrse.flight_data as flight_data


def plotVectors(ax, ts, vs, label_fmt=None):
    xs = np.array([v.x for v in vs])
    ys = np.array([v.y for v in vs])
    zs = np.array([v.z for v in vs])

    ax.plot(ts, xs, label=label_fmt.format(axis='x') if label_fmt is not None else None)
    ax.plot(ts, ys, label=label_fmt.format(axis='y') if label_fmt is not None else None)
    ax.plot(ts, zs, label=label_fmt.format(axis='z') if label_fmt is not None else None)
    
    if label_fmt is not None:
        ax.legend()


def plotQuaternions(ax, ts, qs, label_fmt=None):
    ws = np.array([q.w for q in qs])
    xs = np.array([q.x for q in qs])
    ys = np.array([q.y for q in qs])
    zs = np.array([q.z for q in qs])

    ax.plot(ts, ws, label=label_fmt.format(axis='w') if label_fmt is not None else None)    
    ax.plot(ts, xs, label=label_fmt.format(axis='x') if label_fmt is not None else None)
    ax.plot(ts, ys, label=label_fmt.format(axis='y') if label_fmt is not None else None)
    ax.plot(ts, zs, label=label_fmt.format(axis='z') if label_fmt is not None else None)

    if label_fmt is not None:
        ax.legend()


def plotFlightOverview(fds, show_events=True, show_event_types = ['Ignition', 'Launch', 'Burnout', 'Apogee'], aligned=False, align_event_types = ['Ignition', 'Launch'], align_ref=None):
    if isinstance(fds, flight_data.FlightData):
        fds = [fds]
    
    t_offs = np.array([0] * len(fds)) if not aligned else flight_data.alignment_offsets(fds, event_types=align_event_types, ref=align_ref)
    fig, axs = plt.subplots(3, layout='constrained', sharex=True)

    axs[0].set_title('Altitude (m)')
    axs[0].axhline(0, c='k', alpha=0.2)

    axs[1].set_title('Vertical Velocity (m/s)')
    axs[1].axhline(0, c='k', alpha=0.2)

    axs[2].set_title('Vertical Acceleration (m/s^2)')
    axs[2].axhline(0, c='k', alpha=0.2)

    # print('t_offs = {}'.format(t_offs))
    fig.suptitle('Flight Overview')
    for idx, fd in enumerate(fds):
        t_shift = -t_offs[idx]
        ts = fd['t'].values + t_shift
        hs = fd['h'].values
        hs_raw = fd['hraw'].values
        vzs = fd['Vz'].values
        vzs_raw = fd['Vzraw'].values
        azs = fd['az'].values
        sns.lineplot(x=ts, y=hs, c='g', ax=axs[0], label='Filtered')
        sns.lineplot(x=ts, y=hs_raw, c='k', ax=axs[0], label='Raw')

        sns.lineplot(x=ts, y=vzs, c='g', ax=axs[1], label='Filtered')
        sns.lineplot(x=ts, y=vzs_raw, c='k', ax=axs[1], label='Raw')

        sns.lineplot(x=ts, y=azs, ax=axs[2], label='_')

        if show_events:
            for k, event in fd.events.items():       
                if (show_event_types is None) or (k in show_event_types):
                    t, c, alpha = (event.t + t_shift), event.color, 0.15
                    axs[0].axvline(t, c=c, alpha=alpha)
                    axs[1].axvline(t, c=c, alpha=alpha)
                    axs[2].axvline(t, c=c, alpha=alpha)
        for ax in axs:
            ax.legend()
    return fig

