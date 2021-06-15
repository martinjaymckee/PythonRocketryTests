# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:28:46 2020

@author: marti
"""

# import math

import matplotlib.pyplot as plt
import numpy as np
# import scipy
# import scipy.interpolate

import engines
import rocket_components

orbiter_engine_file = '../Apogee_F10.eng'
srb_engine_file = '../Cesaroni_F30.eng'


def plotEngine(ax, ts, eng):
    Ts = []
    ws = []
    for t in ts:
        T = eng.thrust(t)
        w = eng.mass(t)
        Ts.append(T)
        ws.append(w)
    ax.plot(ts, Ts, '-b', label='{} Thrust (N)'.format(eng.name))
    ax.axhline(eng.average_thrust, c='b', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(ts, ws, '-g', label='{} Mass (g)'.format(eng.name))
    ax.legend()


def getTWR(ship, t, engines):
    T_tot = 0
    for eng in engines:
        T = eng.thrust(t)
        T_tot += T
    return T_tot / (9.80655 * ship.mass(t))


def plotFlight(ts, configurations, engines):
    env = rocket_components.Environment()
    g = 9.80655
    t_last = ts[0]
    v = 0
    h = 0
    Ts = []
    Ds = []
    cgs = []
    mmois = []
    TWRs = []
    ws = []
    accels = []
    vs = []
    hs = []
    ts_stage = []

    for t in ts:
        ship = None
        for t_end, configuration in configurations:
            if t < t_end:
                if (len(ts_stage) == 0) or (ts_stage[-1] < t_end):
                    ts_stage.append(t_end)
                ship = configuration
                break
        T = 0
        for eng in engines:
            T_i = eng.thrust(t)
            T += T_i
        D = ship.drag(v, env)

        mass = ship.mass(t)
        cg = ship.cg(t)
        mmoi = ship.mmoi(t)
        dt = t - t_last
        a = ((T-D) / mass) - g
        h += ((dt*v) + (0.5*a*(dt**2)))
        v += (dt*a)
        t_last = t

        Ts.append(T)
        Ds.append(D)
        TWR = getTWR(ship, t, engines)
        TWRs.append(TWR)
        ws.append(mass)
        cgs.append(cg)
        mmois.append(mmoi)

        accels.append(a)
        vs.append(v)
        hs.append(h)

    fig, axs = plt.subplots(7, 1, figsize=(15, 12), sharex=True)

    axs[0].set_title('TWR')
    axs[0].plot(ts, TWRs, '-b')
    axs[0].axhline(1.0, c='b', alpha=0.25)
    axs[1].set_title('Drag (blue) - Thrust (green)')
    axs[1].plot(ts, Ds, '-b')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(ts, Ts, '-g')
    xs = [cg[0] for cg in cgs]
    zs = [cg[2] for cg in cgs]
    axs[2].set_title('CG.x (blue) - CG.y (green)')
    axs[2].plot(ts, xs, '-b')
    ax2_twin = axs[2].twinx()
    ax2_twin.plot(ts, zs, '-g')
    axs[3].set_title('Velocity (blue) - Altitude (green)')
    axs[3].plot(ts, vs, '-b')
    ax3_twin = axs[3].twinx()
    ax3_twin.plot(ts, hs, '-g')
    xs = [mmoi[0] for mmoi in mmois]
    ys = [mmoi[1] for mmoi in mmois]
    zs = [mmoi[2] for mmoi in mmois]
    axs[4].set_title('MMOI.x (blue) - MMOI.y (green) - MMOI.z (cyan)')
    axs[4].plot(ts, xs, '-b')
    axs[5].plot(ts, ys, '-g')
    axs[6].plot(ts, zs, '-c')

    ax2_twin = axs[2].twinx()
    for ax in axs:
        for t in ts_stage:
            if t <= ts[-1]:
                ax.axvline(t, c='m', alpha=0.25)


external_tank = rocket_components.AeroBody(.75, .147, .3, .295, pos=np.array([-.375, 0, 0]))
srb_left = rocket_components.AeroBody(.726, .066, .15, .295, pos=np.array([-.539, -0.132, 0]))
srb_right = rocket_components.AeroBody(.726, .066, .15, .295, pos=np.array([-.539, 0.132, 0]))
orbiter = rocket_components.AeroBody(.594, .099, .35, .31, pos=np.array([-.572, 0, .1375]))

orbiter_engine = engines.Engine.RASP(orbiter_engine_file)
orbiter_engine.pos = np.array([-.242, 0, 0])

srb_left_engine = engines.Engine.RASP(srb_engine_file)
srb_right_engine = engines.Engine.RASP(srb_engine_file)
srb_offset = np.array([-.35, 0, 0])
srb_right_engine.pos = np.array([-.35, 0, 0])
srb_left_engine.pos = np.array([-.35, 0, 0])

print('SRB Engine Mass = {} g, Orbiter Engine Mass = {} g'.format(srb_left_engine.mass(0)*1000, orbiter_engine.mass(0)*1000))

orbiter.add(orbiter_engine)
srb_left.add(srb_left_engine)
srb_right.add(srb_right_engine)

full_stack = rocket_components.EmptyComponent(children=[external_tank, srb_left, srb_right, orbiter])
sustainer = rocket_components.EmptyComponent(children=[external_tank, orbiter])

#ts = np.linspace(0, max(orbiter_engine.burn_time, srb_left_engine.burn_time), 10)
#print(ts)

#fig, axs = plt.subplots(2, 1, figsize = (15, 10), sharey=True)

#plotEngine(axs[0], ts, orbiter_engine)
#plotEngine(axs[1], ts, srb_left_engine)
#plt.show()

print('Launch Mass = {} g'.format(1000 * full_stack.mass(0)))
print('Launch CG = {}'.format(full_stack.cg(0)))
print('Launch TWR = {}'.format(getTWR(full_stack, 0.1, [orbiter_engine, srb_left_engine, srb_right_engine])))
print('SRB Burnout CG = {}'.format(full_stack.cg(srb_left_engine.burn_time)))
print('SRB Burnout TWR = {}'.format(getTWR(full_stack, srb_left_engine.burn_time, [orbiter_engine, srb_left_engine, srb_right_engine])))
print('Sustainer Initial CG = {}'.format(sustainer.cg(srb_left_engine.burn_time)))
print('Sustainer Initial TWR = {}'.format(getTWR(sustainer, srb_left_engine.burn_time, [orbiter_engine, srb_left_engine, srb_right_engine])))
print('Sustainer Final CG = {}'.format(sustainer.cg(srb_left_engine.burn_time)))

for child in full_stack.children:
    print('{} -> {} g'.format(child, 1000*child.mass(0)))
    for subchild in child.children:
        print('\t{} -> {} g'.format(subchild, 1000*subchild.mass(0)))

ts = np.linspace(0, max(orbiter_engine.burn_time, srb_left_engine.burn_time), 100)

plotFlight(
        ts,
        [
                (srb_left_engine.burn_time+0.25, full_stack),
                (10.0, sustainer)
        ],
        (orbiter_engine, srb_left_engine, srb_right_engine)
)

plt.show()
