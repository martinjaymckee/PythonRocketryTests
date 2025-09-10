import math

import ambiance
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import seaborn as sns


import pyrse.engines as engines


import eiffel_tower_v2_drag_est_processing as cd_est_proc


class CdEstRegressor:
    def __init__(self, results, debug=False):
        self.__processed = None
        self.__Vmin = None
        self.__Vmax = None
        #self.__cd_func = lambda x, a, b, c, d, e, f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f          
        self.__cd_func = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d        
        #self.__cd_func = lambda x, a, b, c: a*x**2 + b*x + c        
        #self.__cd_func = lambda x, a, b, c: a*np.exp(b*x + c)        
        
        self.__params = None
        self.__debug = debug

        self.__do_regression(results)
    
    @property
    def valid(self):
        return self.__processed

    @property
    def func(self):
        class wrapped_func:
            def __init__(self, Vmin, Vmax, func):
                self.__Vmin = Vmin
                self.__Vmax = Vmax
                self.__func = func

            def __call__(self, V):
                if V > self.__Vmax:
                    V = self.__Vmax
                if V < self.__Vmin:
                    V = self.__Vmin
                return self.__func(V)
            
        return wrapped_func(self.__Vmin, self.__Vmax, lambda V: self.__cd_func(V, *self.__params))

    def __do_regression(self, results, N=50000):
        xresults = []
        yresults = []
        fig, ax = None, None

        if self.__debug:
            fig, ax = plt.subplots(1, layout='constrained')

            fig.suptitle('$C_d$ vs. $V$ Regression')
            ax.set_xlabel(r'Velocity ($m s^{-1}$)')
            ax.set_ylabel(r'Coefficient of Drag')

        for result in results:
            cfg, ts, cds, Vs, Ts = result            
            if self.__debug:
                sns.scatterplot(x=Vs, y=cds, color=(0.80, 0.80, 0.80), alpha=0.1, ax=ax)

            Vmin, Vmax = np.min(Vs), np.max(Vs)
            if (self.__Vmin is None) or (Vmin < self.__Vmin):
                self.__Vmin = Vmin
            if (self.__Vmax is None) or (Vmax > self.__Vmax):
                self.__Vmax = Vmax

            xresults.append(np.array(Vs))
            yresults.append(np.array(cds))
        
        xdata = np.concatenate(xresults)
        ydata = np.concatenate(yresults)
        popt, pcov = optimize.curve_fit(self.__cd_func, xdata, ydata, maxfev=N)
        # print(popt)

        if self.__debug:
            Vs = np.linspace(self.__Vmin, self.__Vmax, 100)
            est_cds = [self.__cd_func(V, *popt) for V in Vs]
            sns.lineplot(x=Vs, y=est_cds, color='g', ax=ax)
            eq = ''
            print(popt)
            for idx, coeff in enumerate(popt):
                idx = len(popt) - idx - 1
                eq += (' + ' if coeff > 0 else ' - ') if len(eq) > 0 else ''
                coeff = abs(coeff) if len(eq) > 0 else coeff
                if idx == 0:
                    eq += '{:f}'.format(coeff)
                elif idx == 1:
                    eq += '{:f} V'.format(coeff)
                else:
                    eq += '{:f} V^{:d}'.format(coeff, idx)
            ax.text(0.2, 0.8, '$C_{{d}} = {}$'.format(eq), transform=ax.transAxes)
           # fig.savefig(r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images\part_II_regression_I65_J180_J145.png', dpi=300)


        self.__processed = True
        self.__params = popt


def sim_3d_flight(dt, t_ignition, eng, m0, Sref, vToCd, h0=1600, off_axis=None):
    g0 = -9.80665
    h = h0
    h_last = h
    Vx = 0
    Vz = 0
    az = 0
    t = t_ignition
    eng.start(t_ignition)

    ts = []
    azs = []
    Vxs = []
    Vzs = []
    hs = []
    ms = []
    alphas = []
    Ts = []
    Ds = []
    cds = []
    done = False
    ascent = False
    idx = 0
    t_burn = eng.burn_time
    while not done:
        m = eng.calc_mass(t) + m0
        T = eng.thrust(t)
        alpha = 0 if off_axis is None else off_axis[idx]
        ascent = True if not ascent and T > m*g0 else ascent
        D = 0
        cd = 0
        if ascent:
            #print(h)
            rho = ambiance.Atmosphere(h).density[0]
            V_mag = math.sqrt(Vx**2 + Vz**2)
            cd = vToCd(V_mag)
            D = 0.5 * rho * (V_mag**2) * Sref * cd
            a_aero = (T-D) / m
            az_aero = a_aero * math.cos(alpha)
            ax_aero = a_aero * math.sin(alpha)
            az = az_aero + g0
            ax = ax_aero
            #print('cd = {}, alpha = {}, m = {}, T = {}, D = {}, az = {}, Vz = {}'.format(cd, 57.3 * alpha, m, T, D, az, Vz))
            h = h + dt * Vz + (dt**2 / 2) * az
            Vx = Vx + dt * ax
            Vz = Vz + dt * az
            done = (t > t_burn) and (h < h_last)
            h_last = h
        ts.append(t)
        hs.append(h)
        azs.append(az+g0)
        Vxs.append(Vx)
        Vzs.append(Vz)
        ms.append(m)
        Ts.append(T)
        Ds.append(D)
        alphas.append(alpha)
        cds.append(cd)
        t += dt
        idx += 1
    ts = np.array(ts)
    azs = np.array(azs)
    Vxs = np.array(Vxs)
    Vzs = np.array(Vzs)
    hs = np.array(hs)
    alphas = np.array(alphas)
    Ts = np.array(Ts)
    Ds = np.array(Ds)
    cds = np.array(cds)

    return ts, azs, Vzs, hs, Ts, Ds, cds, alphas 


if __name__ == '__main__':
    estimation_output_dir = 'D:\Workspace\Rockets\HPR\Eiffel Tower v2\Cd Estimation Outputs'    
    input_handler = cd_est_proc.CdEstFileHandler()

    cd_est_results = input_handler.load_dir(estimation_output_dir)

    regressor = CdEstRegressor(cd_est_results, debug = True)

    dt = .02
    t_ignition = 0
    #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_I65W.rse"        
    engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_K185W.rse"
    #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_J180T.rse"
    #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\Cesaroni_699J145-19A.rse"      
    #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_K270W.rse"
    # engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_K270W.eng"
    #engine_path = r"D:\Workspace\Rockets\PythonRocketryTests\Engines\AeroTech_K400C.rse"
   
    eng = engines.Engine.RSE(engine_path)  
    # eng = engines.Engine.RASP(engine_path)  

    h0 = 2300
    m0 = 953/1000 #750 / 1000 # kg
    Sref = .155 * .155 # m^2
    ts, azs, Vzs, hs, Ts, Ds, cds, alphas = sim_3d_flight(dt, t_ignition, eng, m0, Sref, regressor.func, h0=h0) 

    print('Maximum Altitude (agl) = {} ft, Maximum Velocity = {} m/s'.format(3.28 * (np.max(hs) - h0), np.max(Vzs)))

    plt.show()