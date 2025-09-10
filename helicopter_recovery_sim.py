
import math

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import numpy as np
import scipy.integrate as integrate

import pyrse.aero
import pyrse.airfoil

class BladeAssembly:
    def __init__(self, num, l, c, theta, t, r_min=0.01, blade_density=415.0):
        self.__num = num
        self.__l = l
        self.__t = t
        self.__c = c
        self.__theta = theta
        self.__r_min = r_min # If r_min is None, calculate this from the c value and num
        self.__blade_density = blade_density

    @property
    def num(self):
        return self.__num
    
    @property
    def r_min(self):
        return self.__r_min
    
    @property
    def r_max(self):
        return self.__l
    
    @property
    def density(self):
        return self.__blade_density
    
    @property
    def mass(self):
        rho_mat = self.__blade_density
        def integrand(r):
            return rho_mat * self.t(r) * self.c(r)
        
        m, _ = integrate.quad(integrand, self.__r_min, self.__l)
        return self.__num * m
        
    @property
    def mmoi(self):
        rho_mat = self.__blade_density
        def integrand(r):
            return rho_mat * self.t(r) * self.c(r) * (r**2)
        
        I, _ = integrate.quad(integrand, self.__r_min, self.__l)

        return self.__num * I

    @property
    def mmoi_deploy(self):
        r_min = self.__r_min
        rho_mat = self.__blade_density
        def integrand(r):
            return rho_mat * self.t(r) * self.c(r) * ((r-r_min)**2)
        
        I, _ = integrate.quad(integrand, r_min, self.__l)

        return I
    
    def t(self, r):
        try:
            return self.__t(r)
        except:
            pass
        return self.__t
    
    def c(self, r):
        try:
            return self.__c(r)
        except:
            pass
        return self.__c
    
    def theta(self, r):
        try:
            return self.__theta(r)
        except:
            pass
        return self.__theta    
    

class HelicopterDescentState:
    def __init__(self, t, Vd, omega):
        self.t = t
        self.h = None
        self.Vd = Vd
        self.a = None
        self.omega = omega
        self.domega = None
        self.lift = None
        self.Q_net = None
        self.angle_of_attack_distribution = None
        self.velocity_distribution = None
        self.re_distribution = None
        self.cl_distribution = None
        self.lift_distribution = None
        self.cd_distribution = None
        self.drag_distribution = None
        self.torque_distribution = None


class HelicopterDescentSimulation:
    def __init__(self, blades, mtot):
        self.__blades = blades
        self.__mtot = mtot
        self.__env = pyrse.aero.StandardISAConditions

    def __call__(self, dt=0.01, N_discritize=100, t_min=1.0, t_timeout=30):
        blades = self.__blades
        env = self.__env
        done = False
        t = 0 # s
        h = 0 # m
        Vd = 0 # m/s
        omega = 0 # rad/s
        g0 = 9.80665 # m/s^2

        rs = np.linspace(blades.r_min, blades.r_max, N_discritize)
        dr = (blades.r_max - blades.r_min) / N_discritize
        airfoils = [pyrse.airfoil.FlatPlateAirfoil(blades.t(r)/blades.c(r)) for r in rs]

        results = []
        mmoi = blades.mmoi

        while not done:
            L = 0
            Q_net = 0
            angle_of_attack_distribution = []
            velocity_distribution = []
            re_distribution = []
            cl_distribution = []
            lift_distribution = []
            cd_distribution = []
            torque_distribution = []

            for r, airfoil in zip(rs, airfoils):
                Vr = omega*r
                inflow_angle = math.atan2(-Vd, Vr)
                aoa = inflow_angle + blades.theta(r)
                angle_of_attack_distribution.append(aoa)

                Vmag = math.sqrt(Vd**2 + Vr**2)
                velocity_distribution.append(Vmag)
                c = blades.c(r)
                
                Re = pyrse.aero.Re(Vmag, c)
                re_distribution.append(Re)

                Cl = airfoil.Cl(aoa, Re)
                Cd = airfoil.Cd(aoa, Re)
                cl_distribution.append(Cl)
                cd_distribution.append(Cd)

                dL = 0.5 * env.rho * (Vmag**2) * c * Cl * dr
                dD = 0.5 * env.rho * (Vmag**2) * c * Cd * dr
                dT = (dL*math.cos(inflow_angle)) + (dD * math.sin(inflow_angle))
                L += dT
                dQ = r * ((dL * math.sin(inflow_angle) - dD*math.cos(inflow_angle)))
                # print('Vd = {}, Vr = {}, dL = {}, dD = {}, inflow = {}, cos(inflow) = {}, sin(inflow) = {}'.format(Vd, Vr, dL, dD, math.degrees(inflow_angle), math.cos(inflow_angle), math.sin(inflow_angle)))
                # print('aoa = {}, inflow = {}, Vd = {}, Vr = {}, Cl = {}, Cd = {}, dT = {}, dQ = {}'.format(math.degrees(aoa), math.degrees(inflow_angle), Vd, Vr, Cl, Cd, dT, dQ))                
                Q_net += dQ                
                lift_distribution.append(dT)
                torque_distribution.append(dQ)

            
            a = (L / self.__mtot) - g0
            h = h + (dt * Vd) + ((dt**2 / 2) * a)
            Vd += dt * a
            domega = Q_net / mmoi
            omega += dt * domega
            t += dt
            # print('t = {}, L = {}, a = {}, Vd = {}, domega = {}, omega = {}, Q_net = {}, mmoi = {}'.format(t, L, a, Vd, domega, omega, Q_net, mmoi))

            sample_state = HelicopterDescentState(t, Vd, omega)
            sample_state.a = a
            sample_state.domega = domega
            sample_state.Q_net = Q_net
            sample_state.angle_of_attack_distribution = np.array(angle_of_attack_distribution)
            sample_state.velocity_distribution = np.array(velocity_distribution)
            sample_state.re_distribution = np.array(re_distribution)
            sample_state.cl_distribution = np.array(cl_distribution)
            sample_state.lift_distribution = np.array(lift_distribution)
            sample_state.cd_distribution = np.array(cd_distribution)
            sample_state.torque_distribution = np.array(torque_distribution)
            sample_state.h = h
            results.append(sample_state) 
            done = (t > t_min) and ((abs(Q_net) < 1e-4) or (t > t_timeout)) 
    
        return rs, results


def find_settle_idx(xs, tol=0.01):
    x_tgt = xs[-1]
    abs_tol = abs(x_tgt) * tol

    deviation = np.abs(xs - x_tgt)
    within_tol = deviation <= abs_tol

    for i in range(len(xs)):
        if np.all(within_tol[i:]):
            return i
    return None


def blade_energy(mmoi, omega):
    return 0.5 * mmoi * (omega**2)


class HelicopterDeploymentResult:
    def __init__(self):
        self.V = None
        self.ts = []
        self.thetas = []
        self.omegas = []
        self.Es = []
        self.Qs = []
        self.F_max = None


class HelicopterDeploymentSim:
    def __init__(self, blades):
        self.__blades = blades
        self.__env = pyrse.aero.StandardISAConditions

    def __call__(self, v_range, dt = 0.001, N_v=25, N_discritize=100, t_decel=0.0005, V_min = 0.5, theta_deployed=math.pi/2, theta_min=math.radians(5)):
        results = []
        Cd = 1.28 # NOTE: THIS IS APPROXIMATELY THE DRAG FOR A FLAT PLATE. THIS IS A WORST-CASE CONDITION FOR THE FORCES, BUT IS LIKLEY TO UNDERESTIMATE THE DEPLOYMENT TIME
        env = self.__env
        blades = self.__blades
        density = blades.density
        Vs = np.linspace(max(V_min, v_range[0]), v_range[1], N_v)
        rs = np.linspace(blades.r_min, blades.r_max, N_discritize)
        dr = (blades.r_max - blades.r_min) / N_discritize
        r_min = blades.r_min
        Q_additional = 0.1 # NOTE: THIS IS FOR ADDING RUBBRBAND TENSION OR SIMILAR
        for V in Vs:
            theta = theta_min
            omega = 0
            t = 0          
            result = HelicopterDeploymentResult()
            result.V = V
            # print(f'V = {V}m/s')

            while theta < theta_deployed:
                E = 0
                Q = 0               
                sin_theta = math.cos(theta)
                dr_eff = sin_theta * dr

                #print(sin_theta, dr_eff, rs)
                for r in rs:
                    c = blades.c(r)
                    dQ = 0.5 * env.rho * (V**2) * c * dr_eff * Cd * sin_theta * (r - r_min)
                    Q += dQ
                    #print(env.rho, V, c, dr_eff, (r-r_min))
                    dE = 0.5 * density * c * dr_eff * ((r - r_min)**2) * (omega**2)
                    # print(f'density = {density}, c = {c}, dr_eff = {dr_eff}, r_eff = {r - r_min}, omega = {omega}')
                    # print(dE)
                    E += dE
                # print(E)
                Q += Q_additional
                t += dt
                omega += dt * (Q / blades.mmoi_deploy)
                theta += dt * omega
                # print(f'\ttheta = {theta}, omega = {omega}, mmoi = {blades.mmoi_deploy}, Q = {Q}')
                result.ts.append(t)
                result.Es.append(E)
                result.Qs.append(Q)
                result.thetas.append(theta)
                result.omegas.append(omega)
            result.ts = np.array(result.ts)
            result.Es = np.array(result.Es)
            result.Qs = np.array(result.Qs)
            result.thetas = np.array(result.thetas)
            result.omegas = np.array(result.omegas)
            result.F_max = np.max(result.Qs) / (np.max(result.omegas) * t_decel)
            results.append(result)
            # NOTE: THE SHEAR STRESS REQUIREMENTS OF THE PIN CAN BE CALCULATED AS, F_MAX / (PIN AREA) WHERE THE SHEAR STRENGTH IS OFTEN ~0.6 X YIELD STRENGTH
        return results


if __name__ == '__main__':
    L_tot = 0.615
    fps = 180

    def blade_chord(r):
        C0 = 0.060 #0.070
        r0 = C0 / 2
        C1 = 0.030 #0.035
        r1 = C1 / 2
        L_taper = 0.140
        R = ((L_taper**2) + ((r0 - r1)**2)) / (2*(r0 - r1))

        r_taper_begin = L_tot - L_taper
        if r <= r_taper_begin:
            return C0
        elif r > L_tot:
            return 0
        else:
            x = (r - r_taper_begin)
            return 2*(math.sqrt(R**2 - x**2) - R) + C0
               
    blades = BladeAssembly(3, L_tot, blade_chord, math.radians(-12), .003, r_min=0.090)

    sim = HelicopterDescentSimulation(blades, 0.3605)

    rs, results = sim(dt=0.01)
    # print('Num Results = {}'.format(len(results)))
    ts = np.array([result.t for result in results])
    hs = np.array([result.h for result in results])    
    Vds = np.array([result.Vd for result in results])
    omegas = 360 *  np.array([result.omega for result in results]) / (2*math.pi)

    fig_summary, axs_summary = plt.subplots(3, layout='constrained', sharex=True)
    axs_summary[0].plot(ts, hs)
    axs_summary[1].plot(ts, Vds)
    axs_summary[2].plot(ts, omegas)
    idx_vd_settle = find_settle_idx(Vds)
    if idx_vd_settle is not None:
        t_vd_settle = ts[idx_vd_settle]
        axs_summary[0].axvline(t_vd_settle)
        axs_summary[1].axvline(t_vd_settle)
        axs_summary[2].axvline(t_vd_settle)
        print('Settled State (@ {} s): Altitude Drop = {} m, Vertical Rate = {} m/s, Angular Rate = {} Hz'.format(t_vd_settle, hs[idx_vd_settle], Vds[idx_vd_settle], omegas[idx_vd_settle]))

    fig_torque = plt.figure(layout='constrained')
    ax_torque = fig_torque.add_subplot(projection='3d')
    ax_torque.set_xlabel('Span Location (m)')
    ax_torque.set_ylabel('Time (s)')
    ax_torque.set_zlabel('Torque Density (Nm)')

    N_torques = min(50, len(results)) if idx_vd_settle is None else idx_vd_settle+1

    X, Y = np.meshgrid(rs, ts)
    Z = []
    for idx_t in range(len(ts)):
        Z.append(results[idx_t].torque_distribution)
    Z = np.array(Z)
    ax_torque.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

    # for idx in range(N_torques):
    #     ys = np.array([ts[idx]]*len(rs)) # STILL NEED TO FIGURE OUT HOW TO PLOT THIS....
    #     torque_distribution = results[idx].torque_distribution
    #     colors = 'm' #[1 if Q > 0 else 0 for Q in torque_distribution]

    #     ax_torque.plot(rs, ys, torque_distribution, c=colors, alpha=0.25)#, cmap='plasma')
    #ax_torque.axhline(0)

    print('Final: Descent Rate = {} m/s, Frames per Revolution = {}, Blade Energy = {} J'.format(Vds[-1], fps/omegas[-1], blade_energy(blades.mmoi, 2 * math.pi * omegas[-1])))

    deployment = HelicopterDeploymentSim(blades)
    results = deployment([0, 30])
    strength_max = 350e6 # Pa
    fig_deploy, axs_deploy = plt.subplots(3, layout='constrained', sharex=True)
    Vs = np.array([r.V for r in results])
    ts = np.array([np.max(r.ts) for r in results])
    Fs_max = np.array([r.F_max for r in results])
    ds_req = 2 * np.sqrt((2 * Fs_max / strength_max) / math.pi)
    axs_deploy[0].plot(Vs, ts)
    axs_deploy[0].set_ylabel('Deployment Time (s)')
    axs_deploy[1].plot(Vs, Fs_max)
    axs_deploy[1].set_ylabel('Stopping Force (N)')
    axs_deploy[2].plot(Vs, 1000 * ds_req)
    axs_deploy[2].set_xlabel('Velocity (m/s)')
    axs_deploy[2].set_ylabel('Required Pin Diameter (mm)')
    plt.show()



    

