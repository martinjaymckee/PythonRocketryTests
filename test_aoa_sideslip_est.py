import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R

class RocketEKFJacobian:
    def __init__(self, mass, thrust_curve, S_ref, rho=1.225, dt=0.002):
        self.mass = mass
        self.thrust_curve = thrust_curve
        self.S = S_ref
        self.rho = rho
        self.dt = dt

        # State: vx, vy, vz, phi, theta, psi
        self.x = np.zeros(6)
        self.P = np.eye(6)*0.1

        # Vertical position (integrated separately)
        self.z_pos = 0.0

        # Process and measurement noise
        self.Q = np.eye(6)*0.01
        self.R_accel = np.eye(3)*0.1
        self.R_baro = 0.5**2

    def set_aero_coefficients(self, C_l_func, C_d_func, C_y_func):
        self.C_l_func = C_l_func
        self.C_d_func = C_d_func
        self.C_y_func = C_y_func

    def euler_to_rot(self, phi, theta, psi):
        return R.from_euler('xyz', [phi, theta, psi]).as_matrix()

    def compute_aero_forces(self, vx, vy, vz):
        V = np.linalg.norm([vx, vy, vz])
        if V < 1e-6:
            return np.zeros(3)
        alpha = np.arctan2(vz, vx)
        beta = np.arcsin(vy / V)

        F_drag = 0.5 * self.rho * V**2 * self.S * self.C_d_func(alpha)
        F_lift = 0.5 * self.rho * V**2 * self.S * self.C_l_func(alpha)
        F_side = 0.5 * self.rho * V**2 * self.S * self.C_y_func(beta)

        return np.array([-F_drag, F_side, -F_lift])

    def f(self, x, u, t):
        vx, vy, vz, phi, theta, psi = x
        p, q, r = u
        m = self.mass(t)
        T = self.thrust_curve(t)

        F_aero = self.compute_aero_forces(vx, vy, vz)
        Rb2i = self.euler_to_rot(phi, theta, psi)
        g_b = Rb2i.T @ np.array([0,0,9.81])
        F_thrust = np.array([T,0,0])
        v_dot = (F_thrust + F_aero - m*g_b)/m - np.cross([p,q,r],[vx,vy,vz])

        # Euler angles
        phi_dot = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
        theta_dot = np.cos(phi)*q - np.sin(phi)*r
        psi_dot = np.sin(phi)/np.cos(theta)*q + np.cos(phi)/np.cos(theta)*r

        vx += v_dot[0]*self.dt
        vy += v_dot[1]*self.dt
        vz += v_dot[2]*self.dt
        phi += phi_dot*self.dt
        theta += theta_dot*self.dt
        psi += psi_dot*self.dt

        return np.array([vx, vy, vz, phi, theta, psi])

    def compute_F_jacobian(self, x, u, t, eps=1e-6):
        F = np.zeros((6,6))
        fx = self.f(x, u, t)
        for i in range(6):
            x_eps = x.copy()
            x_eps[i] += eps
            fx_eps = self.f(x_eps, u, t)
            F[:,i] = (fx_eps - fx)/eps
        return F

    def ekf_predict(self, u, t):
        self.x = self.f(self.x, u, t)
        F = self.compute_F_jacobian(self.x, u, t)
        self.P = F @ self.P @ F.T + self.Q
        self.z_pos += self.x[2]*self.dt  # integrate vz for altitude

    def ekf_update_accel(self, a_meas):
        H = np.zeros((3,6))
        H[:,0:3] = np.eye(3)
        y = a_meas - self.x[0:3]
        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def ekf_update_baro(self, h_meas):
        H = np.zeros((1,6))
        H[0,2] = 1.0  # measurement corresponds to vz integration
        y = h_meas - self.z_pos
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T / S
        self.x += (K * y).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P
        print (dt*(K * y)[2])
        self.z_pos += (K * y)[2]  # correct integrated altitude

    def get_aero_angles(self):
        vx, vy, vz = self.x[0:3]
        V = np.linalg.norm([vx, vy, vz])
        if V < 1e-6:
            return 0.0, 0.0
        alpha = np.arctan2(vz, vx)
        beta = np.arcsin(vy / V)
        return alpha, beta



if __name__ == '__main__':
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    from pyrse.analysis.coefficients import CoefficientMapping
    import pyrse.analysis.regression as pyrse_regress
    import pyrse.engines as engines
    from pyrse.flight_data import loadOpenRocketExport, loadBlueRavenLog


    m0 = 0.750
    eng = engines.Engine.RSE(r'D:\User Data\Documents\Rockets\PythonRocketryTests\Engines\AeroTech_G74W.rse')
    or_flight_data = loadOpenRocketExport(r"D:\User Data\Documents\Rockets\HPR\Saturn I Block 2 (SA-5)\OpenRocket\Exports\or_sa_5_g74_1.csv")
    br_flight_data = loadBlueRavenLog(
                                        r"D:\User Data\Documents\Rockets\HPR\Saturn I Block 2 (SA-5)\Flight Data\Boilerplate\Flight 1\MJM SA-5_summary_09-06-2025_10_03_11_.csv",
                                        r"D:\User Data\Documents\Rockets\HPR\Saturn I Block 2 (SA-5)\Flight Data\Boilerplate\Flight 1\MJM SA-5 LR_09-06-2025_10_03_11.csv",
                                        r"D:\User Data\Documents\Rockets\HPR\Saturn I Block 2 (SA-5)\Flight Data\Boilerplate\Flight 1\MJM SA-5 HR_09-06-2025_10_03_11.csv"
                                    )   
    br_flight_data.updateEvents()

    marker_size = 10
    num_est_points = 25
    fig, axs = plt.subplots(3, layout='constrained')

    cd_mapping = CoefficientMapping.FromFlightData(or_flight_data, 'Cd', ['aoa'])#, pyrse_regress.KNearestNeighborRegressor())
    cl_mapping = CoefficientMapping.FromFlightData(or_flight_data, 'Cl', ['aoa'])#, pyrse_regress.KNearestNeighborRegressor())
    cm_mapping = CoefficientMapping.FromFlightData(or_flight_data, 'Cm', ['aoa'])#, pyrse_regress.KNearestNeighborRegressor())

    aoas = or_flight_data['aoa'].values
    est_aoas = np.linspace(0, np.nanmax(aoas), num_est_points)
    
    axs[0].scatter(57.3 * aoas, or_flight_data['Cd'].values, s=marker_size)
    cds_est = np.array([cd_mapping({'aoa': alpha})[0] for alpha in est_aoas])
    axs[0].scatter(57.3 * est_aoas, cds_est, s=marker_size)

    axs[1].scatter(57.3 * aoas, or_flight_data['Cl'].values, s=marker_size)
    cls_est = np.array([cl_mapping({'aoa': alpha})[0] for alpha in est_aoas])
    axs[1].scatter(57.3 * est_aoas, cls_est, s=marker_size)

    axs[2].scatter(57.3 * aoas, or_flight_data['Cm'].values, s=marker_size)
    cms_est = np.array([cm_mapping({'aoa': alpha})[0] for alpha in est_aoas])
    axs[2].scatter(57.3 * est_aoas, cms_est, s=marker_size)

    # Define thrust and mass
    def thrust(t): return eng.thrust(t)
    def mass(t): return eng.calc_mass(t) + m0

    # Aerodynamic coefficients
    def C_l(alpha): return cl_mapping({'aoa': alpha})
    def C_d(alpha): return cd_mapping({'aoa': alpha})
    def C_y(beta): return cm_mapping({'aoa': 0})

    fig_summary, axs_summary = plt.subplots(3, layout='constrained', sharex=True)
    fig_summary.suptitle('Flight Summary')

    ts = br_flight_data['t'].values
    axs_summary[0].plot(ts, br_flight_data['az'].values, label='az')
    axs_summary[0].set_title('Vertical Acceleration')
    axs_summary[0].set_ylabel('Acceleration ($m/s^2$)')
    axs_summary[1].plot(ts, br_flight_data['Vz'].values, label='Vz')
    axs_summary[1].set_title('Vertical Velocity')
    axs_summary[1].set_ylabel('Velocity ($m/s$)')
    axs_summary[2].plot(ts, br_flight_data['h'].values, label='h')
    axs_summary[2].set_title('Altitude')
    axs_summary[2].set_ylabel('Altitude ($m$)')
    axs_summary[2].set_xlabel('Time (s)')
    axs_summary[0].grid()

    print(br_flight_data.events)
    for idx, (name, evt) in enumerate(br_flight_data.events.items()):
        for ax in axs_summary:
            t = evt.t
            c = evt.color
            ax.axvline(t, color=c, linestyle='--')
            height = ax.get_ylim()[1] * 0.9 if (idx % 2) == 0 else ax.get_ylim()[1] * 0.4
            ax.text(t, height, name, rotation=90, verticalalignment='top')

    fig_detail, axs_detail = plt.subplots(3, layout='constrained', sharex=True)
    fig_detail.suptitle('Ascent Summary')

    ts = br_flight_data['t'].values
    t_apogee = br_flight_data.events['Apogee'].t if 'Apogee' in br_flight_data.events else ts[-1]
    idx_apogee = np.searchsorted(ts, t_apogee)
    ts = ts[:idx_apogee]
    axs_detail[0].plot(ts, br_flight_data['az'].values[:idx_apogee], label='az')
    axs_detail[0].set_title('Vertical Acceleration')
    axs_detail[0].set_ylabel('Acceleration ($m/s^2$)')
    axs_detail[1].plot(ts, br_flight_data['Vz'].values[:idx_apogee], label='Vz')
    axs_detail[1].set_title('Vertical Velocity')
    axs_detail[1].set_ylabel('Velocity ($m/s$)')
    axs_detail[2].plot(ts, br_flight_data['h'].values[:idx_apogee], label='h')
    axs_detail[2].set_title('Altitude')
    axs_detail[2].set_ylabel('Altitude ($m$)')
    axs_detail[2].set_xlabel('Time (s)')
    axs_detail[0].grid()

    for idx, (name, evt) in enumerate(br_flight_data.events.items()):
        for ax in axs_detail:
            t = evt.t
            c = evt.color
            ax.axvline(t, color=c, linestyle='--')
            height = ax.get_ylim()[1] * 0.9 if (idx % 2) == 0 else ax.get_ylim()[1] * 0.4
            ax.text(t, height, name, rotation=90, verticalalignment='top')
    plt.show()

    # ekf = RocketEKFJacobian(mass, thrust, S_ref=0.05)
    # ekf.set_aero_coefficients(C_l, C_d, C_y)

    # dt = 0.002
    # time = np.arange(0,5,dt)
    # alphas = []
    # betas = []
    # for t in time:
    #     gyro = np.array([0,0,0])  # replace with measurements
    #     accel = np.array([0,0,0]) # replace with measurements
    #     baro = 0                   # replace with altitude measurement

    #     ekf.ekf_predict(gyro, t)
    #     ekf.ekf_update_accel(accel)
    #     ekf.ekf_update_baro(baro)

    #     alpha, beta = ekf.get_aero_angles()
    #     alphas.append(alpha)
    #     betas.append(beta)
    #     print(f"t={t:.2f} s, AoA={np.degrees(alpha):.2f} deg, Sideslip={np.degrees(beta):.2f} deg")
    # alphas = np.array(alphas)
    # betas = np.array(betas)

    # fig, axs = plt.subplots(2, layout='constrained')
    # axs[0].plot(time, alphas)
    # axs[1].plot(time, betas)

    # plt.show()