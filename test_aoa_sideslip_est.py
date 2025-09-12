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
        self.z_pos += (K * y).item()  # correct integrated altitude

    def get_aero_angles(self):
        vx, vy, vz = self.x[0:3]
        V = np.linalg.norm([vx, vy, vz])
        if V < 1e-6:
            return 0.0, 0.0
        alpha = np.arctan2(vz, vx)
        beta = np.arcsin(vy / V)
        return alpha, beta



# Define thrust and mass
def thrust(t): return 500.0 if t<2.0 else 0.0
def mass(t): return 50.0

# Aerodynamic coefficients
def C_l(alpha): return 2*np.pi*alpha
def C_d(alpha): return 0.1 + 0.5*alpha**2
def C_y(beta): return 2*np.pi*beta

ekf = RocketEKFJacobian(mass, thrust, S_ref=0.05)
ekf.set_aero_coefficients(C_l, C_d, C_y)

dt = 0.002
time = np.arange(0,5,dt)
for t in time:
    gyro = np.array([0,0,0])  # replace with measurements
    accel = np.array([0,0,0]) # replace with measurements
    baro = 0                   # replace with altitude measurement

    ekf.ekf_predict(gyro, t)
    ekf.ekf_update_accel(accel)
    ekf.ekf_update_baro(baro)

    alpha, beta = ekf.get_aero_angles()
    print(f"t={t:.2f} s, AoA={np.degrees(alpha):.2f} deg, Sideslip={np.degrees(beta):.2f} deg")



# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # Create PDF
# with PdfPages('/mnt/data/multi_sensor_flight_report_layout.pdf') as pdf:

#     # Page 1: Vertical Flight
#     fig, axes = plt.subplots(4, 1, figsize=(8.5, 11))
#     fig.suptitle('Rocket Flight Report - Vertical Flight', fontsize=16)

#     # Panel 1: Altitude & Vertical Velocity
#     axes[0].set_title('Altitude & Vertical Velocity')
#     axes[0].set_xlabel('Time [s]')
#     axes[0].set_ylabel('Altitude [m] / Vertical Velocity [m/s]')
#     axes[0].text(0.5, 0.5, 'Barometer: Altitude\nIMU: Vertical Velocity\nGPS: Altitude/Vertical Velocity',
#                  ha='center', va='center', fontsize=12, alpha=0.3)

#     # Panel 2: Acceleration
#     axes[1].set_title('Acceleration')
#     axes[1].set_xlabel('Time [s]')
#     axes[1].set_ylabel('m/s^2')
#     axes[1].text(0.5, 0.5, 'IMU: Longitudinal & Lateral Acceleration',
#                  ha='center', va='center', fontsize=12, alpha=0.3)

#     # Panel 3: Angular Rates / Orientation
#     axes[2].set_title('Angular Rates / Orientation')
#     axes[2].set_xlabel('Time [s]')
#     axes[2].set_ylabel('deg/s / deg')
#     axes[2].text(0.5, 0.5, 'IMU: Roll, Pitch, Yaw Rates\nOrientation (Integrated)',
#                  ha='center', va='center', fontsize=12, alpha=0.3)

#     # Panel 4: Lateral Displacement (IMU) - Placeholder
#     axes[3].set_title('Lateral Displacement (IMU)')
#     axes[3].set_xlabel('Time [s]')
#     axes[3].set_ylabel('m')
#     axes[3].text(0.5, 0.5, 'Derived from IMU or N/A', ha='center', va='center', fontsize=12, alpha=0.3)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     pdf.savefig(fig)
#     plt.close()

#     # Page 2: GPS / Trajectory
#     fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
#     fig.suptitle('Rocket Flight Report - GPS & Trajectory', fontsize=16)

#     # Panel 1: 2D Horizontal Trajectory
#     axes[0].set_title('2D Horizontal Trajectory (X vs Y)')
#     axes[0].set_xlabel('X [m]')
#     axes[0].set_ylabel('Y [m]')
#     axes[0].text(0.5, 0.5, 'GPS: Horizontal Path\nOverlay IMU Lateral Displacement',
#                  ha='center', va='center', fontsize=12, alpha=0.3)

#     # Panel 2: 3D Trajectory (schematic)
#     axes[1].set_title('3D Trajectory (X,Y,Z)')
#     axes[1].axis('off')
#     axes[1].text(0.5, 0.5, '3D Flight Path Schematic\nGPS/IMU', ha='center', va='center', fontsize=12, alpha=0.3)

#     # Panel 3: Ground Speed / Track Angle
#     axes[2].set_title('Ground Speed / Track Angle vs Time')
#     axes[2].set_xlabel('Time [s]')
#     axes[2].set_ylabel('m/s / deg')
#     axes[2].text(0.5, 0.5, 'GPS: Speed & Track Angle', ha='center', va='center', fontsize=12, alpha=0.3)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     pdf.savefig(fig)
#     plt.close()

#     # Page 3: Summary Metrics
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     fig.suptitle('Rocket Flight Report - Summary Metrics', fontsize=16)
#     ax.axis('off')
#     metrics_text = (
#         'Key Metrics (Placeholder):\n\n'
#         '- Max Altitude (Barometer/GPS)\n'
#         '- Max Vertical Velocity (IMU / GPS)\n'
#         '- Max Lateral Δv (IMU)\n'
#         '- Max Angular Rates (IMU)\n'
#         '- Flight Duration\n'
#         '- Powered Flight Duration\n'
#         '- Apogee Time\n'
#         '- Total Horizontal Displacement (GPS)\n'
#         '- Course Deviation Metric (GPS)'
#     )
#     ax.text(0.05, 0.95, metrics_text, fontsize=12, va='top')
#     pdf.savefig(fig)
#     plt.close()

# print("PDF layout generated: /mnt/data/multi_sensor_flight_report_layout.pdf")