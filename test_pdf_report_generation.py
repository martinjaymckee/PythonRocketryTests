import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------
# Input: replace these with your actual flight data
# -------------------------------
events = [(1.2, 'Ignition'), (2.5, 'Liftoff'), (5.0, 'Burnout'), (10.0, 'Apogee'), (15.0, 'Landing')]
deviations = [(3.5, 4.2)]  # start and end times of course deviations

t_ignition = events[0][0]
t_burnout_last = events[2][0]
t_apogee = events[3][0]
t_landing = events[4][0]

# Flight phases and colors
phase_colors = {
    'Preflight': 'darkgray',
    'Thrust': 'red',
    'Coast': 'blue',
    'Descent': 'green'
}
phase_times = [(0, t_ignition), (t_ignition, t_burnout_last), (t_burnout_last, t_apogee), (t_apogee, t_landing)]
phase_names = ['Preflight', 'Thrust', 'Coast', 'Descent']

# Mock time series data for placeholders
t = np.linspace(0, t_landing, 300)
altitude = 100*t/15
vertical_velocity = np.gradient(altitude, t)
acceleration = np.gradient(vertical_velocity, t)
roll = 5*np.sin(0.5*t)
pitch = 2*np.sin(0.3*t)
yaw = 3*np.sin(0.2*t)
lateral_displacement = 10*np.sin(0.1*t)
ground_speed = np.gradient(np.sqrt(lateral_displacement**2 + altitude**2), t)
track_angle = np.arctan2(lateral_displacement, altitude) * 180/np.pi

# Mock trajectory data
x = 50*np.sin(0.05*t)
y = 50*np.cos(0.05*t)
z = altitude

# PDF output path
pdf_path = 'multi_sensor_full_flight_report.pdf'

# -------------------------------
# Helper functions
# -------------------------------
def add_event_markers(ax, events):
    for te, name in events:
        ax.axvline(te, color='black', linestyle='--', alpha=0.7)
        ax.text(te, ax.get_ylim()[1]*0.95, name, rotation=90, color='black', ha='right', va='top', fontsize=8)

def add_deviation_regions(ax, deviations):
    for start, end in deviations:
        ax.axvspan(start, end, color='yellow', alpha=0.3)

def add_flight_phases(ax):
    for (start, end), phase in zip(phase_times, phase_names):
        ax.axvspan(start, end, color=phase_colors[phase], alpha=0.2)

# -------------------------------
# Create PDF
# -------------------------------
with PdfPages(pdf_path) as pdf:

    # --- Page 1: Vertical Flight Panels ---
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11))
    fig.suptitle('Rocket Flight Report - Vertical Flight', fontsize=16)

    # Panel 1: Altitude & Vertical Velocity
    axes[0].set_title('Altitude & Vertical Velocity')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Altitude [m] / Vertical Velocity [m/s]')
    axes[0].plot(t, altitude, label='Altitude', color='blue')
    axes[0].plot(t, vertical_velocity, label='Vertical Velocity', color='orange')
    add_event_markers(axes[0], events)
    add_deviation_regions(axes[0], deviations)
    add_flight_phases(axes[0])
    axes[0].legend()

    # Panel 2: Acceleration
    axes[1].set_title('Acceleration')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('m/s²')
    axes[1].plot(t, acceleration, color='purple')
    add_event_markers(axes[1], events)
    add_deviation_regions(axes[1], deviations)
    add_flight_phases(axes[1])

    # Panel 3: Angular Rates
    axes[2].set_title('Angular Rates / Orientation')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('deg/s / deg')
    axes[2].plot(t, roll, label='Roll', color='red')
    axes[2].plot(t, pitch, label='Pitch', color='green')
    axes[2].plot(t, yaw, label='Yaw', color='blue')
    add_event_markers(axes[2], events)
    add_deviation_regions(axes[2], deviations)
    add_flight_phases(axes[2])
    axes[2].legend()

    # Panel 4: Lateral Displacement
    axes[3].set_title('Lateral Displacement (IMU)')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('m')
    axes[3].plot(t, lateral_displacement, color='brown')
    add_event_markers(axes[3], events)
    add_deviation_regions(axes[3], deviations)
    add_flight_phases(axes[3])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Ground Speed / Track Angle ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.suptitle('Ground Speed / Track Angle vs Time', fontsize=16)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('m/s / deg')
    ax.plot(t, ground_speed, label='Ground Speed', color='blue')
    ax.plot(t, track_angle, label='Track Angle', color='orange')
    add_event_markers(ax, events)
    add_deviation_regions(ax, deviations)
    add_flight_phases(ax)
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: 2D Horizontal Trajectory ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.suptitle('2D Horizontal Trajectory with Flight Phases', fontsize=16)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    for (start, end), phase in zip(phase_times, phase_names):
        idx = (t >= start) & (t <= end)
        ax.plot(x[idx], y[idx], color=phase_colors[phase], label=phase, linewidth=2)
    for te, name in events:
        idx = np.argmin(np.abs(t - te))
        ax.plot(x[idx], y[idx], 'o', color='black')
        ax.text(x[idx], y[idx]+2, name, fontsize=8, ha='center')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # --- Page 4: 3D Trajectory ---
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('3D Trajectory with Flight Phases', fontsize=16)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    for (start, end), phase in zip(phase_times, phase_names):
        idx = (t >= start) & (t <= end)
        ax.plot(x[idx], y[idx], z[idx], color=phase_colors[phase], linewidth=2)
    for te, name in events:
        idx = np.argmin(np.abs(t - te))
        ax.scatter(x[idx], y[idx], z[idx], color='black', s=30)
        ax.text(x[idx], y[idx], z[idx]+2, name, fontsize=8, ha='center')
    pdf.savefig(fig)
    plt.close()

    # --- Page 5: Summary Metrics ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.suptitle('Rocket Flight Report - Summary Metrics', fontsize=16)
    ax.axis('off')
    metrics_text = (
        'Key Metrics (Placeholder):\n\n'
        '- Max Altitude (Barometer/GPS)\n'
        '- Max Vertical Velocity (IMU / GPS)\n'
        '- Max Lateral Δv (IMU)\n'
        '- Max Angular Rates (IMU)\n'
        '- Flight Duration\n'
        '- Powered Flight Duration\n'
        '- Apogee Time\n'
        '- Total Horizontal Displacement (GPS)\n'
        '- Course Deviation Metric (GPS)'
    )
    ax.text(0.05, 0.95, metrics_text, fontsize=12, va='top')
    pdf.savefig(fig)
    plt.close()

    # --- Page 6: Error Detection & Validation ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.suptitle('Error Detection & Validation', fontsize=16)
    ax.axis('off')

    error_text = (
        "Suggested Error Detection & Validation Metrics by Sensor Combination:\n\n"

        "Barometer Only:\n"
        "- Max Δaltitude per timestep (detect spikes)\n"
        "- Rate of climb vs expected thrust profile\n"
        "- Apogee consistency check\n\n"

        "Barometer + IMU:\n"
        "- Integrated velocity/altitude vs barometer\n"
        "- Lateral displacement consistency\n"
        "- Max tilt / angular rates vs expected safe values\n\n"

        "Barometer + IMU + GPS:\n"
        "- Horizontal displacement consistency vs IMU integration\n"
        "- Ground speed vs vertical speed ratios\n"
        "- Track deviation from intended flight path\n\n"

        "Multi-Barometer / Multi-IMU:\n"
        "- Inter-sensor bias / drift check\n"
        "- RMS error between sensors\n"
        "- Noise level / sensor agreement"
    )

    ax.text(0.05, 0.95, error_text, fontsize=12, va='top')
    pdf.savefig(fig)
    plt.close()

print(f"Full multi-sensor flight report generated: {pdf_path}")


# # Example input: replace with your actual event times
# events = [(1.2, 'Ignition'), (2.5, 'Liftoff'), (5.0, 'Burnout'), (10.0, 'Apogee'), (15.0, 'Landing')]
# deviations = [(3.5, 4.2)]  # start and end times of deviation

# # Flight phases derived from events
# t_ignition = events[0][0]
# t_burnout_last = events[2][0]
# t_apogee = events[3][0]
# t_landing = events[4][0]

# # Phase colors
# phase_colors = {
#     'Preflight': 'red', #'lightgray',
#     'Thrust': 'lightcoral',
#     'Coast': 'lightblue',
#     'Descent': 'lightgreen'
# }

# pdf_path = 'multi_sensor_flight_report_layout_with_phases.pdf'

# def add_event_markers(ax, events):
#     for t, name in events:
#         ax.axvline(t, color='red', linestyle='--', alpha=0.7)
#         ax.text(t, ax.get_ylim()[1]*0.95, name, rotation=90, color='red',
#                 ha='right', va='top', fontsize=8, alpha=0.7)

# def add_deviation_regions(ax, deviations):
#     for start, end in deviations:
#         ax.axvspan(start, end, color='yellow', alpha=0.3)

# def add_flight_phases(ax):
#     # Preflight
#     ax.axvspan(0, t_ignition, color=phase_colors['Preflight'], alpha=0.2, label='Preflight')
#     # Thrust
#     ax.axvspan(t_ignition, t_burnout_last, color=phase_colors['Thrust'], alpha=0.2, label='Thrust')
#     # Coast
#     ax.axvspan(t_burnout_last, t_apogee, color=phase_colors['Coast'], alpha=0.2, label='Coast')
#     # Descent
#     ax.axvspan(t_apogee, t_landing, color=phase_colors['Descent'], alpha=0.2, label='Descent')

# with PdfPages(pdf_path) as pdf:

#     # --- Page 1: Vertical Flight ---
#     fig, axes = plt.subplots(4, 1, figsize=(8.5, 11))
#     fig.suptitle('Rocket Flight Report - Vertical Flight', fontsize=16)

#     for ax, title, text in zip(
#         axes,
#         ['Altitude & Vertical Velocity', 'Acceleration', 'Angular Rates / Orientation', 'Lateral Displacement (IMU)'],
#         ['Barometer: Altitude\nIMU: Vertical Velocity\nGPS: Altitude/Vertical Velocity',
#          'IMU: Longitudinal & Lateral Acceleration',
#          'IMU: Roll, Pitch, Yaw Rates\nOrientation (Integrated)',
#          'Derived from IMU or N/A']):
#         ax.set_title(title)
#         ax.set_xlabel('Time [s]')
#         ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, alpha=0.3)
#         add_event_markers(ax, events)
#         add_deviation_regions(ax, deviations)
#         add_flight_phases(ax)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     pdf.savefig(fig)
#     plt.close()

#     # --- Page 2: GPS / Trajectory ---
#     fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
#     fig.suptitle('Rocket Flight Report - GPS & Trajectory', fontsize=16)

#     axes[0].set_title('2D Horizontal Trajectory (X vs Y)')
#     axes[0].set_xlabel('X [m]')
#     axes[0].set_ylabel('Y [m]')
#     axes[0].text(0.5, 0.5, 'GPS: Horizontal Path\nOverlay IMU Lateral Displacement',
#                  ha='center', va='center', fontsize=12, alpha=0.3)

#     axes[1].set_title('3D Trajectory (X,Y,Z)')
#     axes[1].axis('off')
#     axes[1].text(0.5, 0.5, '3D Flight Path Schematic\nGPS/IMU', ha='center', va='center', fontsize=12, alpha=0.3)

#     axes[2].set_title('Ground Speed / Track Angle vs Time')
#     axes[2].set_xlabel('Time [s]')
#     axes[2].set_ylabel('m/s / deg')
#     axes[2].text(0.5, 0.5, 'GPS: Speed & Track Angle', ha='center', va='center', fontsize=12, alpha=0.3)
#     add_event_markers(axes[2], events)
#     add_deviation_regions(axes[2], deviations)
#     add_flight_phases(axes[2])

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     pdf.savefig(fig)
#     plt.close()

#     # --- Page 3: Summary Metrics ---
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

# print(f"PDF layout with flight phases, events, and deviations generated: {pdf_path}")
