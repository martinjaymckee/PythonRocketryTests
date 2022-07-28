import math


def gimbal_cycle_properties(f_cycle, theta_offset):
    t_cycle = 1./f_cycle
    rate = 2 * math.pi / t_cycle
    # theta = theta_offset * sin(rate * t)
    # dtheta = theta_offset * rate * cos(rate * t)
    # ddtheta = -theta_offset * (rate**2) cos(rate * t)
    dtheta_max = theta_offset * rate
    ddtheta_max = theta_offset * (rate**2)
    return t_cycle, dtheta_max, ddtheta_max


def required_gimbal_torque(I_gimbal, ddtheta_max): return I_gimbal * ddtheta_max


def kfg_cm_to_N_m(q): return 0.0980665 * q


def gimbal_moi(m, r, h): return (1 / 12) * ((3 * r**2) + (h**2))


if __name__ == '__main__':
    # Gimbal Cycle Properties
    f_cycle = 4.25  # Hz
    gimbal_offset = math.radians(12)

    # Servo Properties -- ES9051
    t_sweep = 0.09  # s
    Q_servo = kfg_cm_to_N_m(.85)  # Given in kgf.cm.
    servo_mass = 4.3/1000  # kg

    # Servo Properties -- ES9251ii
    t_sweep = 0.07  # s
    Q_servo = kfg_cm_to_N_m(.3)  # Given in kgf.cm.
    servo_mass = 3.42/1000  # kg

    # Engine Properties
    engine_mass = 29/1000  # kg
    engine_diameter = 18/1000  # m
    engine_length = 72/1000  # m

    # Mechanism Properties
    l_servo = 7/1000  # 10/1000  # m
    l_gimbal = 18/1000  # m
    C_design = 0.1 # 10%
    # Calculations
    t_cycle, dtheta_max, ddtheta_max = gimbal_cycle_properties(f_cycle, gimbal_offset)
    mechanical_advantage = l_gimbal / l_servo
    Q_available = Q_servo * mechanical_advantage
    servo_dtheta_max = math.radians(60) / (t_sweep * mechanical_advantage)
    I_gimbal = gimbal_moi(engine_mass, engine_diameter/2, engine_length)
    Q_required = required_gimbal_torque(I_gimbal, ddtheta_max)

    print('Maximum servo angular rate = {} rad/s'.format(servo_dtheta_max))
    print('Maximum cycle angular rate = {} rad/s'.format(dtheta_max))
    print('Maximum cycle acceleration = {} rad/s^2'.format(ddtheta_max))
    print('Gimbal MOI = {} kg m^2'.format(I_gimbal))
    print('Available Torque = {} Nm'.format(Q_available))
    print('Required Torque = {} Nm'.format(Q_required))

    print()

    if Q_available >= Q_required:
        print('This design will work')
