import pid

class AngleControlPID:
    def __init__(self, output_constraints=None):
        self.__theta_pid = pid.PID(Kp=150, Ki=0, Kd=1.5, e=0.99)
        self.__omega_pid = pid.PID(Kp=0.5, Ki=0, Kd=0.0, e=0.9)
        self.__output_constraints = output_constraints

    def __str__(self):
        return 'AngleControl[{}, {}]'.format(self.__theta_pid, self.__omega_pid)

    @property
    def sp(self): return self.__theta_pid.sp

    @sp.setter
    def sp(self, _sp):
        self.__theta_pid.sp = _sp
        return self.__theta_pid.sp

    @property
    def v_sp(self): return None

    @v_sp.setter
    def v_sp(self, _):
        return None

    @property
    def theta_pid(self): return self.__theta_pid

    @property
    def omega_pid(self): return self.__omega_pid

    def __call__(self, dt, theta, omega, domega):
        # Process Angle (theta) Control
        omega_sp = self.__theta_pid(dt, theta)
        # print('*** theta = {}, omega = {}, omega_sp = {}'.format(theta, omega, omega_sp))

        # Process Angular Rate (omega) Control
        self.__omega_pid.sp = omega_sp
        domega_sp = self.__omega_pid(dt, omega)
        # print('*** omega = {}, domega = {}, domega_sp = {}'.format(omega, domega, domega_sp))

        if self.__output_constraints is not None:
            if domega_sp > self.__output_constraints[1]:
                return self.__output_constraints[1]
            elif domega_sp < self.__output_constraints[0]:
                return self.__output_constraints[0]
        return domega_sp


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    v_min, v_max = (0, 10)
    t_max = 10
    t_change = 1.5
    N = 100
    dt = t_max / N
    ts = np.linspace(0, t_max, N)

    pid1 = pid.PID(Kp=1, Ki=1, Kd=0.5, e=0.875)
    pid2 = pid.PID(Kp=1, Ki=1, Kd=1, e=0.875)
    pid3 = pid.PID(Kp=1, Ki=1, Kd=0.25, e=0.875)

    vals1 = []
    vals2 = []
    vals3 = []
    tgts = []

    last_val1 = v_min
    last_val2 = v_min
    last_val3 = v_min
    tgt = v_min

    for t in ts:
        tgt = (tgt + (v_min if t < t_change else v_max)) / 2
        tgts.append(tgt)
        pid1.sp = tgt
        pid2.sp = tgt
        pid3.sp = tgt
        new_val = pid1(dt, last_val1)
        last_val1 = new_val #( last_val1 + new_val ) / 2
        vals1.append(new_val)
        new_val = pid2(dt, last_val2)
        last_val2 = new_val #( last_val2 + new_val ) / 2
        vals2.append(new_val)
        new_val = pid3(dt, last_val3)
        last_val3 = new_val#( last_val3 + new_val ) / 2
        vals3.append(new_val)

    fig, ax = plt.subplots(1, figsize=(15, 12))
    N_end = N
    ax.plot(ts[:N_end], tgts[:N_end], c='k')
    ax.plot(ts[:N_end], vals1[:N_end])
    ax.plot(ts[:N_end], vals2[:N_end])
    ax.plot(ts[:N_end], vals3[:N_end])
    plt.show()
