import numpy as np

class RK45:
    def __init__(self, y0, dy_dt, t0=0, dt0=0.1):
        self.__y = y0
        self.__dy_dt = dy_dt
        self.__t = t0
        self.__dt = dt0
        
    def __call__(self):
        done = False
        while not done:
            k1 = h * f(t, y)
            k2 = h * f(t + h/2, y + k1/2)
            k3 = h * f(t + h/2, y + k2/2)
            k4 = h * f(t + h, y + k3)

            # Compute the fifth stage of Runge-Kutta 5
            k5 = h * f(t + h/2, y + (k1 + 2*k2 + 2*k3 + k4)/6)
    
            # Estimate the local truncation error
            error = np.abs((k1 + 2*k2 + 2*k3 + k4 + k5) / 6)

            # Update the solution if the error is within tolerance
            if error <= tol:
                t += h
                y += (k1 + 2*k2 + 2*k3 + k4) / 6
    
            # Update the step size for the next iteration
            h = 0.9 * h * (tol / error)**0.2
               
                # State Update - Euler Integration
                dv = accel.timestep(dt) if t >= 0 else utils.VelocityVector3D()
                new_v = state.vel + dv
                state.vel = new_v
                dpos = new_v.timestep(dt)
                state.pos += dpos

        
        
# def rk45_adaptive(f, t0, y0, t_end, h0, tol):
#     """
#     Runge-Kutta 4/5 method with adaptive step size.

#     Parameters:
#         f: The derivative function dy/dt = f(t, y), where t is the independent variable and y is the dependent variable.
#         t0: Initial value of t.
#         y0: Initial value of y.
#         t_end: End value of t.
#         h0: Initial step size.
#         tol: Tolerance for adaptive step size control.

#     Returns:
#         t_values: Array of t values.
#         y_values: Array of y values.
#     """

#     # Initialize arrays to store t and y values
#     t_values = [t0]
#     y_values = [y0]

#     t = t0
#     y = y0
#     h = h0

#     while t < t_end:
#         # Adjust step size if necessary
#         if t + h > t_end:
#             h = t_end - t

#         # Compute the four stages of Runge-Kutta 4
#         k1 = h * f(t, y)
#         k2 = h * f(t + h/2, y + k1/2)
#         k3 = h * f(t + h/2, y + k2/2)
#         k4 = h * f(t + h, y + k3)

#         # Compute the fifth stage of Runge-Kutta 5
#         k5 = h * f(t + h/2, y + (k1 + 2*k2 + 2*k3 + k4)/6)

#         # Estimate the local truncation error
#         error = np.abs((k1 + 2*k2 + 2*k3 + k4 + k5) / 6)

#         # Update the solution if the error is within tolerance
#         if error <= tol:
#             t += h
#             y += (k1 + 2*k2 + 2*k3 + k4) / 6

#         # Update the step size for the next iteration
#         h = 0.9 * h * (tol / error)**0.2

#         # Store the t and y values
#         t_values.append(t)
#         y_values.append(y)

#     return np.array(t_values), np.array(y_values)

# import numpy as np

# def dp_adaptive(f, t0, y0, t_end, h0, tol):
#     """
#     Dormand-Prince method with adaptive step size.

#     Parameters:
#         f: The derivative function dy/dt = f(t, y), where t is the independent variable and y is the dependent variable.
#         t0: Initial value of t.
#         y0: Initial value of y.
#         t_end: End value of t.
#         h0: Initial step size.
#         tol: Tolerance for adaptive step size control.

#     Returns:
#         t_values: Array of t values.
#         y_values: Array of y values.
#     """

#     # Coefficients for the Dormand-Prince method
#     c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=np.float64)
#     a = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [1/5, 0, 0, 0, 0, 0, 0],
#         [3/40, 9/40, 0, 0, 0, 0, 0],
#         [44/45, -56/15, 32/9, 0, 0, 0, 0],
#         [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
#         [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
#         [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
#     ], dtype=np.float64)
#     b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=np.float64)
#     b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float64)
#     e = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40], dtype=np.float64)

#     # Initialize arrays to store t and y values
#     t_values = [t0]
#     y_values = [y0]

#     t = t0
#     y = y0
#     h = h0

#     while t < t_end:
#         # Adjust step size if necessary
#         if t + h > t_end:
#             h = t_end - t

#         # Compute the stages and stage derivatives
#         stages = np.zeros((7, len(y)), dtype=np.float64)
#         stage_derivatives = np.zeros((7, len(y)), dtype=np.float64)
#         for i in range(7):
#             stages[i] = y + h * np.dot(a[i], stage_derivatives[:i])
#             stage_derivatives[i] = f(t + c[i] * h, stages[i])

#         # Estimate the local truncation error
#         error = np.linalg.norm(h * np.dot(e, stage_derivatives))

#         # Update the solution if the error is within tolerance
#         if error <= tol:
#             t += h
#             y += h * np.dot(b, stage_derivatives)

#         # Update the step size for the next iteration
#         h = 0.9 * h * (tol / error)**0.2

#         # Store the t and y values
#         t_values.append(t)
#         y_values.append(y)

#     return np.array(t_values), np.array(y_values)

# import numpy as np

# def ck_adaptive(f, t0, y0, t_end, h0, tol):
#     """
#     Cash-Karp 7(6) method with adaptive step size.

#     Parameters:
#         f: The derivative function dy/dt = f(t, y), where t is the independent variable and y is the dependent variable.
#         t0: Initial value of t.
#         y0: Initial value of y.
#         t_end: End value of t.
#         h0: Initial step size.
#         tol: Tolerance for adaptive step size control.

#     Returns:
#         t_values: Array of t values.
#         y_values: Array of y values.
#     """

#     # Coefficients for the Cash-Karp 7(6) method
#     c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8, 1], dtype=np.float64)
#     a = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [1/5, 0, 0, 0, 0, 0, 0],
#         [3/40, 9/40, 0, 0, 0, 0, 0],
#         [3/10, -9/10, 6/5, 0, 0, 0, 0],
#         [-11/54, 5/2, -70/27, 35/27, 0, 0, 0],
#         [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0, 0],
#         [2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4, 0, 0]
#     ], dtype=np.float64)
#     b = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771, 1/10], dtype=np.float64)
#     b_star = np.array([2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4, 0, 0], dtype=np.float64)
#     e = np.array([-277/64512, 0, 6925/370944, -6925/202752, 277/14336, -277/7084, 
