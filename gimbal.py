import math
import random


import numpy as np
import numpy.linalg


import mmoi_utils


def rotation_matrix(alpha, beta):
    s_a, c_a = math.sin(alpha), math.cos(alpha)
    s_b, c_b = math.sin(beta), math.cos(beta)
    return np.array([
        [c_a*c_b, -s_a, c_a*s_b],
        [s_a*c_b, c_a, s_a*s_b],
        [-s_b, 0, c_b]
    ])


def thrust_vector(T, alpha, beta):
    return np.matmul(rotation_matrix(alpha, beta), np.array([T, 0, 0]))


def angular_acceleration(p_cg, p_F, T, alpha, beta, I):
    I_inv = np.linalg.inv(I)
    r = p_F - p_cg
    F = thrust_vector(T, alpha, beta)
    # print('F = {}'.format(F))
    domega = np.matmul(I_inv, np.cross((F - (r * (np.dot(F, r) / (T**2)))), r))
    return domega


def linearized_angular_acceleration(p_cg, p_F, T, alpha, beta, I):
    #  It is also possible to just estimate the cosines as 1... at lower accuracy
    s_a, c_a = alpha, (1 - (alpha**2/2))
    s_b, c_b = beta, (1 - (beta**2/2))
    I_inv = np.linalg.inv(I)
    r = p_F - p_cg
    #F_actual = thrust_vector(T, alpha, beta)
    #print('Actual F = {}'.format(F_actual))
    F = T * np.array([c_a*c_b, s_a*c_b, -s_b])
    #print('Optimized Estimate F = {}'.format(F))
    #F_err = F_actual - F
    #print('Estimation Error = {}'.format(100 * (F_err / F_actual)))
    domega = np.matmul(I_inv, np.cross((F - (r * (np.dot(F, r) / (T**2)))), r))
    return domega


def gimbal_angles(domega, p_cg, p_F, T, alpha_init, beta_init, I, dalpha=1e-7, N=3, linearized=False):
    aa = linearized_angular_acceleration if linearized else angular_acceleration

    # Calculate the Jacobian of the acceleration equations using a central difference approximation
    def deriv_aa_alpha(alpha, beta):
        a = aa(p_cg, p_F, T, alpha-dalpha, beta, I)
        b = aa(p_cg, p_F, T, alpha+dalpha, beta, I)
        return (b-a) / (2*dalpha)

    def deriv_aa_beta(alpha, beta):
        a = aa(p_cg, p_F, T, alpha, beta-dalpha, I)
        b = aa(p_cg, p_F, T, alpha, beta+dalpha, I)
        return (b-a) / (2*dalpha)

    def central_difference_jacobian(alpha, beta):
        da = deriv_aa_alpha(alpha, beta)
        db = deriv_aa_beta(alpha, beta)
        return np.array([
            [da[1], db[1]],
            [da[2], db[2]]
        ])

    def analytic_jacobian(alpha, beta):
        I_inv = np.linalg.inv(I)
        r = p_F - p_cg
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        df_da = T * np.array([-sa*cb, ca*cb, 0])
        df_db = T * np.array([-ca*sb, -sa*sb, -cb])
        dfr_da = np.dot(df_da, r)
        dfr_db = np.dot(df_db, r)
        da = np.matmul(I_inv, np.cross((df_da - r*dfr_da/(T**2)), r))
        db = np.matmul(I_inv, np.cross((df_db - r*dfr_db/(T**2)), r))
        return np.array([
            [da[1], db[1]],
            [da[2], db[2]]
        ])

    def linearized_analytic_jacobian(alpha, beta):
        I_inv = np.linalg.inv(I)
        r = p_F - p_cg
        sa, ca = alpha, (1 - (alpha**2/2))
        sb, cb = beta, (1 - (beta**2/2))
        df_da = T * np.array([-sa*cb, ca*cb, 0])
        df_db = T * np.array([-ca*sb, -sa*sb, -cb])
        dfr_da = np.dot(df_da, r)
        dfr_db = np.dot(df_db, r)
        da = np.matmul(I_inv, np.cross((df_da - r*dfr_da/(T**2)), r))
        db = np.matmul(I_inv, np.cross((df_db - r*dfr_db/(T**2)), r))
        return np.array([
            [da[1], db[1]],
            [da[2], db[2]]
        ])

    x = np.array([0, alpha_init, beta_init])
    for i in range(N):
        # J = central_difference_jacobian(x[1], x[2])
        # print('Central Difference Jacobian:\n\tJ = {}'.format(J))

        J = linearized_analytic_jacobian(x[1], x[2])
        # print('Linearized Analytic Jacobian:\n\tJ = {}'.format(J))
        #
        # J = analytic_jacobian(x[1], x[2])
        # print('Analytic Jacobian:\n\tJ = {}'.format(J))

        #print('J = {}'.format(J))
        f = (domega - aa(p_cg, p_F, T, x[1], x[2], I))[1:]
        dx = np.matmul(np.linalg.inv(J), f.transpose())
        #print('f = {}'.format(f))
        #print('dx = {}'.format(dx))
        x = np.array([x[0], x[1]+dx[0], x[2]+dx[1]])
        #print('x = {}'.format(x))

    return x[1], x[2]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    T = 2.3
    alpha_tgt = math.radians(12.5)
    beta_tgt = math.radians(-5)

    # TODO: THIS IS A HACK FOR THE MMOI AND CG
    I = np.diag(mmoi_utils.cylindrical_shell(0.5, 0.01, 0.050))
    p_cg = np.array([0.25, 0, 0])
    p_F = np.array([0.45, 0, 0])

    F = thrust_vector(T, alpha_tgt, beta_tgt)
#    print(F)
    #print(I)
    domega = angular_acceleration(p_cg, p_F, T, alpha_tgt, beta_tgt, I)
    print('Correct domega = {}'.format(domega))
    domega_lin = linearized_angular_acceleration(p_cg, p_F, T, alpha_tgt, beta_tgt, I)
    print('Linearized domega = {}'.format(domega_lin))

    print('domega err = {}'.format(100 * ((domega - domega_lin)/domega)))

    alpha, beta = gimbal_angles(domega, p_cg, p_F, T, math.radians(0), math.radians(0), I)
    print('alpha = {:0.3f} ({:0.3f} %)'.format(alpha, 100 * (alpha - alpha_tgt)))
    print('beta = {:0.3f} ({:0.3f} %)'.format(beta, 100 * (beta - beta_tgt)))
