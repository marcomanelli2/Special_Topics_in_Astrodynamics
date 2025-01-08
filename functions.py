import os

import numpy as np
import numpy.linalg as LA
import scipy

# Assignment constants
M_SUN = 1.9891e+30  # kg
M_EARTH = 6.0477e+24  # kg
AU = 149597870.7e+3  # m
R_SUN = 695700e+3  # m
R_EARTH = 6378e+3  # m

MU = M_EARTH / (M_SUN + M_EARTH)
MU_JPL = 3.05420e-6
X0_JPL = np.array(
    [9.91416770988543e-01, 3.90169410418171e-23,  1.10992377703143e-02,
     2.44188247527455e-15, 1.53784488470937e-02, -1.03706638670444e-14])
L1_JPL = np.array([0.98997092, 0.00000000, 0.00000000])
T_JPL = 2.73985394771395

DIR = os.getcwd()


def equilibrium_eqn(x):
    return x - (1 - MU) / abs((x + MU)) ** 3 * (x + MU) - MU / abs(x - (1 - MU)) ** 3 * (x - (1 - MU))


def motion_eqn(t, x):
    x_dot = np.zeros(6, )
    x_3d = np.zeros(6, )
    x_3d[:2] = x[:2]
    x_3d[3:5] = x[2:]
    x = x_3d
    r1 = np.zeros(3, )
    r1[:2] = x[:2] + np.array([MU, 0])
    r2 = np.zeros(3, )
    r2[:2] = x[:2] - np.array([1 - MU, 0])
    x_dot[:3] = x[3:]
    # noinspection PyUnreachableCode
    x_dot[3:] = -(r1 * (1 - MU) / LA.norm(r1) ** 3 + r2 * MU / LA.norm(r2) ** 3) - 2 * np.cross(
        np.array([0, 0, 1]), x[3:]) - np.cross(np.array([0, 0, 1]), np.cross(np.array([0, 0, 1]), x[:3]))

    return x_dot[[0, 1, 3, 4]]


def motion_eqn_jpl(t, x):
    x_dot = np.zeros(6, )
    r1 = x[:3] - np.array([-MU_JPL, 0, 0])
    r2 = x[:3] - np.array([1 - MU_JPL, 0, 0])
    x_dot[:3] = x[3:]
    # noinspection PyUnreachableCode
    x_dot[3:] = -(r1 * (1 - MU_JPL) / LA.norm(r1) ** 3 + r2 * MU_JPL / LA.norm(r2) ** 3) - 2 * np.cross(
        np.array([0, 0, 1]), x[3:]) - np.cross(np.array([0, 0, 1]), np.cross(np.array([0, 0, 1]), x[:3]))

    return x_dot


def variational_eqn(t, x, mu, beta):
    r1 = LA.norm(x[:3] - np.array([-mu, 0, 0]))
    r1_dir = (x[:3] - np.array([-mu, 0, 0])) / r1
    r2 = LA.norm(x[:3] - np.array([1 - mu, 0, 0]))

    n = np.array([1, 0, 0])
    cos_alpha = np.dot(r1_dir, n)
    a_sail = beta * (1 - mu) / r1 ** 2 * cos_alpha ** 2

    x_dot = np.zeros(42, )

    # state derivative
    x_dot[0] = x[3]
    x_dot[1] = x[4]
    x_dot[2] = x[5]
    x_dot[3] = 2 * x[4] + x[0] - (1 - mu) * (mu + x[0]) / r1 ** 3 + mu * (1 - mu - x[0]) / r2 ** 3 + a_sail
    x_dot[4] = -2 * x[3] + x[1] * (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3)
    x_dot[5] = -((1 - mu) / r1 ** 3 + mu / r2 ** 3) * x[2]

    # state transition matrix
    Phi = x[6:].reshape(6, 6)

    # solar sail acceleration derivatives
    a_xx = -2 * beta * (1 - mu) * (mu + x[0]) * (2 * (mu + x[0]) ** 2 - r1 ** 2) / r1 ** 6
    a_xy = -4 * beta * (1 - mu) * (x[0] + mu) ** 2 / r1 ** 6 * x[1]
    a_xz = -4 * beta * (1 - mu) * (x[0] + mu) ** 2 / r1 ** 6 * x[2]

    # hessian matrix of the potential
    U_xx = 1 + (1 - mu) / r1 ** 5 * (2 * (x[0] + mu) ** 2 - x[1] ** 2 - x[2] ** 2) + mu / r2 ** 5 * (
                2 * (x[0] - (1 - mu)) ** 2
                - x[1] ** 2 - x[2] ** 2)
    U_yy = 1 - (1 - mu) / r1 ** 5 * ((x[0] + mu) ** 2 - 2 * x[1] ** 2 + x[2] ** 2) - mu / r2 ** 5 * (
                (x[0] - (1 - mu)) ** 2
                - 2 * x[1] ** 2 + x[2] ** 2)
    U_zz = -((1 - mu) * ((x[0] + mu) ** 2 + x[1] ** 2 - 2 * x[2] ** 2) / r1 ** 5 + mu * (
                (1 - mu - x[0]) ** 2 + x[1] ** 2
                - 2 * x[2] ** 2) / r2 ** 5)
    U_xy = 3 * x[1] * (1 - mu) * (x[0] + mu) / r1 ** 5 + 3 * mu * x[1] * (x[0] - (1 - mu)) / r2 ** 5

    U_xz = 3 * (1 - mu) * x[2] * (x[0] + mu) / r1 ** 5 + 3 * mu * x[2] * (x[0] - (1 - mu)) / r2 ** 5

    U_yz = 3 * (1 - mu) * x[1] * x[2] / r1 ** 5 + 3 * mu * x[1] * x[2] / r2 ** 5

    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [U_xx + a_xx, U_xy + a_xy, U_xz + a_xz, 0, 2, 0],
                  [U_xy, U_yy, U_yz, -2, 0, 0],
                  [U_xz, U_yz, U_zz, 0, 0, 0]])

    # state transition matrix derivative
    x_dot[6:] = (A @ Phi).reshape(36, )

    return x_dot


def diff_correction(x_0, mu, beta):
    delta = np.array([0, 0])
    err = 1
    it = 0
    while err > 1e-12:
        # correction
        x_0[4] = x_0[4] + delta[1]  # vy_0
        x_0[0] = x_0[0] + delta[0]  # x_0

        event = lambda t, x: x[1]
        event.terminal = True

        sol = scipy.integrate.solve_ivp(lambda t, x: variational_eqn(t, x, mu, beta), [0, 2 * T_JPL], x_0,
                                        rtol=1e-12, atol=1e-12, events=event)

        period = 2 * sol.t_events[0][0]
        state_half = sol.y_events[0][0, :6]
        phi_half = sol.y_events[0][0, 6:].reshape(6, 6)

        # acceleration
        r1 = np.sqrt((mu + state_half[0]) ** 2 + state_half[1] ** 2 + state_half[2] ** 2)
        r2 = np.sqrt((1 - mu - state_half[0]) ** 2 + state_half[1] ** 2 + state_half[2] ** 2)
        acc_x_half = 2 * state_half[4] + state_half[0] - (1 - mu) * (mu + state_half[0]) / r1 ** 3 + mu * (
                    1 - mu - state_half[0]) / r2 ** 3
        acc_z_half = -((1 - mu) / r1 ** 3 + mu / r2 ** 3) * state_half[2]
        vel_y_half = state_half[4]

        # change x0 and vy0
        M = np.array([[phi_half[3, 0], phi_half[3, 4]], [phi_half[5, 0], phi_half[5, 4]]]) - 1 / vel_y_half * (
            np.array([[acc_x_half * phi_half[1, 0], acc_x_half * phi_half[1, 4]],
                      [acc_z_half * phi_half[1, 0], acc_z_half * phi_half[1, 4]]]))
        b = -np.array([state_half[3], state_half[5]])
        delta = np.linalg.solve(M, b)

        # error computation
        err = np.sqrt(state_half[3] ** 2 + state_half[5] ** 2)

        it += 1
        print(it, '\tError:\t', err)

    return x_0, period


if __name__ == '__main__':
    print(MU)
    print(MU_JPL)
