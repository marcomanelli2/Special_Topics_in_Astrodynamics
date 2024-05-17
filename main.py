import numpy as np
import scipy.integrate
from numpy import linalg as LA
import cmath
import matplotlib.pyplot as plt
from scipy import optimize

M_SUN = 1.98847e+30
M_EARTH = 5.9722e+24
MU = M_EARTH / (M_SUN + M_EARTH)


def equilibrium_eqn(x):
    return x - (1 - MU) / abs((x + MU)) ** 3 * (x + MU) - MU / abs(x - (1 - MU)) ** 3 * (x - (1 - MU))


def motion_eqn(t, x):
    x_dot = np.zeros(6,)
    x_3d = np.zeros(6,)
    x_3d[:2] = x[:2]
    x_3d[3:5] = x[2:]
    x = x_3d
    r1 = np.zeros(3,)
    r1[:2] = x[:2] + np.array([MU, 0])
    r2 = np.zeros(3,)
    r2[:2] = x[:2] - np.array([1 - MU, 0])
    x_dot[:3] = x[3:]
    x_dot[3:] = -(r1 * (1 - MU) / LA.norm(r1) ** 3 + r2 * MU / LA.norm(r2) ** 3) - 2 * np.cross(
        np.array([0, 0, 1]), x[3:]) - np.cross(np.array([0, 0, 1]), np.cross(np.array([0, 0, 1]), x[:3]))
    return x_dot[[0, 1, 3, 4]]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1.1a
    x0 = 0.98
    x_L1 = optimize.newton(equilibrium_eqn, x0)
    print(x_L1)

    #1.2a
    r_art = np.array([x_L1 - 0.01, 0, 0.01])
    r1_art = r_art + np.array([MU, 0, 0])
    r2_art = r_art + np.array([-(1-MU), 0, 0])

    # calculate the gradient of the potential
    U_x = (1 - MU) / LA.norm(r1_art)**3 * r1_art[0] + MU / LA.norm(r2_art)**3 * r2_art[0] - r_art[0]
    U_y = 0
    U_z = (1 - MU) / LA.norm(r1_art)**3 * r1_art[2] + MU / LA.norm(r2_art)**3 * r2_art[2]
    grad_U = np.array([U_x, U_y, U_z])

    n = grad_U / LA.norm(grad_U)

    beta = LA.norm(r1_art)**2 / (1 - MU) * np.dot(grad_U, n) / np.dot(r1_art/LA.norm(r1_art), n)

    print(n.tolist())
    print(beta)

    print(np.rad2deg(np.arccos(np.dot(n, r1_art))))


    # 2.1a
    K = (1 - MU) / abs(x_L1 + MU) ** 3 + MU / abs(x_L1 - (1 - MU)) ** 3
    Uxx = -2 * K - 1
    Uyy = K - 1

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-Uxx, 0, 0, 2],
        [0, -Uyy, -2, 0]
    ])

    lam, zeta = LA.eig(A)
    print('Numerical solution:', lam[0], lam[1], lam[2], lam[3])

    # 2.1b Verification:
    # analytical solution
    lam_1 = -np.sqrt(1 + 2 * np.sqrt(7))
    lam_2 = np.sqrt(1 + 2 * np.sqrt(7))
    lam_3 = np.sqrt(2 * np.sqrt(7) - 1) * 1j
    lam_4 = -np.sqrt(2 * np.sqrt(7) - 1) * 1j

    print('Analytical solution:', lam_1, lam_2, lam_3, lam_4)

    # 2.2
    plt.figure()

    # Stable manifold
    x_0_stable1 = np.array([x_L1, 0, 0, 0]) + 1e-5 * np.real(zeta[0])
    x_0_stable2 = np.array([x_L1, 0, 0, 0]) - 1e-5 * np.real(zeta[0])
    sol1 = scipy.integrate.solve_ivp(motion_eqn, [0, -6 * np.pi], x_0_stable1, rtol=1e-12, atol=1e-12)
    sol2 = scipy.integrate.solve_ivp(motion_eqn, [0, -6 * np.pi], x_0_stable2, rtol=1e-12, atol=1e-12)

    plt.plot(sol1.y[0], sol1.y[1], 'C0')
    plt.plot(sol2.y[0], sol2.y[1], 'C0')

    # Unstable manifold
    x_0_unstable1 = np.array([x_L1, 0, 0, 0]) + 1e-5 * np.real(zeta[1])
    x_0_unstable2 = np.array([x_L1, 0, 0, 0]) - 1e-5 * np.real(zeta[1])
    sol1 = scipy.integrate.solve_ivp(motion_eqn, [0, 6 * np.pi], x_0_unstable1, rtol=1e-12, atol=1e-12)
    sol2 = scipy.integrate.solve_ivp(motion_eqn, [0, 6 * np.pi], x_0_unstable2, rtol=1e-12, atol=1e-12)

    plt.plot(sol1.y[0], sol1.y[1], 'C1')
    plt.plot(sol2.y[0], sol2.y[1], 'C1')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(['Stable', '_nolegend_', 'Unstable'])
    plt.tight_layout()

    plt.savefig('manifolds.png')
