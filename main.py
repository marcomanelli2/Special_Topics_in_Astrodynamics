from scipy import integrate, optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from functions import *


# 1.1a
x0 = 0.98
x_L1 = optimize.newton(equilibrium_eqn, x0)
print(x_L1)

part1 = 1
if part1:
    # 1.2a
    r_art = np.array([x_L1 - 0.01, 0, 0.01])
    r1_art = r_art - np.array([-MU, 0, 0])
    r2_art = r_art - np.array([1-MU, 0, 0])

    # gradient of the potential
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
    U_xx = -2 * K - 1
    U_yy = K - 1

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-U_xx, 0, 0, 2],
        [0, -U_yy, -2, 0]
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
    fig, ax = plt.subplots(figsize=(6, 5))

    # Stable manifold
    x_0_stable1 = np.array([x_L1, 0, 0, 0]) + 1e-5 * np.real(zeta[0])
    x_0_stable2 = np.array([x_L1, 0, 0, 0]) - 1e-5 * np.real(zeta[0])
    sol1 = integrate.solve_ivp(motion_eqn, [0, -6 * np.pi], x_0_stable1, rtol=1e-12, atol=1e-12)
    sol2 = integrate.solve_ivp(motion_eqn, [0, -6 * np.pi], x_0_stable2, rtol=1e-12, atol=1e-12)

    ax.plot(sol1.y[0], sol1.y[1], 'C0', label='Stable 1')
    ax.plot(sol2.y[0], sol2.y[1], 'C2', label='Stable 2')

    x1, x2, y1, y2 = 0.98, 1.01, -0.01, 0.01
    axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins.plot(sol1.y[0], sol1.y[1], 'C0')
    axins.plot(sol2.y[0], sol2.y[1], 'C2')

    # Unstable manifold
    x_0_unstable1 = np.array([x_L1, 0, 0, 0]) + 1e-5 * np.real(zeta[1])
    x_0_unstable2 = np.array([x_L1, 0, 0, 0]) - 1e-5 * np.real(zeta[1])
    sol1 = integrate.solve_ivp(motion_eqn, [0, 6 * np.pi], x_0_unstable1, rtol=1e-12, atol=1e-12)
    sol2 = integrate.solve_ivp(motion_eqn, [0, 6 * np.pi], x_0_unstable2, rtol=1e-12, atol=1e-12)

    ax.plot(sol1.y[0], sol1.y[1], 'C1', label='Unstable 1')
    ax.plot(sol2.y[0], sol2.y[1], 'C3', label='Unstable 2')

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim([0, 2])

    ax.scatter(1 - MU, 0, marker='o', color='lightseagreen', label='Earth', zorder=2.5)
    ax.scatter(x_L1, 0, marker='.', color='black', label='L1', zorder=2.5)
    ax.scatter(-MU, 0, marker='*', color='gold', label='Sun', zorder=2.5)

    axins.plot(sol1.y[0], sol1.y[1], 'C1')
    axins.plot(sol2.y[0], sol2.y[1], 'C3')

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_aspect('equal', adjustable='datalim')
    axins.grid()

    axins.scatter(1 - MU, 0, marker='o', color='lightseagreen', label='Earth', zorder=2.5)
    axins.scatter(x_L1, 0, marker='.', color='black', label='L1', zorder=2.5)
    axins.scatter(-MU, 0, marker='*', color='gold', label='Sun', zorder=2.5)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.legend(loc='best')

    plt.tight_layout()

    plt.savefig('figures/manifolds.png', dpi=200)

part2 = 1
if part2:
    # 3.1
    x_0_jpl = np.array([9.9140721635723439E-1,
                        3.9579496236656470E-23,
                        1.1087703436848695E-2,
                        2.2425423430179550E-15,
                        1.5379010802073182E-2,
                        -9.5449845213924625E-15])

    # propagate trajectory
    sol_jpl = integrate.solve_ivp(motion_eqn_jpl, [0, T_JPL], x_0_jpl, rtol=1e-12, atol=1e-12)

    # create the inflated Earth
    phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    inflated_earth_r = 956700000 / AU

    x_earth = inflated_earth_r * np.sin(theta) * np.cos(phi) + 1 - MU_JPL
    y_earth = inflated_earth_r * np.sin(theta) * np.sin(phi)
    z_earth = inflated_earth_r * np.cos(theta)

    # plot
    fig = plt.figure(figsize=(9, 9.5))
    # plt.rc('font', size=15)
    gs = gridspec.GridSpec(3, 2, fig, height_ratios=[1, 1, 0.15])

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot(sol_jpl.y[0], sol_jpl.y[1], sol_jpl.y[2], label='JPL trajectory')
    ax1.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.4, rstride=5, cstride=5, label='"Inflated" Earth')
    ax1.scatter(*L1_JPL, color='red', label='L1')
    ax1.set_title('3D trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_aspect('equal')

    # (x, y)-projection
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(sol_jpl.y[0], sol_jpl.y[1], label='JPL trajectory', zorder=2)
    ax2.pcolormesh(x_earth, y_earth, np.ones_like(z_earth), alpha=0.4, shading='auto', zorder=5)
    ax2.scatter(L1_JPL[0], L1_JPL[1], color='red', label='L1', zorder=6)
    ax2.grid(zorder=1)
    ax2.set_title('(x, y)-projection')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # (x, z)-projection
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(sol_jpl.y[0], sol_jpl.y[2], label='JPL trajectory', zorder=3)
    ax3.pcolormesh(x_earth, z_earth, np.ones_like(y_earth), alpha=0.4, shading='auto', zorder=2)
    ax3.scatter(L1_JPL[0], L1_JPL[2], color='red', label='L1', zorder=6)
    ax3.grid(zorder=1)
    ax3.set_title('(x, z)-projection')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_aspect('equal')
    ax3.set_xticks(np.array([0.9900, 0.9925, 0.9950, 0.9975, 1.0000, 1.0025, 1.0050, 1.0075]))
    ax3.set_xticklabels(np.array(['0.9900', '0.9925', '0.9950', '0.9975', '1.0000', '1.0025', '1.0050', '1.0075']),
                        rotation=45)

    # (y, z)-projection
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(sol_jpl.y[1], sol_jpl.y[2], label='JPL trajectory', zorder=2)
    ax4.pcolormesh(y_earth, z_earth, np.ones_like(x_earth), alpha=0.4, shading='auto', zorder=5)
    ax4.scatter(L1_JPL[1], L1_JPL[2], color='red', label='L1', zorder=6)
    ax4.grid(zorder=1)
    ax4.set_title('(y, z)-projection')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.set_aspect('equal')
    plt.xticks(rotation=45)

    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    lines, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(lines, labels, loc='center', ncol=2)

    fig.tight_layout()
    plt.savefig('figures/northern_halo.png', dpi=200)

    # 3.2a
    # sol_uncorrected = integrate.solve_ivp(motion_eqn, [0, period], x_0_jpl, rtol=1e-12, atol=1e-12)

    Phi_0 = np.eye(6).reshape(36)
    x_0 = np.append(x_0_jpl, Phi_0)

    x_0_corrected, period_corrected = diff_correction(x_0, MU, 0)
    sol_corrected = scipy.integrate.solve_ivp(lambda t, x: variational_eqn(t, x, MU, 0), [0, period_corrected],
                                              x_0_corrected, rtol=1e-12, atol=1e-12)

    # plot
    fig = plt.figure(figsize=(9, 9.5))
    # plt.rc('font', size=15)
    gs = gridspec.GridSpec(3, 2, fig, height_ratios=[1, 1, 0.15])

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot(sol_jpl.y[0], sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory')
    ax1.plot(sol_corrected.y[0], sol_corrected.y[1], sol_corrected.y[2], '--', lw=3, label='Corrected trajectory')
    ax1.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.4, rstride=5, cstride=5, label='"Inflated" Earth')
    ax1.scatter(*L1_JPL, color='red', label='L1')
    ax1.set_title('3D trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_aspect('equal')

    # (x, y)-projection
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(sol_jpl.y[0], sol_jpl.y[1], lw=2, label='JPL trajectory', zorder=2)
    ax2.plot(sol_corrected.y[0], sol_corrected.y[1], '--', lw=3, label='Corrected trajectory', zorder=3)
    ax2.pcolormesh(x_earth, y_earth, np.ones_like(z_earth), alpha=0.4, shading='auto', zorder=5)
    ax2.scatter(L1_JPL[0], L1_JPL[1], color='red', label='L1', zorder=6)
    ax2.grid(zorder=1)
    ax2.set_title('(x, y)-projection')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # (x, z)-projection
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(sol_jpl.y[0], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=3)
    ax3.plot(sol_corrected.y[0, :len(sol_corrected.y[0]) // 2], sol_corrected.y[2, :len(sol_corrected.y[0]) // 2], '--',
             lw=3, label='Corrected trajectory', zorder=4)
    ax3.pcolormesh(x_earth, z_earth, np.ones_like(y_earth), alpha=0.4, shading='auto', zorder=2)
    ax3.scatter(L1_JPL[0], L1_JPL[2], color='red', label='L1', zorder=6)
    ax3.grid(zorder=1)
    ax3.set_title('(x, z)-projection')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_aspect('equal')
    ax3.set_xticks(np.array([0.9900, 0.9925, 0.9950, 0.9975, 1.0000, 1.0025, 1.0050, 1.0075]))
    ax3.set_xticklabels(np.array(['0.9900', '0.9925', '0.9950', '0.9975', '1.0000', '1.0025', '1.0050', '1.0075']),
                        rotation=45)

    # (y, z)-projection
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=2)
    ax4.plot(sol_corrected.y[1], sol_corrected.y[2], '--', lw=3, label='Corrected trajectory', zorder=3)
    ax4.pcolormesh(y_earth, z_earth, np.ones_like(x_earth), alpha=0.4, shading='auto', zorder=5)
    ax4.scatter(L1_JPL[1], L1_JPL[2], color='red', label='L1', zorder=6)
    ax4.grid(zorder=1)
    ax4.set_title('(y, z)-projection')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.set_aspect('equal')
    plt.xticks(rotation=45)

    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    lines, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(lines, labels, loc='center', ncol=2)

    fig.tight_layout()
    plt.savefig('figures/northern_halo_corrected.png', dpi=200)

    # 3.2b

    beta = 0.009
    beta_vec = np.arange(0.0005, beta + 0.0005, 0.0005)
    x_0_inter = np.copy(x_0_corrected)
    for beta_inter in beta_vec:
        print('Computing beta:\t', beta_inter)
        x_0_inter, period_inter = diff_correction(x_0_inter, MU, beta_inter)
    x_0_corrected_beta = x_0_inter
    period_corrected_beta = period_inter
    sol_corrected_beta = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta),
                                                   [0, period_corrected_beta], x_0_corrected_beta, rtol=1e-12,
                                                   atol=1e-12)

    # plot
    fig = plt.figure(figsize=(9.5, 9.5))
    # plt.rc('font', size=15)
    gs = gridspec.GridSpec(3, 2, fig, height_ratios=[1, 1, 0.15])


    # 3D Trajectory Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot(sol_jpl.y[0], sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory')
    ax1.plot(sol_corrected.y[0], sol_corrected.y[1], sol_corrected.y[2], '--', lw=3,
             label='Corrected trajectory')
    ax1.plot(sol_corrected_beta.y[0], sol_corrected_beta.y[1], sol_corrected_beta.y[2], ':', lw=3,
             label='Corrected trajectory w/ solar sail')
    ax1.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.4, rstride=5, cstride=5, label='"Inflated" Earth')
    ax1.scatter(*L1_JPL, color='red', label='L1')
    ax1.set_title('3D trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_aspect('equal')

    # (x, y)-projection
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(sol_jpl.y[0], sol_jpl.y[1], lw=2, label='JPL trajectory', zorder=2)
    ax2.plot(sol_corrected.y[0], sol_corrected.y[1], '--', lw=3, label='Corrected trajectory', zorder=3)
    ax2.plot(sol_corrected_beta.y[0], sol_corrected_beta.y[1], ':', lw=3,
             label='Corrected trajectory w/ solar sail', zorder=4)
    ax2.pcolormesh(x_earth, y_earth, np.ones_like(z_earth), alpha=0.4, shading='auto', zorder=5)
    ax2.scatter(L1_JPL[0], L1_JPL[1], color='red', label='L1', zorder=6)
    ax2.grid(zorder=1)
    ax2.set_title('(x, y)-projection')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # (x, z)-projection
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(sol_jpl.y[0], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=3)
    ax3.plot(sol_corrected.y[0, :len(sol_corrected.y[0])//2], sol_corrected.y[2, :len(sol_corrected.y[0])//2], '--',
             lw=3, label='Corrected trajectory', zorder=4)
    ax3.plot(sol_corrected_beta.y[0, :len(sol_corrected.y[0])//2], sol_corrected_beta.y[2, :len(sol_corrected.y[0])//2],
             ':', lw=3, label='Corrected trajectory w/ solar sail', zorder=5)
    ax3.pcolormesh(x_earth, z_earth, np.ones_like(y_earth), alpha=0.4, shading='auto', zorder=2)
    ax3.scatter(L1_JPL[0], L1_JPL[2], color='red', label='L1', zorder=6)
    ax3.grid(zorder=1)
    ax3.set_title('(x, z)-projection')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_aspect('equal')
    ax3.set_xticks(np.array([0.9875, 0.9900, 0.9925, 0.9950, 0.9975, 1.0000, 1.0025, 1.0050, 1.0075]))
    ax3.set_xticklabels(np.array(['0.9875', '0.9900', '0.9925', '0.9950', '0.9975', '1.0000', '1.0025', '1.0050',
                                  '1.0075']), rotation=45)

    # (y, z)-projection
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=2)
    ax4.plot(sol_corrected.y[1], sol_corrected.y[2], '--', lw=3, label='Corrected trajectory', zorder=3)
    ax4.plot(sol_corrected_beta.y[1], sol_corrected_beta.y[2], ':', lw=3,
             label='Corrected trajectory w/ solar sail', zorder=4)
    ax4.pcolormesh(y_earth, z_earth, np.ones_like(x_earth), alpha=0.4, shading='auto', zorder=5)
    ax4.scatter(L1_JPL[1], L1_JPL[2], color='red', label='L1', zorder=6)
    ax4.grid(zorder=1)
    ax4.set_title('(y, z)-projection')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.set_aspect('equal')
    plt.xticks(rotation=45)

    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    lines, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(lines, labels, loc='center', ncol=2)

    fig.tight_layout()
    plt.savefig('figures/northern_halo_corrected_beta.png', dpi=200)

    # 3.2c

    radius = inflated_earth_r + 1
    x_0_inter = np.copy(x_0_corrected_beta)

    it = 0
    while radius > inflated_earth_r:
        print('Computing step:\t', it)
        x_0_inter[2] -= 0.0005
        x_0_inter, period_inter = diff_correction(x_0_inter, MU, beta)
        sol_corrected_beta_shade = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta),
                                                             [0, period_inter], x_0_inter, rtol=1e-12, atol=1e-12)
        radius = np.max(np.sqrt(sol_corrected_beta_shade.y[1]**2 + sol_corrected_beta_shade.y[2]**2))
        it += 1
    x_0_corrected_beta_shade = x_0_inter
    period_corrected_beta_shade = period_inter

    # plot
    fig = plt.figure(figsize=(9.5, 9.5))
    # plt.rc('font', size=15)
    gs = gridspec.GridSpec(3, 2, fig, height_ratios=[1, 1, 0.15])

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot(sol_jpl.y[0], sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory')
    ax1.plot(sol_corrected.y[0], sol_corrected.y[1], sol_corrected.y[2], '--', lw=3,
             label='Corrected trajectory')
    ax1.plot(sol_corrected_beta.y[0], sol_corrected_beta.y[1], sol_corrected_beta.y[2], ':', lw=3,
             label='Corrected trajectory w/ solar sail')
    ax1.plot(sol_corrected_beta_shade.y[0], sol_corrected_beta_shade.y[1], sol_corrected_beta_shade.y[2], '-.', lw=3,
             label='Corrected trajectory w/ solar sail and $z_0$ continuation')
    ax1.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.4, rstride=5, cstride=5, label='"Inflated" Earth')
    ax1.scatter(*L1_JPL, color='red', label='L1')
    ax1.set_title('3D trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_aspect('equal')

    # (x, y)-projection
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(sol_jpl.y[0], sol_jpl.y[1], lw=2, label='JPL trajectory', zorder=3)
    ax2.plot(sol_corrected.y[0], sol_corrected.y[1], '--', lw=3, label='Corrected trajectory', zorder=4)
    ax2.plot(sol_corrected_beta.y[0], sol_corrected_beta.y[1], ':', lw=3,
             label='Corrected trajectory w/ solar sail', zorder=5)
    ax2.plot(sol_corrected_beta_shade.y[0], sol_corrected_beta_shade.y[1], '-.', lw=3,
             label='Corrected trajectory w/ solar sail and $z_0$ continuation', zorder=2)
    ax2.pcolormesh(x_earth, y_earth, np.ones_like(z_earth), alpha=0.4, shading='auto', zorder=6)
    ax2.scatter(L1_JPL[0], L1_JPL[1], color='red', label='L1', zorder=7)
    ax2.grid(zorder=1)
    ax2.set_title('$(x, y)$-projection')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # (x, z)-projection
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(sol_jpl.y[0], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=4)
    ax3.plot(sol_corrected.y[0, :len(sol_corrected.y[0]) // 2], sol_corrected.y[2, :len(sol_corrected.y[0]) // 2], '--',
             lw=3, label='Corrected trajectory', zorder=5)
    ax3.plot(sol_corrected_beta.y[0, :len(sol_corrected.y[0]) // 2],
             sol_corrected_beta.y[2, :len(sol_corrected.y[0]) // 2],
             ':', lw=3, label='Corrected trajectory w/ solar sail', zorder=6)
    ax3.plot(sol_corrected_beta_shade.y[0, :len(sol_corrected_beta_shade.y[0]) // 2],
             sol_corrected_beta_shade.y[2, :len(sol_corrected_beta_shade.y[0]) // 2],
             '-.', lw=3, label='Corrected trajectory w/ solar sail and $z_0$ continuation', zorder=3)
    ax3.pcolormesh(x_earth, z_earth, np.ones_like(y_earth), alpha=0.4, shading='auto', zorder=2)
    ax3.scatter(L1_JPL[0], L1_JPL[2], color='red', label='L1', zorder=7)
    ax3.grid(zorder=1)
    ax3.set_title('$(x, z)$-projection')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_aspect('equal')
    ax3.set_xticks(np.array([0.9875, 0.9900, 0.9925, 0.9950, 0.9975, 1.0000, 1.0025, 1.0050, 1.0075]))
    ax3.set_xticklabels(np.array(['0.9875', '0.9900', '0.9925', '0.9950', '0.9975', '1.0000', '1.0025', '1.0050',
                                  '1.0075']), rotation=45)

    # (y, z)-projection
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(sol_jpl.y[1], sol_jpl.y[2], lw=2, label='JPL trajectory', zorder=3)
    ax4.plot(sol_corrected.y[1], sol_corrected.y[2], '--', lw=3, label='Corrected trajectory', zorder=4)
    ax4.plot(sol_corrected_beta.y[1], sol_corrected_beta.y[2], ':', lw=3,
             label='Corrected trajectory w/ solar sail', zorder=5)
    ax4.plot(sol_corrected_beta_shade.y[1], sol_corrected_beta_shade.y[2], '-.', lw=3,
             label='Corrected trajectory w/ solar sail and $z_0$ continuation', zorder=2)
    ax4.pcolormesh(y_earth, z_earth, np.ones_like(x_earth), alpha=0.4, shading='auto', zorder=6)
    ax4.scatter(L1_JPL[1], L1_JPL[2], color='red', label='L1', zorder=7)
    ax4.grid(zorder=1)
    ax4.set_title('$(y, z)$-projection')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.set_aspect('equal')
    plt.xticks(rotation=45)

    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    lines, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(lines, labels, loc='center', ncol=2)

    fig.tight_layout()
    plt.savefig('figures/northern_halo_corrected_beta_shade.png', dpi=200)

    # 3.3

    max_dist = np.max(LA.norm(sol_corrected_beta_shade.y[:3] - np.array([[1 - MU], [0], [0]]), axis=0))
    R_disk = R_SUN * max_dist * np.sqrt(1.7 / 100)
    A_disk = np.pi * R_disk ** 2
    print('Max distance from Earth:\t', max_dist)
    print('Disk radius:\t\t\t\t', R_disk, '[m]')
    print('Disk area:\t\t\t\t', A_disk, '[$m^2$]')

    # 4.1

    event = lambda t, x: x[1]
    event.terminal = False
    sol_stability = scipy.integrate.solve_ivp(lambda t, x: variational_eqn(t, x, MU, beta),
                                              [0, 2*period_corrected_beta_shade], x_0_corrected_beta_shade,
                                              events=event, rtol=1e-12, atol=1e-12)
    monodromy_matrix = sol_stability.y_events[0][1, 6:].reshape(6, 6)
    lam, zeta = LA.eig(monodromy_matrix)
    print('Eigenvalues:', lam[0], lam[1], lam[2], lam[3], lam[4], lam[5])

    v_1 = (lam[0] + 1 / lam[1]) / 2
    v_2 = (lam[2] + 1 / lam[3]) / 2
    v_3 = (lam[4] + 1 / lam[5]) / 2
    print('Stability index v1: ', np.abs(v_1))
    print('Stability index v2: ', np.abs(v_2))
    print('Stability index v3: ', np.abs(v_3))

    # 4.2

    period = 5 * 2 * np.pi
    number_of_points = 10
    delta_time = period_corrected_beta_shade / number_of_points

    real_zeta = np.real(zeta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    x1, x2, y1, y2 = 0.98, 1.01, -0.01, 0.01

    # unstable
    x_0_unstable1 = np.append(x_0_corrected_beta_shade[:6] + 1e-8 * real_zeta[:, 0], Phi_0.reshape(-1,))
    x_0_unstable2 = np.append(x_0_corrected_beta_shade[:6] - 1e-8 * real_zeta[:, 0], Phi_0.reshape(-1,))
    manifold1 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, 10 * np.pi],
                                          x_0_unstable1, rtol=1e-12, atol=1e-12)
    manifold2 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, 10 * np.pi],
                                          x_0_unstable2, rtol=1e-12, atol=1e-12)

    # stable
    x_0_stable1 = np.append(x_0_corrected_beta_shade[:6] + 1e-8 * real_zeta[:, 1], Phi_0.reshape(-1,))
    x_0_stable2 = np.append(x_0_corrected_beta_shade[:6] - 1e-8 * real_zeta[:, 1], Phi_0.reshape(-1,))
    manifold3 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, -10 * np.pi],
                                          x_0_stable1, rtol=1e-12, atol=1e-12)
    manifold4 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, -10 * np.pi],
                                          x_0_stable2, rtol=1e-12, atol=1e-12)

    ax1.plot(manifold3.y[0], manifold3.y[1], 'C0')
    ax1.plot(manifold4.y[0], manifold4.y[1], 'C2')
    ax1.plot(manifold1.y[0], manifold1.y[1], 'C1')
    ax1.plot(manifold2.y[0], manifold2.y[1], 'C3')
    ax1.grid()
    ax1.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    ax1.scatter(-MU, 0, marker='*', color='gold', zorder=2.5)
    ax1.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    axins1 = ax1.inset_axes([0.35, 0.35, 0.3, 0.3])
    axins1.plot(manifold3.y[0], manifold3.y[1], 'C0')
    axins1.plot(manifold4.y[0], manifold4.y[1], 'C2')
    axins1.plot(manifold1.y[0], manifold1.y[1], 'C1')
    axins1.plot(manifold2.y[0], manifold2.y[1], 'C3')
    axins1.grid()
    axins1.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    axins1.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.set_aspect('equal', adjustable='datalim')

    ax2.plot(manifold3.y[0], manifold3.y[2], 'C0', label='Stable 1')
    ax2.plot(manifold4.y[0], manifold4.y[2], 'C2', label='Stable 2')
    ax2.plot(manifold1.y[0], manifold1.y[2], 'C1', label='Unstable 1')
    ax2.plot(manifold2.y[0], manifold2.y[2], 'C3', label='Unstable 2')
    ax2.scatter(1 - MU, 0, marker='o', color='lightseagreen', label='Earth', zorder=2.5)
    ax2.scatter(-MU, 0, marker='*', color='gold', label='Sun', zorder=2.5)
    ax2.scatter(x_L1, 0, marker='.', color='black', label='L1', zorder=2.5)
    ax2.grid()
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$z$')

    axins2 = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins2.plot(manifold3.y[0], manifold3.y[2], 'C0')
    axins2.plot(manifold4.y[0], manifold4.y[2], 'C2')
    axins2.plot(manifold1.y[0], manifold1.y[2], 'C1')
    axins2.plot(manifold2.y[0], manifold2.y[2], 'C3')
    axins2.grid()
    axins2.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    axins2.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.set_aspect('equal', adjustable='datalim')

    ax1.indicate_inset_zoom(axins1, edgecolor="black")
    ax2.indicate_inset_zoom(axins2, edgecolor="black")

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.savefig('figures/manifolds_w_sail.png', dpi=200)

    init_cond = scipy.integrate.solve_ivp(lambda t, x: variational_eqn(t, x, MU, beta),
                                          [0, 2 * period_corrected_beta_shade], x_0_corrected_beta_shade, rtol=1e-12,
                                          atol=1e-12, t_eval=delta_time*np.array(range(10)))

    # all points
    dist_unstable = np.empty(20,)
    dist_stable = np.empty(20,)

    vel_unstable = np.empty(20,)
    vel_stable = np.empty(20,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    axins1 = ax1.inset_axes([0.35, 0.35, 0.3, 0.3])
    axins2 = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])

    ax2.scatter(1 - MU, 0, marker='o', color='lightseagreen')
    ax2.scatter(-MU, 0, marker='*', color='gold')
    ax2.scatter(x_L1, 0, marker='.', color='black')

    for i in range(10):
        # transition matrix from initial point
        Phi_0 = init_cond.y[6:, i].reshape(6, 6)
        # initial state at given point
        x_0 = init_cond.y[:6, i]

        # unstable
        x_0_unstable1 = np.append(x_0 + 1e-8 * real_zeta[:, 0], Phi_0.reshape(-1, ))
        x_0_unstable2 = np.append(x_0 - 1e-8 * real_zeta[:, 0], Phi_0.reshape(-1, ))
        manifold1 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, 10 * np.pi],
                                              x_0_unstable1, rtol=1e-12, atol=1e-12)
        manifold2 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, 10 * np.pi],
                                              x_0_unstable2, rtol=1e-12, atol=1e-12)

        idx = LA.norm(manifold1.y[:3] - np.array([[1-MU], [0], [0]]), axis=0).argmin()
        dist_unstable[i] = LA.norm(manifold1.y[:3, idx] - np.array([1-MU, 0, 0]))
        vel_unstable[i] = LA.norm(manifold1.y[3:6, idx])

        idx = LA.norm(manifold2.y[:3] - np.array([[1-MU], [0], [0]]), axis=0).argmin()
        dist_unstable[10+i] = LA.norm(manifold2.y[:3, idx] - np.array([1-MU, 0, 0]))
        vel_unstable[10+i] = LA.norm(manifold2.y[3:6, idx])

        # stable
        x_0_stable1 = np.append(x_0 + 1e-8 * real_zeta[:, 1], Phi_0.reshape(-1, ))
        x_0_stable2 = np.append(x_0 - 1e-8 * real_zeta[:, 1], Phi_0.reshape(-1, ))
        manifold3 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, -10 * np.pi],
                                              x_0_stable1, rtol=1e-12, atol=1e-12)
        manifold4 = scipy.integrate.solve_ivp(lambda t, y: variational_eqn(t, y, MU, beta), [0, -10 * np.pi],
                                              x_0_stable2, rtol=1e-12, atol=1e-12)

        idx = LA.norm(manifold3.y[:3] - np.array([[1 - MU], [0], [0]]), axis=0).argmin()
        dist_stable[i] = LA.norm(manifold3.y[:3, idx] - np.array([1 - MU, 0, 0]))
        vel_stable[i] = LA.norm(manifold3.y[3:6, idx])

        idx = LA.norm(manifold4.y[:3] - np.array([[1 - MU], [0], [0]]), axis=0).argmin()
        dist_stable[10 + i] = LA.norm(manifold4.y[:3, idx] - np.array([1 - MU, 0, 0]))
        vel_stable[10 + i] = LA.norm(manifold4.y[3:6, idx])

        ax1.plot(manifold3.y[0], manifold3.y[1], 'C0', label='Stable 1')
        ax1.plot(manifold4.y[0], manifold4.y[1], 'C2', label='Stable 2')
        axins1.plot(manifold3.y[0], manifold3.y[1], 'C0')
        axins1.plot(manifold4.y[0], manifold4.y[1], 'C2')

        ax2.plot(manifold3.y[0], manifold3.y[2], 'C0', label='Stable 1')
        ax2.plot(manifold4.y[0], manifold4.y[2], 'C2', label='Stable 2')
        axins2.plot(manifold3.y[0], manifold3.y[2], 'C0')
        axins2.plot(manifold4.y[0], manifold4.y[2], 'C2')

        ax1.plot(manifold1.y[0], manifold1.y[1], 'C1', label='Unstable 1')
        ax1.plot(manifold2.y[0], manifold2.y[1], 'C3', label='Unstable 2')
        axins1.plot(manifold1.y[0], manifold1.y[1], 'C1')
        axins1.plot(manifold2.y[0], manifold2.y[1], 'C3')

        ax2.plot(manifold1.y[0], manifold1.y[2], 'C1', label='Unstable 1')
        ax2.plot(manifold2.y[0], manifold2.y[2], 'C3', label='Unstable 2')
        axins2.plot(manifold1.y[0], manifold1.y[2], 'C1')
        axins2.plot(manifold2.y[0], manifold2.y[2], 'C3')

    ax1.grid()
    ax2.grid()
    axins1.grid()
    axins2.grid()

    ax1.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    ax1.scatter(-MU, 0, marker='*', color='gold', zorder=2.5)
    ax1.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    ax2.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    ax2.scatter(-MU, 0, marker='*', color='gold', zorder=2.5)
    ax2.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    axins1.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    axins1.scatter(-MU, 0, marker='*', color='gold', zorder=2.5)
    axins1.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)
    axins2.scatter(1 - MU, 0, marker='o', color='lightseagreen', zorder=2.5)
    axins2.scatter(-MU, 0, marker='*', color='gold', zorder=2.5)
    axins2.scatter(x_L1, 0, marker='.', color='black', zorder=2.5)

    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$z$')

    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.set_aspect('equal', adjustable='datalim')

    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.set_aspect('equal', adjustable='datalim')

    ax1.indicate_inset_zoom(axins1, edgecolor="black")
    ax2.indicate_inset_zoom(axins2, edgecolor="black")

    ax2.legend(['Earth', 'Sun', 'L1', 'Stable 1', 'Stable 2', 'Unstable 1', 'Unstable 2'], loc='center left',
               bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    plt.savefig('figures/manifolds_w_sail_all.png', dpi=200)

    # 4.3
    idx = dist_stable.argmin()
    print('Closest distance on stable manifold:\t', dist_stable[idx])
    print('Associated speed:\t', vel_stable[idx])

    idx = dist_unstable.argmin()
    print('Closest distance on unstable manifold:\t', dist_unstable[idx])
    print('Associated speed:\t', vel_unstable[idx])








    # r1 = LA.norm(x_0_corrected_beta_shade[:3] - np.array([-MU, 0, 0]))
    # r2 = LA.norm(x_0_corrected_beta_shade[:3] - np.array([1 - MU, 0, 0]))
    #
    # U_xx = 1 + (1 - MU) / r1 ** 5 * (2 * (x_0_corrected_beta_shade[0] + MU) ** 2 - x_0_corrected_beta_shade[1] ** 2
    #     - x_0_corrected_beta_shade[2] ** 2) + MU / r2 ** 5 * (2 * (x_0_corrected_beta_shade[0] - (1 - MU)) ** 2
    #     - x_0_corrected_beta_shade[1] ** 2 - x_0_corrected_beta_shade[2] ** 2)
    # U_yy = 1 - (1 - MU) / r1 ** 5 * ((x_0_corrected_beta_shade[0] + MU) ** 2 - 2 * x_0_corrected_beta_shade[1] ** 2
    #     + x_0_corrected_beta_shade[2] ** 2) - MU / r2 ** 5 * ((x_0_corrected_beta_shade[0] - (1 - MU)) ** 2
    #     - 2 * x_0_corrected_beta_shade[1] ** 2 + x_0_corrected_beta_shade[2] ** 2)
    # U_zz = -((1 - MU) * ((x_0_corrected_beta_shade[0] + MU) ** 2 + x_0_corrected_beta_shade[1] ** 2
    #     - 2 * x_0_corrected_beta_shade[2] ** 2) / r1 ** 5 + MU * ((1 - MU - x_0_corrected_beta_shade[0]) ** 2
    #     + x_0_corrected_beta_shade[1] ** 2 - 2 * x_0_corrected_beta_shade[2] ** 2) / r2 ** 5)
    # U_xy = (3 * x_0_corrected_beta_shade[1] * (1 - MU) * (x_0_corrected_beta_shade[0] + MU) / r1 ** 5 + 3 * MU *
    #         x_0_corrected_beta_shade[1] * (x_0_corrected_beta_shade[0] - (1 - MU)) / r2 ** 5)
    #
    # U_xz = (3 * (1 - MU) * x_0_corrected_beta_shade[2] * (x_0_corrected_beta_shade[0] + MU) / r1 ** 5 + 3 * MU *
    #         x_0_corrected_beta_shade[2] * (x_0_corrected_beta_shade[0] - (1 - MU)) / r2 ** 5)
    #
    # U_yz = (3 * (1 - MU) * x_0_corrected_beta_shade[1] * x_0_corrected_beta_shade[2] / r1 ** 5 + 3 * MU *
    #         x_0_corrected_beta_shade[1] * x_0_corrected_beta_shade[2] / r2 ** 5)
    #
    # # solar sail acceleration derivatives
    # a_xx = -2 * beta * (1 - MU) * (MU + x_0_corrected_beta_shade[0]) * (2 * (MU + x_0_corrected_beta_shade[0]) ** 2 - r1 ** 2) / r1 ** 6
    # a_xy = -4 * beta * (1 - MU) * (x_0_corrected_beta_shade[0] + MU) ** 2 / r1 ** 6 * x_0_corrected_beta_shade[1]
    # a_xz = -4 * beta * (1 - MU) * (x_0_corrected_beta_shade[0] + MU) ** 2 / r1 ** 6 * x_0_corrected_beta_shade[2]
    #
    # A = np.array([[0, 0, 0, 1, 0, 0],
    #               [0, 0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 0, 1],
    #               [U_xx + a_xx, U_xy + a_xy, U_xz + a_xz, 0, 2, 0],
    #               [U_xy, U_yy, U_yz, -2, 0, 0],
    #               [U_xz, U_yz, U_zz, 0, 0, 0]])
    #
    # lam, zeta = LA.eig(A)
    # print('Eigenvalues:', lam[0], lam[1], lam[2], lam[3], lam[4], lam[5])
