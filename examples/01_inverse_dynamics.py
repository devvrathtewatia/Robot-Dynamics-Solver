"""Minimal example: the torque needed to hold and accelerate an arm.

Shows the whole point of inverse dynamics in a few lines, and confirms that the
independent backends agree.
"""
from __future__ import annotations

import numpy as np
from _common import style

import robot_dynamics as rd


def main() -> None:
    robot = rd.planar_chain(3, length=0.4, mass=1.0)
    print(robot)

    q = np.array([0.3, -0.6, 0.9])
    qd = np.array([0.5, 0.2, -0.4])
    qdd = np.array([1.0, -0.5, 0.3])

    print('\nTerm by term:')
    print('  M(q) =\n', np.round(robot.mass_matrix(q), 5))
    print('  C(q,qd) qd =', np.round(robot.coriolis(q, qd) @ qd, 5))
    print('  G(q)       =', np.round(robot.gravity_torque(q), 5))

    tau_rnea = robot.inverse_dynamics(q, qd, qdd, method='rnea')
    tau_lag = robot.inverse_dynamics(q, qd, qdd, method='lagrangian')
    print('\ntau (RNEA)       =', np.round(tau_rnea, 6))
    print('tau (Lagrangian) =', np.round(tau_lag, 6))
    print(f'agreement        = {np.max(np.abs(tau_rnea - tau_lag)):.2e}')

    qdd_back = robot.forward_dynamics(q, qd, tau_rnea)
    print('\nround trip: |qdd_recovered - qdd| = '
          f'{np.max(np.abs(qdd_back - qdd)):.2e}')

    # gravity torque against configuration - a physically readable result
    plt = style()
    if plt is None:
        return
    angles = np.linspace(-np.pi, np.pi, 400)
    G = np.array([robot.gravity_torque([a, 0.0, 0.0]) for a in angles])
    fig, ax = plt.subplots(figsize=(6, 3))
    for j in range(robot.n):
        ax.plot(np.degrees(angles), G[:, j], label=f'joint {j + 1}')
    ax.set_xlabel('shoulder angle (deg)')
    ax.set_ylabel('gravity torque (N m)')
    ax.set_title('Gravity torque of a 3-link planar arm, elbow and wrist straight')
    ax.legend()
    fig.tight_layout()
    from _common import figure_path
    out = figure_path('gravity_torque.png')
    fig.savefig(out)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
