"""Phase 10: model-based control, and the tracking error it buys.

Runs PD-plus-gravity and computed torque on the same arm with the same gains and the
same trajectory, so the difference is attributable to the dynamics model alone.
"""
from __future__ import annotations

import numpy as np
from _common import figure_path, style

import robot_dynamics as rd
from robot_dynamics.control import ComputedTorque, FeedforwardPD, PDGravity
from robot_dynamics.simulation import simulate, tracking_error
from robot_dynamics.trajectory import quintic


def main() -> None:
    robot = rd.planar_chain(2, length=0.5, mass=1.0)
    q0 = np.zeros(robot.n)
    traj = quintic(q0, np.array([1.2, -0.8]), duration=1.0)

    # Gains are limited by the SIMULATION, not the controller: a PD loop leaves a
    # fast pole near kd/M_min, so kd is set from the smallest inertia to keep a 1 ms
    # step stable. Computed torque has no such constraint - it cancels M.
    m_min = np.linalg.eigvalsh(robot.mass_matrix(q0)).min()
    kp = 100.0
    kd = 2.0 * np.sqrt(kp * m_min)
    print(f'M_min = {m_min:.5f} kg m^2  ->  kp = {kp:.0f}, kd = {kd:.3f}')

    runs = {}
    for name, ctrl in (('PD + gravity', PDGravity(robot, kp, kd, traj)),
                       ('feedforward + PD', FeedforwardPD(robot, kp, kd, traj)),
                       ('computed torque', ComputedTorque(robot, kp, kd, traj))):
        res = simulate(robot, q0, np.zeros(robot.n), duration=1.0, dt=1e-3,
                       control=ctrl)
        err = tracking_error(res, traj)
        runs[name] = (res, err)
        print(f'  {name:<18} max |error| = {err["max_abs"]:.3e} rad   '
              f'rms = {np.round(err["rms_per_joint"], 8)}')

    plt = style()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    res_ct, err_ct = runs['computed torque']
    axes[0].plot(res_ct['t'], err_ct['q_desired'][:, 0], 'k--',
                 label='desired q1')
    axes[0].plot(res_ct['t'], err_ct['q_desired'][:, 1], 'k:', label='desired q2')
    for name, (res, _) in runs.items():
        axes[0].plot(res['t'], res['q'][:, 0], alpha=0.8, label=f'{name} q1')
    axes[0].set_title('Tracking')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('rad')
    axes[0].legend(fontsize=6)

    for name, (res, err) in runs.items():
        axes[1].semilogy(res['t'], np.abs(err['error']).max(axis=1) + 1e-16,
                         label=name)
    axes[1].set_title('Tracking error (log scale)')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('|error| (rad)')
    axes[1].legend(fontsize=7)

    for j in range(robot.n):
        axes[2].plot(res_ct['t'], res_ct['tau'][:, j], label=f'tau{j + 1}')
    axes[2].set_title('Computed-torque command')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('N m')
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    out = figure_path('control_tracking.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
