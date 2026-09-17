"""Phase 9's deliverable: a trajectory, and the joint torques it demands.

Plots q(t), qd(t), qdd(t) and tau(t) for every joint - the figure the README asks
for. Reading the torque plot is the point: the peaks say what the actuators must
deliver, which is the first engineering question a dynamics model answers.
"""
from __future__ import annotations

import numpy as np
from _common import figure_path, style

import robot_dynamics as rd
from robot_dynamics.trajectory import quintic


def main() -> None:
    robot = rd.spatial_chain(4, length=0.3, mass=1.2)
    q0 = np.zeros(robot.n)
    qf = np.array([1.2, -0.9, 0.7, -0.5])
    traj = quintic(q0, qf, duration=2.0)

    ts, qs, qds, qdds = traj.sample(dt=0.005)
    taus = np.array([robot.inverse_dynamics(q, qd, qdd)
                     for q, qd, qdd in zip(qs, qds, qdds, strict=True)])

    print('peak |tau| per joint (N m):', np.round(np.max(np.abs(taus), axis=0), 4))
    print('static hold at the start  :', np.round(robot.gravity_torque(q0), 4))
    print('static hold at the end    :', np.round(robot.gravity_torque(qf), 4))

    plt = style()
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    labels = [f'joint {j + 1}' for j in range(robot.n)]
    for j in range(robot.n):
        axes[0, 0].plot(ts, qs[:, j], label=labels[j])
        axes[0, 1].plot(ts, qds[:, j], label=labels[j])
        axes[1, 0].plot(ts, qdds[:, j], label=labels[j])
        axes[1, 1].plot(ts, taus[:, j], label=labels[j])
    for ax, title, ylab in ((axes[0, 0], 'Desired position q(t)', 'rad'),
                            (axes[0, 1], 'Velocity qd(t)', 'rad/s'),
                            (axes[1, 0], 'Acceleration qdd(t)', 'rad/s^2'),
                            (axes[1, 1], 'Required torque tau(t)', 'N m')):
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel('time (s)')
        ax.legend(fontsize=7)
    fig.suptitle('Quintic trajectory through inverse dynamics, 4-DOF spatial chain')
    fig.tight_layout()
    out = figure_path('trajectory_torques.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
