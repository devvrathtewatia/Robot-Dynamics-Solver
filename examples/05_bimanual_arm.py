"""The 6-DOF arm from the companion Gazebo mobile-manipulator project.

That project fixed the arm's geometry - two 0.16 m links, a 0.07 m gripper, a
spherical wrist - and validated its *kinematics*. This asks the next question:
what torque must each servo actually deliver?

The answer is an engineering result, not a demonstration. The peak torque over a
realistic motion is what a servo has to be specified against, and the static hold
torque is what it must sustain indefinitely without overheating.
"""
from __future__ import annotations

import numpy as np
from _common import figure_path, style

import robot_dynamics as rd
from robot_dynamics.trajectory import quintic


def main() -> None:
    robot = rd.bimanual_arm()
    print(robot)
    print(f'total arm mass: {robot.total_mass:.3f} kg')

    # worst-case static hold: arm extended horizontally
    stretched = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    G = robot.gravity_torque(stretched)
    print('\nstatic hold, arm straight out (N m):')
    for joint, g in zip(robot.joints, G, strict=True):
        print(f'  {joint.name:<10} {g: .4f}')
    print(f'  worst case  {np.max(np.abs(G)):.4f} N m at '
          f'{robot.joints[int(np.argmax(np.abs(G)))].name}')

    # a 1-second reach, of the kind the pick-and-place cycle performs
    q0 = np.array([0.0, -0.6, 1.2, 0.0, 0.5, 0.0])
    qf = np.array([0.8, 0.3, -0.9, 1.0, -0.7, 0.4])
    traj = quintic(q0, qf, duration=1.0)
    ts, qs, qds, qdds = traj.sample(dt=0.004)
    taus = np.array([robot.inverse_dynamics(q, qd, qdd)
                     for q, qd, qdd in zip(qs, qds, qdds, strict=True)])
    peak = np.max(np.abs(taus), axis=0)
    print('\npeak torque over a 1 s reach (N m):')
    for joint, p in zip(robot.joints, peak, strict=True):
        print(f'  {joint.name:<10} {p: .4f}')

    # what a 0.5 kg payload at the gripper adds
    payload = np.array([0.0, 0.0, -0.5 * 9.81, 0.0, 0.0, 0.0])
    with_load = robot.inverse_dynamics(stretched, np.zeros(6), np.zeros(6),
                                       external_wrenches={'gripper': payload})
    print('\nholding a 0.5 kg payload at the gripper (N m, and the extra load):')
    for joint, total, base in zip(robot.joints, with_load, G, strict=True):
        extra = abs(total) - abs(base)
        print(f'  {joint.name:<10} {total: .4f}   ({extra:+.4f} vs unloaded)')
    print(f'  the shoulder pitch carries {abs(with_load[1]):.3f} N m loaded, '
          f'{abs(with_load[1]) / abs(G[1]):.2f}x the unloaded hold')

    plt = style()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for j, joint in enumerate(robot.joints):
        axes[0].plot(ts, qs[:, j], label=joint.name)
        axes[1].plot(ts, taus[:, j], label=joint.name)
    axes[0].set_title('Joint trajectory')
    axes[0].set_ylabel('rad')
    axes[1].set_title('Required torque')
    axes[1].set_ylabel('N m')
    for ax in axes[:2]:
        ax.set_xlabel('time (s)')
        ax.legend(fontsize=6)
    idx = np.arange(robot.n)
    axes[2].bar(idx - 0.2, np.abs(G), 0.4, label='static hold')
    axes[2].bar(idx + 0.2, peak, 0.4, label='peak, 1 s reach')
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels([j.name for j in robot.joints], rotation=45,
                            ha='right', fontsize=6)
    axes[2].set_ylabel('N m')
    axes[2].set_title('Torque budget per joint')
    axes[2].legend(fontsize=7)
    fig.suptitle('6-DOF bimanual arm: what the servos must deliver')
    fig.tight_layout()
    out = figure_path('bimanual_torques.png')
    fig.savefig(out)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
