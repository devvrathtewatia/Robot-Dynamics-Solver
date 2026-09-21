"""A torque-free arm falling under gravity, and the energy check it enables.

The energy trace gives a numerical check on the dynamics because integration drift
accumulates over many time steps.
"""
from __future__ import annotations

import numpy as np
from _common import figure_path, style

import robot_dynamics as rd
from robot_dynamics.simulation import energy_history, simulate


def main() -> None:
    robot = rd.planar_chain(2, length=0.5, mass=1.0)
    result = simulate(robot, q0=[np.pi / 3, -np.pi / 4], qd0=[0.0, 0.0],
                      duration=4.0, dt=1e-3)
    E = energy_history(robot, result)
    drift = abs(E[-1] - E[0]) / abs(E[0])
    print(f'energy at t=0    : {E[0]: .9f} J')
    print(f'energy at t=end  : {E[-1]: .9f} J')
    print(f'relative drift   : {drift:.2e}  (RK4, dt = 1 ms)')

    euler = simulate(robot, q0=[np.pi / 3, -np.pi / 4], qd0=[0.0, 0.0],
                     duration=4.0, dt=1e-3, integrator='euler')
    Ee = energy_history(robot, euler)
    print(f'same with Euler  : {abs(Ee[-1] - Ee[0]) / abs(Ee[0]):.2e}')

    plt = style()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for j in range(robot.n):
        axes[0].plot(result['t'], result['q'][:, j], label=f'q{j + 1}')
        axes[1].plot(result['t'], result['qd'][:, j], label=f'qd{j + 1}')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('angle (rad)')
    axes[0].set_title('Passive fall: joint angles')
    axes[0].legend()
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('rate (rad/s)')
    axes[1].set_title('Joint velocities')
    axes[1].legend()
    axes[2].plot(result['t'], E - E[0], label='RK4')
    axes[2].plot(euler['t'], Ee - Ee[0], label='Euler', alpha=0.8)
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('energy error (J)')
    axes[2].set_title('Energy drift')
    axes[2].legend()
    fig.tight_layout()
    out = figure_path('passive_energy.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
