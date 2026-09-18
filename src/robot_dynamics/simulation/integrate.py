# Numerical integration and the simulation loop.
# RK4 with a fixed step. Fourth order matters here: with explicit Euler the energy
# drift of a passive manipulator is large enough to be mistaken for a bug in the
# dynamics, whereas RK4 conserves energy well enough that simulate can be used as a
# test of the dynamics rather than merely a demonstration of it.

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np  # pyright:ignore[reportMissingImports]
from numpy.typing import ArrayLike, NDArray  # pyright:ignore[reportMissingImports]

from ..models.robot import Robot

StateFn = Callable[[float, NDArray], NDArray]
ControlFn = Callable[[float, NDArray, NDArray], NDArray]


def rk4_step(f: StateFn, t: float, y: NDArray, dt: float):
    # One classical fourth-order Runge-Kutta step of dy/dt = f(t, y).
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f: StateFn, t: float, y: NDArray, dt: float):
    # One explicit Euler step. Present for comparison only.
    return y + dt * f(t, y)


def simulate(robot: Robot, q0: ArrayLike, qd0: ArrayLike,
             duration: float, dt: float = 1e-3,
             control: ControlFn | None = None,
             external_wrenches: Mapping[object, ArrayLike] | None = None,
             method: str = 'aba',
             integrator: str = 'rk4'):
    # Integrate the robot's motion forward in time.
    # control: f(t, q, qd) -> tau. None means zero torque (free fall under gravity).
    # method: Forward-dynamics backend, 'aba' or 'lagrangian'.
    # integrator: 'rk4' (default) or 'euler'.
    # Returns a dict containing t, q, qd, qdd, tau.
    q0, qd0 = robot.check_q(q0, qd0)
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError('duration and dt must both be positive')

    step_fn = {'rk4': rk4_step, 'euler': euler_step}.get(integrator)
    if step_fn is None:
        raise ValueError(f'unknown integrator {integrator!r}')

    n = robot.n
    n_steps = round(duration / dt)
    ts = np.zeros(n_steps + 1)
    qs = np.zeros((n_steps + 1, n))
    qds = np.zeros((n_steps + 1, n))
    qdds = np.zeros((n_steps + 1, n))
    taus = np.zeros((n_steps + 1, n))

    def torque(t: float, q: NDArray, qd: NDArray):
        return np.zeros(n) if control is None else np.asarray(
            control(t, q, qd), dtype=float).reshape(n)

    def rhs(t: float, y: NDArray):
        q, qd = y[:n], y[n:]
        tau = torque(t, q, qd)
        qdd = robot.forward_dynamics(q, qd, tau, method=method,
                                     external_wrenches=external_wrenches)
        return np.concatenate([qd, qdd])

    y = np.concatenate([q0, qd0])
    for k in range(n_steps + 1):
        t = k * dt
        q, qd = y[:n], y[n:]
        tau = torque(t, q, qd)
        ts[k] = t
        qs[k], qds[k] = q, qd
        taus[k] = tau
        qdds[k] = robot.forward_dynamics(q, qd, tau, method=method,
                                         external_wrenches=external_wrenches)
        if k < n_steps:
            y = step_fn(rhs, t, y, dt)

    return {'t': ts, 'q': qs, 'qd': qds, 'qdd': qdds, 'tau': taus}


def energy_history(robot: Robot, result: Mapping[str, NDArray]):
    # Total energy T + V at each stored step of a simulation result.
    return np.array([robot.total_energy(q, qd)
                     for q, qd in zip(result['q'], result['qd'], strict=True)])


def tracking_error(result: Mapping[str, NDArray],
                   trajectory: object):
    # Compare a simulation against the trajectory it was asked to follow.
    # Returns per-step error, its RMS per joint, and the worst absolute error.
    q_des = np.array([trajectory.evaluate(t)[0] for t in result['t']])
    err = q_des - result['q']
    return {'q_desired': q_des,
            'error': err,
            'rms_per_joint': np.sqrt(np.mean(err ** 2, axis=0)),
            'max_abs': float(np.max(np.abs(err)))}
