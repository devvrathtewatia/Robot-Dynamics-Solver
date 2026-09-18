# Model-based controllers.
# Two controllers chosen because the contrast between them is the clearest
# demonstration of what a dynamics model buys you:
# PD plus gravity compensation cancels only gravity. It guarantees asymptotic tracking
# of a constant setpoint, but during motion inertial and Coriolis terms act as disturbances.
# Computed torque additionally cancels M and C, collapsing the error
# system to a linear, decoupled plant.

from __future__ import annotations

from typing import Protocol

import numpy as np  # pyright:ignore[reportMissingImports]
from numpy.typing import ArrayLike  # pyright:ignore[reportMissingImports]

from ..models.robot import Robot


class Trajectory(Protocol):
    # Anything providing evaluate(t) -> (q_d, qd_d, qdd_d).

    def evaluate(self, t: float): ...


def _gains(value: ArrayLike, n: int, label: str):
    # Accept a scalar, a length-n vector, or a full matrix; return (n, n).
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(n) * float(arr)
    if arr.shape == (n,):
        return np.diag(arr)
    if arr.shape == (n, n):
        return arr
    raise ValueError(f'{label} must be a scalar, ({n},) or ({n},{n}); '
                     f'got shape {arr.shape}')


class PDGravity:
    # PD control with gravity compensation.
    # tau = Kp * e + Kd * ed + G(q), where e = q_d - q

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory | None = None,
                 setpoint: ArrayLike | None = None,
                 model: Robot | None = None):
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory
        self.setpoint = (np.zeros(robot.n) if setpoint is None
                         else np.asarray(setpoint, float).reshape(robot.n))

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if self.trajectory is None:
            q_des, qd_des = self.setpoint, np.zeros_like(q)
        else:
            q_des, qd_des, _ = self.trajectory.evaluate(t)
        e, ed = q_des - q, qd_des - qd
        return self.kp @ e + self.kd @ ed + self.model.gravity_torque(q)


class ComputedTorque:
    # Computed-torque (inverse-dynamics) control.
    # tau = M(q)*v + C(q, qd)*qd + G(q)
    # v = qdd_d + Kd*(qd_d - qd) + Kp*(q_d - q)

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory, model: Robot | None = None):
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory

    @classmethod
    def critically_damped(cls, robot: Robot, natural_frequency: float,
                          trajectory: Trajectory,
                          model: Robot | None = None):
        # Build gains for a critically damped error response (Kd = 2*omega, Kp = omega^2).
        w = float(natural_frequency)
        return cls(robot, kp=w ** 2, kd=2.0 * w, trajectory=trajectory,
                   model=model)

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_des, qd_des, qdd_des = self.trajectory.evaluate(t)
        v = qdd_des + self.kd @ (qd_des - qd) + self.kp @ (q_des - q)
        return self.model.inverse_dynamics(q, qd, v)


class FeedforwardPD:
    # Feedforward inverse dynamics plus a PD correction.
    # tau = tau_ff(q_d, qd_d, qdd_d) + Kp * e + Kd * ed

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory, model: Robot | None = None):
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_des, qd_des, qdd_des = self.trajectory.evaluate(t)
        tau_ff = self.model.inverse_dynamics(q_des, qd_des, qdd_des)
        return tau_ff + self.kp @ (q_des - q) + self.kd @ (qd_des - qd)
