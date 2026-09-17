"""Model-based controllers.

Two controllers, chosen because the contrast between them is the clearest
demonstration of what a dynamics model buys you:

**PD plus gravity compensation** cancels only the configuration-dependent gravity
term. It guarantees asymptotic tracking of a *constant* setpoint, but during motion
the inertial and Coriolis terms act as unmodelled disturbances, so tracking error
grows with speed.

**Computed torque** additionally cancels :math:`M` and :math:`C`. Substituting

.. math::

    \\tau = M(q)\\big[\\ddot q_d + K_d \\dot e + K_p e\\big]
           + C(q,\\dot q)\\dot q + G(q)

into the equation of motion collapses it to :math:`\\ddot e + K_d \\dot e + K_p e = 0` —
a *linear, decoupled* error system, identical for every joint regardless of
configuration. The nonlinear plant has been converted into ``n`` independent second
order systems, so :math:`K_p` and :math:`K_d` can be chosen analytically for a chosen
damping ratio instead of tuned by hand.

That collapse is exact only if the model is exact. Both controllers therefore accept
a ``model`` distinct from the simulated ``robot``, which is how the test suite
measures sensitivity to modelling error rather than assuming it away.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models.robot import Robot


class Trajectory(Protocol):
    """Anything providing ``evaluate(t) -> (q_d, qd_d, qdd_d)``."""

    def evaluate(self, t: float) -> tuple[NDArray, NDArray, NDArray]: ...


def _gains(value: ArrayLike, n: int, label: str) -> NDArray:
    """Accept a scalar, a length-``n`` vector, or a full matrix; return ``(n, n)``."""
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
    r"""PD control with gravity compensation.

    .. math:: \tau = K_p e + K_d \dot e + G(q), \qquad e = q_d - q
    """

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory | None = None,
                 setpoint: ArrayLike | None = None,
                 model: Robot | None = None) -> None:
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory
        self.setpoint = (np.zeros(robot.n) if setpoint is None
                         else np.asarray(setpoint, float).reshape(robot.n))

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike) -> NDArray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if self.trajectory is None:
            q_des, qd_des = self.setpoint, np.zeros_like(q)
        else:
            q_des, qd_des, _ = self.trajectory.evaluate(t)
        e, ed = q_des - q, qd_des - qd
        return self.kp @ e + self.kd @ ed + self.model.gravity_torque(q)


class ComputedTorque:
    r"""Computed-torque (inverse-dynamics) control.

    .. math::

        \tau &= M(q)\,v + C(q,\dot q)\dot q + G(q) \\
        v &= \ddot q_d + K_d(\dot q_d - \dot q) + K_p(q_d - q)

    For critical damping choose :math:`K_d = 2\sqrt{K_p}`, which
    :meth:`critically_damped` does.
    """

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory, model: Robot | None = None) -> None:
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory

    @classmethod
    def critically_damped(cls, robot: Robot, natural_frequency: float,
                          trajectory: Trajectory,
                          model: Robot | None = None) -> ComputedTorque:
        r"""Build gains for a critically damped error response.

        With :math:`K_p = \omega^2` and :math:`K_d = 2\omega` the closed-loop error
        obeys :math:`\ddot e + 2\omega\dot e + \omega^2 e = 0`, giving the fastest
        response without overshoot. Possible *because* the model linearises the
        error dynamics — no equivalent analytic choice exists for a plain PD loop on
        a nonlinear plant.
        """
        w = float(natural_frequency)
        return cls(robot, kp=w ** 2, kd=2.0 * w, trajectory=trajectory,
                   model=model)

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike) -> NDArray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_des, qd_des, qdd_des = self.trajectory.evaluate(t)
        v = qdd_des + self.kd @ (qd_des - qd) + self.kp @ (q_des - q)
        return self.model.inverse_dynamics(q, qd, v)


class FeedforwardPD:
    r"""Feedforward inverse dynamics plus a PD correction.

    .. math::

        \tau = \underbrace{\tau_{ff}(q_d,\dot q_d,\ddot q_d)}_{\text{open loop}}
               + K_p e + K_d \dot e

    Cheaper than computed torque in a real controller — the feedforward term depends
    only on the *planned* trajectory, so it can be computed offline once, leaving a
    fixed-gain PD loop at the servo rate. The trade-off is that the error dynamics
    are only linearised near the planned trajectory, not globally.
    """

    def __init__(self, robot: Robot, kp: ArrayLike, kd: ArrayLike,
                 trajectory: Trajectory, model: Robot | None = None) -> None:
        self.robot = robot
        self.model = model or robot
        self.kp = _gains(kp, robot.n, 'kp')
        self.kd = _gains(kd, robot.n, 'kd')
        self.trajectory = trajectory

    def __call__(self, t: float, q: ArrayLike, qd: ArrayLike) -> NDArray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_des, qd_des, qdd_des = self.trajectory.evaluate(t)
        tau_ff = self.model.inverse_dynamics(q_des, qd_des, qdd_des)
        return tau_ff + self.kp @ (q_des - q) + self.kd @ (qd_des - qd)
