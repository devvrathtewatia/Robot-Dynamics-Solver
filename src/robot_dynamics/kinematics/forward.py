"""Generic forward kinematics and Jacobians for a serial chain of any length.

Everything here is dtype-transparent: pass a complex ``q`` and complex transforms
come back. That is what lets the Lagrangian backend take a complex-step derivative
of the mass matrix without a separate symbolic path.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models.robot import Robot
from ..utils.spatial import plucker


def link_transforms(robot: Robot, q: ArrayLike) -> list[NDArray]:
    r"""Return :math:`[T_0^1, T_0^2, \ldots, T_0^n]`, one per link.

    Each is the 4x4 pose of that link's frame in the base frame.
    """
    q = np.asarray(q).reshape(-1)
    if q.shape != (robot.n,):
        raise ValueError(f'expected {robot.n} joint values, got {q.shape}')
    out: list[NDArray] = []
    T = np.eye(4, dtype=np.result_type(q.dtype, np.float64))
    for i, joint in enumerate(robot.joints):
        T = T @ joint.origin @ joint.motion_transform(q[i])
        out.append(T)
    return out


def forward_kinematics(robot: Robot, q: ArrayLike) -> NDArray:
    r"""End-effector pose :math:`T_0^n(q)`."""
    return link_transforms(robot, q)[-1]


def com_positions(robot: Robot, q: ArrayLike) -> list[NDArray]:
    """World-frame centre-of-mass position of every link."""
    Ts = link_transforms(robot, q)
    return [T[:3, :3] @ link.com + T[:3, 3]
            for T, link in zip(Ts, robot.links, strict=True)]


def _axes_and_origins(robot: Robot, Ts: list[NDArray]
                      ) -> tuple[list[NDArray], list[NDArray]]:
    """World-frame joint axis and frame origin for every joint."""
    axes, origins = [], []
    for i, joint in enumerate(robot.joints):
        R, p = Ts[i][:3, :3], Ts[i][:3, 3]
        # A revolute joint rotates about its own axis, so the axis is identical
        # whether taken before or after that rotation - using frame i is safe.
        axes.append(R @ joint.axis)
        origins.append(p)
    return axes, origins


def point_jacobian(robot: Robot, q: ArrayLike, point: ArrayLike,
                   up_to: int | None = None) -> NDArray:
    r"""Jacobian of a world-frame ``point`` rigidly attached to link ``up_to``.

    Returns a ``(6, n)`` matrix in the **geometric** ordering

    .. math:: \begin{bmatrix} \dot p \\ \omega \end{bmatrix} = J(q)\,\dot q

    Columns for joints beyond ``up_to`` are zero: a joint downstream of a body
    cannot move it. That zero pattern is what makes the per-link COM Jacobians
    assemble into a correct mass matrix.
    """
    q = np.asarray(q).reshape(-1)
    Ts = link_transforms(robot, q)
    axes, origins = _axes_and_origins(robot, Ts)
    last = robot.n - 1 if up_to is None else up_to
    point = np.asarray(point).reshape(3)

    dtype = np.result_type(q.dtype, np.float64)
    J = np.zeros((6, robot.n), dtype=dtype)
    for j in range(last + 1):
        if robot.joints[j].joint_type == 'revolute':
            J[:3, j] = np.cross(axes[j], point - origins[j])
            J[3:, j] = axes[j]
        else:
            J[:3, j] = axes[j]
    return J


def jacobian(robot: Robot, q: ArrayLike, link: int = -1,
             point_local: ArrayLike | None = None) -> NDArray:
    r"""Geometric Jacobian of a link frame (or a point fixed in it).

    Parameters
    ----------
    link:
        Index of the link, default ``-1`` for the end effector.
    point_local:
        Optional point in that link's own frame; defaults to its origin.
    """
    idx = robot.n - 1 if link == -1 else link
    Ts = link_transforms(robot, q)
    T = Ts[idx]
    if point_local is None:
        point = T[:3, 3]
    else:
        point = T[:3, :3] @ np.asarray(point_local).reshape(3) + T[:3, 3]
    return point_jacobian(robot, q, point, up_to=idx)


def com_jacobians(robot: Robot, q: ArrayLike) -> list[NDArray]:
    """Geometric Jacobian of every link's centre of mass.

    Returned once and reused by the mass matrix, the gravity vector and the
    external-wrench mapping - all three need exactly this quantity.
    """
    q = np.asarray(q).reshape(-1)
    Ts = link_transforms(robot, q)
    axes, origins = _axes_and_origins(robot, Ts)
    dtype = np.result_type(q.dtype, np.float64)

    out = []
    for i, link in enumerate(robot.links):
        c = Ts[i][:3, :3] @ link.com + Ts[i][:3, 3]
        J = np.zeros((6, robot.n), dtype=dtype)
        for j in range(i + 1):
            if robot.joints[j].joint_type == 'revolute':
                J[:3, j] = np.cross(axes[j], c - origins[j])
                J[3:, j] = axes[j]
            else:
                J[:3, j] = axes[j]
        out.append(J)
    return out


def joint_transforms_spatial(robot: Robot, q: ArrayLike) -> list[NDArray]:
    """Per-joint spatial motion transforms ``X[i]``: parent coordinates to link ``i``.

    These are the ``X_{i,\\lambda(i)}`` of Featherstone's recursions.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    return [plucker(joint.origin @ joint.motion_transform(q[i]))
            for i, joint in enumerate(robot.joints)]


def base_to_link_spatial(robot: Robot, q: ArrayLike) -> list[NDArray]:
    """Composed spatial motion transforms from the base frame to each link."""
    Xs = joint_transforms_spatial(robot, q)
    out, acc = [], np.eye(6)
    for X in Xs:
        acc = X @ acc
        out.append(acc)
    return out


def jacobian_dot(robot: Robot, q: ArrayLike, qd: ArrayLike,
                 link: int = -1, h: float = 1e-20) -> NDArray:
    r""":math:`\dot J(q,\dot q)`, by complex-step differentiation along ``qd``.

    Uses the directional derivative
    :math:`\dot J = \sum_k (\partial J/\partial q_k)\dot q_k`, evaluated in one shot
    as :math:`\mathrm{Im}\,J(q + i h \dot q)/h`. Exact to machine precision, and one
    function evaluation rather than ``2n``.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    qd = np.asarray(qd, dtype=float).reshape(-1)
    Jc = jacobian(robot, q + 1j * h * qd, link=link)
    return np.imag(Jc) / h
