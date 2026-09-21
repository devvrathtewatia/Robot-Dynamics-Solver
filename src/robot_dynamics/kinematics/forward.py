# Forward kinematics and Jacobians for a serial chain of arbitrary length.
# The routines preserve float and complex dtypes so complex-step differentiation
# can use the same numerical code path.

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models.robot import Robot
from ..utils.spatial import plucker


def link_transforms(robot: Robot, q: ArrayLike):
    # Returns [T_0^1, T_0^2,... T_0^n], one per link where each is the
    # 4x4 pose of that link's frame in the base frame.
    q = np.asarray(q).reshape(-1)
    if q.shape != (robot.n,):
        raise ValueError(f'expected {robot.n} joint values, got {q.shape}')
    out: list[NDArray] = []
    T = np.eye(4, dtype=np.result_type(q.dtype, np.float64))
    for i, joint in enumerate(robot.joints):
        T = T @ joint.origin @ joint.motion_transform(q[i])
        out.append(T)
    return out


def forward_kinematics(robot: Robot, q: ArrayLike):
    # End-effector pose : T_0^n(q)
    return link_transforms(robot, q)[-1]


def com_positions(robot: Robot, q: ArrayLike):
    # World frame COM position of every link.
    Ts = link_transforms(robot, q)
    return [T[:3, :3] @ link.com + T[:3, 3]
            for T, link in zip(Ts, robot.links, strict=True)]


def _axes_and_origins(robot: Robot, Ts: list[NDArray]):
    # Joint axis and frame origin for every joint wrt to world frame.
    # Used to assemble geometric Jacobians.
    axes, origins = [], []
    for i, joint in enumerate(robot.joints):
        R, p = Ts[i][:3, :3], Ts[i][:3, 3]
        # A revolute joint rotates about its own axis, so the axis is identical
        # whether taken before or after that rotation - using frame i is safe.
        axes.append(R @ joint.axis)
        origins.append(p)
    return axes, origins


def point_jacobian(robot: Robot, q: ArrayLike, point: ArrayLike, up_to: int):
    # Geometric Jacobian of a point attached to a link, in the world frame.
    # Columns after `up_to` remain zero because those joints do not affect the point.
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
             point_local: ArrayLike | None = None):
    # Returns Jacobian of a link frame.
    """
    Parameters -->
    link: Index of the link, with a -1 for end effector.
    point_local: Optional point in that link's own frame.
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
    # Jacobian of every link's centre of mass
    # It is calculated once and used by mass matrix, gravity vector, and
    # even the external wrench mapping.

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


def joint_transforms_spatial(robot: Robot, q: ArrayLike):
    # Per-joint spatial motion transforms X[i] from the parent link
    # to the child link, in the base frame.
    # These are the Featherstone's recursions.
    q = np.asarray(q, dtype=float).reshape(-1)
    return [plucker(joint.origin @ joint.motion_transform(q[i]))
            for i, joint in enumerate(robot.joints)]


def base_to_link_spatial(robot: Robot, q: ArrayLike):
    # Composed spatial motion transforms from the base frame to each link.
    Xs = joint_transforms_spatial(robot, q)
    out, acc = [], np.eye(6)
    for X in Xs:
        acc = X @ acc
        out.append(acc)
    return out


def jacobian_dot(
    robot: Robot,
    q: ArrayLike,
    qd: ArrayLike,
    link: int = -1,
    h: float = 1e-20,
):
    """Compute Jdot using complex-step differentiation along qd.

    Jdot(q, qd) = Im[J(q + i*h*qd)] / h.

    This requires one Jacobian evaluation instead of 2n finite-difference
    evaluations and is numerically stable.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    qd = np.asarray(qd, dtype=float).reshape(-1)
    Jc = jacobian(robot, q + 1j * h * qd, link=link)
    return np.imag(Jc) / h
