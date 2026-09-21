"""Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics and dynamics validation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..kinematics.forward import (
    joint_transforms_spatial,
    link_transforms,
)
from ..models.robot import Robot
from ..utils.spatial import crf, crm, geometric_to_force, plucker, spatial_inertia
from ..utils.transforms import transform


def _spatial_inertias(robot: Robot):
    return [spatial_inertia(link.mass, link.com, link.inertia)
            for link in robot.links]


def _external_in_link_frames(robot: Robot, q: ArrayLike,
                             external_wrenches: Mapping[int | str, ArrayLike]):
    # Convert base-frame wrenches to link-frame spatial forces
    Ts = link_transforms(robot, np.asarray(q, dtype=float).reshape(-1))
    by_name = {link.name: i for i, link in enumerate(robot.links) if link.name}
    out: dict[int, NDArray] = {}
    for key, wrench in external_wrenches.items():
        if isinstance(key, str):
            if key not in by_name:
                raise KeyError(f'no link named {key!r}')
            idx = by_name[key]
        else:
            idx = key

        f_base = geometric_to_force(np.asarray(wrench, dtype=float).reshape(6))

        # Use a rotation-only Plucker transform (block-diagonal R^T, no moment shift)
        # because the incoming moment is already referenced to the link origin.
        R_rot = plucker(transform(Ts[idx][:3, :3], np.zeros(3)))
        contribution = R_rot @ f_base
        out[idx] = out[idx] + contribution if idx in out else contribution
    return out


def inverse_dynamics(robot: Robot, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike,
                     external_wrenches: Mapping[int | str, ArrayLike] | None = None, gravity: bool = True):
    # Compute inverse dynamics using the Recursive Newton-Euler Algorithm (RNEA)
    q, qd, qdd = robot.check_q(q, qd, qdd)
    n = robot.n
    Xs = joint_transforms_spatial(robot, q)
    Is = _spatial_inertias(robot)
    Ss = [j.motion_subspace() for j in robot.joints]
    ext = (_external_in_link_frames(robot, q, external_wrenches)
           if external_wrenches else {})

    # Base acceleration carries gravity: a_0 = -g, expressed spatially.
    a_base = np.zeros(6)
    if gravity:
        a_base[3:] = -robot.gravity

    v = [np.zeros(6) for _ in range(n)]
    a = [np.zeros(6) for _ in range(n)]
    f = [np.zeros(6) for _ in range(n)]

    # Outward pass: compute velocity and acceleration for every link.
    for i in range(n):
        p = robot.parent[i]
        vp = v[p] if p >= 0 else np.zeros(6)
        ap = a[p] if p >= 0 else a_base
        v[i] = Xs[i] @ vp + Ss[i] * qd[i]
        a[i] = Xs[i] @ ap + Ss[i] * qdd[i] + crm(v[i]) @ (Ss[i] * qd[i])
        f[i] = Is[i] @ a[i] + crf(v[i]) @ Is[i] @ v[i]
        if i in ext:
            f[i] = f[i] - ext[i]

    # Inward pass: gather forces and project onto each joint axis.
    tau = np.zeros(n)
    for i in range(n - 1, -1, -1):
        tau[i] = Ss[i] @ f[i]
        p = robot.parent[i]
        if p >= 0:
            f[p] = f[p] + Xs[i].T @ f[i]
    return tau


def gravity_torque(robot: Robot, q: ArrayLike):
    # Compute the gravity torque vector G(q) using RNEA at rest.
    z = np.zeros(robot.n)
    return inverse_dynamics(robot, q, z, z)


def coriolis_vector(robot: Robot, q: ArrayLike, qd: ArrayLike):
    # Compute velocity-product torques C(q, qd) * qd using RNEA with gravity off.
    return inverse_dynamics(robot, q, qd, np.zeros(robot.n), gravity=False)


def mass_matrix(robot: Robot, q: ArrayLike):
    # Build the mass matrix M(q) column by column using RNEA (unit-acceleration method).
    q, = robot.check_q(q)
    n = robot.n
    z = np.zeros(n)
    M = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = inverse_dynamics(robot, q, z, e, gravity=False)
    return (M + M.T) / 2.0
