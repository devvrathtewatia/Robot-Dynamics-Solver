"""Composite Rigid Body Algorithm (CRBA) and Articulated Body Algorithm (ABA)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..kinematics.forward import joint_transforms_spatial
from ..models.robot import Robot
from ..utils.spatial import crf, crm
from .rnea import _external_in_link_frames, _spatial_inertias


def mass_matrix(robot: Robot, q: ArrayLike) -> NDArray:
    """Compute the mass matrix using the Composite Rigid Body Algorithm (CRBA)."""
    q, = robot.check_q(q)
    n = robot.n
    Xs = joint_transforms_spatial(robot, q)
    Ss = [j.motion_subspace() for j in robot.joints]
    IC = _spatial_inertias(robot)

    M = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        p = robot.parent[i]
        if p >= 0:
            IC[p] = IC[p] + Xs[i].T @ IC[i] @ Xs[i]
        F = IC[i] @ Ss[i]
        M[i, i] = Ss[i] @ F
        j = i
        while robot.parent[j] >= 0:
            F = Xs[j].T @ F
            j = robot.parent[j]
            M[i, j] = F @ Ss[j]
            M[j, i] = M[i, j]
    return M


def forward_dynamics(robot: Robot, q: ArrayLike, qd: ArrayLike, tau: ArrayLike,
                     external_wrenches: Mapping[int | str, ArrayLike] | None = None
                     ) -> NDArray:
    """Compute joint accelerations using the Articulated Body Algorithm (ABA)."""
    q, qd, tau = robot.check_q(q, qd, tau)
    n = robot.n
    Xs = joint_transforms_spatial(robot, q)
    Ss = [j.motion_subspace() for j in robot.joints]
    IA = _spatial_inertias(robot)
    ext = (_external_in_link_frames(robot, q, external_wrenches)
           if external_wrenches else {})

    # Base acceleration carries gravity
    a_base = np.zeros(6)
    a_base[3:] = -robot.gravity

    v = [np.zeros(6) for _ in range(n)]
    c = [np.zeros(6) for _ in range(n)]
    pA = [np.zeros(6) for _ in range(n)]

    # Pass 1: Outward sweep for velocities, velocity products, and bias forces
    for i in range(n):
        p = robot.parent[i]
        vp = v[p] if p >= 0 else np.zeros(6)
        v[i] = Xs[i] @ vp + Ss[i] * qd[i]
        c[i] = crm(v[i]) @ (Ss[i] * qd[i])
        pA[i] = crf(v[i]) @ IA[i] @ v[i]
        if i in ext:
            pA[i] = pA[i] - ext[i]

    U = [np.zeros(6) for _ in range(n)]
    d = np.zeros(n)
    u = np.zeros(n)

    # Pass 2: Inward sweep to update articulated inertias and bias forces
    for i in range(n - 1, -1, -1):
        U[i] = IA[i] @ Ss[i]
        d[i] = Ss[i] @ U[i]
        if abs(d[i]) < 1e-15:
            raise np.linalg.LinAlgError(
                f'joint {i} has no articulated inertia along its axis '
                f'(d = {d[i]:.3e}); the model is degenerate')
        u[i] = tau[i] - Ss[i] @ pA[i]
        p = robot.parent[i]
        if p >= 0:
            Ia = IA[i] - np.outer(U[i], U[i]) / d[i]
            pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / d[i])
            IA[p] = IA[p] + Xs[i].T @ Ia @ Xs[i]
            pA[p] = pA[p] + Xs[i].T @ pa

    qdd = np.zeros(n)
    a = [np.zeros(6) for _ in range(n)]

    # Pass 3: Outward sweep to compute joint accelerations
    for i in range(n):
        p = robot.parent[i]
        ap = a[p] if p >= 0 else a_base
        a_prime = Xs[i] @ ap + c[i]
        qdd[i] = (u[i] - U[i] @ a_prime) / d[i]
        a[i] = a_prime + Ss[i] * qdd[i]

    return qdd
