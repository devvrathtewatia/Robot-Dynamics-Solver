"""Recursive Newton-Euler Algorithm (RNEA) — Featherstone, Table 5.1.

An independent second implementation of inverse dynamics. Where the Lagrangian
backend assembles :math:`M`, :math:`C` and :math:`G` and multiplies them out, RNEA
never forms them: it propagates velocities and accelerations outward from the base,
then gathers forces inward from the tip, giving :math:`\\tau` directly in
:math:`O(n)` instead of :math:`O(n^3)`.

That independence is the point. The two backends share only the kinematic model, so
their agreeing to :math:`10^{-12}` is real evidence of correctness — a single bug
would have to be replicated in two different formalisms to hide.

Gravity is handled by Featherstone's trick of giving the base a fictitious upward
acceleration :math:`a_0 = -g`. Every link then inherits it through the same
recursion that handles real acceleration, so no separate gravity term is needed.
"""
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


def _spatial_inertias(robot: Robot) -> list[NDArray]:
    return [spatial_inertia(link.mass, link.com, link.inertia)
            for link in robot.links]


def _external_in_link_frames(robot: Robot, q: ArrayLike,
                             external_wrenches: Mapping[int | str, ArrayLike]
                             ) -> dict[int, NDArray]:
    """Convert base-frame ``[force; moment]`` wrenches to link-frame spatial forces.

    **Convention, and the subtlety in it.** A wrench is supplied as
    ``[fx, fy, fz, nx, ny, nz]`` in base *orientation*, acting at the target link's
    frame **origin** — so ``n`` is already the moment about that point. Converting to
    link coordinates is therefore a pure **rotation**.

    It is tempting to reach for the spatial force transform ``X^{-T}`` here, since
    that is what maps force vectors between frames. But ``X^{-T}`` also performs the
    moment shift ``n -> n - r x f``, which assumes the incoming moment was referenced
    to the *base* origin. Applying it to a wrench already referenced to the link
    origin double-counts the shift. The symptom is precise and worth recording: pure
    moments stay correct while pure forces come out wrong, so a test using only
    moments passes and the bug survives. A pure force applied at a link's own origin
    must produce exactly zero torque at that link's joint - zero moment arm - and
    that is the check which catches it.

    The moment shift for the *parents* still happens, correctly, in the backward pass
    via ``X.T``.
    """
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
        # rotation-only Plucker transform: block-diagonal R^T, no moment shift
        R_rot = plucker(transform(Ts[idx][:3, :3], np.zeros(3)))
        contribution = R_rot @ f_base
        out[idx] = out[idx] + contribution if idx in out else contribution
    return out


def inverse_dynamics(robot: Robot, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike,
                     external_wrenches: Mapping[int | str, ArrayLike] | None = None,
                     gravity: bool = True) -> NDArray:
    r"""Inverse dynamics by the recursive Newton-Euler algorithm.

    Parameters
    ----------
    gravity:
        Set ``False`` to omit gravity, which turns the result into
        :math:`M\ddot q + C\dot q` and is how :func:`coriolis_vector` is built.
    """
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

    # Outward pass: velocity and acceleration of every link.
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


def gravity_torque(robot: Robot, q: ArrayLike) -> NDArray:
    r""":math:`G(q)` — RNEA at rest, :math:`\dot q = \ddot q = 0`."""
    z = np.zeros(robot.n)
    return inverse_dynamics(robot, q, z, z)


def coriolis_vector(robot: Robot, q: ArrayLike, qd: ArrayLike) -> NDArray:
    r""":math:`C(q,\dot q)\dot q` — RNEA with gravity off and :math:`\ddot q = 0`.

    Exact and :math:`O(n)`, with no derivative of :math:`M` anywhere. Comparing this
    against the Christoffel-symbol result from the Lagrangian backend is the sharpest
    test in the suite: it checks the mass matrix, its derivative and the Coriolis
    factorisation all at once.
    """
    return inverse_dynamics(robot, q, qd, np.zeros(robot.n), gravity=False)


def mass_matrix(robot: Robot, q: ArrayLike) -> NDArray:
    r"""Build :math:`M(q)` column by column using RNEA — the "unit acceleration" method.

    With :math:`\dot q = 0`, gravity off and :math:`\ddot q = e_j`, inverse dynamics
    returns exactly the ``j``-th column of :math:`M`. Costs :math:`O(n^2)`, so
    :mod:`~robot_dynamics.dynamics.crba` is the one to use in anger; this exists as a
    third independent route to :math:`M` for the test suite.
    """
    q, = robot.check_q(q)
    n = robot.n
    z = np.zeros(n)
    M = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = inverse_dynamics(robot, q, z, e, gravity=False)
    return (M + M.T) / 2.0
