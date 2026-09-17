"""Spatial vector algebra, following Featherstone's *Rigid Body Dynamics Algorithms*.

A spatial vector packs angular and linear parts into one 6-vector. **Angular part
first**, which is Featherstone's convention:

.. math::

    \\hat v = \\begin{bmatrix} \\omega \\\\ v \\end{bmatrix}
    \\qquad
    \\hat f = \\begin{bmatrix} n \\\\ f \\end{bmatrix}

Motion vectors (velocity, acceleration) and force vectors (force, momentum)
transform *differently*, which is the whole point of the formalism and the reason
RNEA, CRBA and ABA come out as short recursions. If ``X`` transforms motion
vectors from frame A to frame B, then forces transform from B back to A by
``X.T`` — that single identity is what the backward passes rely on.

This module deliberately uses ``[angular; linear]`` while
:mod:`robot_dynamics.utils.transforms` uses ``[linear; angular]`` for geometric
Jacobians. Both conventions are standard in their own context; keeping them in
separate modules with this note is safer than silently picking one.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotations import skew


def plucker(T: ArrayLike) -> NDArray:
    """Spatial **motion** transform from a homogeneous transform.

    ``T`` expresses the child frame in the parent frame (as produced by forward
    kinematics). The returned ``X`` maps motion vectors from parent coordinates to
    child coordinates:

    .. math::

        X = \\begin{bmatrix} E & 0 \\\\ -E\\,[r]_\\times & E \\end{bmatrix},
        \\qquad E = R^\\top,\\; r = p

    The linear block is what accounts for the change of reference point:
    :math:`v_{\\text{child}} = E(v_{\\text{parent}} - r \\times \\omega)`.
    """
    T = np.asarray(T)
    R, p = T[:3, :3], T[:3, 3]
    E = R.T
    X = np.zeros((6, 6), dtype=np.result_type(T.dtype, np.float64))
    X[:3, :3] = E
    X[3:, :3] = -E @ skew(p)
    X[3:, 3:] = E
    return X


def plucker_inverse(X: ArrayLike) -> NDArray:
    """Inverse of a spatial motion transform, in closed form."""
    X = np.asarray(X)
    E = X[:3, :3]
    # X[3:, :3] = -E @ skew(r)  =>  skew(r) = -E^T @ X[3:, :3]
    from .rotations import unskew
    r = unskew(-E.T @ X[3:, :3])
    Xi = np.zeros_like(X)
    Xi[:3, :3] = E.T
    Xi[3:, :3] = skew(r) @ E.T
    Xi[3:, 3:] = E.T
    return Xi


def crm(v: ArrayLike) -> NDArray:
    """Spatial cross product for **motion** vectors, :math:`v \\times`.

    .. math::

        v\\times = \\begin{bmatrix} [\\omega]_\\times & 0 \\\\
                                    [v_l]_\\times & [\\omega]_\\times \\end{bmatrix}
    """
    v = np.asarray(v).reshape(6)
    w, vl = v[:3], v[3:]
    M = np.zeros((6, 6), dtype=np.result_type(v.dtype, np.float64))
    M[:3, :3] = skew(w)
    M[3:, :3] = skew(vl)
    M[3:, 3:] = skew(w)
    return M


def crf(v: ArrayLike) -> NDArray:
    """Spatial cross product for **force** vectors.

    :math:`v \\times^* = -(v\\times)^\\top`
    """
    return -crm(v).T


def spatial_inertia(mass: float, com: ArrayLike, inertia_com: ArrayLike) -> NDArray:
    """Build the 6x6 spatial inertia of a body about its **frame origin**.

    .. math::

        I = \\begin{bmatrix}
            I_c + m\\,[c]_\\times [c]_\\times^\\top & m\\,[c]_\\times \\\\
            m\\,[c]_\\times^\\top & m\\,1
            \\end{bmatrix}

    ``inertia_com`` is the 3x3 rotational inertia about the centre of mass; the
    :math:`m[c]_\\times[c]_\\times^\\top` term is the parallel-axis shift to the frame
    origin.
    """
    c = np.asarray(com, dtype=float).reshape(3)
    Ic = np.asarray(inertia_com, dtype=float).reshape(3, 3)
    S = skew(c)
    I = np.zeros((6, 6))
    I[:3, :3] = Ic + mass * (S @ S.T)
    I[:3, 3:] = mass * S
    I[3:, :3] = mass * S.T
    I[3:, 3:] = mass * np.eye(3)
    return I


def motion_to_geometric(v: ArrayLike) -> NDArray:
    """Reorder a spatial motion vector ``[omega; v]`` to geometric ``[v; omega]``."""
    v = np.asarray(v).reshape(6)
    return np.concatenate([v[3:], v[:3]])


def force_to_geometric(f: ArrayLike) -> NDArray:
    """Reorder a spatial force ``[n; f]`` to ``[f; n]``."""
    f = np.asarray(f).reshape(6)
    return np.concatenate([f[3:], f[:3]])


def geometric_to_force(w: ArrayLike) -> NDArray:
    """Reorder a wrench ``[f; n]`` to the spatial force convention ``[n; f]``."""
    w = np.asarray(w).reshape(6)
    return np.concatenate([w[3:], w[:3]])
