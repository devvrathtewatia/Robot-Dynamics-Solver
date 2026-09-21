
"""Spatial-vector helpers used by the Featherstone algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotations import skew


def plucker(T: ArrayLike):
    # spatial motion transform from a homogeneous transform `T`.
    # Maps motion vectors from parent to child coordinates.
    T = np.asarray(T)
    R, p = T[:3, :3], T[:3, 3]
    E = R.T
    X = np.zeros((6, 6), dtype=np.result_type(T.dtype, np.float64))
    X[:3, :3] = E
    X[3:, :3] = -E @ skew(p)
    X[3:, 3:] = E
    return X


def plucker_inverse(X: ArrayLike):
    # Closed-form inverse of a spatial motion transform.
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
    # Spatial cross-product matrix for motion vectors.
    v = np.asarray(v).reshape(6)
    w, vl = v[:3], v[3:]
    M = np.zeros((6, 6), dtype=np.result_type(v.dtype, np.float64))
    M[:3, :3] = skew(w)
    M[3:, :3] = skew(vl)
    M[3:, 3:] = skew(w)
    return M


def crf(v: ArrayLike) -> NDArray:
    # Spatial cross-product matrix for force vectors.
    return -crm(v).T


def spatial_inertia(mass: float, com: ArrayLike, inertia_com: ArrayLike) -> NDArray:
    # Build the 6x6 spatial inertia matrix of a body about its frame origin.
    # Includes the parallel-axis shift from the center of mass to the origin.

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
    # Reorder a spatial motion vector from [omega, v] to geometric [v, omega]
    v = np.asarray(v).reshape(6)
    return np.concatenate([v[3:], v[:3]])


def force_to_geometric(f: ArrayLike) -> NDArray:
    # Reorder a spatial force from [n, f] to [f, n].
    f = np.asarray(f).reshape(6)
    return np.concatenate([f[3:], f[:3]])


def geometric_to_force(w: ArrayLike) -> NDArray:
    # Reorder a wrench from [f, n] to the spatial force convention [n, f].
    w = np.asarray(w).reshape(6)
    return np.concatenate([w[3:], w[:3]])


# Spatial motion vectors use [angular; linear]. Public geometric Jacobians
# use [linear; angular]. Conversion functions above keep the conventions explicit.
