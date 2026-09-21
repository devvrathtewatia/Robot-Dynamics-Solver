"""Rotation and cross-product utilities used by the kinematics code."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rot_x(theta: float):
    # Rotation of theta radians about the x-axis
    c, s = np.cos(theta), np.sin(theta)
    z, o = 0.0 * c, 1.0 + 0.0 * c          # keeps the dtype if theta is complex
    return np.array([[o, z, z],
                     [z, c, -s],
                     [z, s, c]])


def rot_y(theta: float):
    # Rotation of theta radians about the y-axis.
    c, s = np.cos(theta), np.sin(theta)
    z, o = 0.0 * c, 1.0 + 0.0 * c
    return np.array([[c, z, s],
                     [z, o, z],
                     [-s, z, c]])


def rot_z(theta: float) -> NDArray:
    # Rotation of theta radians about the z-axis.
    c, s = np.cos(theta), np.sin(theta)
    z, o = 0.0 * c, 1.0 + 0.0 * c
    return np.array([[c, -s, z],
                     [s, c, z],
                     [z, z, o]])


def skew(v: ArrayLike):
    """
    Skew-symmetric matrix of a 3-vector, such that skew(v) @ w == np.cross(v, w)
    """
    v = np.asarray(v)
    if v.shape != (3,):
        raise ValueError(f'skew expects a 3-vector, got shape {v.shape}')
    z = 0.0 * v[0]
    return np.array([[z, -v[2], v[1]],
                     [v[2], z, -v[0]],
                     [-v[1], v[0], z]])


def unskew(S: ArrayLike):
    """Inverse of :func:`skew`: recover the vector from a skew-symmetric matrix."""
    S = np.asarray(S)
    if S.shape != (3, 3):
        raise ValueError(f'unskew expects a 3x3 matrix, got shape {S.shape}')
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot_axis(axis: ArrayLike, theta: float) -> NDArray:
    """
    Rotation of theta about an arbitrary axis, via Rodrigues' formula.
    The axis is not re-normalized to protect complex-step perturbations.
    Validate it during model construction instead
    """
    a = np.asarray(axis, dtype=float)
    K = skew(a)
    c, s = np.cos(theta), np.sin(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def is_rotation(R: ArrayLike, tol: float = 1e-9):
    # True if R is orthonormal with determinant +1.
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol)
                and abs(np.linalg.det(R) - 1.0) < tol)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> NDArray:
    # Fixed-axis roll-pitch-yaw, R = Rz(yaw) Ry(pitch) Rx(roll).
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
