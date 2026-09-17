"""Homogeneous (4x4) transformations.

.. math::

    T = \\begin{bmatrix} R & p \\\\ 0 & 1 \\end{bmatrix}
    \\qquad
    T^{-1} = \\begin{bmatrix} R^\\top & -R^\\top p \\\\ 0 & 1 \\end{bmatrix}

:func:`transform_inverse` uses that closed form rather than a general matrix
inverse: it is exact, cheaper, and cannot drift away from being a valid rigid
transform the way a numerically inverted matrix can.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def transform(R: ArrayLike, p: ArrayLike) -> NDArray:
    """Assemble a 4x4 homogeneous transform from a rotation and a translation."""
    R = np.asarray(R)
    p = np.asarray(p).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f'transform expects a 3x3 rotation, got {R.shape}')
    T = np.zeros((4, 4), dtype=np.result_type(R.dtype, p.dtype, np.float64))
    T[:3, :3] = R
    T[:3, 3] = p
    T[3, 3] = 1.0
    return T


def transform_inverse(T: ArrayLike) -> NDArray:
    """Inverse of a homogeneous transform, in closed form."""
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f'transform_inverse expects 4x4, got {T.shape}')
    R, p = T[:3, :3], T[:3, 3]
    return transform(R.T, -(R.T @ p))


def rotation_of(T: ArrayLike) -> NDArray:
    """Extract the rotation block."""
    return np.asarray(T)[:3, :3]


def translation_of(T: ArrayLike) -> NDArray:
    """Extract the translation block."""
    return np.asarray(T)[:3, 3]


def transform_point(T: ArrayLike, p: ArrayLike) -> NDArray:
    """Apply a homogeneous transform to a 3D point."""
    T = np.asarray(T)
    return T[:3, :3] @ np.asarray(p).reshape(3) + T[:3, 3]


def compose(*transforms: ArrayLike) -> NDArray:
    """Left-to-right product of homogeneous transforms."""
    if not transforms:
        return np.eye(4)
    out = np.asarray(transforms[0])
    for T in transforms[1:]:
        out = out @ np.asarray(T)
    return out


def adjoint(T: ArrayLike) -> NDArray:
    """The 6x6 adjoint mapping a twist ``[v; omega]`` between frames.

    Note the ordering: this library uses ``[linear; angular]`` for *geometric*
    Jacobians and twists, but ``[angular; linear]`` for the *spatial* (Plucker)
    vectors of :mod:`robot_dynamics.utils.spatial`, following Featherstone. The two
    conventions are kept in separate modules on purpose; mixing them is the classic
    source of sign errors in dynamics code.
    """
    T = np.asarray(T)
    from .rotations import skew  # local: avoid a cycle
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6), dtype=np.result_type(T.dtype, np.float64))
    A[:3, :3] = R
    A[:3, 3:] = skew(p) @ R
    A[3:, 3:] = R
    return A
