"""The robot model: :class:`Link`, :class:`Joint`, :class:`Robot`.

The model carries no notion of how many degrees of freedom it has. ``Robot.n`` is
simply ``len(joints)``, and every algorithm in the library loops over that. There
is no ``if n == ...`` branch anywhere in the package — the 2-DOF planar arm and a
random 30-DOF chain go down exactly the same code path.

**Geometry convention.** Joint ``i`` connects link ``i-1`` to link ``i``. Its
``origin`` is the fixed 4x4 transform from the previous link's frame to the joint
frame; the joint's own motion is then applied about/along ``axis``:

.. math:: T_0^i = T_0^{i-1}\\; \\text{origin}_i \\; T_{\\text{joint}}(q_i)

This is the URDF/SDF layout, so a model can be transcribed from a robot
description without re-deriving Denavit-Hartenberg parameters. Link ``i``'s inertial
properties (``com``, ``inertia``) are expressed in link ``i``'s own frame.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.rotations import rot_axis
from ..utils.transforms import transform

JointType = Literal['revolute', 'prismatic']


@dataclass
class Link:
    """A rigid body.

    Parameters
    ----------
    mass:
        Mass in kg. Must be positive.
    com:
        Centre of mass in the **link frame**, metres.
    inertia:
        3x3 rotational inertia **about the centre of mass**, in link-frame axes.
        Must be symmetric positive definite and satisfy the triangle inequalities
        on its principal moments, or it does not describe a physical body.
    name:
        Optional label, used for external-wrench lookup and error messages.
    """

    mass: float
    com: NDArray = field(default_factory=lambda: np.zeros(3))
    inertia: NDArray = field(default_factory=lambda: np.eye(3) * 1e-6)
    name: str = ''

    def __post_init__(self) -> None:
        self.com = np.asarray(self.com, dtype=float).reshape(3)
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        if self.mass <= 0.0:
            raise ValueError(f'link {self.name!r}: mass must be positive, '
                             f'got {self.mass}')
        if not np.allclose(self.inertia, self.inertia.T, atol=1e-12):
            raise ValueError(f'link {self.name!r}: inertia must be symmetric')

    def validate_physical(self, tol: float = 1e-12) -> None:
        """Raise if the inertia tensor cannot belong to a real rigid body.

        Checks positive definiteness and the triangle inequality on the principal
        moments, :math:`I_1 + I_2 \\ge I_3`. An inertia that fails these can make
        the mass matrix indefinite, which then looks like a bug in the dynamics
        rather than a bad model — worth catching at construction time.
        """
        eig = np.linalg.eigvalsh(self.inertia)
        if eig.min() <= tol:
            raise ValueError(f'link {self.name!r}: inertia is not positive '
                             f'definite (eigenvalues {eig})')
        a, b, c = np.sort(eig)
        if a + b < c - tol:
            raise ValueError(f'link {self.name!r}: principal moments {eig} '
                             'violate the triangle inequality')


@dataclass
class Joint:
    """A one-degree-of-freedom joint.

    Parameters
    ----------
    joint_type:
        ``'revolute'`` (rotates about ``axis``) or ``'prismatic'`` (translates
        along it).
    axis:
        Unit axis in the joint frame. Normalised on construction.
    origin:
        Fixed 4x4 transform from the previous link's frame to this joint's frame.
    limits:
        Optional ``(lower, upper)``. Recorded for trajectory and workspace code;
        the dynamics never enforces them, which is stated as a limitation.
    """

    joint_type: JointType = 'revolute'
    axis: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    origin: NDArray = field(default_factory=lambda: np.eye(4))
    limits: tuple[float, float] | None = None
    name: str = ''

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        norm = np.linalg.norm(self.axis)
        if norm < 1e-12:
            raise ValueError(f'joint {self.name!r}: axis must be non-zero')
        self.axis = self.axis / norm
        self.origin = np.asarray(self.origin, dtype=float).reshape(4, 4)
        if self.joint_type not in ('revolute', 'prismatic'):
            raise ValueError(f'joint {self.name!r}: unknown type '
                             f'{self.joint_type!r}')

    def motion_transform(self, q: float) -> NDArray:
        """The joint's own 4x4 transform at coordinate ``q``."""
        if self.joint_type == 'revolute':
            return transform(rot_axis(self.axis, q), np.zeros(3))
        return transform(np.eye(3), self.axis * q)

    def motion_subspace(self) -> NDArray:
        """Featherstone's ``S``: the spatial motion vector this joint generates.

        ``[axis; 0]`` for a revolute joint, ``[0; axis]`` for a prismatic one, in
        the ``[angular; linear]`` ordering of :mod:`robot_dynamics.utils.spatial`.
        """
        S = np.zeros(6)
        if self.joint_type == 'revolute':
            S[:3] = self.axis
        else:
            S[3:] = self.axis
        return S


class Robot:
    """A fixed-base serial chain of arbitrary length.

    Parameters
    ----------
    links, joints:
        Equal-length sequences. ``joints[i]`` drives ``links[i]``.
    gravity:
        Gravitational **acceleration** in the base frame, default
        ``[0, 0, -9.81]``. Set it to zeros to switch gravity off (Phase 11's
        ``g = 0 => G(q) = 0`` check).
    name:
        Optional label.

    Notes
    -----
    Dynamics methods live on this class but are implemented in
    :mod:`robot_dynamics.dynamics`; they are attached at import time to keep the
    model free of algorithm code. The ``method=`` argument selects between
    independent backends, which is how the test suite cross-validates them.
    """

    def __init__(self, links: Sequence[Link], joints: Sequence[Joint],
                 gravity: ArrayLike = (0.0, 0.0, -9.81),
                 name: str = 'robot') -> None:
        if len(links) != len(joints):
            raise ValueError(f'need one joint per link: got {len(links)} links '
                             f'and {len(joints)} joints')
        if not links:
            raise ValueError('a robot needs at least one link')
        self.links = list(links)
        self.joints = list(joints)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.name = name
        # Serial chain: link i's parent is link i-1, and -1 denotes the fixed base.
        # Stored explicitly so the recursive algorithms are written against a
        # parent array and would extend to a branched tree without change.
        self.parent = [i - 1 for i in range(self.n)]

    @property
    def n(self) -> int:
        """Number of degrees of freedom."""
        return len(self.joints)

    def __repr__(self) -> str:
        kinds = ''.join('R' if j.joint_type == 'revolute' else 'P'
                        for j in self.joints)
        return (f'<Robot {self.name!r} n={self.n} [{kinds}] '
                f'mass={self.total_mass:.3f} kg>')

    @property
    def total_mass(self) -> float:
        return float(sum(link.mass for link in self.links))

    def validate(self) -> None:
        """Check every link is physically realisable."""
        for link in self.links:
            link.validate_physical()

    def check_q(self, *arrays: ArrayLike) -> tuple[NDArray, ...]:
        """Coerce joint-space arrays to shape ``(n,)`` floats, or raise.

        Centralised so every entry point gives the same error message instead of
        failing later inside a matrix product with an unhelpful shape mismatch.
        """
        out = []
        for a in arrays:
            v = np.asarray(a, dtype=float).reshape(-1)
            if v.shape != (self.n,):
                raise ValueError(f'expected an array of {self.n} joint values, '
                                 f'got shape {np.shape(a)}')
            out.append(v)
        return tuple(out)

    def random_configuration(self, rng: np.random.Generator | None = None,
                             scale: float = 1.0) -> NDArray:
        """A random ``q``, respecting joint limits where they are defined."""
        rng = rng or np.random.default_rng()
        q = np.empty(self.n)
        for i, j in enumerate(self.joints):
            if j.limits is not None:
                lo, hi = j.limits
                q[i] = rng.uniform(lo, hi)
            elif j.joint_type == 'revolute':
                q[i] = rng.uniform(-np.pi, np.pi) * scale
            else:
                q[i] = rng.uniform(-1.0, 1.0) * scale
        return q
