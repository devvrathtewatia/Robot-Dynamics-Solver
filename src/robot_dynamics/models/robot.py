# Joint i connects link i-1 to link i. The fixed origin transform is applied
# first, followed by the joint motion along the joint axis.
#
# The model uses a URDF-like frame convention so each link inertia and COM are
# expressed in the corresponding link frame.
#
# T_0^i = T_0^(i-1) @ origin_i @ joint_motion_i(q_i)

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

    Parameters:

    mass: Mass in kg. Must be positive.
    com: Centre of mass in the link frame, in metres.
    inertia: 3x3 rotational inertia about the centre of mass, expressed in
             link-frame axes. It must be symmetric positive definite and satisfy
             the triangle inequalities on its principal moments.
    name: Optional label used for external-wrench lookup and error messages.
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
        """
        Raise if the inertia tensor cannot belong to a real rigid body.
        We check positive definiteness and the triangle inequality on the principal
        moments. An inertia that fails these can make the mass matrix indefinite,
        which can otherwise be mistaken for a dynamics failure.
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
    """
    Parameters:
    joint_type: revolute or prismatic.
    axis: Unit axis in the joint frame. Normalised on construction.
    origin: Fixed 4x4 transform from the previous link's frame to this joint's frame.
    limits: Optional (lower, upper). It is recorded for trajectory and workspace code;
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
        # The joint's own 4x4 transform at coordinate q, in the joint frame.
        # This is the "joint motion" in the URDF convention.
        if self.joint_type == 'revolute':
            return transform(rot_axis(self.axis, q), np.zeros(3))
        return transform(np.eye(3), self.axis * q)

    def motion_subspace(self) -> NDArray:
        # Featherstone's ``S``: the spatial motion vector this joint generates.

        S = np.zeros(6)
        if self.joint_type == 'revolute':
            S[:3] = self.axis
        else:
            S[3:] = self.axis
        return S


class Robot:
    """
    A fixed-base serial chain of arbitrary length.

    Parameters:
    links, joints: Equal-length sequences, joints[i] drives links[i].
    gravity: Gravitational acceleration in the base frame, default
             [0, 0, -9.81].
    name: Optional label.

    Dynamics methods live on this class but are implemented in
    robot_dynamics.dynamics; they are attached at import time to keep the
    model free of algorithm code.
    """

    def __init__(self, links: Sequence[Link], joints: Sequence[Joint],
                 gravity: ArrayLike = (0.0, 0.0, -9.81),
                 name: str = 'robot'):
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
        # Stored here so the algorithms are written against a parent array.
        self.parent = [i - 1 for i in range(self.n)]

    @property
    def n(self) -> int:
        # Number of degrees of freedom
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
        # Checking every link is physically valid.
        for link in self.links:
            link.validate_physical()

    def check_q(self, *arrays: ArrayLike) -> tuple[NDArray, ...]:
        """Convert joint-space inputs to arrays of shape ``(n,)``.

        A single check keeps shape errors consistent across the public API.
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
        # A random q, which gives joint limits where they are defined.
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
