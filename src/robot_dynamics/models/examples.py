"""Ready-made robot models, plus closed-form references to validate against.

The builders take ``n`` as an argument rather than existing in per-DOF variants —
:func:`planar_chain` and :func:`spatial_chain` generate any number of joints, which
is how the test suite runs one identical API over ``n = 2..8``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.rotations import rpy_to_matrix
from ..utils.transforms import transform
from .robot import Joint, Link, Robot


def _cylinder_inertia(mass: float, radius: float, length: float,
                      axis: str = 'x') -> NDArray:
    """Inertia of a solid cylinder about its centre of mass.

    Used instead of an idealised thin rod so the tensor is strictly positive
    definite and satisfies the triangle inequality — a rod with zero transverse
    inertia makes the mass matrix singular for spatial motion, which then looks
    like a solver bug.
    """
    I_axial = 0.5 * mass * radius ** 2
    I_trans = mass * (3.0 * radius ** 2 + length ** 2) / 12.0
    diag = {'x': (I_axial, I_trans, I_trans),
            'y': (I_trans, I_axial, I_trans),
            'z': (I_trans, I_trans, I_axial)}[axis]
    return np.diag(diag)


def _box_inertia(mass: float, dx: float, dy: float, dz: float) -> NDArray:
    """Inertia of a solid cuboid about its centre of mass. Always physical."""
    return np.diag([mass * (dy ** 2 + dz ** 2) / 12.0,
                    mass * (dx ** 2 + dz ** 2) / 12.0,
                    mass * (dx ** 2 + dy ** 2) / 12.0])


def planar_chain(n: int, length: float = 0.4, mass: float = 1.0,
                 radius: float = 0.03,
                 gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
                 ) -> Robot:
    """An ``n``-link planar arm: revolute joints about z, links along x.

    Gravity defaults to ``-y`` so that it acts **in** the plane of motion, matching
    the usual textbook planar derivation. With the default ``-z`` of
    :class:`~robot_dynamics.models.robot.Robot` this arm is horizontal and
    :math:`G(q) = 0` identically — a valid physical result the test suite checks
    separately, but not what a planar example should show.
    """
    if n < 1:
        raise ValueError('planar_chain needs at least one link')
    links, joints = [], []
    for i in range(n):
        origin = transform(np.eye(3),
                           [0.0 if i == 0 else length, 0.0, 0.0])
        joints.append(Joint('revolute', [0.0, 0.0, 1.0], origin,
                            name=f'q{i + 1}'))
        links.append(Link(mass=mass,
                          com=[length / 2.0, 0.0, 0.0],
                          inertia=_cylinder_inertia(mass, radius, length, 'x'),
                          name=f'link{i + 1}'))
    return Robot(links, joints, gravity=gravity, name=f'planar{n}')


def spatial_chain(n: int, length: float = 0.3, mass: float = 1.2,
                  radius: float = 0.035) -> Robot:
    """An ``n``-joint genuinely 3D chain, axes cycling through z, y, x.

    Planar chains hide whole classes of bug: the inertia rotation
    :math:`R I R^\\top` is nearly irrelevant when all motion shares one axis, and
    the angular part of the Coriolis term stays small. This alternates axes so every
    term is exercised.
    """
    if n < 1:
        raise ValueError('spatial_chain needs at least one joint')
    axes = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    links, joints = [], []
    for i in range(n):
        origin = transform(np.eye(3), [0.0 if i == 0 else length, 0.0, 0.0])
        joints.append(Joint('revolute', axes[i % 3], origin, name=f'q{i + 1}'))
        links.append(Link(mass=mass,
                          com=[length / 2.0, 0.0, 0.0],
                          inertia=_cylinder_inertia(mass, radius, length, 'x'),
                          name=f'link{i + 1}'))
    return Robot(links, joints, name=f'spatial{n}')


def random_robot(n: int, seed: int = 0, prismatic_fraction: float = 0.0
                 ) -> Robot:
    """A randomly generated ``n``-DOF chain — random axes, frames and inertias.

    The strongest generality test available: nothing about the geometry is aligned
    or symmetric, so any accidental dependence on a special case shows up. Inertias
    are built from random boxes, which guarantees they are physically valid.
    """
    rng = np.random.default_rng(seed)
    links, joints = [], []
    for i in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = rpy_to_matrix(*rng.uniform(-np.pi, np.pi, size=3))
        p = rng.uniform(-0.3, 0.3, size=3)
        kind = ('prismatic'
                if rng.random() < prismatic_fraction else 'revolute')
        joints.append(Joint(kind, axis, transform(R, p), name=f'q{i + 1}'))
        m = float(rng.uniform(0.3, 2.5))
        dims = rng.uniform(0.05, 0.35, size=3)
        links.append(Link(mass=m,
                          com=rng.uniform(-0.15, 0.15, size=3),
                          inertia=_box_inertia(m, *dims),
                          name=f'link{i + 1}'))
    return Robot(links, joints, name=f'random{n}')


def two_dof_planar(m1: float = 1.0, m2: float = 1.0,
                   l1: float = 1.0, l2: float = 1.0,
                   lc1: float | None = None, lc2: float | None = None,
                   I1: float = 0.1, I2: float = 0.1,
                   g: float = 9.81) -> Robot:
    """The textbook 2-DOF planar arm, parameterised to match an analytic solution.

    Pairs with :func:`analytic_two_dof` for the Phase 19 check against a hand
    derivation. Inertias are set as pure ``Izz`` values with tiny transverse terms,
    exactly as the planar derivation assumes.
    """
    lc1 = l1 / 2.0 if lc1 is None else lc1
    lc2 = l2 / 2.0 if lc2 is None else lc2
    tiny = 1e-9
    links = [Link(m1, [lc1, 0, 0], np.diag([tiny, tiny, I1]), 'link1'),
             Link(m2, [lc2, 0, 0], np.diag([tiny, tiny, I2]), 'link2')]
    joints = [Joint('revolute', [0, 0, 1], np.eye(4), name='q1'),
              Joint('revolute', [0, 0, 1],
                    transform(np.eye(3), [l1, 0, 0]), name='q2')]
    return Robot(links, joints, gravity=(0.0, -g, 0.0), name='two_dof_planar')


def analytic_two_dof(q: NDArray, qd: NDArray,
                     m1: float = 1.0, m2: float = 1.0,
                     l1: float = 1.0, l2: float = 1.0,
                     lc1: float | None = None, lc2: float | None = None,
                     I1: float = 0.1, I2: float = 0.1,
                     g: float = 9.81) -> tuple[NDArray, NDArray, NDArray]:
    r"""Closed-form :math:`M`, :math:`C`, :math:`G` for the 2-DOF planar arm.

    Derived by hand from the Euler-Lagrange equations, independent of anything in
    this library. This is the anchor for the whole validation suite: the property
    tests prove self-consistency, but only an external derivation proves the model
    is the *intended* one.

    .. math::

        M_{11} &= m_1 l_{c1}^2 + I_1 + I_2
                  + m_2(l_1^2 + l_{c2}^2 + 2 l_1 l_{c2}\cos q_2) \\
        M_{12} &= m_2(l_{c2}^2 + l_1 l_{c2}\cos q_2) + I_2 \\
        M_{22} &= m_2 l_{c2}^2 + I_2

    with :math:`h = -m_2 l_1 l_{c2}\sin q_2` giving
    :math:`C = \begin{bmatrix} h\dot q_2 & h(\dot q_1+\dot q_2) \\
    -h\dot q_1 & 0\end{bmatrix}`.
    """
    lc1 = l1 / 2.0 if lc1 is None else lc1
    lc2 = l2 / 2.0 if lc2 is None else lc2
    q1, q2 = float(q[0]), float(q[1])
    qd1, qd2 = float(qd[0]), float(qd[1])
    c2, s2 = np.cos(q2), np.sin(q2)

    M = np.array([
        [m1 * lc1 ** 2 + I1
         + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I2,
         m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2],
        [m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2,
         m2 * lc2 ** 2 + I2]])

    h = -m2 * l1 * lc2 * s2
    C = np.array([[h * qd2, h * (qd1 + qd2)],
                  [-h * qd1, 0.0]])

    G = np.array([
        (m1 * lc1 + m2 * l1) * g * np.cos(q1)
        + m2 * lc2 * g * np.cos(q1 + q2),
        m2 * lc2 * g * np.cos(q1 + q2)])
    return M, C, G


def bimanual_arm(side: int = +1) -> Robot:
    """The 6-DOF arm from the companion Gazebo mobile-manipulator project.

    Kinematics as designed there: J1 yaw about z, J2 and J3 pitch about y,
    then a spherical wrist J4 roll / J5 pitch / J6 roll whose axes intersect at a
    single point. Links ``L1 = L2 = 0.16`` m and a ``0.07`` m gripper give the
    ``0.32`` m reach that project fixed as a requirement.

    Present so the dynamics library can be pointed at a robot that actually exists:
    the joint torques it predicts are the ones that arm's servos would have to
    deliver, which is the natural next question once the Gazebo simulation runs.
    """
    L1 = L2 = 0.16
    LG = 0.07
    shoulder = transform(np.eye(3), [0.08, 0.12 * side, 0.24])

    masses = [0.35, 0.30, 0.26, 0.16, 0.13, 0.11]
    joints = [
        Joint('revolute', [0, 0, 1], shoulder,
              limits=(-np.deg2rad(150), np.deg2rad(150)), name='j1_yaw'),
        Joint('revolute', [0, 1, 0], np.eye(4),
              limits=(-np.deg2rad(115), np.deg2rad(115)), name='j2_pitch'),
        Joint('revolute', [0, 1, 0], transform(np.eye(3), [L1, 0, 0]),
              limits=(-np.deg2rad(145), np.deg2rad(145)), name='j3_elbow'),
        Joint('revolute', [1, 0, 0], transform(np.eye(3), [L2, 0, 0]),
              limits=(-np.pi, np.pi), name='j4_roll'),
        Joint('revolute', [0, 1, 0], np.eye(4),
              limits=(-np.deg2rad(140), np.deg2rad(140)), name='j5_pitch'),
        Joint('revolute', [1, 0, 0], np.eye(4),
              limits=(-np.pi, np.pi), name='j6_roll'),
    ]
    links = [
        Link(masses[0], [0.0, 0.0, 0.02],
             _box_inertia(masses[0], 0.06, 0.06, 0.05), 'shoulder_yaw'),
        Link(masses[1], [L1 / 2, 0.0, 0.0],
             _cylinder_inertia(masses[1], 0.022, L1, 'x'), 'upper_arm'),
        Link(masses[2], [L2 / 2, 0.0, 0.0],
             _cylinder_inertia(masses[2], 0.020, L2, 'x'), 'forearm'),
        Link(masses[3], [0.01, 0.0, 0.0],
             _box_inertia(masses[3], 0.04, 0.04, 0.04), 'wrist_roll'),
        Link(masses[4], [0.01, 0.0, 0.0],
             _box_inertia(masses[4], 0.035, 0.035, 0.035), 'wrist_pitch'),
        Link(masses[5], [LG / 2, 0.0, 0.0],
             _box_inertia(masses[5], LG, 0.05, 0.03), 'gripper'),
    ]
    return Robot(links, joints, name=f'bimanual_{"L" if side > 0 else "R"}')


def prismatic_chain(n: int = 3, mass: float = 1.0) -> Robot:
    """A Cartesian gantry: all prismatic joints along x, y, z in turn.

    Included because a prismatic joint's mass-matrix contribution is
    configuration-independent, so :math:`\\partial M/\\partial q` should vanish for
    the pure-translation part. Any Coriolis term appearing here would be a bug.
    """
    axes = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    links, joints = [], []
    for i in range(n):
        joints.append(Joint('prismatic', axes[i % 3], np.eye(4),
                            limits=(-0.5, 0.5), name=f'd{i + 1}'))
        links.append(Link(mass, [0.0, 0.0, 0.0],
                          _box_inertia(mass, 0.1, 0.1, 0.1), f'slider{i + 1}'))
    return Robot(links, joints, name=f'gantry{n}')
