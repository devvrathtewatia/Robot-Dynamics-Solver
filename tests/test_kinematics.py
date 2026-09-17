"""Phase 1 and 3: primitives, transforms, kinematics and Jacobians."""
from __future__ import annotations

import numpy as np
import pytest

from robot_dynamics import Joint, Link, Robot, planar_chain, random_robot
from robot_dynamics.kinematics import (
    com_jacobians,
    forward_kinematics,
    jacobian,
    jacobian_dot,
    link_transforms,
)
from robot_dynamics.utils import (
    is_rotation,
    plucker,
    plucker_inverse,
    rot_axis,
    rot_x,
    rot_y,
    rot_z,
    skew,
    spatial_inertia,
    transform,
    transform_inverse,
    unskew,
)

RNG = np.random.default_rng(2024)


# --------------------------------------------------------------- rotations
@pytest.mark.parametrize('fn', [rot_x, rot_y, rot_z])
def test_rotations_are_valid_rotation_matrices(fn):
    for theta in RNG.uniform(-np.pi, np.pi, size=12):
        assert is_rotation(fn(theta))


@pytest.mark.parametrize('fn,axis', [(rot_x, [1, 0, 0]), (rot_y, [0, 1, 0]),
                                     (rot_z, [0, 0, 1])])
def test_axis_is_fixed_by_its_own_rotation(fn, axis):
    a = np.array(axis, dtype=float)
    for theta in RNG.uniform(-np.pi, np.pi, size=6):
        assert np.allclose(fn(theta) @ a, a)


def test_rotations_compose_additively():
    a, b = 0.31, -1.27
    assert np.allclose(rot_z(a) @ rot_z(b), rot_z(a + b))


def test_skew_reproduces_the_cross_product():
    """The defining property, checked against NumPy as the spec asks."""
    for _ in range(50):
        v, w = RNG.normal(size=3), RNG.normal(size=3)
        assert np.allclose(skew(v) @ w, np.cross(v, w))


def test_skew_is_antisymmetric_and_invertible():
    v = RNG.normal(size=3)
    S = skew(v)
    assert np.allclose(S, -S.T)
    assert np.allclose(unskew(S), v)
    assert np.allclose(S @ v, 0.0)          # v x v = 0


def test_rot_axis_matches_the_elementary_rotations():
    for theta in RNG.uniform(-np.pi, np.pi, size=6):
        assert np.allclose(rot_axis([1, 0, 0], theta), rot_x(theta))
        assert np.allclose(rot_axis([0, 1, 0], theta), rot_y(theta))
        assert np.allclose(rot_axis([0, 0, 1], theta), rot_z(theta))


def test_skew_rejects_bad_shapes():
    with pytest.raises(ValueError):
        skew(np.zeros(4))


# ------------------------------------------------------------- transforms
def test_transform_inverse_gives_identity():
    r"""The Phase 1 requirement: :math:`T T^{-1} = I`."""
    for _ in range(50):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)         # must be the SAME vector, normalised
        R = rot_axis(axis, RNG.uniform(-np.pi, np.pi))
        assert is_rotation(R)                # guard: a bad axis would void the test
        T = transform(rot_z(RNG.uniform()) @ R, RNG.normal(size=3))
        assert np.allclose(T @ transform_inverse(T), np.eye(4))
        assert np.allclose(transform_inverse(T) @ T, np.eye(4))


def test_transform_inverse_matches_a_general_matrix_inverse():
    T = transform(rot_y(0.4) @ rot_x(-1.1), [0.2, -0.7, 1.3])
    assert np.allclose(transform_inverse(T), np.linalg.inv(T))


def test_transform_blocks_round_trip():
    R, p = rot_z(0.9), np.array([1.0, 2.0, 3.0])
    T = transform(R, p)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], p)
    assert np.allclose(T[3], [0, 0, 0, 1])


# ---------------------------------------------------------------- spatial
def test_plucker_inverse_is_a_true_inverse():
    T = transform(rot_x(0.7) @ rot_z(-0.3), [0.1, 0.2, -0.4])
    X = plucker(T)
    assert np.allclose(X @ plucker_inverse(X), np.eye(6))


def test_spatial_inertia_is_symmetric_and_has_the_right_mass():
    I = spatial_inertia(2.5, [0.1, -0.2, 0.05], np.diag([0.3, 0.4, 0.5]))
    assert np.allclose(I, I.T)
    assert np.allclose(I[3:, 3:], 2.5 * np.eye(3))


# ------------------------------------------------------------- kinematics
def test_two_dof_forward_kinematics_against_hand_derivation():
    """Manually derived 2-DOF planar case, as the spec requires."""
    l1 = 1.0
    robot = planar_chain(2, length=l1)
    for q1, q2 in [(0.0, 0.0), (np.pi / 2, 0.0), (0.3, -0.8), (1.1, 2.2)]:
        T = forward_kinematics(robot, [q1, q2])
        # The chain's last frame sits at the *base* of link 2, one link out.
        expected = np.array([l1 * np.cos(q1), l1 * np.sin(q1), 0.0])
        assert np.allclose(T[:3, 3], expected, atol=1e-12)
        assert np.allclose(T[:3, :3], rot_z(q1 + q2), atol=1e-12)


def test_three_dof_tip_position_against_hand_derivation():
    """Tip of a 3-link planar arm, from the closed-form sum of link vectors."""
    L = 0.4
    robot = planar_chain(3, length=L)
    q = np.array([0.2, -0.5, 0.9])
    T = link_transforms(robot, q)[-1]
    tip = T[:3, :3] @ np.array([L, 0, 0]) + T[:3, 3]
    c = np.cumsum(q)
    expected = np.array([L * (np.cos(c[0]) + np.cos(c[1]) + np.cos(c[2])),
                         L * (np.sin(c[0]) + np.sin(c[1]) + np.sin(c[2])),
                         0.0])
    assert np.allclose(tip, expected, atol=1e-12)


def test_all_link_transforms_are_rigid():
    robot = random_robot(6, seed=5)
    for T in link_transforms(robot, robot.random_configuration(RNG)):
        assert is_rotation(T[:3, :3])
        assert np.allclose(T[3], [0, 0, 0, 1])


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 8])
def test_jacobian_matches_a_numerical_derivative(n):
    r"""Verify :math:`\dot x = J\dot q` against a finite-difference of the pose.

    The Jacobian is the bridge to control, so an error here would silently corrupt
    the mass matrix as well - both are built from the same columns.
    """
    robot = random_robot(n, seed=n)
    q = robot.random_configuration(RNG)
    J = jacobian(robot, q)
    h = 1e-7
    for k in range(n):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        Tp, Tm = forward_kinematics(robot, qp), forward_kinematics(robot, qm)
        dp = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        assert np.allclose(J[:3, k], dp, atol=1e-6)
        # angular part: dR/dt R^T is the angular-velocity skew matrix
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h)
        W = dR @ forward_kinematics(robot, q)[:3, :3].T
        assert np.allclose(J[3:, k], unskew((W - W.T) / 2), atol=1e-6)


def test_jacobian_columns_beyond_the_link_are_zero():
    """A joint downstream of a body cannot move it."""
    robot = random_robot(5, seed=9)
    q = robot.random_configuration(RNG)
    for i in range(robot.n):
        J = jacobian(robot, q, link=i)
        assert np.allclose(J[:, i + 1:], 0.0)


def test_com_jacobians_match_finite_differences():
    from robot_dynamics.kinematics import com_positions
    robot = random_robot(4, seed=13)
    q = robot.random_configuration(RNG)
    Js = com_jacobians(robot, q)
    h = 1e-7
    for k in range(robot.n):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        cp, cm = com_positions(robot, qp), com_positions(robot, qm)
        for i in range(robot.n):
            assert np.allclose(Js[i][:3, k], (cp[i] - cm[i]) / (2 * h),
                               atol=1e-6)


def test_jacobian_dot_matches_finite_differences():
    robot = random_robot(4, seed=17)
    q = robot.random_configuration(RNG)
    qd = RNG.normal(size=robot.n)
    Jd = jacobian_dot(robot, q, qd)
    h = 1e-7
    fd = (jacobian(robot, q + h * qd) - jacobian(robot, q - h * qd)) / (2 * h)
    assert np.allclose(Jd, fd, atol=1e-6)


def test_prismatic_joint_translates_along_its_axis():
    robot = Robot([Link(1.0, inertia=np.eye(3) * 0.01)],
                  [Joint('prismatic', [0, 0, 1])])
    T = forward_kinematics(robot, [0.37])
    assert np.allclose(T[:3, 3], [0, 0, 0.37])
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(jacobian(robot, [0.37])[:3, 0], [0, 0, 1])


# --------------------------------------------------------- model validation
def test_model_rejects_unphysical_inputs():
    with pytest.raises(ValueError):
        Link(mass=-1.0)
    with pytest.raises(ValueError):
        Link(mass=1.0, inertia=np.array([[1.0, 0.5, 0], [0.0, 1, 0], [0, 0, 1]]))
    with pytest.raises(ValueError):
        Joint('revolute', [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        Joint('helical', [0, 0, 1])
    with pytest.raises(ValueError):
        Robot([Link(1.0)], [Joint(), Joint()])


def test_inertia_triangle_inequality_is_enforced():
    bad = Robot([Link(1.0, inertia=np.diag([0.01, 0.01, 10.0]))], [Joint()])
    with pytest.raises(ValueError, match='triangle inequality'):
        bad.validate()


def test_joint_axis_is_normalised():
    j = Joint('revolute', [0.0, 0.0, 3.0])
    assert np.allclose(j.axis, [0, 0, 1])


def test_wrong_sized_configuration_is_rejected():
    robot = planar_chain(3)
    with pytest.raises(ValueError):
        robot.mass_matrix([0.1, 0.2])
