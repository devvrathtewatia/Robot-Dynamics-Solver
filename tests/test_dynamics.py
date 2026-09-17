"""Phases 7, 8, 11, 12, 13, 14 and 19: the dynamics validation suite.

Organised by *what kind of evidence* each test provides, because that is what
determines how much it is worth:

* **analytic** — against an independent hand derivation. The anchor.
* **structural** — properties any correct model must have.
* **isolation** — each term switched off in turn.
* **round trip** — inverse then forward dynamics recovers the input.
* **cross-backend** — independent implementations agree.

Property tests alone can only prove self-consistency: a model could satisfy every
one of them and still describe the wrong robot. The analytic and symbolic tests are
what rule that out.
"""
from __future__ import annotations

import numpy as np
import pytest

from robot_dynamics import (
    analytic_two_dof,
    bimanual_arm,
    planar_chain,
    prismatic_chain,
    random_robot,
    spatial_chain,
    two_dof_planar,
)
from robot_dynamics.dynamics import crba, lagrangian, rnea

RNG = np.random.default_rng(31337)
TIGHT = 1e-10

# One identical API exercised over every DOF count and topology. The spec's
# "your test suite should literally instantiate 2, 3, 4, 5, 6 DOF" requirement,
# extended past 6 to make the point that nothing is special about six.
ROBOTS = [
    ('planar2', planar_chain(2)),
    ('planar3', planar_chain(3)),
    ('planar4', planar_chain(4)),
    ('planar5', planar_chain(5)),
    ('planar6', planar_chain(6)),
    ('spatial4', spatial_chain(4)),
    ('spatial6', spatial_chain(6)),
    ('spatial9', spatial_chain(9)),
    ('random2', random_robot(2, seed=2)),
    ('random5', random_robot(5, seed=5)),
    ('random6', random_robot(6, seed=6)),
    ('random8', random_robot(8, seed=8)),
    ('mixed6', random_robot(6, seed=42, prismatic_fraction=0.4)),
    ('gantry3', prismatic_chain(3)),
    ('bimanual6', bimanual_arm()),
]
IDS = [name for name, _ in ROBOTS]


def _state(robot, scale=0.8):
    q = robot.random_configuration(RNG)
    return q, RNG.normal(size=robot.n) * scale, RNG.normal(size=robot.n) * scale


# =====================================================================
# ANALYTIC — against an independent derivation
# =====================================================================
@pytest.mark.parametrize('q1,q2,qd1,qd2', [
    (0.0, 0.0, 0.0, 0.0), (0.3, -0.7, 0.5, 0.2), (1.9, 2.4, -1.1, 0.8),
    (-2.2, 0.15, 0.03, -2.0), (np.pi / 2, np.pi / 2, 1.0, 1.0),
])
def test_two_dof_matches_the_closed_form(q1, q2, qd1, qd2):
    """The single most valuable test: an external hand derivation."""
    params = {'m1': 1.3, 'm2': 0.8, 'l1': 0.9, 'l2': 0.6,
              'I1': 0.11, 'I2': 0.07}
    robot = two_dof_planar(**params)
    q, qd = np.array([q1, q2]), np.array([qd1, qd2])
    M, C, G = analytic_two_dof(q, qd, **params)

    assert np.allclose(robot.mass_matrix(q), M, atol=1e-9)
    assert np.allclose(robot.coriolis(q, qd), C, atol=1e-9)
    assert np.allclose(robot.gravity_torque(q), G, atol=1e-9)

    qdd = np.array([0.4, -0.9])
    tau_expected = M @ qdd + C @ qd + G
    assert np.allclose(robot.inverse_dynamics(q, qd, qdd), tau_expected,
                       atol=1e-9)


def test_single_pendulum_against_the_textbook_equation():
    r"""A one-link arm must satisfy
    :math:`(I + m l_c^2)\ddot\theta + m g l_c\cos\theta = \tau`.
    """
    m, L, lc, Izz = 2.0, 1.0, 0.5, 0.15
    robot = two_dof_planar(m1=m, l1=L, lc1=lc, I1=Izz)
    robot.links = robot.links[:1]
    robot.joints = robot.joints[:1]
    robot.parent = [-1]
    for theta in (-1.2, 0.0, 0.4, 2.7):
        expected_M = Izz + m * lc ** 2
        assert np.allclose(robot.mass_matrix([theta])[0, 0], expected_M)
        assert np.allclose(robot.gravity_torque([theta])[0],
                           m * 9.81 * lc * np.cos(theta))
        # a pendulum has no Coriolis term at all: M is constant
        assert np.allclose(robot.coriolis([theta], [1.7]), 0.0)


# =====================================================================
# STRUCTURAL — properties any correct model must have
# =====================================================================
@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_mass_matrix_is_symmetric(name, robot):
    r""":math:`M = M^\top`, checked before the library's own symmetrisation."""
    for _ in range(5):
        q = robot.random_configuration(RNG)
        for method in ('lagrangian', 'crba', 'rnea'):
            M = robot.mass_matrix(q, method=method)
            assert np.max(np.abs(M - M.T)) < 1e-11, f'{name}/{method}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_mass_matrix_is_positive_definite(name, robot):
    r""":math:`x^\top M x > 0` for all nonzero x, i.e. every eigenvalue positive.

    Physically: no direction of joint motion can have zero or negative kinetic
    energy. This is also the precondition for the Cholesky solve in forward dynamics.
    """
    for _ in range(5):
        q = robot.random_configuration(RNG)
        M = robot.mass_matrix(q)
        eig = np.linalg.eigvalsh(M)
        assert eig.min() > 0.0, f'{name}: eigenvalues {eig}'
        for _ in range(10):
            x = RNG.normal(size=robot.n)
            assert x @ M @ x > 0.0


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_mdot_minus_2c_is_skew_symmetric(name, robot):
    r""":math:`\dot M - 2C` skew-symmetric — the passivity property.

    Not required merely for elegance: it is what makes the standard Lyapunov proof of
    computed-torque stability valid, and it holds only for the Christoffel
    factorisation of C. A different but numerically valid factorisation of the same
    :math:`C\dot q` would fail here, so this test pins down the factorisation, not
    just the product.
    """
    q, qd, _ = _state(robot)
    C = robot.coriolis(q, qd)
    dM = lagrangian.mass_matrix_derivative(robot, q)
    Mdot = dM @ qd
    S = Mdot - 2.0 * C
    assert np.max(np.abs(S + S.T)) < 1e-9, f'{name}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_kinetic_energy_is_consistent_with_the_mass_matrix(name, robot):
    q, qd, _ = _state(robot)
    assert np.isclose(robot.kinetic_energy(q, qd),
                      0.5 * qd @ robot.mass_matrix(q) @ qd)
    assert robot.kinetic_energy(q, np.zeros(robot.n)) == pytest.approx(0.0)
    assert robot.kinetic_energy(q, qd) > 0.0


# =====================================================================
# ISOLATION — switch each term off in turn
# =====================================================================
@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_at_rest_the_torque_is_pure_gravity(name, robot):
    r""":math:`\dot q = \ddot q = 0 \Rightarrow \tau = G(q)`."""
    q = robot.random_configuration(RNG)
    z = np.zeros(robot.n)
    G = robot.gravity_torque(q)
    for method in ('lagrangian', 'rnea'):
        assert np.allclose(robot.inverse_dynamics(q, z, z, method=method), G,
                           atol=TIGHT), f'{name}/{method}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_no_gravity_means_no_gravity_torque(name, robot):
    r""":math:`g = 0 \Rightarrow G(q) = 0`."""
    saved = robot.gravity.copy()
    try:
        robot.gravity = np.zeros(3)
        q = robot.random_configuration(RNG)
        assert np.allclose(robot.gravity_torque(q, method='lagrangian'), 0.0)
        assert np.allclose(robot.gravity_torque(q, method='rnea'), 0.0,
                           atol=TIGHT)
    finally:
        robot.gravity = saved


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_zero_acceleration_leaves_coriolis_plus_gravity(name, robot):
    r""":math:`\ddot q = 0 \Rightarrow \tau = C\dot q + G`."""
    q, qd, _ = _state(robot)
    z = np.zeros(robot.n)
    expected = robot.coriolis(q, qd) @ qd + robot.gravity_torque(q)
    assert np.allclose(robot.inverse_dynamics(q, qd, z), expected, atol=1e-9)


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_zero_velocity_leaves_inertia_plus_gravity(name, robot):
    r""":math:`\dot q = 0 \Rightarrow \tau = M\ddot q + G`."""
    q, _, qdd = _state(robot)
    z = np.zeros(robot.n)
    expected = robot.mass_matrix(q) @ qdd + robot.gravity_torque(q)
    assert np.allclose(robot.inverse_dynamics(q, z, qdd), expected, atol=1e-9)


def test_a_gantry_has_no_velocity_product_terms():
    """A purely prismatic chain has a constant M, so C must vanish identically."""
    robot = prismatic_chain(3)
    q, qd = RNG.uniform(-0.4, 0.4, 3), RNG.normal(size=3)
    assert np.allclose(robot.coriolis(q, qd), 0.0, atol=1e-12)
    assert np.allclose(lagrangian.mass_matrix_derivative(robot, q), 0.0,
                       atol=1e-12)


# =====================================================================
# ROUND TRIP — the Phase 7/8 consistency requirement
# =====================================================================
@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_inverse_dynamics_equals_the_assembled_terms(name, robot):
    r"""Phase 7: :math:`\tau_{\text{direct}} \approx M\ddot q + C\dot q + G`."""
    q, qd, qdd = _state(robot)
    assembled = (robot.mass_matrix(q) @ qdd
                 + robot.coriolis(q, qd) @ qd
                 + robot.gravity_torque(q))
    for method in ('lagrangian', 'rnea'):
        tau = robot.inverse_dynamics(q, qd, qdd, method=method)
        assert np.allclose(tau, assembled, atol=1e-9), f'{name}/{method}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
@pytest.mark.parametrize('fd_method', ['aba', 'lagrangian'])
def test_forward_inverse_round_trip(name, robot, fd_method):
    r"""Phase 8: :math:`\ddot q \to \tau \to \ddot q` recovers the input."""
    for _ in range(3):
        q, qd, qdd = _state(robot)
        tau = robot.inverse_dynamics(q, qd, qdd)
        recovered = robot.forward_dynamics(q, qd, tau, method=fd_method)
        assert np.allclose(recovered, qdd, atol=1e-8), f'{name}/{fd_method}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_round_trip_the_other_way(name, robot):
    r""":math:`\tau \to \ddot q \to \tau`."""
    q, qd, _ = _state(robot)
    tau = RNG.normal(size=robot.n)
    qdd = robot.forward_dynamics(q, qd, tau)
    assert np.allclose(robot.inverse_dynamics(q, qd, qdd), tau, atol=1e-8)


# =====================================================================
# CROSS-BACKEND — Phases 12, 13, 14
# =====================================================================
@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_lagrange_agrees_with_rnea(name, robot):
    r"""Phase 12: :math:`\tau_{\text{Lagrange}} \approx \tau_{\text{RNEA}}`.

    Two formalisms, sharing only the kinematic model. The strongest single piece of
    evidence in the suite.
    """
    for _ in range(5):
        q, qd, qdd = _state(robot)
        tl = lagrangian.inverse_dynamics(robot, q, qd, qdd)
        tr = rnea.inverse_dynamics(robot, q, qd, qdd)
        rel = np.max(np.abs(tl - tr)) / max(1.0, np.max(np.abs(tr)))
        assert rel < 1e-10, f'{name}: relative difference {rel:.2e}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_lagrange_agrees_with_crba(name, robot):
    r"""Phase 13: :math:`M_{\text{Lagrange}} \approx M_{\text{CRBA}}`."""
    for _ in range(5):
        q = robot.random_configuration(RNG)
        Ml = lagrangian.mass_matrix(robot, q)
        Mc = crba.mass_matrix(robot, q)
        Mr = rnea.mass_matrix(robot, q)
        scale = max(1.0, np.max(np.abs(Ml)))
        assert np.max(np.abs(Ml - Mc)) / scale < 1e-11, f'{name} crba'
        assert np.max(np.abs(Ml - Mr)) / scale < 1e-11, f'{name} rnea'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_lagrange_agrees_with_aba(name, robot):
    r"""Phase 14: :math:`\ddot q_{\text{Lagrange}} \approx \ddot q_{\text{ABA}}`."""
    for _ in range(5):
        q, qd, _ = _state(robot)
        tau = RNG.normal(size=robot.n)
        qa = crba.forward_dynamics(robot, q, qd, tau)
        ql = lagrangian.forward_dynamics(robot, q, qd, tau)
        scale = max(1.0, np.max(np.abs(qa)))
        assert np.max(np.abs(qa - ql)) / scale < 1e-9, f'{name}'


@pytest.mark.parametrize('name,robot', ROBOTS, ids=IDS)
def test_coriolis_matrix_product_matches_the_recursive_vector(name, robot):
    r""":math:`C(q,\dot q)\dot q` from Christoffel symbols vs from RNEA.

    The sharpest test of the Lagrangian layer: it exercises M, its derivative and the
    factorisation simultaneously. A wrong axis permutation in the Christoffel tensor
    is invisible to every other test here and shows up as ~1e-1 in this one.
    """
    for _ in range(5):
        q, qd, _ = _state(robot)
        via_matrix = robot.coriolis(q, qd) @ qd
        via_rnea = rnea.coriolis_vector(robot, q, qd)
        scale = max(1.0, np.max(np.abs(via_rnea)))
        assert np.max(np.abs(via_matrix - via_rnea)) / scale < 1e-10, f'{name}'


@pytest.mark.parametrize('method', ['central', 'forward'])
def test_complex_step_derivative_agrees_with_finite_differences(method):
    """Validate the exact derivative against the approximate ones.

    Confirms the complex step is right, and quantifies how much accuracy a finite
    difference would have cost: central loses about six digits, forward about eight.
    """
    robot = spatial_chain(5)
    q = robot.random_configuration(RNG)
    exact = lagrangian.mass_matrix_derivative(robot, q, method='complex')
    approx = lagrangian.mass_matrix_derivative(robot, q, method=method)
    tol = 1e-7 if method == 'central' else 1e-5
    assert np.max(np.abs(exact - approx)) < tol


def test_unknown_backend_names_are_rejected():
    robot = planar_chain(2)
    q = np.zeros(2)
    for call in (lambda: robot.mass_matrix(q, method='nope'),
                 lambda: robot.gravity_torque(q, method='nope'),
                 lambda: robot.inverse_dynamics(q, q, q, method='nope'),
                 lambda: robot.forward_dynamics(q, q, q, method='nope'),
                 lambda: robot.coriolis(q, q, method='rnea')):
        with pytest.raises(ValueError):
            call()


def test_degenerate_model_raises_rather_than_returning_nonsense():
    """A massless-in-effect model should fail loudly, not return garbage."""
    from robot_dynamics import Joint, Link, Robot
    # A link whose inertia about the joint axis is below the solver's threshold:
    # the joint cannot resist any torque, so the acceleration is unbounded.
    robot = Robot([Link(1.0, com=[0, 0, 0], inertia=np.eye(3) * 1e-18)],
                  [Joint('revolute', [0, 0, 1])])
    with pytest.raises(np.linalg.LinAlgError):
        robot.forward_dynamics([0.0], [0.0], [1.0], method='aba')
    # and the model validator should have flagged it before it got that far
    with pytest.raises(ValueError):
        robot.validate()
