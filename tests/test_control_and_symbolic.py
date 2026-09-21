"""Phases 5, 9, 10 and 15: symbolic validation, wrenches, trajectories, control."""

from __future__ import annotations

import numpy as np
import pytest

from robot_dynamics import (
    bimanual_arm,
    planar_chain,
    random_robot,
    spatial_chain,
    two_dof_planar,
)
from robot_dynamics.control import ComputedTorque, FeedforwardPD, PDGravity
from robot_dynamics.dynamics import lagrangian, rnea
from robot_dynamics.dynamics import symbolic as symmod
from robot_dynamics.kinematics import jacobian
from robot_dynamics.simulation import energy_history, simulate, tracking_error
from robot_dynamics.trajectory import (
    MultiPointTrajectory,
    PolynomialTrajectory,
    quintic,
)

RNG = np.random.default_rng(5150)
sympy_required = pytest.mark.skipif(not symmod.HAVE_SYMPY,
                                    reason='SymPy not installed')
@sympy_required
@pytest.mark.parametrize('robot', [planar_chain(2), two_dof_planar(),
                                   spatial_chain(2), planar_chain(3)],
                         ids=['planar2', 'twodof', 'spatial2', 'planar3'])
def test_symbolic_derivation_matches_the_numerical_layer(robot):
    # An independent derivation route: literal differentiation of L = T - V.
    # Compare the numerical M, C, and G with direct symbolic differentiation.
    model = symmod.symbolic_model(robot)
    qs, qds = model['q'], model['qd']
    for _ in range(4):
        q = robot.random_configuration(RNG)
        qd = RNG.normal(size=robot.n)
        subs = {**dict(zip(qs, q, strict=True)), **dict(zip(qds, qd, strict=True))}
        M_sym = np.array(model['M'].subs(subs).evalf(), dtype=float)
        C_sym = np.array(model['C'].subs(subs).evalf(), dtype=float)
        G_sym = np.array(model['G'].subs(subs).evalf(), dtype=float).ravel()
        assert np.allclose(robot.mass_matrix(q), M_sym, atol=1e-9)
        assert np.allclose(robot.coriolis(q, qd), C_sym, atol=1e-9)
        assert np.allclose(robot.gravity_torque(q), G_sym, atol=1e-9)


@sympy_required
def test_literal_euler_lagrange_matches_the_mcg_split():
    # The M, C, G split is a faithful rearrangement.
    # Confirms tau_i = d/dt (dL/dqd_i) - dL/dq_i gives the same torque as
    # Check the standard M*qdd + C*qd + G form directly.
    import sympy as sp
    robot = planar_chain(2)
    model = symmod.euler_lagrange(robot)
    diff = sp.simplify(model['tau_euler_lagrange'] - model['tau'])
    assert all(sp.simplify(e) == 0 for e in diff), f'residual {diff}'


@sympy_required
def test_lambdified_model_is_free_of_sympy_and_still_correct():
    # The intended use of the symbolic layer: derive once, evaluate numerically.
    robot = planar_chain(2)
    compiled = symmod.lambdify_model(robot)
    q, qd = RNG.normal(size=2), RNG.normal(size=2)
    assert np.allclose(np.array(compiled['mass_matrix'](q), dtype=float),
                       robot.mass_matrix(q), atol=1e-9)
    assert np.allclose(np.array(compiled['coriolis'](q, qd), dtype=float),
                       robot.coriolis(q, qd), atol=1e-9)
    assert np.allclose(np.array(compiled['gravity'](q), dtype=float).ravel(),
                       robot.gravity_torque(q), atol=1e-9)


def test_numerical_layer_does_not_import_sympy():
    # Enforce the architectural boundary the spec insists on.
    # Checked mechanically, not by convention: a stray import sympy in a hot path
    # is exactly the regression that turns a library back into a notebook.
    import inspect
    for module in (lagrangian, rnea):
        src = inspect.getsource(module)
        assert 'sympy' not in src, f'{module.__name__} must not import SymPy'
@pytest.mark.parametrize('robot', [planar_chain(3), spatial_chain(4),
                                   random_robot(5, seed=21), bimanual_arm()],
                         ids=['planar3', 'spatial4', 'random5', 'bimanual6'])
def test_external_wrench_enters_as_minus_jacobian_transpose_force(robot):
    # Check tau = M*qdd + C*qd + G - J^T * f_ext.
    # Validates the spatial-force transform inside RNEA against the independent
    # Jacobian route. The two use entirely different machinery - Plucker transforms
    # versus geometric Jacobian columns - and the [angular; linear] versus
    # [linear; angular] ordering difference between them is precisely the kind of
    # convention slip this catches.
    q, qd, qdd = (RNG.normal(size=robot.n) * 0.6 for _ in range(3))
    link = robot.n - 1
    wrench = RNG.normal(size=6) * 5.0            # [force; moment] in the base frame

    tau_free = robot.inverse_dynamics(q, qd, qdd)
    expected = tau_free - jacobian(robot, q, link=link).T @ wrench
    for method in ('rnea', 'lagrangian'):
        tau = robot.inverse_dynamics(q, qd, qdd, method=method,
                                     external_wrenches={link: wrench})
        assert np.allclose(tau, expected, atol=1e-8), method


def test_external_wrench_is_consistent_between_forward_and_inverse():
    robot = spatial_chain(4)
    q, qd, qdd = (RNG.normal(size=4) * 0.5 for _ in range(3))
    ext = {3: np.array([2.0, -1.0, 4.0, 0.3, -0.2, 0.1])}
    tau = robot.inverse_dynamics(q, qd, qdd, external_wrenches=ext)
    for method in ('aba', 'lagrangian'):
        got = robot.forward_dynamics(q, qd, tau, method=method,
                                     external_wrenches=ext)
        assert np.allclose(got, qdd, atol=1e-8), method


def test_wrench_can_be_addressed_by_link_name():
    robot = bimanual_arm()
    q, z = robot.random_configuration(RNG), np.zeros(robot.n)
    w = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
    by_name = robot.inverse_dynamics(q, z, z, external_wrenches={'gripper': w})
    by_index = robot.inverse_dynamics(q, z, z, external_wrenches={5: w})
    assert np.allclose(by_name, by_index)
    with pytest.raises(KeyError):
        robot.inverse_dynamics(q, z, z, external_wrenches={'nonexistent': w})


def test_a_vertical_force_at_the_tip_adds_the_expected_static_torque():
    # A physical sanity check with a hand-computable answer.
    # A 2-link planar arm held straight out along x with a downward tip force must show
    # a joint torque equal to force times moment arm.
    robot = planar_chain(2, length=0.5, mass=1.0)
    q, z = np.zeros(2), np.zeros(2)
    force = 10.0
    w = np.array([0.0, -force, 0.0, 0.0, 0.0, 0.0])
    tau_g = robot.gravity_torque(q)
    tau = robot.inverse_dynamics(q, z, z, external_wrenches={1: w})
    # link 1's frame origin sits 0.5 m out; joint 1 sees 0.5 m of arm, joint 2 zero
    assert np.allclose(tau - tau_g, [force * 0.5, 0.0], atol=1e-9)
@pytest.mark.parametrize('order', [3, 5])
def test_polynomial_trajectory_meets_its_boundary_conditions(order):
    q0, qf = np.array([0.0, -1.0, 0.5]), np.array([1.5, 0.3, -0.8])
    T = 2.5
    traj = PolynomialTrajectory(q0, qf, T, order=order)
    q_start, qd_start, qdd_start = traj.evaluate(0.0)
    q_end, qd_end, qdd_end = traj.evaluate(T)
    assert np.allclose(q_start, q0)
    assert np.allclose(q_end, qf)
    assert np.allclose(qd_start, 0.0)
    assert np.allclose(qd_end, 0.0)
    if order == 5:
        # the quintic also zeroes the endpoint accelerations, so torque is
        # continuous at the start of the motion; the cubic does not
        assert np.allclose(qdd_start, 0.0)
        assert np.allclose(qdd_end, 0.0)


def test_quintic_derivatives_are_self_consistent():
    # Check qd and qdd really are the derivatives of q.
    traj = quintic(np.array([0.0, 1.0]), np.array([2.0, -1.0]), 3.0)
    h = 1e-6
    for t in (0.4, 1.5, 2.9):
        _, qd, qdd = traj.evaluate(t)
        q_p, qd_p, _ = traj.evaluate(t + h)
        q_m, qd_m, _ = traj.evaluate(t - h)
        assert np.allclose((q_p - q_m) / (2 * h), qd, atol=1e-6)
        assert np.allclose((qd_p - qd_m) / (2 * h), qdd, atol=1e-6)


def test_trajectory_is_clamped_outside_its_window():
    traj = quintic(np.zeros(2), np.ones(2), 1.0)
    assert np.allclose(traj.evaluate(-5.0)[0], np.zeros(2))
    assert np.allclose(traj.evaluate(99.0)[0], np.ones(2))


def test_multipoint_trajectory_passes_through_every_waypoint():
    wp = np.array([[0.0, 0.0], [1.0, -0.5], [0.5, 1.0], [-1.0, 0.2]])
    traj = MultiPointTrajectory(wp, [1.0, 1.5, 0.8])
    for k, t in enumerate(traj.starts):
        assert np.allclose(traj.evaluate(t)[0], wp[k], atol=1e-9)


def test_trajectory_rejects_bad_input():
    with pytest.raises(ValueError):
        quintic(np.zeros(2), np.ones(2), -1.0)
    with pytest.raises(ValueError):
        PolynomialTrajectory(np.zeros(2), np.ones(3), 1.0)
    with pytest.raises(ValueError):
        PolynomialTrajectory(np.zeros(2), np.ones(2), 1.0, order=4)
    with pytest.raises(ValueError):
        MultiPointTrajectory(np.zeros((3, 2)), [1.0])


def test_torques_along_a_trajectory_are_finite_and_smooth():
    robot = planar_chain(3)
    traj = quintic(np.zeros(3), np.array([1.0, -0.7, 0.4]), 2.0)
    _, qs, qds, qdds = traj.sample(dt=0.01)
    taus = np.array([robot.inverse_dynamics(q, qd, qdd)
                     for q, qd, qdd in zip(qs, qds, qdds, strict=True)])
    assert np.all(np.isfinite(taus))
    # no step changes: successive samples of a smooth torque stay close
    assert np.max(np.abs(np.diff(taus, axis=0))) < 1.0
def test_a_passive_system_conserves_energy():
    # RK4 on a torque-free arm must conserve T + V.
    # Doubles as an integration test of forward dynamics: an error in M, C or G shows
    # up as energy drift even when every algebraic identity still holds, because those
    # identities are all evaluated at a single instant and this one accumulates.
    robot = planar_chain(2, length=0.5, mass=1.0)
    result = simulate(robot, [0.4, -0.9], [0.0, 0.0], duration=2.0, dt=1e-3)
    E = energy_history(robot, result)
    drift = abs(E[-1] - E[0]) / max(1e-12, abs(E[0]))
    assert drift < 1e-6, f'relative energy drift {drift:.2e}'


def test_euler_integration_drifts_much_more_than_rk4():
    # Check the selected integration step against the observed tracking behaviour.
    robot = planar_chain(2)
    args = {'q0': [0.4, -0.9], 'qd0': [0.0, 0.0],
            'duration': 2.0, 'dt': 1e-3}
    drifts = {}
    for integ in ('rk4', 'euler'):
        res = simulate(robot, integrator=integ, **args)
        E = energy_history(robot, res)
        drifts[integ] = abs(E[-1] - E[0]) / abs(E[0])
    assert drifts['rk4'] < drifts['euler'] / 100.0, drifts


def test_a_pendulum_released_from_rest_falls_the_right_way():
    # Cheap physical plausibility check on the sign of gravity.
    robot = planar_chain(1, length=0.5, mass=1.0)
    result = simulate(robot, [0.2], [0.0], duration=0.2, dt=1e-3)
    assert result['q'][-1, 0] < result['q'][0, 0]      # swings down, towards -y


def test_computed_torque_tracks_far_better_than_pd_gravity():
    # Both controllers get identical gains, trajectory and step size, so the difference
    # is attributable to the dynamics model and nothing else. Measured: computed torque
    # reaches ~1e-11 while PD-plus-gravity plateaus near 3e-2, a factor of about 1e9.
    # Gain choice is constrained by the simulation, not the controller. A PD loop
    # leaves a fast closed-loop pole at roughly kd/M_min, so a large
    # kd on a chain with a small minimum inertia demands a tiny step. Here
    # kd = 2*sqrt(kp * M_min) keeps the stiff direction critically damped and
    # the step at 1 ms. See test_pd_gain_limit_is_set_by_the_smallest_inertia.
    robot = planar_chain(2)
    q0 = np.zeros(2)
    traj = quintic(q0, np.array([1.0, -0.6]), 1.0)
    m_min = np.linalg.eigvalsh(robot.mass_matrix(q0)).min()
    kp = 100.0
    kd = 2.0 * np.sqrt(kp * m_min)

    res_ct = simulate(robot, q0, np.zeros(2), duration=1.0, dt=1e-3,
                      control=ComputedTorque(robot, kp, kd, traj))
    res_pd = simulate(robot, q0, np.zeros(2), duration=1.0, dt=1e-3,
                      control=PDGravity(robot, kp, kd, traj))
    err_ct = tracking_error(res_ct, traj)['max_abs']
    err_pd = tracking_error(res_pd, traj)['max_abs']
    assert np.all(np.isfinite(res_pd['q'])), 'PD simulation went unstable'
    assert err_ct < 1e-6, f'computed torque should track tightly, got {err_ct:.2e}'
    assert err_ct < err_pd / 1e3, f'ct {err_ct:.2e} vs pd {err_pd:.2e}'


def test_pd_gain_limit_is_set_by_the_smallest_inertia():
    # Record why fixed-gain PD is awkward on a manipulator.
    # The mass matrix of a serial chain is badly conditioned - the eigenvalue spread is
    # ~40x for a 2-link planar arm and >1300x for a 3-joint spatial one, because a wrist
    # roll about a slender link's own axis has almost no inertia. A single gain pair
    # therefore cannot suit every direction: whatever damps the heavy direction creates a
    # pole at kd/M_min in the light one, and the integrator step must resolve it.
    # Computed torque escapes this entirely, since cancelling M makes the error dynamics
    # identical in every direction regardless of configuration. That is the practical
    # This effect is checked numerically below.
    ratios = {}
    for robot in (planar_chain(2), planar_chain(3), spatial_chain(3)):
        eig = np.linalg.eigvalsh(robot.mass_matrix(np.zeros(robot.n)))
        ratios[robot.name] = eig.max() / eig.min()
    assert ratios['planar2'] > 10.0
    assert ratios['spatial3'] > 100.0

    # and the promise: computed torque's error decay is inertia-independent
    robot = spatial_chain(3)
    traj = quintic(np.zeros(3), np.zeros(3), 1.0)
    ctrl = ComputedTorque.critically_damped(robot, natural_frequency=8.0,
                                            trajectory=traj)
    res = simulate(robot, [0.2, 0.2, 0.2], np.zeros(3), duration=1.0, dt=1e-3,
                   control=ctrl)
    assert np.all(np.isfinite(res['q']))
    # every joint decays at the same designed rate despite a 1300x inertia spread
    final = np.abs(res['q'][-1])
    assert final.max() < 1e-3, final


def test_computed_torque_error_decays_at_the_designed_rate():
    # The closed loop should behave like qdd_e + Kd*qd_e + Kp*e = 0.
    # Started deliberately off the trajectory, the error must decay towards zero at
    # roughly the designed natural frequency, not just stability.
    robot = planar_chain(2)
    traj = quintic(np.zeros(2), np.zeros(2), 2.0)      # hold the origin
    ctrl = ComputedTorque.critically_damped(robot, natural_frequency=10.0,
                                            trajectory=traj)
    res = simulate(robot, [0.3, -0.3], np.zeros(2), duration=2.0, dt=5e-4,
                   control=ctrl)
    err = np.abs(res['q'])
    assert err[-1].max() < 1e-4
    # critically damped at w = 10 means the error is well down by t = 5/w = 0.5 s
    half = int(0.5 / 5e-4)
    assert err[half].max() < 0.3 * err[0].max()


def test_feedforward_pd_also_tracks_well():
    # Feedforward plus PD: nearly computed-torque accuracy at a fraction of the cost.
    # The feedforward term uses only the planned trajectory, so it can be evaluated
    # offline; what runs at servo rate is a fixed-gain PD loop.
    robot = planar_chain(2)
    traj = quintic(np.zeros(2), np.array([0.8, -0.5]), 1.0)
    m_min = np.linalg.eigvalsh(robot.mass_matrix(np.zeros(2))).min()
    kp = 100.0
    res = simulate(robot, np.zeros(2), np.zeros(2), duration=1.0, dt=1e-3,
                   control=FeedforwardPD(robot, kp, 2.0 * np.sqrt(kp * m_min),
                                         traj))
    assert np.all(np.isfinite(res['q']))
    assert tracking_error(res, traj)['max_abs'] < 1e-3


def test_a_deliberately_wrong_model_degrades_tracking():
    # Computed torque is only exact when the model is - quantify the sensitivity.
    # Guards against the test suite flattering the controller by always handing it a
    # perfect model, which is never true on hardware.
    import copy
    robot = planar_chain(2)
    wrong = copy.deepcopy(robot)
    for link in wrong.links:
        link.mass *= 1.5                     # 50% mass error
    traj = quintic(np.zeros(2), np.array([1.0, -0.6]), 1.0)
    m_min = np.linalg.eigvalsh(robot.mass_matrix(np.zeros(2))).min()
    kp = 100.0
    kd = 2.0 * np.sqrt(kp * m_min)
    good = simulate(robot, np.zeros(2), np.zeros(2), duration=1.0, dt=1e-3,
                    control=ComputedTorque(robot, kp, kd, traj))
    bad = simulate(robot, np.zeros(2), np.zeros(2), duration=1.0, dt=1e-3,
                   control=ComputedTorque(robot, kp, kd, traj, model=wrong))
    e_good = tracking_error(good, traj)['max_abs']
    e_bad = tracking_error(bad, traj)['max_abs']
    assert e_good < e_bad / 100.0, (e_good, e_bad)
    # the feedback term still bounds the error - the model error is not catastrophic
    assert e_bad < 0.5, f'a 50% mass error should stay bounded, got {e_bad:.3f}'
    assert np.all(np.isfinite(bad['q']))


def test_simulate_records_consistent_arrays():
    robot = planar_chain(2)
    res = simulate(robot, [0.1, 0.2], [0.0, 0.0], duration=0.05, dt=1e-3)
    n_steps = len(res['t'])
    for key in ('q', 'qd', 'qdd', 'tau'):
        assert res[key].shape == (n_steps, 2)
    assert np.allclose(res['tau'], 0.0)          # passive
    assert np.isclose(res['t'][1] - res['t'][0], 1e-3)


def test_simulate_rejects_bad_arguments():
    robot = planar_chain(2)
    with pytest.raises(ValueError):
        simulate(robot, [0, 0], [0, 0], duration=-1.0)
    with pytest.raises(ValueError):
        simulate(robot, [0, 0], [0, 0], duration=1.0, integrator='verlet')


def test_gain_shapes_are_flexible():
    robot = planar_chain(3)
    traj = quintic(np.zeros(3), np.ones(3), 1.0)
    for kp in (100.0, np.array([100.0, 50.0, 25.0]), np.eye(3) * 100.0):
        ctrl = ComputedTorque(robot, kp, 20.0, traj)
        assert ctrl(0.0, np.zeros(3), np.zeros(3)).shape == (3,)
    with pytest.raises(ValueError):
        ComputedTorque(robot, np.ones((2, 2)), 20.0, traj)
