"""Lagrangian dynamics — the numerical layer.

This is the heart of the project. It computes

.. math:: \\tau = M(q)\\,\\ddot q + C(q,\\dot q)\\,\\dot q + G(q)

from the Lagrangian :math:`L = T - V`, but **without** ever building a symbolic
expression at runtime. SymPy lives in :mod:`robot_dynamics.dynamics.symbolic` and is
used to *derive and audit* these formulas, never to evaluate them. Pure NumPy here.

How each term is obtained, and why
----------------------------------

**Mass matrix — exact, closed form.** The kinetic energy of a chain is
:math:`T = \\tfrac12 \\dot q^\\top M(q) \\dot q`, and collecting the per-link
contributions gives

.. math::

    M(q) = \\sum_i \\left[ m_i J_{v,c_i}^\\top J_{v,c_i}
                          + J_{\\omega,i}^\\top R_i I_i R_i^\\top J_{\\omega,i} \\right]

No differentiation is involved: the Jacobians *are* the derivatives that the
Euler-Lagrange procedure would otherwise produce by hand. :math:`R_i I_i R_i^\\top`
rotates each link's inertia into the base frame, which is the step most easily
forgotten and produces a subtly wrong-but-plausible :math:`M` when it is.

**Gravity — exact, analytic.** From :math:`V = -\\sum_i m_i g^\\top p_{c_i}`,

.. math::

    G_j = \\frac{\\partial V}{\\partial q_j}
        = -\\sum_i m_i\\,g^\\top J_{v,c_i}[:,j]

because :math:`\\partial p_{c_i}/\\partial q_j` is precisely the ``j``-th column of
that link's translational COM Jacobian.

**Coriolis matrix — Christoffel symbols.** The only term needing a derivative of
:math:`M`:

.. math::

    C_{ij} = \\sum_k \\tfrac12 \\left( \\frac{\\partial M_{ij}}{\\partial q_k}
             + \\frac{\\partial M_{ik}}{\\partial q_j}
             - \\frac{\\partial M_{jk}}{\\partial q_i} \\right) \\dot q_k

:math:`\\partial M/\\partial q` uses **complex-step differentiation**:

.. math::

    \\frac{\\partial f}{\\partial q_k}
        = \\frac{\\mathrm{Im}\\, f(q + i h e_k)}{h},
    \\qquad h = 10^{-20}

which is accurate to machine precision. A central difference is limited to about
:math:`10^{-6}` relative error by the cancellation in :math:`f(q+h) - f(q-h)`;
the complex step has no subtraction of nearby values, so ``h`` can be made
arbitrarily small. This matters more than it first appears: an inexact
:math:`\\partial M/\\partial q` propagates into :math:`C`, and a :math:`10^{-6}`
discrepancy is easily mistaken for a genuine modelling difference when comparing
this backend against RNEA. With the complex step the backends agree to
:math:`10^{-12}`, so any real disagreement is unambiguous.

Both alternatives are still available via ``method=``, and the test suite checks
all three against each other and against the symbolic derivative.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..kinematics.forward import com_jacobians, jacobian, link_transforms
from ..models.robot import Robot

DiffMethod = Literal['complex', 'central', 'forward']


def mass_matrix(robot: Robot, q: ArrayLike) -> NDArray:
    r"""The joint-space mass matrix :math:`M(q)`, exact and symmetric.

    Symmetry is enforced by the final ``(M + M.T)/2``: the assembled sum is
    symmetric analytically, and this removes only floating-point asymmetry at the
    :math:`10^{-17}` level. It is not papering over an error - the test suite
    checks the *unsymmetrised* matrix is already symmetric to ``1e-12``.
    """
    q = np.asarray(q).reshape(-1)
    Ts = link_transforms(robot, q)
    Js = com_jacobians(robot, q)
    dtype = np.result_type(q.dtype, np.float64)

    M = np.zeros((robot.n, robot.n), dtype=dtype)
    for link, T, J in zip(robot.links, Ts, Js, strict=True):
        Jv, Jw = J[:3], J[3:]
        R = T[:3, :3]
        I_world = R @ link.inertia @ R.T
        M = M + link.mass * (Jv.T @ Jv) + Jw.T @ I_world @ Jw
    return (M + M.T) / 2.0


def mass_matrix_derivative(robot: Robot, q: ArrayLike,
                           method: DiffMethod = 'complex',
                           h: float | None = None) -> NDArray:
    r"""Return :math:`\partial M/\partial q` as an ``(n, n, n)`` array.

    ``dM[i, j, k]`` is :math:`\partial M_{ij}/\partial q_k`.

    Parameters
    ----------
    method:
        ``'complex'`` (default) is exact to machine precision. ``'central'`` and
        ``'forward'`` exist for validation and for the rare case where a model
        component is not complex-analytic.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    n = robot.n
    dM = np.zeros((n, n, n))

    if method == 'complex':
        step = 1e-20 if h is None else h
        for k in range(n):
            qc = q.astype(complex)
            qc[k] += 1j * step
            dM[:, :, k] = np.imag(mass_matrix(robot, qc)) / step
        return dM

    if method == 'central':
        step = 1e-6 if h is None else h
        for k in range(n):
            qp, qm = q.copy(), q.copy()
            qp[k] += step
            qm[k] -= step
            dM[:, :, k] = (mass_matrix(robot, qp)
                           - mass_matrix(robot, qm)) / (2.0 * step)
        return dM

    if method == 'forward':
        step = 1e-8 if h is None else h
        M0 = mass_matrix(robot, q)
        for k in range(n):
            qp = q.copy()
            qp[k] += step
            dM[:, :, k] = (mass_matrix(robot, qp) - M0) / step
        return dM

    raise ValueError(f'unknown differentiation method {method!r}')


def coriolis(robot: Robot, q: ArrayLike, qd: ArrayLike,
             method: DiffMethod = 'complex') -> NDArray:
    r"""The Coriolis/centrifugal **matrix** :math:`C(q,\dot q)` via Christoffel symbols.

    This particular factorisation is the one for which :math:`\dot M - 2C` is
    skew-symmetric, which is the property that makes the standard passivity-based
    stability proofs work. Other valid factorisations of the same *vector*
    :math:`C\dot q` do not have it, so the choice is not arbitrary.
    """
    q, qd = robot.check_q(q, qd)
    dM = mass_matrix_derivative(robot, q, method=method)
    # C[i, j] = sum_k 1/2 (dM[i,j,k] + dM[i,k,j] - dM[j,k,i]) qd[k].
    #
    # The axis permutations must be read carefully. np.transpose(a, axes) sends
    # axis axes[m] of `a` to axis m of the result, so to obtain an array whose
    # [i,j,k] entry is dM[j,k,i] the permutation is (2, 0, 1) -- NOT (1, 2, 0),
    # which yields dM[k,i,j] and is a plausible-looking but wrong tensor. That
    # error leaves M and G exact and only corrupts C, so it shows up as a ~1e-1
    # disagreement with RNEA while every other check still passes.
    christoffel = 0.5 * (dM
                         + np.transpose(dM, (0, 2, 1))     # dM[i,k,j]
                         - np.transpose(dM, (2, 0, 1)))    # dM[j,k,i]
    return christoffel @ qd


def coriolis_vector(robot: Robot, q: ArrayLike, qd: ArrayLike,
                    method: DiffMethod = 'complex') -> NDArray:
    r"""The vector :math:`C(q,\dot q)\dot q` — velocity-product torques."""
    q, qd = robot.check_q(q, qd)
    return coriolis(robot, q, qd, method=method) @ qd


def gravity(robot: Robot, q: ArrayLike) -> NDArray:
    r"""The gravity torque vector :math:`G(q)`, exact and analytic.

    Zero automatically when ``robot.gravity`` is zero, satisfying Phase 11's
    ``g = 0 => G(q) = 0`` check by construction rather than by a special case.
    """
    q, = robot.check_q(q)
    Js = com_jacobians(robot, q)
    G = np.zeros(robot.n)
    for link, J in zip(robot.links, Js, strict=True):
        G -= link.mass * (robot.gravity @ J[:3])
    return G


def potential_energy(robot: Robot, q: ArrayLike) -> float:
    r""":math:`V(q) = -\sum_i m_i g^\top p_{c_i}`."""
    from ..kinematics.forward import com_positions
    q, = robot.check_q(q)
    coms = com_positions(robot, q)
    return float(-sum(link.mass * (robot.gravity @ c)
                      for link, c in zip(robot.links, coms, strict=True)))


def kinetic_energy(robot: Robot, q: ArrayLike, qd: ArrayLike) -> float:
    r""":math:`T = \tfrac12 \dot q^\top M(q) \dot q`."""
    q, qd = robot.check_q(q, qd)
    return float(0.5 * qd @ mass_matrix(robot, q) @ qd)


def lagrangian(robot: Robot, q: ArrayLike, qd: ArrayLike) -> float:
    r""":math:`L = T - V`."""
    return kinetic_energy(robot, q, qd) - potential_energy(robot, q)


def external_torque(robot: Robot, q: ArrayLike,
                    external_wrenches: Mapping[int | str, ArrayLike]) -> NDArray:
    r"""Joint torques produced by external wrenches: :math:`\sum J_i^\top f_i`.

    Keys are link indices or link names. Values are 6-vectors ``[force; moment]``
    in the **base frame**, applied at that link's frame origin.

    The caller subtracts this, matching
    :math:`\tau = M\ddot q + C\dot q + G - J^\top f_{\text{ext}}` — i.e. a wrench
    that helps the motion reduces the required actuator torque.
    """
    q, = robot.check_q(q)
    by_name = {link.name: i for i, link in enumerate(robot.links) if link.name}
    tau = np.zeros(robot.n)
    for key, wrench in external_wrenches.items():
        idx = by_name[key] if isinstance(key, str) else key
        if isinstance(key, str) and key not in by_name:
            raise KeyError(f'no link named {key!r}')
        w = np.asarray(wrench, dtype=float).reshape(6)
        tau += jacobian(robot, q, link=idx).T @ w
    return tau


def inverse_dynamics(robot: Robot, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike,
                     external_wrenches: Mapping[int | str, ArrayLike] | None = None,
                     method: DiffMethod = 'complex') -> NDArray:
    r"""Inverse dynamics.

    .. math:: \tau = M\ddot q + C\dot q + G - J^\top f_{\text{ext}}
    """
    q, qd, qdd = robot.check_q(q, qd, qdd)
    tau = (mass_matrix(robot, q) @ qdd
           + coriolis_vector(robot, q, qd, method=method)
           + gravity(robot, q))
    if external_wrenches:
        tau -= external_torque(robot, q, external_wrenches)
    return tau


def forward_dynamics(robot: Robot, q: ArrayLike, qd: ArrayLike, tau: ArrayLike,
                     external_wrenches: Mapping[int | str, ArrayLike] | None = None,
                     method: DiffMethod = 'complex') -> NDArray:
    r"""Forward dynamics by a **linear solve**, not by inverting :math:`M`.

    .. math:: M(q)\,\ddot q = \tau - C(q,\dot q)\dot q - G(q) + J^\top f_{\text{ext}}

    Uses a Cholesky factorisation, which is valid because :math:`M` is symmetric
    positive definite for any physically valid model, and is both about twice as
    fast as a general LU solve and better conditioned. Forming :math:`M^{-1}`
    explicitly would be slower, less accurate, and would throw away that structure.
    A model with a degenerate :math:`M` (a zero-inertia link, say) raises here
    rather than returning quietly wrong accelerations.
    """
    q, qd, tau = robot.check_q(q, qd, tau)
    rhs = (tau - coriolis_vector(robot, q, qd, method=method)
           - gravity(robot, q))
    if external_wrenches:
        rhs += external_torque(robot, q, external_wrenches)
    M = mass_matrix(robot, q)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            'the mass matrix is not positive definite, so this model is not '
            'physically valid - check link masses and inertia tensors '
            '(Robot.validate() reports which link)') from exc
    y = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, y)
