"""Symbolic layer — SymPy derivation of the equations of motion.

**This module is never imported by the numerical layer.** SymPy appears nowhere in
:mod:`robot_dynamics.dynamics.lagrangian`, :mod:`~robot_dynamics.dynamics.rnea` or
:mod:`~robot_dynamics.dynamics.crba`, and nothing here runs inside a simulation loop.
That separation is deliberate and is the difference between a dynamics *library* and
a notebook that prints a large expression: a solver whose runtime path is symbolic is
orders of magnitude too slow to integrate and impossible to profile.

What the symbolic layer is for
------------------------------
1. **Independent validation.** :func:`symbolic_model` derives :math:`M`, :math:`C`
   and :math:`G` by literal differentiation of :math:`L = T - V`. Comparing against
   the numerical layer checks the closed-form Jacobian assembly *and* the
   complex-step :math:`\\partial M/\\partial q` against exact symbolic derivatives.
2. **Inspection.** For a 2- or 3-DOF arm the printed equations can be read and
   checked against a textbook, which is how a modelling error is distinguished from
   an arithmetic one.
3. **Optional code generation.** :func:`lambdify_model` compiles the derived
   expressions into plain NumPy callables — the symbolic work happens once, offline,
   and what runs afterwards has no SymPy in it.

Cost: symbolic derivation grows steeply with DOF (a full 6-DOF spatial chain
generates very large expressions and can take minutes). Use it on small models to
verify, and trust the recursive backends at scale. That is stated as a limitation
rather than hidden.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.robot import Robot

try:
    import sympy as sp
    HAVE_SYMPY = True
except ImportError:                                    # pragma: no cover
    sp = None                                          # type: ignore[assignment]
    HAVE_SYMPY = False


def _require_sympy() -> None:
    if not HAVE_SYMPY:
        raise ImportError(
            'the symbolic layer needs SymPy: pip install "robot_dynamics[symbolic]" '
            '(the numerical layer does not need it)')


def _sym_skew(v: Any) -> Any:
    return sp.Matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


def _sym_rot_axis(axis: np.ndarray, theta: Any) -> Any:
    """Rodrigues' rotation formula, symbolically."""
    a = sp.Matrix([sp.nsimplify(float(x), rational=True) for x in axis])
    K = _sym_skew(a)
    return sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * (K @ K)


def _sym_joint_transform(joint: Any, qi: Any) -> Any:
    T = sp.eye(4)
    if joint.joint_type == 'revolute':
        T[:3, :3] = _sym_rot_axis(joint.axis, qi)
    else:
        a = sp.Matrix([sp.nsimplify(float(x), rational=True)
                       for x in joint.axis])
        T[:3, 3] = a * qi
    return T


def _sym_matrix(M: np.ndarray) -> Any:
    return sp.Matrix([[sp.nsimplify(float(x), rational=True) for x in row]
                      for row in np.atleast_2d(M)])


def symbolic_model(robot: Robot, simplify: bool = False,
                   trig_simplify: bool = False) -> dict[str, Any]:
    r"""Derive :math:`M(q)`, :math:`C(q,\dot q)` and :math:`G(q)` symbolically.

    Follows the Euler-Lagrange procedure literally:

    1. forward kinematics to every link frame;
    2. link COM velocities and angular velocities in terms of :math:`\dot q`;
    3. :math:`T=\sum_i \tfrac12 m_i v_{c_i}^\top v_{c_i}
       + \tfrac12\omega_i^\top I_i^0 \omega_i` and
       :math:`V=-\sum_i m_i g^\top p_{c_i}`;
    4. :math:`M_{ij}=\partial^2 T/\partial\dot q_i \partial\dot q_j`,
       :math:`G_i=\partial V/\partial q_i`, and :math:`C` from Christoffel symbols.

    Step 4's identity for :math:`M` holds because :math:`T` is exactly quadratic in
    :math:`\dot q` for a fixed-base chain — a second derivative extracts the
    coefficient matrix with no assumption about which terms survive.

    Parameters
    ----------
    simplify:
        Apply :func:`sympy.simplify` to every entry. Correct but slow; on a 6-DOF
        chain it can take minutes and is rarely worth it.
    trig_simplify:
        Cheaper middle ground using :func:`sympy.trigsimp`.

    Returns
    -------
    dict
        ``q``, ``qd``, ``qdd`` (symbol tuples), ``M``, ``C``, ``G``, ``T``, ``V``,
        ``L``, and ``tau`` :math:`= M\ddot q + C\dot q + G`.
    """
    _require_sympy()
    n = robot.n
    q = sp.symbols(f'q1:{n + 1}', real=True)
    qd = sp.symbols(f'qd1:{n + 1}', real=True)
    qdd = sp.symbols(f'qdd1:{n + 1}', real=True)
    g = sp.Matrix([sp.nsimplify(float(x), rational=True)
                   for x in robot.gravity])

    # 1. forward kinematics
    T_cum = sp.eye(4)
    frames = []
    for i, joint in enumerate(robot.joints):
        T_cum = (T_cum @ _sym_matrix(joint.origin)
                 @ _sym_joint_transform(joint, q[i]))
        if trig_simplify:
            T_cum = sp.trigsimp(T_cum)
        frames.append(T_cum)

    # 2-3. energies
    Tkin = sp.S.Zero
    Vpot = sp.S.Zero
    for link, F in zip(robot.links, frames, strict=True):
        R = F[:3, :3]
        p = F[:3, 3]
        c_world = R @ _sym_matrix(link.com.reshape(3, 1)) + p

        # COM velocity by the chain rule, and angular velocity from R_dot R^T
        v = sp.zeros(3, 1)
        for j in range(n):
            v += sp.diff(c_world, q[j]) * qd[j]

        Rdot = sp.zeros(3, 3)
        for j in range(n):
            Rdot += sp.diff(R, q[j]) * qd[j]
        W = Rdot @ R.T                      # skew-symmetric angular velocity
        omega = sp.Matrix([W[2, 1], W[0, 2], W[1, 0]])

        I_world = R @ _sym_matrix(link.inertia) @ R.T
        m = sp.nsimplify(float(link.mass), rational=True)
        Tkin += (m * (v.T @ v)[0, 0] + (omega.T @ I_world @ omega)[0, 0]) / 2
        Vpot += -m * (g.T @ c_world)[0, 0]

    if trig_simplify:
        Tkin, Vpot = sp.trigsimp(Tkin), sp.trigsimp(Vpot)

    # 4. M from the Hessian of T in qd; G from the gradient of V
    M = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            entry = sp.diff(Tkin, qd[i], qd[j])
            M[i, j] = M[j, i] = entry
    G = sp.Matrix([sp.diff(Vpot, q[i]) for i in range(n)])

    C = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            expr = sp.S.Zero
            for k in range(n):
                expr += (sp.diff(M[i, j], q[k])
                         + sp.diff(M[i, k], q[j])
                         - sp.diff(M[j, k], q[i])) * qd[k] / 2
            C[i, j] = expr

    if simplify:
        M, C, G = sp.simplify(M), sp.simplify(C), sp.simplify(G)
    elif trig_simplify:
        M, C, G = sp.trigsimp(M), sp.trigsimp(C), sp.trigsimp(G)

    tau = M @ sp.Matrix(qdd) + C @ sp.Matrix(qd) + G
    return {'q': q, 'qd': qd, 'qdd': qdd, 'M': M, 'C': C, 'G': G,
            'T': Tkin, 'V': Vpot, 'L': Tkin - Vpot, 'tau': tau}


def euler_lagrange(robot: Robot, simplify: bool = False) -> dict[str, Any]:
    r"""Derive :math:`\tau` by the **literal** Euler-Lagrange equation.

    .. math:: \tau_i = \frac{d}{dt}\frac{\partial L}{\partial \dot q_i}
                       - \frac{\partial L}{\partial q_i}

    The total time derivative is expanded by the chain rule,
    :math:`\frac{d}{dt} = \sum_k \dot q_k \partial_{q_k}
    + \ddot q_k \partial_{\dot q_k}`.

    Slower and less structured than :func:`symbolic_model`, and kept because it is
    the definition. Agreement between the two confirms that the
    :math:`M`/:math:`C`/:math:`G` split is a faithful rearrangement of
    :math:`L = T - V` and not a formula asserted without proof.
    """
    _require_sympy()
    model = symbolic_model(robot, simplify=False)
    q, qd, qdd, L = model['q'], model['qd'], model['qdd'], model['L']
    n = robot.n

    tau = []
    for i in range(n):
        dL_dqd = sp.diff(L, qd[i])
        total = sp.S.Zero
        for k in range(n):
            total += sp.diff(dL_dqd, q[k]) * qd[k]
            total += sp.diff(dL_dqd, qd[k]) * qdd[k]
        tau.append(sp.simplify(total - sp.diff(L, q[i])) if simplify
                   else total - sp.diff(L, q[i]))
    model['tau_euler_lagrange'] = sp.Matrix(tau)
    return model


def lambdify_model(robot: Robot, simplify: bool = False) -> dict[str, Any]:
    """Compile the symbolic model into plain NumPy callables.

    Returns ``mass_matrix(q)``, ``coriolis(q, qd)``, ``gravity(q)`` — closures with
    no SymPy inside them. This is the intended way to *use* the symbolic layer: derive
    once, then evaluate numerically forever.

    For a small arm the compiled functions can beat the general recursive algorithms,
    since every configuration-independent simplification has already been applied.
    Expression size grows steeply with DOF, though, so this is a small-model
    optimisation and not a replacement for RNEA.
    """
    _require_sympy()
    model = symbolic_model(robot, simplify=simplify)
    q, qd = model['q'], model['qd']
    return {
        'mass_matrix': sp.lambdify((q,), model['M'], 'numpy'),
        'coriolis': sp.lambdify((q, qd), model['C'], 'numpy'),
        'gravity': sp.lambdify((q,), model['G'], 'numpy'),
        'symbolic': model,
    }


def print_equations(robot: Robot, simplify: bool = True) -> None:
    """Pretty-print the derived equations. Intended for a 2- or 3-DOF arm."""
    _require_sympy()
    model = symbolic_model(robot, simplify=simplify)
    print(f'=== {robot.name}: n = {robot.n} ===')
    for label in ('M', 'C', 'G'):
        print(f'\n{label}(q) =')
        sp.pprint(model[label])
