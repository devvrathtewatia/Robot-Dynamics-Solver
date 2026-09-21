"""Symbolic layer for deriving equations of motion using SymPy."""

from __future__ import annotations

from typing import Any
from ..models.robot import Robot

try:
    import sympy as sp
    HAVE_SYMPY = True
except ImportError:                                    # pragma: no cover
    sp = None
    HAVE_SYMPY = False


def _require_sympy():
    if not HAVE_SYMPY:
        raise ImportError(
            'the symbolic layer needs SymPy: pip install "robot_dynamics[symbolic]" '
            '(the numerical layer does not need it)')


def _sym_skew(v: Any):
    return sp.Matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


def _sym_rot_axis(axis: Any, theta: Any):
    # Rodrigues' rotation formula symbolically.
    a = sp.Matrix([sp.nsimplify(float(x), rational=True) for x in axis])
    K = _sym_skew(a)
    return sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * (K @ K)


def _sym_joint_transform(joint: Any, qi: Any):
    T = sp.eye(4)
    if joint.joint_type == 'revolute':
        T[:3, :3] = _sym_rot_axis(joint.axis, qi)
    else:
        a = sp.Matrix([sp.nsimplify(float(x), rational=True)
                       for x in joint.axis])
        T[:3, 3] = a * qi
    return T


def _sym_matrix(M: Any):
    # Convert array-like data to a SymPy matrix without importing NumPy.
    rows = M.tolist() if hasattr(M, 'tolist') else M
    if not rows:
        return sp.Matrix(rows)
    if not hasattr(rows[0], '__iter__') or isinstance(rows[0], (str, bytes)):
        rows = [rows]
    return sp.Matrix([[sp.nsimplify(float(x), rational=True) for x in row]
                      for row in rows])


def symbolic_model(robot: Robot, simplify: bool = False, trig_simplify: bool = False):
    # Derive M(q), C(q, qd), and G(q) symbolically using the Euler-Lagrange approach.
    _require_sympy()
    n = robot.n
    q = sp.symbols(f'q1:{n + 1}', real=True)
    qd = sp.symbols(f'qd1:{n + 1}', real=True)
    qdd = sp.symbols(f'qdd1:{n + 1}', real=True)
    g = sp.Matrix([sp.nsimplify(float(x), rational=True)
                   for x in robot.gravity])

    # 1. Forward kinematics for every link frame
    T_cum = sp.eye(4)
    frames = []
    for i, joint in enumerate(robot.joints):
        T_cum = (T_cum @ _sym_matrix(joint.origin)
                 @ _sym_joint_transform(joint, q[i]))
        if trig_simplify:
            T_cum = sp.trigsimp(T_cum)
        frames.append(T_cum)

    # 2-3. Kinetic and potential energies
    Tkin = sp.S.Zero
    Vpot = sp.S.Zero
    for link, F in zip(robot.links, frames, strict=True):
        R = F[:3, :3]
        p = F[:3, 3]
        com = link.com.tolist() if hasattr(link.com, 'tolist') else link.com
        c_world = R @ sp.Matrix([sp.nsimplify(float(x), rational=True)
                                 for x in com]) + p

        # Compute COM velocity and angular velocity
        v = sp.zeros(3, 1)
        for j in range(n):
            v += sp.diff(c_world, q[j]) * qd[j]

        Rdot = sp.zeros(3, 3)
        for j in range(n):
            Rdot += sp.diff(R, q[j]) * qd[j]
        W = Rdot @ R.T
        omega = sp.Matrix([W[2, 1], W[0, 2], W[1, 0]])

        I_world = R @ _sym_matrix(link.inertia) @ R.T
        m = sp.nsimplify(float(link.mass), rational=True)
        Tkin += (m * (v.T @ v)[0, 0] + (omega.T @ I_world @ omega)[0, 0]) / 2
        Vpot += -m * (g.T @ c_world)[0, 0]

    if trig_simplify:
        Tkin, Vpot = sp.trigsimp(Tkin), sp.trigsimp(Vpot)

    # 4. Extract M from the Hessian of T, G from the gradient of V, and C via Christoffel symbols
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


def euler_lagrange(robot: Robot, simplify: bool = False):
    # Derive torques using the literal Euler-Lagrange equations.
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


def lambdify_model(robot: Robot, simplify: bool = False):
    # Compile the symbolic model into plain NumPy callables.
    _require_sympy()
    model = symbolic_model(robot, simplify=simplify)
    q, qd = model['q'], model['qd']
    return {
        'mass_matrix': sp.lambdify((q,), model['M'], 'numpy'),
        'coriolis': sp.lambdify((q, qd), model['C'], 'numpy'),
        'gravity': sp.lambdify((q,), model['G'], 'numpy'),
        'symbolic': model,
    }


def print_equations(robot: Robot, simplify: bool = True):
    # Pretty-print the derived equations for inspection.
    _require_sympy()
    model = symbolic_model(robot, simplify=simplify)
    print(f'=== {robot.name}: n = {robot.n} ===')
    for label in ('M', 'C', 'G'):
        print(f'\n{label}(q) =')
        sp.pprint(model[label])
