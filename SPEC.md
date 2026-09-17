# Technical Specification — `robot_dynamics`

*Phase 0 deliverable. Written before any implementation.*

## 1. Objective

A general-purpose rigid-body dynamics library for serial manipulators with an
arbitrary number of degrees of freedom, built on the Lagrangian formulation and
cross-validated against Featherstone's recursive algorithms.

The library computes the equation of motion

$$\tau = M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) - J^\top f_{\text{ext}}$$

in both directions — inverse dynamics for control, forward dynamics for
simulation — and supports trajectory generation and model-based control on top.

## 2. Scope

**In scope.** Fixed-base serial (open) chains. Revolute and prismatic joints.
Arbitrary joint axes and link geometry, so 3D chains and not just planar ones.
Arbitrary $n$: the same code path serves $n = 2$ and $n = 30$. Three independent
dynamics backends (Lagrangian, RNEA, CRBA/ABA). External wrenches. Quintic and
cubic trajectories. PD-plus-gravity and computed-torque control. Fixed-step RK4
simulation.

**Explicitly out of scope for v1**, and stated in the README rather than implied:
floating-base dynamics, closed kinematic loops and branched trees, joint friction
and actuator dynamics, contact and collision resolution, joint-limit enforcement
during simulation, and any form of code generation or GPU acceleration.

## 3. Non-negotiable design constraints

1. **No degree-of-freedom branching.** No `if n == 4:` anywhere in the library.
   Every algorithm is written for arbitrary $n$ and validated at $n = 2,3,4,5,6$
   and beyond by running one identical API over all of them.
2. **Forward dynamics solves a linear system.** $\ddot q$ comes from
   `np.linalg.solve(M, rhs)`, never from forming $M^{-1}$ — explicit inversion is
   both slower and numerically worse, and it discards the symmetry of $M$.
3. **The symbolic and numerical layers are separate.** SymPy derives and audits
   the equations; it is never in the loop at runtime. The numerical layer is pure
   NumPy and has no SymPy import. A "solver" that returns a giant symbolic
   expression is not a solver.
4. **Independent implementations, not shared internals.** The Lagrangian backend
   and the Featherstone backends share only the kinematic model. They agreeing to
   $10^{-10}$ is then real evidence, because a common bug would have to appear
   twice in two different formalisms.

## 4. Public API

```python
robot = Robot(links=[...], joints=[...])          # or a builder from models

robot.forward_kinematics(q)   -> list[T_0i]       # 4x4 each
robot.jacobian(q, link=-1)    -> (6, n)           # [linear; angular]

robot.mass_matrix(q,           method="lagrangian" | "crba")
robot.coriolis(q, qd,          method="lagrangian")           # the MATRIX C
robot.gravity(q,               method="lagrangian" | "rnea")
robot.inverse_dynamics(q, qd, qdd, method="lagrangian" | "rnea",
                       external_wrenches=None)    -> tau
robot.forward_dynamics(q, qd, tau, method="lagrangian" | "aba",
                       external_wrenches=None)    -> qdd
```

Every method takes and returns plain NumPy arrays. `method=` selects the backend
and defaults to the fastest correct one; the alternatives exist so the test suite
can cross-check them.

## 5. Mathematical approach

**Mass matrix** from the kinetic energy, assembled per link from its COM Jacobian:

$$M(q)=\sum_i \left[ m_i\,J_{v,c_i}^\top J_{v,c_i} + J_{\omega,i}^\top R_i I_i R_i^\top J_{\omega,i} \right]$$

This is exact and closed-form — no differentiation is involved.

**Gravity** from the potential $V=-\sum_i m_i\,g^\top p_{c_i}$, giving the exact
analytic result $G_j=-\sum_i m_i\,g^\top J_{v,c_i}[:,j]$.

**Coriolis matrix** from Christoffel symbols of the first kind:

$$C_{ij}=\sum_k \frac12\left(\frac{\partial M_{ij}}{\partial q_k}+\frac{\partial M_{ik}}{\partial q_j}-\frac{\partial M_{jk}}{\partial q_i}\right)\dot q_k$$

$\partial M/\partial q$ is obtained by **complex-step differentiation**, which is
accurate to machine precision rather than the $\sim\!10^{-6}$ of a finite
difference, because the subtractive cancellation that limits finite differences
never occurs. The test suite verifies it against both a central difference and the
symbolic derivative.

This choice matters: $C$ built from an inaccurate $\partial M/\partial q$ silently
degrades every downstream result, and a $10^{-6}$ error is large enough to be
mistaken for a real modelling discrepancy when comparing backends.

## 6. Validation strategy

Correctness is asserted by properties and by agreement between independent
implementations, not by inspecting plots:

| Class | Check |
|---|---|
| Structural | $M=M^\top$; $x^\top M x>0$; $\dot M - 2C$ skew-symmetric |
| Isolation | $\dot q=\ddot q=0 \Rightarrow \tau=G$; $g=0 \Rightarrow G=0$ |
| Round trip | $\ddot q \to \tau \to \ddot q$ recovers the input |
| Cross-backend | Lagrange $\equiv$ RNEA $\equiv$ CRBA $\equiv$ ABA |
| Analytic | closed-form 2-DOF planar $M$, $C$, $G$ from a textbook derivation |
| Symbolic | SymPy Euler–Lagrange derivation matches the numerical layer |
| Energy | a passive system conserves $T+V$ under RK4 |

The analytic and symbolic checks are what make the rest trustworthy: property
tests confirm self-consistency, but only an independent derivation confirms the
model is the *intended* one.

## 7. Deliverables

Installable package with type hints and docstrings; test suite covering the
21 items of Phase 19; runnable examples that produce the README's plots;
benchmarks comparing backend cost against $n$; `docs/theory.md` deriving the
equations and `docs/validation.md` reporting measured errors; CI running the
suite on every push.

## 8. Success criteria

The library is done when a randomly generated 6-DOF spatial chain — random axes,
random link frames, random inertias — satisfies every property above, and all
three backends agree to $10^{-10}$, using the same code path that handles a 2-DOF
planar arm.
