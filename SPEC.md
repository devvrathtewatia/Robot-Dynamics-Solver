# Technical Specification — `robot_dynamics`

*Phase 0 deliverable. Written before any implementation.*

## 1. Objective

A general-purpose rigid-body dynamics library for serial manipulators with an
arbitrary number of degrees of freedom, built on the Lagrangian formulation and
cross-validated against Featherstone's recursive algorithms.

The library computes the equation of motion

\(\tau = M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) - J^\top f_{\text{ext}}\)

in both directions — inverse dynamics for control, forward dynamics for
simulation — and supports trajectory generation and model-based control on top.

## 2. Scope

**In scope.** Fixed-base serial (open) chains. Revolute and prismatic joints.
Arbitrary joint axes and link geometry, so 3D chains and not just planar ones.
Arbitrary $n$: the same code path serves $n = 2$ and $n = 30$.

The dynamics layer contains a Lagrangian formulation together with independent
Featherstone algorithms: RNEA for inverse dynamics, CRBA for mass-matrix
construction, and ABA for forward dynamics.

External wrenches. Quintic and cubic trajectories. PD-plus-gravity and
computed-torque control. Fixed-step RK4 simulation.

**Explicitly out of scope for v1**, and stated in the README rather than implied:
floating-base dynamics, closed kinematic loops and branched trees, joint friction
and actuator dynamics, contact and collision resolution, joint-limit enforcement
during simulation, and any form of code generation or GPU acceleration.

## 3. Non-negotiable design constraints

1. **No degree-of-freedom branching.** No `if n == 4:` anywhere in the library.
   Every algorithm is written for arbitrary $n$ and validated at $n = 2,3,4,5,6$
   and beyond by running one identical API over all of them.

2. **Forward dynamics never forms an explicit inverse.** The Lagrangian
   forward-dynamics backend solves

   \(M(q)\ddot q = \tau - C(q,\dot q)\dot q - G(q) + J^\top f_{\text{ext}}\)

   using a Cholesky factorisation of $M$ rather than forming $M^{-1}$. The
   Featherstone ABA backend computes $\ddot q$ recursively without forming $M$.

3. **The symbolic and numerical layers are separate.** SymPy derives and audits
   the equations; it is never in the loop at runtime. The numerical layer is pure
   NumPy and has no SymPy import. A "solver" that returns a giant symbolic
   expression is not a solver.

4. **Independent dynamics formulations.** The Lagrangian formulation and the
   Featherstone algorithms use the same robot and kinematic representation, but
   compute the dynamics through different formulations. This allows the test suite
   to cross-check independent implementations rather than merely exercising
   multiple names for the same calculation.

## 4. Public API

```python
robot = Robot(links=[...], joints=[...])          # or a builder from models

robot.forward_kinematics(q)   -> list[T_0i]       # 4x4 each
robot.jacobian(q, link=-1)    -> (6, n)           # [linear; angular]

robot.mass_matrix(
    q,
    method="crba" | "lagrangian" | "rnea"
) -> M

robot.coriolis(
    q,
    qd,
    method="lagrangian"
) -> C                                               # the MATRIX C

robot.coriolis_vector(
    q,
    qd,
    method="rnea" | "lagrangian"
) -> C qd

robot.gravity_torque(
    q,
    method="lagrangian" | "rnea"
) -> G

robot.inverse_dynamics(
    q,
    qd,
    qdd,
    method="rnea" | "lagrangian",
    external_wrenches=None
) -> tau

robot.forward_dynamics(
    q,
    qd,
    tau,
    method="aba" | "lagrangian",
    external_wrenches=None
) -> qdd
```

The default methods are the fastest intended implementations for each operation:
CRBA for the mass matrix, Lagrangian for the explicit Coriolis matrix,
RNEA for inverse dynamics and the Coriolis-vector calculation, and ABA for
forward dynamics.

Every method takes array-like numerical inputs and returns plain NumPy arrays.
The `method=` argument selects the available formulation so that the test suite
can cross-check corresponding implementations.

## 5. Mathematical approach

**Mass matrix** from the kinetic energy, assembled per link from its COM Jacobian:

\(M(q)=\sum_i \left[ m_i\,J_{v,c_i}^\top J_{v,c_i} + J_{\omega,i}^\top R_i I_i R_i^\top J_{\omega,i} \right]\)

This is exact and closed-form — no differentiation is involved.

**Gravity** from the potential $V=-\sum_i m_i,g^\top p_{c_i}$, giving the exact
analytic result

\(G_j=-\sum_i m_i\,g^\top J_{v,c_i}[:,j]\)

**Coriolis matrix** from Christoffel symbols of the first kind:

\(C_{ij}=\sum_k \frac12\left(\frac{\partial M_{ij}}{\partial q_k}+\frac{\partial M_{ik}}{\partial q_j}-\frac{\partial M_{jk}}{\partial q_i}\right)\dot q_k\)

$\partial M/\partial q$ is obtained by **complex-step differentiation**, which is
accurate to machine precision rather than the $\sim!10^{-6}$ of a finite
difference, because the subtractive cancellation that limits finite differences
never occurs. The test suite verifies it against both a central difference and the
symbolic derivative.

This choice matters: $C$ built from an inaccurate $\partial M/\partial q$ silently
degrades every downstream result, and a $10^{-6}$ error is large enough to be
mistaken for a real modelling discrepancy when comparing backends.

## 6. Validation strategy

Correctness is asserted by properties and by agreement between independent
implementations, not by inspecting plots:

| Class         | Check                                                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Structural    | $M=M^\top$; $x^\top M x>0$; $\dot M - 2C$ skew-symmetric                                                                          |
| Isolation     | $\dot q=\ddot q=0 \Rightarrow \tau=G$; $g=0 \Rightarrow G=0$                                                                      |
| Round trip    | $\ddot q \to \tau \to \ddot q$ recovers the input                                                                                 |
| Cross-backend | Lagrange/RNEA inverse-dynamics results agree; Lagrange/CRBA/RNEA mass matrices agree; Lagrange/ABA forward-dynamics results agree |
| Analytic      | closed-form 2-DOF planar $M$, $C$, $G$ from a textbook derivation                                                                 |
| Symbolic      | SymPy Euler–Lagrange derivation matches the numerical layer                                                                       |
| Energy        | a passive system conserves $T+V$ under RK4                                                                                        |

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
random link frames, random inertias — satisfies every property a

