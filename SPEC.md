# Technical Specification — `robot_dynamics`

Initial project specification. The document was updated during implementation to
keep the stated scope and acceptance criteria aligned with the code.

## 1. Objective

A general-purpose rigid-body dynamics library for serial manipulators with an
arbitrary number of degrees of freedom. The library includes a Lagrangian
formulation and Featherstone's recursive algorithms, with the two routes used to
cross-check the dynamics implementation.

The equation of motion is

$$
\tau = M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) - J^\top f_{\text{ext}}.
$$

The library supports inverse dynamics for control, forward dynamics for
simulation, trajectory generation, and model-based control.

## 2. Scope

**In scope.** Fixed-base serial (open) chains with revolute and prismatic joints.
Joint axes and link frames are arbitrary, so the implementation covers 3D chains
as well as planar examples. The code is N-DOF: the same generic implementation is
used for different chain sizes without degree-of-freedom-specific branches. The
current validation suite covers $n=2$ through $n=9$; the benchmark exercises chains
up to $n=24$.

The dynamics layer contains a Lagrangian formulation together with Featherstone
algorithms: RNEA for inverse dynamics, CRBA for mass-matrix construction, and ABA
for forward dynamics.

Also included are external wrenches, cubic and quintic trajectories, PD-plus-
gravity and computed-torque control, and fixed-step RK4 simulation.

**Out of scope for v1:** floating-base dynamics, closed kinematic loops and
branched trees, joint friction and actuator dynamics, contact and collision
resolution, joint-limit enforcement during simulation, and deployment-oriented
native/GPU code generation.

## 3. Non-negotiable design constraints

1. **No degree-of-freedom branching.** No `if n == 4:` style logic exists in the
   library. Algorithms operate on arbitrary chain length.

2. **Forward dynamics does not form an explicit inverse.** The Lagrangian backend
   solves

   $$
   M(q)\ddot q = \tau - C(q,\dot q)\dot q - G(q) + J^\top f_{\text{ext}}
   $$

   using a Cholesky factorisation of $M$. The ABA backend computes $\ddot q$
   recursively without forming $M$.

3. **The symbolic and numerical layers are separate.** SymPy is used for offline
   derivation and checking. The numerical runtime uses NumPy and does not import
   SymPy. The symbolic layer can also compile expressions into NumPy callables for
   offline evaluation.

4. **Independent dynamics formulations.** The Lagrangian and Featherstone paths
   share the robot and kinematic representation, but use different dynamics
   formulations. The validation suite compares their results as a cross-check.

## 4. Public API

```python
robot = Robot(links=[...], joints=[...])

robot.forward_kinematics(q)   -> list[T_0i]
robot.jacobian(q, link=-1)    -> (6, n)       # [linear; angular]

robot.mass_matrix(
    q,
    method="crba" | "lagrangian" | "rnea"
) -> M

robot.coriolis(q, qd, method="lagrangian") -> C

robot.coriolis_vector(
    q, qd,
    method="rnea" | "lagrangian"
) -> C @ qd

robot.gravity_torque(q, method="lagrangian" | "rnea") -> G

robot.inverse_dynamics(
    q, qd, qdd,
    method="rnea" | "lagrangian",
    external_wrenches=None,
) -> tau

robot.forward_dynamics(
    q, qd, tau,
    method="aba" | "lagrangian",
    external_wrenches=None,
) -> qdd
```

The default methods are CRBA for the mass matrix, Lagrangian for the explicit
Coriolis matrix, RNEA for inverse dynamics and the Coriolis vector, and ABA for
forward dynamics.

## 5. Mathematical approach

**Mass matrix.** The Lagrangian backend forms the kinetic energy from the linear
and angular COM Jacobians:

$$
M(q)=\sum_i \left[m_i J_{v,c_i}^\top J_{v,c_i}
+J_{\omega,i}^\top R_i I_i R_i^\top J_{\omega,i}\right].
$$

The matrix is symmetrised at the end of the numerical assembly to remove roundoff
asymmetry.

**Gravity.** With

$$
V(q)=-\sum_i m_i\,g^\top p_{c_i},
$$

the gravity vector is

$$
G_j(q)=-\sum_i m_i\,g^\top J_{v,c_i}[:,j].
$$

**Coriolis and centrifugal terms.** The Lagrangian backend constructs the matrix
from the Christoffel symbols:

$$
C_{ij}(q,\dot q)=\frac12\sum_k
\left(
\frac{\partial M_{ij}}{\partial q_k}
+\frac{\partial M_{ik}}{\partial q_j}
-\frac{\partial M_{jk}}{\partial q_i}
\right)\dot q_k.
$$

The mass-matrix derivatives can be evaluated with complex-step, central, or
forward finite differences. Complex-step is the default because it avoids the
subtractive cancellation present in finite differences for analytic functions.
The tests compare the complex-step result with finite-difference derivatives and
with symbolic results on small models.

## 6. Validation strategy

The project checks both algebraic properties and independent implementations.

| Class | Check |
| --- | --- |
| Structural | $M=M^\top$; $M$ positive definite on valid models; $\dot M-2C$ skew-symmetric |
| Isolation | $\dot q=\ddot q=0$ gives gravity torque; zero gravity gives zero gravity torque |
| Round trip | $\ddot q\rightarrow\tau\rightarrow\ddot q$ and $\tau\rightarrow\ddot q\rightarrow\tau$ |
| Cross-backend | Lagrange/RNEA inverse dynamics; Lagrange/CRBA/RNEA mass matrices; Lagrange/ABA forward dynamics |
| Analytic | Closed-form 2-DOF planar $M$, $C$, and $G$ |
| Symbolic | SymPy Euler–Lagrange derivation on small models |
| External wrench | RNEA wrench handling versus $-J^\top f$ |
| Simulation | Passive energy drift for RK4 and Euler |
| Control | Tracking error for PD+gravity, feedforward+PD, and computed torque |

Property tests establish internal consistency. The closed-form and symbolic
comparisons provide checks against independent derivations.

## 7. Deliverables

The repository contains an installable package with type hints and docstrings,
a test suite, runnable examples, backend benchmarks, the derivation notes in
`docs/theory.md`, a measured validation report, and CI checks on supported Python
versions.

## 8. Success criteria

The v1 implementation is considered complete when the generic API works for the
supported serial-chain models and the validation suite confirms the following:

- Lagrangian and Featherstone implementations agree within numerical tolerance.
- Structural properties of the mass and Coriolis terms hold on the validation
  models.
- The closed-form 2-DOF model and small SymPy derivations agree with the numerical
  implementation.
- External-wrench mappings are consistent with the Jacobian formulation.
- Passive simulation and model-based control produce the expected numerical
  behaviour.
- The same library code is exercised across several chain sizes and joint types.
