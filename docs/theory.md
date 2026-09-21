# Dynamics theory and implementation notes

This document records the conventions used by the library and the main checks used
to validate the dynamics implementation. It is written around the code in
`src/robot_dynamics`, not around a separate symbolic model.

## 1. Generalized coordinates and frames

The robot is a fixed-base serial chain. Joint `i` connects link `i-1` to link `i`.
Its fixed `origin` transform is followed by the joint motion along the joint axis:

$$
T_0^i = T_0^{i-1} T_{\mathrm{origin},i} T_{\mathrm{joint},i}(q_i).
$$

The public geometric Jacobian uses the ordering

$$
J = \begin{bmatrix}J_v\\J_\omega\end{bmatrix},
$$

so that a point velocity is $v = J_v\dot q$ and an angular velocity is
$\omega = J_\omega\dot q$.

The Featherstone spatial-vector routines use the conventional spatial ordering
$[\omega;v]$ for motion and $[n;f]$ for force. The conversion functions in
`utils/spatial.py` are used where a geometric wrench/Jacobian convention is needed.

## 2. Equation of motion

The library uses the joint-space form

$$
\tau = M(q)\ddot q + C(q,\dot q)\dot q + G(q) - J^\top f_{\mathrm{ext}}.
$$

Here $M$ is the mass matrix, $C\dot q$ is the velocity-dependent torque vector,
$G$ is the gravity torque, and $f_{\mathrm{ext}}$ is an externally applied wrench
in the geometric `[force; moment]` convention used by the public Jacobian.

The corresponding forward-dynamics equation is

$$
M(q)\ddot q = \tau - C(q,\dot q)\dot q - G(q) + J^\top f_{\mathrm{ext}}.
$$

The Lagrangian backend solves this linear system with a Cholesky factorisation.
It does not form $M^{-1}$. The ABA backend computes the acceleration recursively
without first constructing the mass matrix.

## 3. Lagrangian formulation

The starting point is the Lagrangian

$$
L(q,\dot q)=T(q,\dot q)-V(q).
$$

The Euler–Lagrange equations are

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q_i}
-\frac{\partial L}{\partial q_i}
= \tau_i + (J^\top f_{\mathrm{ext}})_i.
$$

The implementation obtains the three standard terms directly from the robot
geometry.

### 3.1 Mass matrix

For link $i$, let $J_{v,c_i}$ and $J_{\omega,i}$ be the translational and angular
Jacobians of its centre of mass. If $R_i$ rotates the link-frame inertia tensor
into the base frame, the link contribution is

$$
M_i = m_i J_{v,c_i}^\top J_{v,c_i}
+J_{\omega,i}^\top R_i I_i R_i^\top J_{\omega,i}.
$$

The full mass matrix is

$$
M = \sum_i M_i.
$$

This calculation is local to each link and does not need numerical differentiation.
The result is symmetrised before it is returned to suppress roundoff asymmetry.

### 3.2 Gravity

The potential energy used by the code is

$$
V(q)=-\sum_i m_i g^\top p_{c_i}.
$$

Taking the gradient gives

$$
G_j(q)=\frac{\partial V}{\partial q_j}
=-\sum_i m_i g^\top J_{v,c_i}[:,j].
$$

The same sign convention is used by the RNEA gravity calculation, so the two
routes can be compared directly.

### 3.3 Coriolis and centrifugal terms

The code represents the velocity-dependent terms with a matrix $C$ satisfying
$C\dot q$ equal to the usual quadratic velocity product. Its entries are built
from the Christoffel symbols of the first kind:

$$
C_{ij}=\frac12\sum_k
\left(
\frac{\partial M_{ij}}{\partial q_k}
+\frac{\partial M_{ik}}{\partial q_j}
-\frac{\partial M_{jk}}{\partial q_i}
\right)\dot q_k.
$$

The default derivative method is complex-step differentiation. For an analytic
function $F$,

$$
\frac{\partial F}{\partial q_k}(q)
\approx \frac{\operatorname{Im}F(q+i h e_k)}{h},
$$

with a very small $h$. The implementation is arranged so the kinematic functions
preserve the complex dtype during this calculation. Central and forward finite
difference versions are also available for comparison tests.

## 4. Featherstone algorithms

The recursive backend uses the spatial-vector formulation described by Featherstone.
The same `Robot`, `Joint`, and kinematic model are passed to the recursive routines,
but the dynamics calculations do not reconstruct $M$, $C$, and $G$ first.

### 4.1 RNEA

The Recursive Newton–Euler Algorithm computes inverse dynamics in two passes.
The outward pass propagates spatial velocity and acceleration from the base through
the chain and computes each link's spatial force. The inward pass accumulates child
forces and projects each link force onto its joint motion subspace:

$$
\tau_i = S_i^\top f_i.
$$

With gravity enabled, the base spatial acceleration is chosen so that gravity enters
the outward recursion. External wrenches are subtracted from the corresponding link
force using the library's force-transform convention.

### 4.2 CRBA

The Composite Rigid Body Algorithm constructs the mass matrix from composite spatial
inertias. It performs the upward accumulation first and then projects the composite
inertias through the joint motion subspaces to fill the matrix. The implementation
uses the matrix symmetry explicitly when returning the final result.

### 4.3 ABA

The Articulated-Body Algorithm computes forward dynamics without constructing the
full mass matrix. It performs an outward pass to compute articulated-body quantities
and a backward/forward recursion to obtain each joint acceleration.

This is the default forward-dynamics method because its computational structure is
linear in the number of joints for a serial chain.

## 5. External wrenches

The public Jacobian is ordered `[linear; angular]`, so an applied wrench is supplied
as

$$
f = \begin{bmatrix}F\\N\end{bmatrix}.
$$

The corresponding joint-space contribution is $J^\top f$. With the sign convention
of the equation of motion, this appears as `-J.T @ f` in inverse dynamics and with
the opposite sign on the right-hand side of the forward-dynamics equation.

The validation code compares an RNEA calculation with the Jacobian mapping. This
check is useful because an error in the force frame or sign can leave ordinary
no-wrench tests unchanged.

## 6. Validation strategy

The test suite uses several kinds of evidence:

1. **Structural properties.** The mass matrix is symmetric and positive definite on
   valid models, and the Christoffel construction satisfies the expected skew-
   symmetry property of $\dot M-2C$.
2. **Round trips.** Inverse and forward dynamics are applied in sequence to recover
   the original acceleration or torque.
3. **Independent calculations.** Lagrangian results are compared with RNEA, CRBA,
   and ABA results, and a 2-DOF planar model is compared with closed-form equations.
4. **Symbolic checks.** Small planar/spatial models are differentiated directly with
   SymPy and compared with the numerical layer.
5. **Physical checks.** External-wrench mappings, passive energy drift, and tracking
   behaviour are exercised on numerical simulations.

The important distinction is between self-consistency and an external reference.
A set of internal algorithms can agree while sharing the same mistake. The closed-form
2-DOF model and the independent SymPy derivation reduce that risk.

## 7. Development notes

Two implementation errors were found while the numerical and recursive backends were
being cross-checked.

- **Coriolis term:** an index error in the matrix construction was exposed when the
  Lagrangian velocity-product vector disagreed with the RNEA result. The closed-form
  2-DOF model also failed until the expression was corrected.
- **External wrench handling:** a frame/sign error in the wrench path caused the
  externally applied moment to be counted incorrectly. Comparing RNEA against the
  independent `-J.T @ f` calculation exposed the problem.

There was also a simulation issue that was not a dynamics-algorithm bug. An early
control run used a time step that was too large for the selected gains. Reducing the
step size restored stable numerical integration. The validation script now reports
energy drift at both RK4 and Euler step sizes so integration error is visible separately
from the dynamics equations.

These notes describe development and debugging. They are not claims of algorithmic
novelty; RNEA, CRBA, ABA, and the underlying equations are established methods.

## 8. Practical limitations

The current model is fixed-base and serial. It does not handle contact resolution,
friction, actuator dynamics, closed loops, branched trees, floating bases, or hard
real-time/native code generation. Joint limits are stored but not enforced by the
simulator. The symbolic expressions become expensive as the number of degrees of
freedom increases, so symbolic checks are intended for small models.
