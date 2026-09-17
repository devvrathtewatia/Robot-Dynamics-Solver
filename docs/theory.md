# Theory

The derivations this library implements, and the reasoning behind each
implementation choice. Notation: $n$ degrees of freedom, $q\in\mathbb R^n$,
gravitational acceleration $g$ in the base frame.

---

## 1. Kinematics

Joint $i$ connects link $i-1$ to link $i$. Its fixed `origin` transform is followed
by the joint's own motion:

$$T_0^i(q) = T_0^{i-1}\,\text{origin}_i\,T_{\text{joint}}(q_i)$$

with $T_{\text{joint}} = \begin{bmatrix} R(\hat a, q_i) & 0 \\ 0 & 1\end{bmatrix}$
for a revolute joint about $\hat a$, and
$\begin{bmatrix} I & \hat a q_i \\ 0 & 1\end{bmatrix}$ for a prismatic one.

This is the URDF layout rather than Denavit–Hartenberg. DH parameters are more
compact but constrain where frames may sit, and transcribing an existing robot into
them is error-prone; here a model can be copied out of a robot description directly.

### Jacobians

For a point $p$ attached to link $i$, the column of the geometric Jacobian
belonging to joint $j \le i$ is

$$
J_{v,j} = \begin{cases} \hat z_j \times (p - o_j) & \text{revolute} \\
\hat z_j & \text{prismatic}\end{cases}
\qquad
J_{\omega,j} = \begin{cases} \hat z_j & \text{revolute} \\ 0 & \text{prismatic}\end{cases}
$$

where $\hat z_j$ is joint $j$'s axis in base coordinates and $o_j$ its frame origin.
Columns for $j > i$ are zero — a joint downstream of a body cannot move it — and that
zero pattern is what makes the per-link Jacobians assemble into a correct mass matrix.

---

## 2. Lagrangian dynamics

### Kinetic energy and the mass matrix

$$T = \sum_i \left[\tfrac12 m_i v_{c_i}^\top v_{c_i}
              + \tfrac12 \omega_i^\top I_i^0 \omega_i\right],
\qquad I_i^0 = R_i I_i R_i^\top$$

Since $v_{c_i} = J_{v,c_i}\dot q$ and $\omega_i = J_{\omega,i}\dot q$, this is
exactly $T = \tfrac12\dot q^\top M(q)\dot q$ with

$$\boxed{M(q) = \sum_i \left[m_i J_{v,c_i}^\top J_{v,c_i}
   + J_{\omega,i}^\top R_i I_i R_i^\top J_{\omega,i}\right]}$$

No differentiation appears: the Jacobians already *are* the derivatives that a
by-hand Euler–Lagrange derivation would produce. The rotation $R_i I_i R_i^\top$ is
the step most often dropped, and dropping it gives a wrong-but-plausible $M$ that
still passes symmetry and positive-definiteness tests — which is why this library
checks against an independent derivation too.

### Potential energy and gravity

$$V = -\sum_i m_i\,g^\top p_{c_i}
\quad\Longrightarrow\quad
\boxed{G_j = \frac{\partial V}{\partial q_j} = -\sum_i m_i\,g^\top J_{v,c_i}[:,j]}$$

exact and analytic, because $\partial p_{c_i}/\partial q_j$ is precisely that
Jacobian column. Setting $g=0$ gives $G=0$ identically, with no special case.

### Coriolis and centrifugal terms

The only place a derivative of $M$ is needed. Christoffel symbols of the first kind:

$$\boxed{C_{ij} = \sum_k \tfrac12\left(
\frac{\partial M_{ij}}{\partial q_k} +
\frac{\partial M_{ik}}{\partial q_j} -
\frac{\partial M_{jk}}{\partial q_i}\right)\dot q_k}$$

Many matrices $C$ satisfy $C\dot q = $ (the correct velocity-product torque); this
particular one is singled out because it makes

$$\dot M(q) - 2C(q,\dot q) \quad \text{skew-symmetric,}$$

which is the property the standard passivity-based stability proof of computed-torque
control relies on. A different valid factorisation would give the same torque and
break that proof, so the test suite checks the skew-symmetry and not merely the
product.

### Why complex-step differentiation

$\partial M/\partial q$ is computed as

$$\frac{\partial f}{\partial q_k} = \frac{\operatorname{Im} f(q + ih\,e_k)}{h},
\qquad h = 10^{-20}$$

For analytic $f$, expanding gives
$f(q+ih) = f(q) + ihf'(q) - \tfrac{h^2}{2}f''(q) + \ldots$, so the imaginary part is
$hf'(q) + O(h^3)$ — and crucially **no subtraction of nearby values occurs**, so $h$
can be made arbitrarily small without cancellation. A central difference is stuck at
about $\sqrt[3]{\epsilon} \approx 10^{-5}$ relative accuracy because shrinking $h$
trades truncation error for cancellation error.

Measured on a 5-joint spatial chain, the central difference disagrees with the
complex step by $\sim\!10^{-10}$ absolute and the forward difference by
$\sim\!10^{-8}$. That error would flow straight into $C$, and at $10^{-6}$ a genuine
modelling discrepancy is indistinguishable from a differentiation artefact. At
machine precision, any disagreement between backends is unambiguous — which is what
makes the cross-validation in `docs/validation.md` meaningful.

This requires every function in the kinematic chain to propagate complex dtype
cleanly. That is why `utils/rotations.py` builds matrices from nested lists instead
of assigning into a `np.zeros` array, and why `rot_axis` does not renormalise its
axis.

### Equation of motion

$$\boxed{\tau = M(q)\ddot q + C(q,\dot q)\dot q + G(q) - J^\top f_{\text{ext}}}$$

Forward dynamics inverts it as a **linear solve**:

$$M(q)\ddot q = \tau - C\dot q - G + J^\top f_{\text{ext}}$$

via Cholesky factorisation, valid because $M$ is symmetric positive definite for any
physical model. Forming $M^{-1}$ explicitly would be about twice the work, less
accurate, and would discard that structure. A model with a degenerate $M$ raises
rather than returning quietly wrong accelerations.

---

## 3. Spatial vector algebra

Featherstone's 6-vectors, **angular part first**:

$$\hat v = \begin{bmatrix}\omega\\v\end{bmatrix},
\qquad \hat f = \begin{bmatrix}n\\f\end{bmatrix}$$

Motion transform for a child frame at $p$ with rotation $R$ (child expressed in
parent), mapping parent coordinates to child coordinates:

$$X = \begin{bmatrix} R^\top & 0 \\ -R^\top[p]_\times & R^\top\end{bmatrix}$$

The lower-left block performs the change of reference point,
$v_c = R^\top(v_p - p\times\omega)$.

Force vectors transform by $X^{*} = X^{-\top}$, and equivalently
$\hat f_{\text{parent}} = X^\top \hat f_{\text{child}}$ — that single identity is
what the backward passes of RNEA and ABA use.

Spatial cross products:

$$\hat v\times = \begin{bmatrix}[\omega]_\times & 0\\ [v]_\times & [\omega]_\times\end{bmatrix},
\qquad \hat v\times^* = -(\hat v\times)^\top$$

Spatial inertia about a body's frame origin, given mass $m$, COM $c$ and rotational
inertia $I_c$ about the COM:

$$I = \begin{bmatrix} I_c + m[c]_\times[c]_\times^\top & m[c]_\times \\
m[c]_\times^\top & m\,1\end{bmatrix}$$

> **Convention warning.** This library uses `[angular; linear]` for spatial vectors
> and `[linear; angular]` for geometric Jacobians. Both are standard in their own
> literature. They are kept in separate modules, and the conversion helpers are
> explicit, because silently mixing them is the classic source of sign errors — and
> one such error did occur during this build (see §6).

---

## 4. Recursive algorithms

### RNEA — inverse dynamics in $O(n)$

Outward, $i = 1\ldots n$:

$$\hat v_i = X_i \hat v_{\lambda(i)} + S_i\dot q_i, \qquad
\hat a_i = X_i \hat a_{\lambda(i)} + S_i\ddot q_i + \hat v_i\times S_i\dot q_i$$
$$\hat f_i = I_i \hat a_i + \hat v_i\times^* I_i \hat v_i - \hat f_i^{\text{ext}}$$

Inward, $i = n\ldots 1$:

$$\tau_i = S_i^\top \hat f_i, \qquad
\hat f_{\lambda(i)} \mathrel{+}= X_i^\top \hat f_i$$

Gravity enters as a fictitious base acceleration $\hat a_0 = -\hat g$. Every link
then inherits it through the same recursion that handles real acceleration, so no
separate gravity term is needed — an elegance worth noticing, since it means one code
path handles what the Lagrangian formulation splits into two.

### CRBA — the mass matrix in $O(n^2)$

Walking inward, $I^C_i$ accumulates the spatial inertia of link $i$ and everything
distal to it. Then for every ancestor $j$ of $i$:

$$M_{ij} = S_i^\top\,{}^iX_j^\top\,I^C_i\,S_j$$

i.e. the projection of that composite inertia onto the two joint axes.

### ABA — forward dynamics in $O(n)$

Three passes. The middle one folds each link's *articulated* inertia into its parent
after removing the component the joint is free to accelerate along:

$$I^a_i = I^A_i - U_i d_i^{-1} U_i^\top,
\qquad U_i = I^A_i S_i, \quad d_i = S_i^\top U_i$$

This is the step that makes the algorithm linear rather than cubic: by the time the
outward pass reaches a link, that link already knows how the rest of the chain will
resist it, so no global linear system is ever assembled.

---

## 5. Control

**PD plus gravity compensation.** $\tau = K_p e + K_d\dot e + G(q)$. Asymptotically
stable for a constant setpoint, but during motion the inertial and Coriolis terms act
as unmodelled disturbances.

**Computed torque.** With

$$\tau = M(q)\big[\ddot q_d + K_d\dot e + K_p e\big] + C(q,\dot q)\dot q + G(q)$$

substitution into the equation of motion collapses it to

$$\ddot e + K_d\dot e + K_p e = 0$$

a *linear, decoupled* error system, identical for every joint in every
configuration. The nonlinear plant has become $n$ independent second-order systems,
so gains follow analytically: $K_p = \omega^2$, $K_d = 2\omega$ for critical damping.

**Why this matters practically.** The eigenvalues of $M$ span roughly $40\times$ on a
2-link planar arm and over $1300\times$ on a 3-joint spatial chain — a wrist roll
about a slender link's own axis has almost no inertia. No single fixed gain pair
suits every direction: whatever damps the heavy direction leaves a fast pole near
$K_d/M_{\min}$ in the light one. Computed torque escapes this entirely by cancelling
$M$. Measured tracking error on the same arm, same gains, same step: $3\times10^{-2}$
rad for PD-plus-gravity against $2\times10^{-11}$ for computed torque.

The collapse is exact only if the model is exact, so both controllers accept a
`model` distinct from the simulated `robot`, and the test suite measures the
degradation under a 50 % mass error rather than assuming it away.

---

## 6. Errors found during development

Recorded because each was caught by a specific test, and each illustrates a failure
mode that would otherwise have been invisible.

**The Christoffel axis permutation.** `np.transpose(dM, (1,2,0))` yields
$\partial M_{ki}/\partial q_j$, not the required $\partial M_{jk}/\partial q_i$; the
correct permutation is `(2,0,1)`. The error left $M$ and $G$ exact and corrupted only
$C$, so symmetry, positive definiteness, the gravity checks and the analytic $M$
comparison all still passed. It surfaced as a $10^{-1}$ disagreement between the
Christoffel $C\dot q$ and the RNEA velocity-product torque — a test that exists
precisely because it exercises $M$, its derivative and the factorisation at once.

**The external-wrench frame convention.** Using the spatial force transform
$X^{-\top}$ to bring a base-frame wrench into link coordinates also performs the
moment shift $n \to n - r\times f$, which assumes the moment was referenced to the
*base* origin. For a wrench already referenced to the link origin this double-counts
the shift. The symptom was diagnostic: pure moments were correct while pure forces
were wrong, so a test using only moments passed. The catching check is physical — a
pure force applied at a link's own frame origin must produce exactly zero torque at
that link's joint, because the moment arm is zero. The fix is a rotation-only
transform; the moment shift for the *parents* still happens correctly in the backward
pass via $X^\top$.

**A stiffness problem misread as a controller bug.** A PD-controlled simulation
diverged to NaN. The cause was neither the controller nor the dynamics but the
integration step: a PD loop leaves a fast closed-loop pole near $K_d/M_{\min}$, and
with $K_d = 20$ and $M_{\min} = 7.4\times10^{-4}$ that pole needs
$\Delta t < 1.2\times10^{-5}$ s, while the simulation used $10^{-3}$. The resolution
was to choose $K_d = 2\sqrt{K_p M_{\min}}$, and the finding is now itself a test.
Worth recording because the instinct on seeing a diverging controller is to suspect
the dynamics.

Two of the *tests* were also wrong — one normalised a vector by a different random
vector's norm, producing a non-unit rotation axis, and one asserted a degeneracy
guard would fire on an inertia above its threshold. Both are noted here because a
test suite is code too, and a passing suite whose tests are wrong is worse than no
suite.

---

## 7. References

- R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008 — Tables 5.1
  (RNEA), 6.2 (CRBA), 7.1 (ABA), and the spatial-vector formalism throughout.
- M. Spong, S. Hutchinson, M. Vidyasagar, *Robot Modeling and Control*, Wiley — the
  Lagrangian formulation, Christoffel symbols and computed-torque control.
- J. Squire, G. Trapp, "Using complex variables to estimate derivatives of real
  functions", *SIAM Review* 40(1), 1998 — complex-step differentiation.
