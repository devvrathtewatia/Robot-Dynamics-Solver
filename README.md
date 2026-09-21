# `robot_dynamics`

**A general N-DOF rigid-body dynamics library for serial manipulators, implementing
the Lagrangian formulation together with Featherstone's recursive algorithms.**

Computes

$$
\tau = M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) - J^\top f_{\text{ext}}
$$

for inverse and forward dynamics, with trajectory generation, simulation, and
model-based control on top.

The implementation is generic in the number of joints. The current validation suite
uses models with $n=2$ through $n=9$, and the benchmark exercises chains up to $n=24$.
There is no degree-of-freedom-specific branch such as `if n == 4` in the library.

---

## What the project is for

This project is an implementation and validation study of rigid-body robot dynamics.
It is intended to make the equations visible in code and to compare a Lagrangian
implementation with recursive rigid-body algorithms.

The Lagrangian backend forms $M$, $C$, and $G$ from kinetic energy, potential energy,
and Christoffel symbols. The recursive backend implements RNEA, CRBA, and ABA using
spatial-vector quantities. The two approaches share the robot model and kinematics,
but the dynamics calculations are different.

The repository also contains a 6-DOF arm model used as a test case from a companion
Gazebo project. This gives the dynamics code a concrete manipulator model in addition
to the analytic and randomly generated validation models.

---

## Installation

```bash
pip install -e ".[dev]"        # tests, SymPy, matplotlib, and development tools
pip install -e .                # numerical layer only; NumPy is the runtime dependency
```

Python 3.10+ is supported.

---

## Minimal example

```python
import numpy as np
import robot_dynamics as rd

robot = rd.planar_chain(3)

q   = np.array([0.3, -0.6,  0.9])
qd  = np.array([0.5,  0.2, -0.4])
qdd = np.array([1.0, -0.5,  0.3])

tau = robot.inverse_dynamics(q, qd, qdd)
qdd_back = robot.forward_dynamics(q, qd, tau)

assert np.allclose(qdd_back, qdd)

M = robot.mass_matrix(q)
C = robot.coriolis(q, qd)
G = robot.gravity_torque(q)

assert np.allclose(tau, M @ qdd + C @ qd + G)
```

Building a model explicitly:

```python
from robot_dynamics import Joint, Link, Robot
from robot_dynamics.utils import transform

robot = Robot(
    links=[
        Link(mass=1.2, com=[0.15, 0, 0],
             inertia=np.diag([1e-3, 2e-2, 2e-2])),
        Link(mass=0.8, com=[0.10, 0, 0],
             inertia=np.diag([1e-3, 1e-2, 1e-2])),
    ],
    joints=[
        Joint('revolute', axis=[0, 0, 1]),
        Joint('revolute', axis=[0, 1, 0],
              origin=transform(np.eye(3), [0.30, 0, 0])),
    ],
    gravity=(0, 0, -9.81),
)
```

Selecting a backend explicitly:

```python
robot.mass_matrix(q, method='crba')
robot.mass_matrix(q, method='lagrangian')

robot.inverse_dynamics(q, qd, qdd, method='rnea')
robot.inverse_dynamics(q, qd, qdd, method='lagrangian')

robot.forward_dynamics(q, qd, tau, method='aba')
robot.forward_dynamics(q, qd, tau, method='lagrangian')
```

---

## Architecture

```text
                         Robot model
                    (Link, Joint, Robot)
                              |
                          Kinematics
              (transforms, Jacobians, COM Jacobians)
                              |
                 +------------+------------+
                 |                         |
            Lagrangian               Featherstone
        M from kinetic energy       RNEA: inverse dynamics
        G from potential energy     CRBA: mass matrix
        C from Christoffel terms    ABA: forward dynamics
                 |                         |
                 +------------+------------+
                              |
                       Forward dynamics
                         (no explicit M^-1)
                              |
                          Simulation
                              |
                           Control
```

The symbolic layer is separate from the numerical path. SymPy is used for offline
derivation and checking; the numerical modules use NumPy only. The symbolic module
can also compile expressions to NumPy callables for offline evaluation.

```text
src/robot_dynamics/

├── utils/          rotations, transforms, spatial-vector utilities
├── models/         Link, Joint, Robot, example models
├── kinematics/     forward kinematics, Jacobians, COM Jacobians
├── dynamics/       Lagrangian, RNEA, CRBA/ABA, symbolic checks
├── trajectory/     cubic, quintic, multi-segment trajectories
├── control/        PD+gravity, feedforward+PD, computed torque
└── simulation/     Euler/RK4 integration, energy and tracking metrics
```

---

## Results

Run the example scripts to regenerate the figures.

### Trajectory and required torques

![trajectory](docs/figures/trajectory_torques.png)

A quintic rest-to-rest motion on a 4-DOF spatial chain. The torque plot shows the
joint effort required by the motion; the example can be used directly when checking
motor or actuator sizing.

### Model-based control

![control](docs/figures/control_tracking.png)

For the 2-link test in the validation script, the maximum tracking errors were:

| controller | max abs error |
| --- | ---: |
| PD + gravity compensation | 5.4 × 10⁻² rad |
| feedforward + PD | 4.2 × 10⁻¹⁰ rad |
| computed torque | 1.6 × 10⁻¹¹ rad |

For this test case, using the full dynamics model reduces the measured error by about
nine orders of magnitude.

### Energy conservation

![energy](docs/figures/passive_energy.png)

A torque-free planar arm is simulated under gravity. RK4 at 1 ms has about
$4	imes10^{-9}$ relative energy drift over 4 s in the validation run; explicit Euler
at the same step has about $7	imes10^{-2}$ drift. The test separates integration
error from the instantaneous algebraic checks used elsewhere.

### Companion 6-DOF arm

![bimanual](docs/figures/bimanual_torques.png)

The repository also includes the 6-DOF arm model from the companion Gazebo project,
used for static and dynamic torque checks.

---

## Validation

The full measured report is [`docs/validation.md`](docs/validation.md). It is
regenerated by `python docs/generate_validation.py`.

The current test suite contains **316 tests** over 13 models spanning $n=2$ through
$n=9$, including planar, spatial, random, mixed revolute/prismatic, prismatic, and
6-DOF arm models.

| check | result in the current validation run |
| --- | --- |
| Lagrange vs RNEA, torque | ≤ 8.3 × 10⁻¹⁶ |
| Lagrange vs CRBA, mass matrix | ≤ 1.7 × 10⁻¹⁵ |
| Lagrange vs ABA, acceleration | ≤ 1.6 × 10⁻¹⁴ |
| Christoffel $C\dot q$ vs RNEA | ≤ 1.5 × 10⁻¹⁵ |
| closed-form 2-DOF model | ≤ 10⁻¹⁵ over 200 random states |
| SymPy Euler–Lagrange comparison | ≤ 10⁻¹⁵ |
| $M=M^	op$, $x^	op Mx>0$ | holds on all validation models |
| $\dot M-2C$ skew-symmetry | ≤ 10⁻⁹ |
| $\ddot q
ightarrow	au
ightarrow\ddot q$ | ≤ 1.6 × 10⁻¹⁴ |
| external wrench vs $-J^	op f$ | ≤ 10⁻¹⁴ |

The property tests establish self-consistency. The closed-form 2-DOF equations and
small SymPy derivations provide independent checks of the numerical model. During
development, two implementation errors were caught by these comparisons; the
third development note in [`docs/theory.md`](docs/theory.md#7-development-notes) is a
numerical integration issue, not an incorrect dynamics equation.

The benchmark measures the expected scaling trends. At the tested Python problem
sizes, constant factors and per-link overhead affect the fitted exponent, so the
measured exponent is not treated as a proof of asymptotic complexity.

| algorithm | measured exponent | expected scaling |
| --- | ---: | ---: |
| RNEA inverse dynamics | 0.98 | 1 |
| ABA forward dynamics | 0.99 | 1 |
| CRBA mass matrix | 1.20 | 2 |
| Lagrangian inverse dynamics | 2.43 | 3 |

---

## Limitations

- **Fixed base only.** No floating-base dynamics.
- **Serial chains only.** No closed loops or branched trees.
- **No contact or collision resolution.** External wrenches are supplied by the caller.
- **No joint friction or actuator dynamics.** Gear ratios and motor inertia are also
  outside the model.
- **Joint limits are stored but not enforced during simulation.**
- **Rigid links only.** No flexibility or backlash.
- **No deployment-oriented code generation.** The symbolic layer can compile SymPy
  expressions to NumPy callables for offline work, but it does not generate native
  code, GPU kernels, or hard-real-time controller code.
- **Symbolic expressions do not scale well.** Symbolic checks are intended for small
  models; recursive numerical methods are the practical route for larger chains.
- **Dense, single-threaded numerical path.** No sparse dynamics implementation.

---

## Project layout

```text
dynamics/
├── SPEC.md                     project scope and acceptance criteria
├── README.md                   this file
├── src/robot_dynamics/         library source
├── tests/                      316 tests
├── examples/                   five runnable examples
├── benchmarks/                 backend scaling benchmark
├── docs/
│   ├── theory.md               equations, conventions, development notes
│   ├── validation.md           measured validation report
│   ├── generate_validation.py  validation report generator
│   └── figures/                generated example figures
├── pyproject.toml
├── LICENSE                     MIT
└── .github/workflows/tests.yml CI: supported Python versions, lint, tests
```

## Running the project

```bash
pytest tests -q
python docs/generate_validation.py > docs/validation.md
python benchmarks/bench_backends.py
cd examples && python 03_trajectory_torques.py
```

## References

R. Featherstone, *Rigid Body Dynamics Algorithms* (2008) for spatial-vector methods
and recursive rigid-body algorithms. Spong, Hutchinson & Vidyasagar, *Robot Modeling
and Control* for the Lagrangian formulation and computed-torque control. Squire &
Trapp, *SIAM Review* 40(1), 1998, for complex-step differentiation.

## License

MIT.
