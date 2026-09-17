Robotics Dynamics Solver

A Python implementation of rigid-body dynamics for serial robot manipulators.

I built this project to understand the mathematics behind robot dynamics and to implement the main algorithms myself rather than treating a dynamics library as a black box.

The project includes Lagrangian dynamics, Featherstone's recursive algorithms, kinematics, trajectory generation, simulation, and model-based control.

What this project does

For a serial manipulator, the dynamics can be written in the general form:

tau = M(q) qdd + C(q, qdot) qdot + G(q) - J^T f_ext

where:

M(q) is the mass matrix
C(q, qdot) contains Coriolis and centrifugal terms
G(q) is the gravity torque
J^T f_ext represents the effect of an external wrench

The library supports both directions of the dynamics problem.

Inverse dynamics

Given:

q, qdot, qdd

the solver calculates the required joint torques:

tau

Forward dynamics

Given:

q, qdot, tau

the solver calculates the resulting joint acceleration:

qdd

The implementation is written to work with different serial manipulators and numbers of degrees of freedom.

Why I built it

I started with the Lagrangian formulation because I wanted to understand how the standard manipulator dynamics equation is obtained from kinetic and potential energy.

I then implemented Featherstone's recursive algorithms independently so that I could compare different formulations of the same dynamics problem.

This gave me two goals for the project:

Understand the mathematics behind robot dynamics.
Validate the implementation by comparing independent algorithms.

This is primarily an educational and research implementation, rather than an attempt to replace established robotics dynamics libraries.

Dynamics algorithms
1. Lagrangian Dynamics

The Lagrangian formulation starts from:

L = T - V

where:

T is kinetic energy
V is potential energy

The implementation calculates:

Mass matrix M(q)
Gravity vector G(q)
Coriolis and centrifugal terms
Inverse dynamics
Forward dynamics

The Lagrangian implementation provides an explicit formulation that can be used to validate the recursive algorithms.

2. Recursive Newton-Euler Algorithm

RNEA is used for inverse dynamics.

It performs:

A forward pass through the robot to calculate velocities and accelerations.
A backward pass to propagate forces and calculate joint torques.

The algorithm has linear computational complexity with respect to the number of joints:

O(n)

3. Composite Rigid Body Algorithm

CRBA calculates the mass matrix using composite rigid-body inertias.

Its computational complexity is:

O(n^2)

The resulting mass matrix can then be used for forward dynamics and other calculations.

4. Articulated Body Algorithm

ABA is used for forward dynamics.

It calculates joint accelerations directly from the robot state and applied torques using articulated-body inertias.

Its computational complexity is:

O(n)

Validation

A major part of the project is comparing independent formulations of the same dynamics problem.

The validation suite includes comparisons between:

Lagrangian inverse dynamics and RNEA
Lagrangian mass matrix and CRBA
Lagrangian forward dynamics and ABA
Christoffel-based Coriolis terms and the recursive formulation
Numerical dynamics and a closed-form 2-DOF derivation
Numerical dynamics and an independent SymPy Euler-Lagrange derivation

Additional checks include:

Mass matrix symmetry
Positive definiteness of the mass matrix
The skew-symmetry property of the dynamics matrices
Forward/inverse dynamics consistency
External wrench handling
Energy conservation during passive simulation

The test suite currently contains 316 tests covering kinematics, dynamics, symbolic derivations, control, simulation, and multiple robot models.

More detailed validation information is available in:

docs/validation.md

Example
import numpy as np
import robot_dynamics as rd

robot = rd.planar_chain(3)

q = np.array([0.3, -0.6, 0.9])
qd = np.array([0.5, 0.2, -0.4])
qdd = np.array([1.0, -0.5, 0.3])

# Inverse dynamics
tau = robot.inverse_dynamics(q, qd, qdd)

# Forward dynamics
qdd_back = robot.forward_dynamics(q, qd, tau)

assert np.allclose(qdd_back, qdd)

# Individual dynamics terms
M = robot.mass_matrix(q)
C = robot.coriolis(q, qd)
G = robot.gravity_torque(q)

assert np.allclose(tau, M @ qdd + C @ qd + G)

Different implementations can also be selected explicitly:

robot.mass_matrix(q, method="crba")
robot.mass_matrix(q, method="lagrangian")

robot.inverse_dynamics(q, qd, qdd, method="rnea")
robot.inverse_dynamics(q, qd, qdd, method="lagrangian")

robot.forward_dynamics(q, qd, tau, method="aba")
robot.forward_dynamics(q, qd, tau, method="lagrangian")
From dynamics to simulation and control

The project also includes trajectory generation, numerical simulation, and model-based control.

Robot Model
    |
    +-- Kinematics
    |     +-- Forward kinematics
    |     +-- Jacobians
    |     +-- COM Jacobians
    |
    +-- Dynamics
    |     +-- Lagrangian
    |     +-- RNEA
    |     +-- CRBA
    |     +-- ABA
    |
    +-- Trajectory Generation
    |     +-- Cubic
    |     +-- Quintic
    |     +-- Multi-waypoint
    |
    +-- Simulation
    |     +-- RK4 integration
    |
    +-- Control
          +-- PD + gravity compensation
          +-- Feedforward + PD
          +-- Computed torque control
Experiments
Trajectory Torque Calculation

The project can generate joint trajectories and calculate the torques required to follow them using the dynamics model.




Model-Based Control

The project includes model-based control experiments using:

PD + gravity compensation
Feedforward + PD
Computed torque control




Energy Conservation

A passive manipulator can be simulated under gravity using numerical integration.

The project compares RK4 and explicit Euler integration and examines the resulting energy drift.




6-DOF Manipulator

The dynamics model also includes the 6-DOF arm used in my mobile bimanual manipulation project.

The solver can be used to investigate:

Joint torque requirements
Static payload requirements
Torque variation during trajectories
Dynamics during arm motion
Model-based control




Project Structure
robot_dynamics/
|
+-- src/robot_dynamics/
|   |
|   +-- models/
|   +-- kinematics/
|   |
|   +-- dynamics/
|   |   +-- lagrangian.py
|   |   +-- rnea.py
|   |   +-- crba.py
|   |   +-- aba.py
|   |   +-- symbolic.py
|   |
|   +-- trajectory/
|   +-- control/
|   +-- simulation/
|   +-- utils/
|
+-- tests/
+-- examples/
+-- benchmarks/
|
+-- docs/
|   +-- theory.md
|   +-- validation.md
|   +-- generate_validation.py
|
+-- SPEC.md
+-- pyproject.toml
+-- requirements.txt
+-- README.md
Installation

Python 3.10+ is required.

For the development environment:

pip install -e ".[dev]"

For the numerical library only:

pip install -e .
Running the examples
python examples/01_inverse_dynamics.py
python examples/02_passive_simulation.py
python examples/03_trajectory_torques.py
python examples/04_computed_torque.py
python examples/05_bimanual_arm.py

To regenerate validation results:

python docs/generate_validation.py

To run the test suite:

pytest
Computational Complexity
Algorithm	Operation	Complexity
RNEA	Inverse dynamics	O(n)
ABA	Forward dynamics	O(n)
CRBA	Mass matrix	O(n²)
Lagrangian	Equation-based dynamics	Higher computational cost

The repository also contains a benchmark suite for measuring the empirical scaling of the different implementations.

Limitations

This is an educational and research implementation rather than a production robotics dynamics engine.

Current limitations include:

Fixed-base serial manipulators only
No closed kinematic loops
No branched robot trees
No contact or collision dynamics
No joint friction or actuator dynamics
No gear-ratio or motor-inertia modelling
Joint limits are stored but not enforced during simulation
Rigid links only
Dense, single-threaded numerical operations
Symbolic expressions become expensive as the number of DOF increases
Python implementation is intended primarily for analysis, simulation, and control experiments rather than high-frequency real-time deployment
Theory and Documentation

The mathematical background, spatial-vector conventions, derivations, and implementation notes are documented in:

SPEC.md
docs/theory.md
docs/validation.md

The recursive dynamics implementation follows the rigid-body dynamics framework developed in Featherstone's work.

Relation to My Robotics Work

This project is connected to my work on a mobile bimanual manipulator.

The dynamics solver gives me a way to investigate the mechanical requirements of the 6-DOF arm, including:

Required joint torques for a given payload
Static loading at different configurations
Torque requirements during trajectories
Dynamics during manipulation
Model-based control

The goal is to use the solver not only as a standalone implementation project, but also as a tool for analysing the robot I am building.

Status

The core dynamics implementations, validation suite, simulation, and control components are implemented.

The project is still evolving, with ongoing work on mathematical documentation, additional validation cases, and applications of the solver to robotic manipulation.
