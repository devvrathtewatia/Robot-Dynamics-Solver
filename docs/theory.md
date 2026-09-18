Theory and Design Choices
1. Kinematics

Joints connect one body part to the next, combining a fixed base layout with the joint's actual movement—whether it is turning around an axis (revolute) or sliding along a line (prismatic). This library relies on standard robot description layouts similar to URDF files rather than older, error-prone math conventions. This design choice makes it straightforward to copy a robot design directly into code without struggling to convert parameters. Jacobians act as the mathematical bridge connecting joint positions (q) and velocities (q dot) to the actual movement speeds of specific points on the robot through space.
2. Lagrangian Dynamics

The mass matrix describes how heavy the robot feels to move in different configurations, seamlessly combining mass, shape, and rotational inertia. Gravity pulls on the robot using its positions and movement Jacobians to find the exact effect on each joint. Coriolis and centrifugal forces naturally appear when parts are already moving, and the library uses a specific mathematical formulation for these forces to ensure the system remains stable and passive.

To calculate how the mass matrix changes as the robot moves, the library uses a clean complex-number differentiation trick that bypasses the rounding errors typical of standard numerical approximations. The core equations of motion link motor torques to the robot's mass, velocity forces, gravity, and external pushes, and they are solved efficiently via linear equations rather than slow, error-prone matrix inversions.
3. Spatial Vectors

Instead of managing linear forces and rotational moments separately everywhere, the library groups them into six-dimensional spatial vectors following standard approaches. Strict care is maintained regarding coordinate frames and rotation transforms to completely prevent classic sign errors that often plague robotics software.
4. Fast Simulation Algorithms

Several recursive algorithms power the library's performance:

    RNEA handles inverse dynamics in linear time to calculate exact motor torques for a desired movement by passing data outward and inward along the chain.

    CRBA builds the mass matrix efficiently by working inward from the tips to the base.

    ABA computes forward dynamics to see how the robot accelerates under specific motor torques without needing to solve a massive global matrix, relying instead on the concept of articulated inertia.

5. Robot Control

Basic PD control with gravity compensation helps hold positions and fight gravity, but it struggles during fast movements because it ignores internal forces like Coriolis effects. In contrast, Computed Torque control uses the robot's mathematical model to cancel out these internal forces on the fly. This transforms a complex, nonlinear robot into a set of independent systems, drastically improving precision and tracking accuracy.
6. Lessons Learned During Development

Building this library involved catching subtle bugs through rigorous testing. Notable examples include a matrix typo in the Coriolis calculations that was invisible to basic tests but caught when cross-validating backends, a frame confusion with external wrenches where moments were accidentally double-counted, and a simulation step-size issue where a fast-moving joint blew up due to integration limits rather than a broken controller.
7. References

    R. Featherstone, Rigid Body Dynamics Algorithms (for spatial vectors and fast recursive algorithms).

    M. Spong, S. Hutchinson, and M. Vidyasagar, Robot Modeling and Control (for Lagrangian mechanics and computed torque control).

    J. Squire and G. Trapp, on using complex numbers for accurate derivative calculations.
