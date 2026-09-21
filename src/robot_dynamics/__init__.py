# robot_dynamics — general N-DOF rigid-body dynamics for serial manipulators.
# Implements the equation of motion
# tau = M(q)*qdd + C(q, qd)*qd + G(q) - J^T * f_ext
# by two independent routes — the Lagrangian formulation and Featherstone's recursive
# algorithms — so that each can validate the other.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike, NDArray

from . import control, dynamics, kinematics, models, simulation, trajectory
from .dynamics import crba as _crba
from .dynamics import lagrangian as _lag
from .dynamics import rnea as _rnea
from .kinematics import forward as _fk
from .models.examples import (
    analytic_two_dof,
    bimanual_arm,
    planar_chain,
    prismatic_chain,
    random_robot,
    spatial_chain,
    two_dof_planar,
)
from .models.robot import Joint, Link, Robot

__version__ = '1.0.0'
# Kinematics
def _forward_kinematics(self: Robot, q: ArrayLike):
    # End-effector pose T_0^n(q) as a 4x4 homogeneous transform.
    return _fk.forward_kinematics(self, q)


def _link_transforms(self: Robot, q: ArrayLike):
    # Poses [T_0^1, ..., T_0^n] of every link frame.
    return _fk.link_transforms(self, q)


def _jacobian(self: Robot, q: ArrayLike, link: int = -1,
              point_local: ArrayLike | None = None):
    # Geometric Jacobian, (6, n), ordered [linear; angular].
    return _fk.jacobian(self, q, link=link, point_local=point_local)


def _jacobian_dot(self: Robot, q: ArrayLike, qd: ArrayLike,
                  link: int = -1):
    # J_dot(q, qd) by complex-step differentiation.
    return _fk.jacobian_dot(self, q, qd, link=link)
# Dynamics — the method= argument selects between independent backends
def _mass_matrix(self: Robot, q: ArrayLike,
                 method: str = 'crba'):
    # The mass matrix M(q).
    # method: 'crba' (default, O(n^2)), 'lagrangian' (from the kinetic
    # energy), or 'rnea' (unit-acceleration columns). All three agree to
    # machine precision; CRBA is simply the cheapest.
    if method == 'crba':
        return _crba.mass_matrix(self, q)
    if method == 'lagrangian':
        return _lag.mass_matrix(self, q)
    if method == 'rnea':
        return _rnea.mass_matrix(self, q)
    raise ValueError(f'unknown mass-matrix method {method!r}; expected '
                     "'crba', 'lagrangian' or 'rnea'")


def _coriolis(self: Robot, q: ArrayLike, qd: ArrayLike,
              method: str = 'lagrangian'):
    # The Coriolis/centrifugal matrix C(q, qd).
    # Only the Lagrangian backend produces the matrix; the recursive algorithms
    # naturally give the product C*qd, available as coriolis_vector().
    if method != 'lagrangian':
        raise ValueError('the Coriolis matrix is only available from the '
                         "'lagrangian' backend; use coriolis_vector() for RNEA")
    return _lag.coriolis(self, q, qd)


def _coriolis_vector(self: Robot, q: ArrayLike, qd: ArrayLike,
                     method: str = 'rnea'):
    # The vector C(q, qd)*qd.
    # 'rnea' (default) is exact and O(n). 'lagrangian' builds the
    # matrix first, which needs dM/dq.
    if method == 'rnea':
        return _rnea.coriolis_vector(self, q, qd)
    if method == 'lagrangian':
        return _lag.coriolis_vector(self, q, qd)
    raise ValueError(f'unknown coriolis method {method!r}')


def _gravity(self: Robot, q: ArrayLike, method: str = 'lagrangian'):
    # The gravity torque vector G(q).
    if method == 'lagrangian':
        return _lag.gravity(self, q)
    if method == 'rnea':
        return _rnea.gravity_torque(self, q)
    raise ValueError(f'unknown gravity method {method!r}')


def _inverse_dynamics(self: Robot, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike,
                      method: str = 'rnea',
                      external_wrenches: Mapping[Any, ArrayLike] | None = None
                      ):
    # Inverse dynamics: the torque needed to produce qdd.
    # tau = M(q)*qdd + C(q, qd)*qd + G(q) - J^T * f_ext
    # method: 'rnea' (default, O(n)) or 'lagrangian' (O(n^3)).
    if method == 'rnea':
        return _rnea.inverse_dynamics(self, q, qd, qdd,
                                      external_wrenches=external_wrenches)
    if method == 'lagrangian':
        return _lag.inverse_dynamics(self, q, qd, qdd,
                                     external_wrenches=external_wrenches)
    raise ValueError(f'unknown inverse-dynamics method {method!r}')


def _forward_dynamics(self: Robot, q: ArrayLike, qd: ArrayLike, tau: ArrayLike,
                      method: str = 'aba',
                      external_wrenches: Mapping[Any, ArrayLike] | None = None
                      ):
    # Forward dynamics: the acceleration produced by tau.
    # 'aba' (default) is O(n) and never forms M.
    # 'lagrangian' solves M*qdd = tau - C*qd - G by Cholesky factorisation.
    if method == 'aba':
        return _crba.forward_dynamics(self, q, qd, tau,
                                      external_wrenches=external_wrenches)
    if method == 'lagrangian':
        return _lag.forward_dynamics(self, q, qd, tau,
                                     external_wrenches=external_wrenches)
    raise ValueError(f'unknown forward-dynamics method {method!r}')


def _kinetic_energy(self: Robot, q: ArrayLike, qd: ArrayLike):
    # T = 0.5 * qd^T * M(q) * qd
    return _lag.kinetic_energy(self, q, qd)


def _potential_energy(self: Robot, q: ArrayLike):
    # V = - sum(m_i * g^T * p_c_i)
    return _lag.potential_energy(self, q)


def _total_energy(self: Robot, q: ArrayLike, qd: ArrayLike):
    # T + V. Conserved by a passive system — used as a simulation check.
    return _lag.kinetic_energy(self, q, qd) + _lag.potential_energy(self, q)


def _lagrangian_value(self: Robot, q: ArrayLike, qd: ArrayLike):
    # L = T - V
    return _lag.lagrangian(self, q, qd)


for _name, _fn in [
    ('forward_kinematics', _forward_kinematics),
    ('link_transforms', _link_transforms),
    ('jacobian', _jacobian),
    ('jacobian_dot', _jacobian_dot),
    ('mass_matrix', _mass_matrix),
    ('coriolis', _coriolis),
    ('coriolis_vector', _coriolis_vector),
    ('gravity_torque', _gravity),
    ('inverse_dynamics', _inverse_dynamics),
    ('forward_dynamics', _forward_dynamics),
    ('kinetic_energy', _kinetic_energy),
    ('potential_energy', _potential_energy),
    ('total_energy', _total_energy),
    ('lagrangian', _lagrangian_value),
]:
    setattr(Robot, _name, _fn)

__all__ = [
    'Joint',
    'Link',
    'Robot',
    '__version__',
    'analytic_two_dof',
    'bimanual_arm',
    'control',
    'dynamics',
    'kinematics',
    'models',
    'planar_chain',
    'prismatic_chain',
    'random_robot',
    'simulation',
    'spatial_chain',
    'trajectory',
    'two_dof_planar',
]
