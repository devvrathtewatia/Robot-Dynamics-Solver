"""``robot_dynamics`` — general N-DOF rigid-body dynamics for serial manipulators.

Implements the equation of motion

.. math::

    \\tau = M(q)\\,\\ddot q + C(q,\\dot q)\\,\\dot q + G(q)
           - J^\\top f_{\\text{ext}}

by two independent routes — the Lagrangian formulation and Featherstone's recursive
algorithms — so that each can validate the other.

Quick start
-----------

>>> >>> from robot_dynamics import planar_chain
>>> robot = planar_chain(3)
>>> q = np.array([0.1, -0.4, 0.7])
>>> qd = np.array([0.2, 0.1, -0.3])
>>> qdd = np.array([0.5, -0.2, 0.1])
>>> tau = robot.inverse_dynamics(q, qd, qdd)
>>> bool(np.allclose(robot.forward_dynamics(q, qd, tau), qdd))
True

Design notes
------------
The dynamics functions are free functions in :mod:`robot_dynamics.dynamics` and are
bound onto :class:`~robot_dynamics.models.robot.Robot` here. That keeps the model
class free of algorithm code while still giving the ergonomic
``robot.mass_matrix(q)`` API, and it makes the backend choice an explicit
``method=`` argument rather than a hidden global.
"""
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


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------
def _forward_kinematics(self: Robot, q: ArrayLike) -> NDArray:
    r"""End-effector pose :math:`T_0^n(q)` as a 4x4 homogeneous transform."""
    return _fk.forward_kinematics(self, q)


def _link_transforms(self: Robot, q: ArrayLike) -> list[NDArray]:
    r"""Poses :math:`[T_0^1, \ldots, T_0^n]` of every link frame."""
    return _fk.link_transforms(self, q)


def _jacobian(self: Robot, q: ArrayLike, link: int = -1,
              point_local: ArrayLike | None = None) -> NDArray:
    r"""Geometric Jacobian, ``(6, n)``, ordered ``[linear; angular]``."""
    return _fk.jacobian(self, q, link=link, point_local=point_local)


def _jacobian_dot(self: Robot, q: ArrayLike, qd: ArrayLike,
                  link: int = -1) -> NDArray:
    r""":math:`\dot J(q, \dot q)` by complex-step differentiation."""
    return _fk.jacobian_dot(self, q, qd, link=link)


# --------------------------------------------------------------------------
# Dynamics — the method= argument selects between independent backends
# --------------------------------------------------------------------------
def _mass_matrix(self: Robot, q: ArrayLike,
                 method: str = 'crba') -> NDArray:
    r"""The mass matrix :math:`M(q)`.

    Parameters
    ----------
    method:
        ``'crba'`` (default, :math:`O(n^2)`), ``'lagrangian'`` (from the kinetic
        energy), or ``'rnea'`` (unit-acceleration columns). All three agree to
        machine precision; CRBA is simply the cheapest.
    """
    if method == 'crba':
        return _crba.mass_matrix(self, q)
    if method == 'lagrangian':
        return _lag.mass_matrix(self, q)
    if method == 'rnea':
        return _rnea.mass_matrix(self, q)
    raise ValueError(f'unknown mass-matrix method {method!r}; expected '
                     "'crba', 'lagrangian' or 'rnea'")


def _coriolis(self: Robot, q: ArrayLike, qd: ArrayLike,
              method: str = 'lagrangian') -> NDArray:
    r"""The Coriolis/centrifugal **matrix** :math:`C(q,\dot q)`.

    Only the Lagrangian backend produces the matrix; the recursive algorithms
    naturally give the product :math:`C\dot q`, available as
    :meth:`coriolis_vector`. The Christoffel factorisation used here is the one for
    which :math:`\dot M - 2C` is skew-symmetric.
    """
    if method != 'lagrangian':
        raise ValueError('the Coriolis matrix is only available from the '
                         "'lagrangian' backend; use coriolis_vector() for RNEA")
    return _lag.coriolis(self, q, qd)


def _coriolis_vector(self: Robot, q: ArrayLike, qd: ArrayLike,
                     method: str = 'rnea') -> NDArray:
    r"""The vector :math:`C(q,\dot q)\,\dot q`.

    ``'rnea'`` (default) is exact and :math:`O(n)`. ``'lagrangian'`` builds the
    matrix first, which needs :math:`\partial M/\partial q`.
    """
    if method == 'rnea':
        return _rnea.coriolis_vector(self, q, qd)
    if method == 'lagrangian':
        return _lag.coriolis_vector(self, q, qd)
    raise ValueError(f'unknown coriolis method {method!r}')


def _gravity(self: Robot, q: ArrayLike, method: str = 'lagrangian') -> NDArray:
    r"""The gravity torque vector :math:`G(q)`."""
    if method == 'lagrangian':
        return _lag.gravity(self, q)
    if method == 'rnea':
        return _rnea.gravity_torque(self, q)
    raise ValueError(f'unknown gravity method {method!r}')


def _inverse_dynamics(self: Robot, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike,
                      method: str = 'rnea',
                      external_wrenches: Mapping[Any, ArrayLike] | None = None
                      ) -> NDArray:
    r"""Inverse dynamics: the torque needed to produce :math:`\ddot q`.

    .. math:: \tau = M(q)\ddot q + C(q,\dot q)\dot q + G(q) - J^\top f_{\text{ext}}

    Parameters
    ----------
    method:
        ``'rnea'`` (default, :math:`O(n)`) or ``'lagrangian'`` (:math:`O(n^3)`,
        assembles the terms explicitly).
    external_wrenches:
        Optional ``{link index or name: [fx, fy, fz, nx, ny, nz]}`` in the base
        frame, applied at that link's frame origin.
    """
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
                      ) -> NDArray:
    r"""Forward dynamics: the acceleration produced by :math:`\tau`.

    ``'aba'`` (default) is :math:`O(n)` and never forms :math:`M`.
    ``'lagrangian'`` solves :math:`M\ddot q = \tau - C\dot q - G` by Cholesky
    factorisation — a linear solve, never an explicit :math:`M^{-1}`.
    """
    if method == 'aba':
        return _crba.forward_dynamics(self, q, qd, tau,
                                      external_wrenches=external_wrenches)
    if method == 'lagrangian':
        return _lag.forward_dynamics(self, q, qd, tau,
                                     external_wrenches=external_wrenches)
    raise ValueError(f'unknown forward-dynamics method {method!r}')


def _kinetic_energy(self: Robot, q: ArrayLike, qd: ArrayLike) -> float:
    r""":math:`T = \tfrac12\dot q^\top M(q)\dot q`."""
    return _lag.kinetic_energy(self, q, qd)


def _potential_energy(self: Robot, q: ArrayLike) -> float:
    r""":math:`V = -\sum_i m_i g^\top p_{c_i}`."""
    return _lag.potential_energy(self, q)


def _total_energy(self: Robot, q: ArrayLike, qd: ArrayLike) -> float:
    r""":math:`T + V`. Conserved by a passive system — used as a simulation check."""
    return _lag.kinetic_energy(self, q, qd) + _lag.potential_energy(self, q)


def _lagrangian_value(self: Robot, q: ArrayLike, qd: ArrayLike) -> float:
    r""":math:`L = T - V`."""
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
