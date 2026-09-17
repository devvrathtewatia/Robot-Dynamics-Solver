"""Joint-space trajectory generation."""
from .polynomial import MultiPointTrajectory, PolynomialTrajectory, cubic, quintic

__all__ = ['MultiPointTrajectory', 'PolynomialTrajectory', 'cubic', 'quintic']
