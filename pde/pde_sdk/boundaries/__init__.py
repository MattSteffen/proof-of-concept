"""
Boundary Conditions Module

Contains boundary condition implementations like Dirichlet and Neumann.
"""

from .base import BoundaryCondition
from .dirichlet import DirichletBC

__all__ = ["BoundaryCondition", "DirichletBC"]
