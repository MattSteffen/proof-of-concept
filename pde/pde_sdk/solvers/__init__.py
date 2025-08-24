"""
Solvers Module

Contains solver implementations like explicit Euler, backward Euler, Crank-Nicolson, and iterative solvers.
"""

from .backward_euler import BackwardEuler1D, BackwardEuler2D
from .crank_nicolson import CrankNicolson1D, CrankNicolson2D
from .explicit_euler import ExplicitEuler1D, ExplicitEuler2D
from .poisson_iterative import JacobiPoisson2D

__all__ = [
    "ExplicitEuler1D",
    "ExplicitEuler2D",
    "BackwardEuler1D",
    "BackwardEuler2D",
    "CrankNicolson1D",
    "CrankNicolson2D",
    "JacobiPoisson2D",
]
