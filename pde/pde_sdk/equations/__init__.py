"""
PDE Equations Module

Contains implementations of common PDE equations like heat and Poisson equations.
"""

from .heat import HeatEquation1D, HeatEquation2D
from .poisson import Poisson2D

__all__ = ["HeatEquation1D", "HeatEquation2D", "Poisson2D"]
