"""
Visualization Module

Contains plotting and visualization utilities for PDE solutions.
"""

from .plot import animate_solution, plot_1d, plot_2d, plot_comparison, plot_error

__all__ = [
    "plot_1d",
    "plot_2d",
    "plot_comparison",
    "plot_error",
    "animate_solution",
]
