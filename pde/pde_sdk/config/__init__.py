"""
Configuration Module

Provides automatic parameter calculation and solver recommendations based on
user's accuracy and speed preferences.
"""

from .calculator import calculate_optimal_dt, calculate_optimal_nx
from .recommendations import SolverConfig, recommend_solver

__all__ = [
    "SolverConfig",
    "recommend_solver",
    "calculate_optimal_dt",
    "calculate_optimal_nx",
]

