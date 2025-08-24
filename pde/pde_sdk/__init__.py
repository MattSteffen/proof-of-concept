"""
PDE SDK - Python Package

Finite-difference SDK for solving common PDEs on uniform 1D/2D grids.
Prototype fast in Python, then switch on a high-performance Rust backend.
"""

__version__ = "0.1.0"

from . import boundaries, domains, equations, rust_backend, solvers, visualization

__all__ = [
    "equations",
    "domains",
    "boundaries",
    "solvers",
    "visualization",
    "rust_backend",
]
