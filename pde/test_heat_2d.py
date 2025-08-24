#!/usr/bin/env python3
"""
Test script for the 2D heat equation implementation
"""

import numpy as np
from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.explicit_euler import ExplicitEuler2D

def test_2d_heat_equation():
    """Test the 2D heat equation implementation"""

    # Parameters
    nx, ny = 21, 21
    length_x, length_y = 1.0, 1.0
    alpha = 0.01
    dt = 5e-5
    t_final = 0.05

    # Setup 2D grid
    grid = UniformGrid2D(nx=nx, ny=ny, length_x=length_x, length_y=length_y)

    # Initial condition: sin(πx) * sin(πy)
    def initial_condition(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # Boundary conditions
    left_bc = DirichletBC(0.0)
    right_bc = DirichletBC(0.0)
    bottom_bc = DirichletBC(0.0)
    top_bc = DirichletBC(0.0)

    # Create 2D heat equation
    eq = HeatEquation2D(
        alpha=alpha,
        grid=grid,
        left_bc=left_bc,
        right_bc=right_bc,
        bottom_bc=bottom_bc,
        top_bc=top_bc,
        initial_condition=initial_condition,
    )

    # Solve
    solver = ExplicitEuler2D(dt=dt)
    u_final = solver.solve(eq, t_final=t_final)

    # Analytical solution: e^{-2π²αt} sin(πx) sin(πy)
    decay_rate = 2 * np.pi**2 * alpha * t_final
    u_exact = np.exp(-decay_rate) * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)

    # Calculate errors
    error = np.abs(u_final - u_exact)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print("2D Heat Equation Test Results:")
    print(f"Grid size: {nx} x {ny}")
    print(".2e")
    print(".2e")
    print(".6f")
    print(".6f")
    print(".2e")

    # Check accuracy
    assert max_error < 1e-4, f"Max error too large: {max_error}"

    # Check boundary conditions
    assert np.allclose(u_final[0, :], 0.0, atol=1e-10), "Left BC not satisfied"
    assert np.allclose(u_final[-1, :], 0.0, atol=1e-10), "Right BC not satisfied"
    assert np.allclose(u_final[:, 0], 0.0, atol=1e-10), "Bottom BC not satisfied"
    assert np.allclose(u_final[:, -1], 0.0, atol=1e-10), "Top BC not satisfied"

    print("✓ 2D heat equation test passed!")

    return u_final, u_exact, error

if __name__ == "__main__":
    test_2d_heat_equation()
