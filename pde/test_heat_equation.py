#!/usr/bin/env python3
"""
Test script for the 1D heat equation implementation
"""

import numpy as np
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

def test_heat_equation():
    """Test the 1D heat equation implementation"""

    # Parameters
    nx = 51
    length = 1.0
    alpha = 0.01
    dt = 1e-4
    t_final = 0.1

    # Setup
    grid = UniformGrid1D(nx=nx, length=length)
    ic = lambda x: np.sin(np.pi * x)

    eq = HeatEquation1D(
        alpha=alpha,
        grid=grid,
        left_bc=DirichletBC(0.0),
        right_bc=DirichletBC(0.0),
        initial_condition=ic,
    )

    solver = ExplicitEuler1D(dt=dt)
    u_final = solver.solve(eq, t_final=t_final)

    # Analytical solution
    x = grid.x
    u_exact = np.exp(-np.pi**2 * alpha * t_final) * np.sin(np.pi * x)

    # Calculate error
    error = np.abs(u_final - u_exact)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print("1D Heat Equation Test Results:")
    print(f"Grid points: {nx}")
    print(".2e")
    print(".2e")
    print(".6f")
    print(".6f")
    print(".2e")

    # Check if solution is reasonable
    assert max_error < 0.01, f"Max error too large: {max_error}"
    assert np.allclose(u_final[0], 0.0, atol=1e-10), "Left BC not satisfied"
    assert np.allclose(u_final[-1], 0.0, atol=1e-10), "Right BC not satisfied"

    print("✓ All tests passed!")

    return u_final, u_exact, error

if __name__ == "__main__":
    test_heat_equation()
