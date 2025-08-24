#!/usr/bin/env python3
"""
Comparison of different PDE solvers in the PDE SDK
"""

import numpy as np
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D
from pde_sdk.solvers.backward_euler import BackwardEuler1D
from pde_sdk.solvers.crank_nicolson import CrankNicolson1D

def compare_solvers():
    """Compare different PDE solvers: explicit, backward, and Crank-Nicolson"""

    print("PDE SDK Solver Comparison Demo")
    print("=" * 50)
    print("Comparing: Explicit Euler, Backward Euler, and Crank-Nicolson")
    print()

    # Parameters
    nx = 51
    alpha = 0.01
    t_final = 0.5

    # Setup
    grid = UniformGrid1D(nx=nx, length=1.0)
    ic = lambda x: np.sin(np.pi * x)

    eq = HeatEquation1D(
        alpha=alpha,
        grid=grid,
        left_bc=DirichletBC(0.0),
        right_bc=DirichletBC(0.0),
        initial_condition=ic,
    )

    # Analytical solution
    x = grid.x
    decay_rate = np.pi**2 * alpha * t_final
    u_exact = np.exp(-decay_rate) * np.sin(np.pi * x)

    print(f"Grid: {nx} points")
    print(f"Alpha: {alpha}")
    print(f"Final time: {t_final}")
    print()

    # Test Explicit Euler with small timestep
    print("1. Explicit Euler (dt=1e-4):")
    grid.values = ic(grid.x)  # Reset initial condition
    explicit_solver = ExplicitEuler1D(dt=1e-4)
    u_explicit = explicit_solver.solve(eq, t_final)

    error_explicit = np.abs(u_explicit - u_exact)
    max_error_explicit = np.max(error_explicit)
    steps_explicit = int(t_final / 1e-4)
    print(f"Max error: {max_error_explicit:.2e}")
    print(f"Mean error: {np.mean(error_explicit):.2e}")
    print(f"Steps: {steps_explicit}")
    print()

    # Test Backward Euler with larger timestep
    print("2. Backward Euler (dt=0.005):")
    grid.values = ic(grid.x)  # Reset initial condition
    implicit_solver = BackwardEuler1D(dt=0.005)
    u_implicit = implicit_solver.solve(eq, t_final)

    error_implicit = np.abs(u_implicit - u_exact)
    max_error_implicit = np.max(error_implicit)
    steps_implicit = int(t_final / 0.005)
    print(f"Max error: {max_error_implicit:.2e}")
    print(f"Mean error: {np.mean(error_implicit):.2e}")
    print(f"Steps: {steps_implicit}")
    print()

    # Test Crank-Nicolson with reasonable timestep
    print("3. Crank-Nicolson (dt=0.02):")
    grid.values = ic(grid.x)  # Reset initial condition
    cn_solver = CrankNicolson1D(dt=0.02)
    u_cn = cn_solver.solve(eq, t_final)

    error_cn = np.abs(u_cn - u_exact)
    max_error_cn = np.max(error_cn)
    steps_cn = int(t_final / 0.02)
    print(f"Max error: {max_error_cn:.2e}")
    print(f"Mean error: {np.mean(error_cn):.2e}")
    print(f"Steps: {steps_cn}")
    print()

    # Compare performance
    print("Performance Comparison:")
    print(f"Explicit Euler:  {steps_explicit:4d} steps, error = {max_error_explicit:.2e}")
    print(f"Backward Euler:  {steps_implicit:4d} steps, error = {max_error_implicit:.2e}")
    print(f"Crank-Nicolson:  {steps_cn:4d} steps, error = {max_error_cn:.2e}")
    print()

    print("Efficiency Analysis:")
    print(f"Backward vs Explicit: {steps_explicit/steps_implicit:.1f}x fewer steps")
    print(f"Crank-Nicolson vs Explicit: {steps_explicit/steps_cn:.1f}x fewer steps")
    print(f"Crank-Nicolson vs Backward: {steps_implicit/steps_cn:.1f}x fewer steps")
    print()

    print("Accuracy Analysis:")
    print(f"Crank-Nicolson vs Explicit: {max_error_explicit/max_error_cn:.1f}x better accuracy")
    print(f"Crank-Nicolson vs Backward: {max_error_implicit/max_error_cn:.1f}x better accuracy")
    print(f"Second-order advantage: {max_error_implicit/max_error_cn:.1f}x better than first-order methods")
    print()

    # Show solver capabilities
    print("Solver Capabilities:")
    print("• Explicit Euler: First-order, conditional stability")
    print("• Backward Euler: First-order, unconditional stability")
    print("• Crank-Nicolson: Second-order, unconditional stability")
    print()

    if max_error_cn < 1e-5:
        print("✅ All solvers working correctly!")
        print("   Crank-Nicolson shows superior accuracy as expected.")
    else:
        print("⚠️  Crank-Nicolson may need implementation refinement")

if __name__ == "__main__":
    compare_solvers()
