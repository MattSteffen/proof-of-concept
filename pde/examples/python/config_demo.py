"""
Demonstration of the configuration system for automatic parameter calculation.

This example shows how to use the SolverConfig class to automatically
determine optimal solver parameters based on accuracy or speed preferences.
"""

import numpy as np
from pde_sdk.boundaries import DirichletBC
from pde_sdk.config import SolverConfig
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
import pde_sdk.solvers as solvers


def main():
    print("=" * 60)
    print("PDE SDK Configuration System Demo")
    print("=" * 60)

    # Example 1: High accuracy configuration
    print("\n1. High Accuracy Configuration")
    print("-" * 60)
    config_high = SolverConfig(
        target_accuracy=0.0001,  # 0.01% accuracy
        problem_type="heat_1d",
        alpha=0.01,
        domain_length=1.0,
    )

    print(f"Target accuracy: {config_high.target_accuracy}")
    print(f"Recommended nx: {config_high.nx}")
    print(f"Recommended dt: {config_high.dt:.2e}")
    print(f"Recommended solver: {config_high.solver_type}")

    # Example 2: Fast configuration
    print("\n2. Fast Configuration")
    print("-" * 60)
    config_fast = SolverConfig(
        speed_preference=0.8,  # 80% speed preference
        problem_type="heat_1d",
        alpha=0.01,
        domain_length=1.0,
    )

    print(f"Speed preference: {config_fast.speed_preference}")
    print(f"Recommended nx: {config_fast.nx}")
    print(f"Recommended dt: {config_fast.dt:.2e}")
    print(f"Recommended solver: {config_fast.solver_type}")

    # Example 3: Complete workflow with configuration
    print("\n3. Complete Workflow")
    print("-" * 60)
    config = SolverConfig(
        target_accuracy=0.001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
    )

    # Create grid and solver using configuration
    grid = UniformGrid1D(**config.get_grid_params())
    solver_class = getattr(solvers, config.solver_type)
    solver = solver_class(**config.get_solver_params())

    # Set up equation
    equation = HeatEquation1D(
        alpha=config.alpha,
        grid=grid,
        left_bc=DirichletBC(0.0),
        right_bc=DirichletBC(0.0),
        initial_condition=lambda x: np.sin(np.pi * x),
    )

    # Solve with progress tracking
    print(f"\nSolving with {config.solver_type}...")
    solution = solver.solve(equation, t_final=0.1, verbosity="summary")

    # Compare with analytical solution
    u_exact = np.exp(-np.pi**2 * config.alpha * 0.1) * np.sin(np.pi * grid.x)
    error = np.max(np.abs(solution - u_exact))
    print(f"\nMaximum error: {error:.2e}")
    print(f"Target accuracy: {config.target_accuracy:.2e}")

    # Example 4: Comparison of different configurations
    print("\n4. Configuration Comparison")
    print("-" * 60)
    configs = [
        SolverConfig(
            target_accuracy=0.01, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        ),
        SolverConfig(
            target_accuracy=0.001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        ),
        SolverConfig(
            target_accuracy=0.0001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        ),
    ]

    print(f"{'Accuracy':<12} {'nx':<6} {'dt':<12} {'Solver':<20}")
    print("-" * 60)
    for cfg in configs:
        print(
            f"{cfg.target_accuracy:<12.4f} {cfg.nx:<6} {cfg.dt:<12.2e} {cfg.solver_type:<20}"
        )

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

