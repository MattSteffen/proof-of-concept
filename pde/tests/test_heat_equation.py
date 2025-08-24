"""
Tests for the 1D heat equation implementation
"""

import numpy as np
import pytest

from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D


def sin_pi_x_initial_condition(x):
    """Initial condition: sin(πx)"""
    return np.sin(np.pi * x)


class TestUniformGrid1D:
    """Test the 1D uniform grid implementation"""

    def test_grid_creation(self):
        """Test basic grid creation"""
        nx = 10
        length = 2.0
        grid = UniformGrid1D(nx=nx, length=length)

        assert grid.nx == nx
        assert grid.length == length
        assert grid.dx == length / (nx - 1)
        assert len(grid.x) == nx
        assert len(grid.values) == nx
        assert np.allclose(grid.x[0], 0.0)
        assert np.allclose(grid.x[-1], length)

    def test_grid_spacing(self):
        """Test grid spacing calculation"""
        nx = 5
        length = 1.0
        grid = UniformGrid1D(nx=nx, length=length)

        expected_dx = 0.25  # (1.0 - 0) / (5 - 1) = 1/4 = 0.25
        assert np.allclose(grid.dx, expected_dx)

        # Check that points are evenly spaced
        for i in range(1, nx):
            assert np.allclose(grid.x[i] - grid.x[i-1], expected_dx)


class TestDirichletBC:
    """Test Dirichlet boundary conditions"""

    def test_dirichlet_bc(self):
        """Test Dirichlet boundary condition application"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bc = DirichletBC(0.0)

        result = bc.apply(values)

        assert np.allclose(result[0], 0.0)
        assert np.allclose(result[-1], 0.0)
        assert np.array_equal(result[1:-1], values[1:-1])


class TestHeatEquation1D:
    """Test the 1D heat equation implementation"""

    def test_equation_creation(self):
        """Test heat equation creation"""
        nx = 10
        grid = UniformGrid1D(nx=nx, length=1.0)
        left_bc = DirichletBC(0.0)
        right_bc = DirichletBC(1.0)
        ic = sin_pi_x_initial_condition

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        assert eq.alpha == 0.01
        assert eq.grid is grid
        assert eq.left_bc is left_bc
        assert eq.right_bc is right_bc

        # Check initial condition was applied
        expected_initial = np.sin(np.pi * grid.x)
        assert np.allclose(grid.values, expected_initial)


class TestExplicitEuler1D:
    """Test the Explicit Euler solver"""

    def test_solver_creation(self):
        """Test solver creation"""
        dt = 1e-4
        solver = ExplicitEuler1D(dt=dt)

        assert solver.dt == dt

    def test_stability_check(self):
        """Test stability check (CFL condition)"""
        nx = 10
        length = 1.0
        grid = UniformGrid1D(nx=nx, length=length)
        left_bc = DirichletBC(0.0)
        right_bc = DirichletBC(0.0)
        ic = sin_pi_x_initial_condition

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        # This should work (stable)
        dt_stable = 1e-5
        solver_stable = ExplicitEuler1D(dt=dt_stable)
        solver_stable.solve(eq, t_final=0.01)

        # This should raise an error (unstable)
        # For nx=10, dx=0.111, alpha=0.01, we need dt > 0.5 * dx^2 / alpha
        # dx^2 / alpha = 0.111^2 / 0.01 = 0.01232 / 0.01 = 1.232
        # So dt > 0.5 * 1.232 = 0.616 to be unstable
        dt_unstable = 1.0  # Definitely unstable
        solver_unstable = ExplicitEuler1D(dt=dt_unstable)

        with pytest.raises(ValueError, match="Unstable timestep"):
            solver_unstable.solve(eq, t_final=0.01)

    def test_heat_equation_accuracy(self):
        """Test accuracy against analytical solution"""
        nx = 51
        length = 1.0
        alpha = 0.01
        dt = 1e-4
        t_final = 0.1

        # Setup
        grid = UniformGrid1D(nx=nx, length=length)
        ic = sin_pi_x_initial_condition

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

        # Check accuracy
        error = np.abs(u_final - u_exact)
        max_error = np.max(error)

        # Should be reasonably accurate
        assert max_error < 1e-4, f"Max error too large: {max_error}"

        # Boundary conditions should be satisfied
        assert np.allclose(u_final[0], 0.0, atol=1e-10)
        assert np.allclose(u_final[-1], 0.0, atol=1e-10)

    def test_conservation_of_boundary_conditions(self):
        """Test that boundary conditions are maintained throughout simulation"""
        nx = 21
        length = 1.0
        alpha = 0.01
        dt = 1e-4
        t_final = 0.1

        # Setup with non-zero BCs
        grid = UniformGrid1D(nx=nx, length=length)
        ic = sin_pi_x_initial_condition

        eq = HeatEquation1D(
            alpha=alpha,
            grid=grid,
            left_bc=DirichletBC(0.5),  # Non-zero BC
            right_bc=DirichletBC(-0.3),  # Non-zero BC
            initial_condition=ic,
        )

        solver = ExplicitEuler1D(dt=dt)
        u_final = solver.solve(eq, t_final=t_final)

        # Boundary conditions should be exactly satisfied
        assert np.allclose(u_final[0], 0.5, atol=1e-10)
        assert np.allclose(u_final[-1], -0.3, atol=1e-10)
