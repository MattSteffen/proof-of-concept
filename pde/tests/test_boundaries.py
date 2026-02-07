"""
Tests for boundary conditions, including Neumann BCs
"""

import numpy as np

from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.boundaries.neumann import NeumannBC
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.equations.heat import HeatEquation1D, HeatEquation2D
from pde_sdk.equations.poisson import Poisson2D
from pde_sdk.solvers.backward_euler import BackwardEuler1D
from pde_sdk.solvers.crank_nicolson import CrankNicolson1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D, ExplicitEuler2D
from pde_sdk.solvers.poisson_iterative import JacobiPoisson2D


class TestNeumannBC:
    """Test Neumann boundary condition class"""

    def test_neumann_bc_creation(self):
        """Test NeumannBC creation"""
        bc = NeumannBC(0.5)
        assert bc.value == 0.5
        assert isinstance(bc, NeumannBC)

    def test_neumann_bc_apply(self):
        """Test NeumannBC apply method (placeholder)"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bc = NeumannBC(0.0)
        result = bc.apply(values)
        # Apply is a placeholder, should return values unchanged
        assert np.array_equal(result, values)


class TestNeumannBCWithExplicitEuler1D:
    """Test Neumann BCs with ExplicitEuler1D solver"""

    def test_neumann_bc_left_boundary(self):
        """Test Neumann BC on left boundary"""
        nx = 21
        grid = UniformGrid1D(nx=nx, length=1.0)
        dx = grid.dx

        # Zero Neumann BC on left, Dirichlet on right
        left_bc = NeumannBC(0.0)  # Zero gradient
        right_bc = DirichletBC(0.0)

        # Initial condition: constant (should remain constant with zero Neumann)
        def ic(x):
            return np.ones_like(x)

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        solver = ExplicitEuler1D(dt=1e-5)
        u_final = solver.solve(eq, t_final=0.01)

        # With zero Neumann on left and Dirichlet zero on right,
        # solution should decay but left boundary should satisfy gradient constraint
        # Check that gradient at left boundary is approximately zero
        gradient_left = (u_final[1] - u_final[0]) / dx
        assert np.abs(gradient_left) < 1e-3

    def test_neumann_bc_both_boundaries(self):
        """Test Neumann BCs on both boundaries"""
        nx = 21
        grid = UniformGrid1D(nx=nx, length=1.0)
        dx = grid.dx

        left_bc = NeumannBC(0.0)
        right_bc = NeumannBC(0.0)

        def ic(x):
            return np.sin(np.pi * x)

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        solver = ExplicitEuler1D(dt=1e-5)
        u_final = solver.solve(eq, t_final=0.01)

        # Check Neumann BCs are satisfied
        gradient_left = (u_final[1] - u_final[0]) / dx
        gradient_right = (u_final[-1] - u_final[-2]) / dx
        assert np.abs(gradient_left) < 1e-3
        assert np.abs(gradient_right) < 1e-3


class TestNeumannBCWithBackwardEuler1D:
    """Test Neumann BCs with BackwardEuler1D solver"""

    def test_neumann_bc_implicit_solver(self):
        """Test Neumann BCs with implicit solver"""
        nx = 21
        grid = UniformGrid1D(nx=nx, length=1.0)
        dx = grid.dx

        left_bc = NeumannBC(0.0)
        right_bc = DirichletBC(0.0)

        def ic(x):
            return np.sin(np.pi * x)

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        solver = BackwardEuler1D(dt=1e-4)
        u_final = solver.solve(eq, t_final=0.01)

        # Check Neumann BC is satisfied
        gradient_left = (u_final[1] - u_final[0]) / dx
        assert np.abs(gradient_left) < 1e-3


class TestNeumannBCWithCrankNicolson1D:
    """Test Neumann BCs with CrankNicolson1D solver"""

    def test_neumann_bc_crank_nicolson(self):
        """Test Neumann BCs with Crank-Nicolson solver"""
        nx = 21
        grid = UniformGrid1D(nx=nx, length=1.0)
        dx = grid.dx

        left_bc = NeumannBC(0.0)
        right_bc = DirichletBC(0.0)

        def ic(x):
            return np.sin(np.pi * x)

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        solver = CrankNicolson1D(dt=1e-4)
        u_final = solver.solve(eq, t_final=0.01)

        # Check Neumann BC is satisfied
        gradient_left = (u_final[1] - u_final[0]) / dx
        assert np.abs(gradient_left) < 1e-3


class TestNeumannBCWithExplicitEuler2D:
    """Test Neumann BCs with ExplicitEuler2D solver"""

    def test_neumann_bc_2d(self):
        """Test Neumann BCs in 2D"""
        nx, ny = 21, 21
        grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)
        dx, _dy = grid.dx, grid.dy

        # Neumann on left, Dirichlet on others
        left_bc = NeumannBC(0.0)
        right_bc = DirichletBC(0.0)
        bottom_bc = DirichletBC(0.0)
        top_bc = DirichletBC(0.0)

        def ic(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        eq = HeatEquation2D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            top_bc=top_bc,
            initial_condition=ic
        )

        solver = ExplicitEuler2D(dt=1e-6)
        u_final = solver.solve(eq, t_final=0.01)

        # Check Neumann BC on left boundary
        for j in range(ny):
            gradient = (u_final[1, j] - u_final[0, j]) / dx
            assert np.abs(gradient) < 1e-2


class TestMixedBoundaryConditions:
    """Test mixed Dirichlet and Neumann boundary conditions"""

    def test_mixed_bc_1d(self):
        """Test mixed BCs in 1D"""
        nx = 21
        grid = UniformGrid1D(nx=nx, length=1.0)

        left_bc = DirichletBC(1.0)
        right_bc = NeumannBC(0.0)

        def ic(x):
            return np.ones_like(x)

        eq = HeatEquation1D(
            alpha=0.01,
            grid=grid,
            left_bc=left_bc,
            right_bc=right_bc,
            initial_condition=ic
        )

        solver = ExplicitEuler1D(dt=1e-5)
        u_final = solver.solve(eq, t_final=0.01)

        # Check Dirichlet BC
        assert np.allclose(u_final[0], 1.0, atol=1e-10)

        # Check Neumann BC
        dx = grid.dx
        gradient_right = (u_final[-1] - u_final[-2]) / dx
        assert np.abs(gradient_right) < 1e-3


class TestNeumannBCWithPoisson:
    """Test Neumann BCs with Poisson equation"""

    def test_poisson_neumann_bc(self):
        """Test Poisson equation with Neumann BCs"""
        nx, ny = 21, 21
        grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)
        dx, dy = grid.dx, grid.dy

        # Zero source term
        def source(x, y):
            return np.zeros_like(x)

        # Neumann BCs on all boundaries
        left_bc = NeumannBC(0.0)
        right_bc = NeumannBC(0.0)
        bottom_bc = NeumannBC(0.0)
        top_bc = NeumannBC(0.0)

        eq = Poisson2D(
            grid=grid,
            f=source,
            left_bc=left_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            top_bc=top_bc
        )

        solver = JacobiPoisson2D(max_iter=1000, tol=1e-6)
        u_final = solver.solve(eq)

        # With zero source and zero Neumann BCs, solution should be constant
        # Check that gradients are approximately zero
        for j in range(ny):
            gradient_left = (u_final[1, j] - u_final[0, j]) / dx
            gradient_right = (u_final[-1, j] - u_final[-2, j]) / dx
            assert np.abs(gradient_left) < 1e-3
            assert np.abs(gradient_right) < 1e-3

        for i in range(nx):
            gradient_bottom = (u_final[i, 1] - u_final[i, 0]) / dy
            gradient_top = (u_final[i, -1] - u_final[i, -2]) / dy
            assert np.abs(gradient_bottom) < 1e-3
            assert np.abs(gradient_top) < 1e-3

