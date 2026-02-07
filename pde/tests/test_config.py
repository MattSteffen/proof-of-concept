"""Tests for configuration system."""

import pytest

from pde_sdk.config import (
    SolverConfig,
    calculate_optimal_dt,
    calculate_optimal_nx,
    recommend_solver,
)


class TestCalculateOptimalNx:
    def test_basic_calculation(self):
        """Test basic nx calculation."""
        nx = calculate_optimal_nx(target_accuracy=0.01, domain_length=1.0)
        assert nx >= 21
        assert nx <= 1001

    def test_high_accuracy(self):
        """Test high accuracy requirement."""
        nx_high = calculate_optimal_nx(target_accuracy=0.0001, domain_length=1.0)
        nx_low = calculate_optimal_nx(target_accuracy=0.01, domain_length=1.0)
        assert nx_high >= nx_low  # Higher accuracy should require more points

    def test_2d_dimension(self):
        """Test 2D dimension handling."""
        nx_1d = calculate_optimal_nx(target_accuracy=0.01, domain_length=1.0, dimension=1)
        nx_2d = calculate_optimal_nx(target_accuracy=0.01, domain_length=1.0, dimension=2)
        # 2D should be more conservative
        assert nx_2d <= nx_1d


class TestCalculateOptimalDt:
    def test_explicit_1d(self):
        """Test explicit method timestep calculation."""
        dt = calculate_optimal_dt(alpha=0.01, dx=0.01, solver_type="explicit", dimension=1)
        assert dt > 0
        # Should satisfy CFL: dt <= 0.5 * dx² / (2*alpha)
        assert dt <= 0.5 * 0.01**2 / (2 * 0.01)

    def test_explicit_2d(self):
        """Test explicit method timestep for 2D."""
        dt = calculate_optimal_dt(alpha=0.01, dx=0.01, solver_type="explicit", dimension=2)
        assert dt > 0
        # Should satisfy CFL: dt <= safety_factor * dx² / (4*alpha) where safety_factor=0.5
        # So: dt <= 0.5 * 0.01² / (4 * 0.01) = 0.5 * 0.0001 / 0.04 = 0.00125
        assert dt <= 0.5 * 0.01**2 / (4 * 0.01)

    def test_implicit(self):
        """Test implicit method timestep."""
        dt = calculate_optimal_dt(alpha=0.01, dx=0.01, solver_type="implicit", dimension=1)
        assert dt > 0
        # Implicit can use larger timesteps
        assert dt >= 0.01**2 / 0.01

    def test_crank_nicolson(self):
        """Test Crank-Nicolson timestep."""
        dt = calculate_optimal_dt(alpha=0.01, dx=0.01, solver_type="crank_nicolson", dimension=1)
        assert dt > 0


class TestSolverConfig:
    def test_target_accuracy(self):
        """Test configuration with target accuracy."""
        config = SolverConfig(
            target_accuracy=0.001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        )
        assert config.nx >= 21
        assert config.dt > 0
        assert config.solver_type in ("ExplicitEuler1D", "CrankNicolson1D", "BackwardEuler1D")

    def test_speed_preference(self):
        """Test configuration with speed preference."""
        config = SolverConfig(
            speed_preference=0.8, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        )
        assert config.nx >= 21
        assert config.dt > 0
        # High speed preference should favor explicit methods
        assert config.solver_type in ("ExplicitEuler1D", "CrankNicolson1D", "BackwardEuler1D")

    def test_heat_2d(self):
        """Test 2D heat equation configuration."""
        config = SolverConfig(
            target_accuracy=0.001, problem_type="heat_2d", alpha=0.01, domain_length=1.0
        )
        assert config.nx >= 21
        assert config.ny >= 21
        assert config.dt > 0

    def test_poisson_2d(self):
        """Test Poisson equation configuration."""
        config = SolverConfig(
            target_accuracy=0.001, problem_type="poisson_2d", domain_length=1.0
        )
        assert config.nx >= 21
        assert config.ny >= 21
        assert config.dt is None  # Poisson doesn't use dt
        assert config.solver_type == "JacobiPoisson2D"

    def test_get_grid_params(self):
        """Test grid parameters retrieval."""
        config = SolverConfig(
            target_accuracy=0.001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        )
        params = config.get_grid_params()
        assert "nx" in params
        assert "length" in params

    def test_get_solver_params(self):
        """Test solver parameters retrieval."""
        config = SolverConfig(
            target_accuracy=0.001, problem_type="heat_1d", alpha=0.01, domain_length=1.0
        )
        params = config.get_solver_params()
        assert "dt" in params

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Both target_accuracy and speed_preference
        with pytest.raises(ValueError):
            SolverConfig(
                target_accuracy=0.001,
                speed_preference=0.5,
                problem_type="heat_1d",
                alpha=0.01,
            )

        # Neither specified
        with pytest.raises(ValueError):
            SolverConfig(problem_type="heat_1d", alpha=0.01)

        # Missing alpha for heat equation
        with pytest.raises(ValueError):
            SolverConfig(target_accuracy=0.001, problem_type="heat_1d")


class TestRecommendSolver:
    def test_heat_1d_low_accuracy(self):
        """Test solver recommendation for low accuracy."""
        solver, reason = recommend_solver("heat_1d", accuracy_requirement="low")
        assert solver == "ExplicitEuler1D"
        assert len(reason) > 0

    def test_heat_1d_high_accuracy(self):
        """Test solver recommendation for high accuracy."""
        solver, reason = recommend_solver("heat_1d", accuracy_requirement="high")
        assert solver == "BackwardEuler1D"
        assert len(reason) > 0

    def test_heat_2d(self):
        """Test solver recommendation for 2D heat."""
        solver, reason = recommend_solver("heat_2d", accuracy_requirement="medium")
        assert solver == "CrankNicolson2D"
        assert len(reason) > 0

    def test_poisson_2d(self):
        """Test solver recommendation for Poisson."""
        solver, reason = recommend_solver("poisson_2d")
        assert solver == "JacobiPoisson2D"
        assert len(reason) > 0

