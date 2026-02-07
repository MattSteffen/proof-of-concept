"""Solver configuration and recommendation engine."""

from typing import Literal

from .calculator import calculate_optimal_dt, calculate_optimal_nx


class SolverConfig:
    """
    Configuration class for automatic parameter calculation.

    Automatically calculates optimal solver parameters (nx, dt, solver type)
    based on user's accuracy and speed preferences.

    Parameters
    ----------
    target_accuracy : float, optional
        Target relative error (e.g., 0.001 for 0.1% accuracy).
        If None, uses speed_preference to determine.
    speed_preference : float, optional
        Value between 0.0 (prioritize accuracy) and 1.0 (prioritize speed).
        If None, uses target_accuracy to determine.
    problem_type : {'heat_1d', 'heat_2d', 'poisson_2d'}
        Type of PDE problem to solve
    alpha : float, optional
        Diffusion coefficient (required for heat equations)
    domain_length : float, default=1.0
        Length of the domain
    domain_length_y : float, optional
        Length in y-direction for 2D problems (defaults to domain_length)

    Examples
    --------
    >>> # High accuracy configuration
    >>> config = SolverConfig(target_accuracy=0.0001, problem_type='heat_1d', alpha=0.01)
    >>> print(f"Recommended nx: {config.nx}, dt: {config.dt:.2e}")
    Recommended nx: 101, dt: 5.00e-05

    >>> # Fast configuration
    >>> config = SolverConfig(speed_preference=0.9, problem_type='heat_1d', alpha=0.01)
    >>> print(f"Recommended solver: {config.solver_type}")
    Recommended solver: ExplicitEuler1D
    """

    def __init__(
        self,
        target_accuracy: float | None = None,
        speed_preference: float | None = None,
        problem_type: Literal["heat_1d", "heat_2d", "poisson_2d"] = "heat_1d",
        alpha: float | None = None,
        domain_length: float = 1.0,
        domain_length_y: float | None = None,
    ):
        if target_accuracy is None and speed_preference is None:
            raise ValueError("Must specify either target_accuracy or speed_preference")

        if target_accuracy is not None and speed_preference is not None:
            raise ValueError("Cannot specify both target_accuracy and speed_preference")

        self.problem_type = problem_type
        self.domain_length = domain_length
        self.domain_length_y = domain_length_y or domain_length

        # Determine dimension
        if problem_type == "heat_1d":
            self.dimension = 1
        elif problem_type in ("heat_2d", "poisson_2d"):
            self.dimension = 2
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

        # Convert between accuracy and speed preference if needed
        if target_accuracy is not None:
            self.target_accuracy = target_accuracy
            # Convert accuracy to speed preference (inverse relationship)
            # Higher accuracy -> lower speed preference
            self.speed_preference = 1.0 - min(1.0, target_accuracy * 100)
        else:
            self.speed_preference = max(0.0, min(1.0, speed_preference))
            # Convert speed preference to accuracy
            # Lower speed preference -> higher accuracy
            self.target_accuracy = (1.0 - self.speed_preference) * 0.01

        # Validate alpha for heat equations
        if problem_type.startswith("heat") and alpha is None:
            raise ValueError(f"alpha is required for {problem_type}")
        self.alpha = alpha

        # Calculate optimal parameters
        self._calculate_parameters()

    def _calculate_parameters(self):
        """Calculate optimal nx, dt, and solver type."""
        # Calculate optimal grid resolution
        self.nx = calculate_optimal_nx(
            self.target_accuracy, self.domain_length, self.dimension
        )
        if self.dimension == 2:
            self.ny = calculate_optimal_nx(
                self.target_accuracy, self.domain_length_y, self.dimension
            )
        else:
            self.ny = None

        # Determine solver type based on speed preference
        if self.problem_type == "heat_1d":
            if self.speed_preference > 0.7:
                self.solver_type = "ExplicitEuler1D"
                solver_str = "explicit"
            elif self.speed_preference > 0.4:
                self.solver_type = "CrankNicolson1D"
                solver_str = "crank_nicolson"
            else:
                self.solver_type = "BackwardEuler1D"
                solver_str = "implicit"

        elif self.problem_type == "heat_2d":
            if self.speed_preference > 0.7:
                self.solver_type = "ExplicitEuler2D"
                solver_str = "explicit"
            elif self.speed_preference > 0.4:
                self.solver_type = "CrankNicolson2D"
                solver_str = "crank_nicolson"
            else:
                self.solver_type = "BackwardEuler2D"
                solver_str = "implicit"

        elif self.problem_type == "poisson_2d":
            self.solver_type = "JacobiPoisson2D"
            solver_str = "iterative"
            # For Poisson, dt is not applicable
            self.dt = None
            return

        # Calculate optimal timestep
        dx = self.domain_length / (self.nx - 1)
        self.dt = calculate_optimal_dt(
            self.alpha, dx, solver_str, self.dimension
        )

    def get_grid_params(self) -> dict:
        """
        Get grid parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary with grid parameters (nx, ny if 2D, length, etc.)
        """
        if self.dimension == 1:
            return {
                "nx": self.nx,
                "length": self.domain_length,
            }
        else:
            return {
                "nx": self.nx,
                "ny": self.ny,
                "length_x": self.domain_length,
                "length_y": self.domain_length_y,
            }

    def get_solver_params(self) -> dict:
        """
        Get solver parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary with solver parameters (dt, max_iter, tol, etc.)
        """
        if self.problem_type == "poisson_2d":
            # For Poisson, estimate iterations needed based on accuracy
            max_iter = int(1000 / (1.0 - self.speed_preference))
            return {
                "max_iter": max_iter,
                "tol": self.target_accuracy,
            }
        else:
            return {"dt": self.dt}


def recommend_solver(
    problem_type: Literal["heat_1d", "heat_2d", "poisson_2d"],
    accuracy_requirement: Literal["low", "medium", "high"] = "medium",
    problem_size: Literal["small", "medium", "large"] = "medium",
) -> tuple[str, str]:
    """
    Recommend a solver based on problem characteristics.

    Parameters
    ----------
    problem_type : {'heat_1d', 'heat_2d', 'poisson_2d'}
        Type of PDE problem
    accuracy_requirement : {'low', 'medium', 'high'}
        Required accuracy level
    problem_size : {'small', 'medium', 'large'}
        Problem size (affects 2D recommendations)

    Returns
    -------
    tuple[str, str]
        Tuple of (solver_name, reason)

    Examples
    --------
    >>> solver, reason = recommend_solver('heat_1d', accuracy_requirement='high')
    >>> print(f"Recommended: {solver} - {reason}")
    Recommended: CrankNicolson1D - Good accuracy, reasonable cost
    """
    if problem_type == "heat_1d":
        if accuracy_requirement == "low":
            return "ExplicitEuler1D", "Fast, simple"
        elif accuracy_requirement == "medium":
            return "CrankNicolson1D", "Good accuracy, reasonable cost"
        else:  # high
            return "BackwardEuler1D", "Stable, good convergence"

    elif problem_type == "heat_2d":
        if problem_size == "small" and accuracy_requirement == "low":
            return "ExplicitEuler2D", "Fast for small problems"
        else:
            return "CrankNicolson2D", "Best balance of speed and accuracy"

    elif problem_type == "poisson_2d":
        return "JacobiPoisson2D", "Standard iterative solver"

    else:
        return "CrankNicolson1D", "Default choice"

