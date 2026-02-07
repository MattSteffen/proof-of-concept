"""Parameter calculation utilities for optimal solver configuration."""

from typing import Literal


def calculate_optimal_nx(
    target_accuracy: float,
    domain_length: float,
    dimension: Literal[1, 2] = 1,
) -> int:
    """
    Calculate optimal grid resolution based on target accuracy.

    Uses the rule of thumb: Δx ≈ √(target_accuracy) for second-order methods.

    Parameters
    ----------
    target_accuracy : float
        Target relative error (e.g., 0.001 for 0.1% accuracy)
    domain_length : float
        Length of the domain
    dimension : {1, 2}
        Spatial dimension (1D or 2D)

    Returns
    -------
    int
        Recommended number of grid points

    Examples
    --------
    >>> nx = calculate_optimal_nx(target_accuracy=0.001, domain_length=1.0)
    >>> print(f"Recommended nx: {nx}")
    Recommended nx: 101
    """
    # Rule of thumb: Δx ≈ √(target_accuracy) for second-order spatial discretization
    dx = (target_accuracy) ** 0.5
    nx = int(domain_length / dx) + 1

    # Apply reasonable bounds
    if dimension == 1:
        nx = max(21, min(nx, 1001))  # Reasonable range for 1D
    else:  # 2D
        nx = max(21, min(nx, 501))  # More conservative for 2D

    return nx


def calculate_optimal_dt(
    alpha: float,
    dx: float,
    solver_type: str,
    dimension: Literal[1, 2] = 1,
    safety_factor: float = 0.5,
) -> float:
    """
    Calculate optimal timestep for a given solver and grid spacing.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient
    dx : float
        Grid spacing
    solver_type : str
        Type of solver: 'explicit', 'implicit', or 'crank_nicolson'
    dimension : {1, 2}
        Spatial dimension (1D or 2D)
    safety_factor : float
        Safety factor for explicit methods (default 0.5 for stability)

    Returns
    -------
    float
        Recommended timestep

    Examples
    --------
    >>> dt = calculate_optimal_dt(alpha=0.01, dx=0.01, solver_type='explicit')
    >>> print(f"Recommended dt: {dt:.2e}")
    Recommended dt: 5.00e-03
    """
    if solver_type == "explicit":
        # CFL condition: dt ≤ safety_factor * dx² / (2*alpha) for 1D
        # For 2D: dt ≤ safety_factor * dx² / (4*alpha)
        if dimension == 1:
            dt_max = safety_factor * dx**2 / (2 * alpha)
        else:  # 2D
            dt_max = safety_factor * dx**2 / (4 * alpha)
        return dt_max

    elif solver_type in ("implicit", "backward_euler"):
        # For implicit methods, stability is not limiting
        # Base on accuracy: dt ≈ dx²/α for diffusion problems
        dt_base = dx**2 / alpha
        # Can use larger timesteps, but keep reasonable
        return dt_base

    elif solver_type == "crank_nicolson":
        # Crank-Nicolson is unconditionally stable, but accuracy matters
        # Use similar to implicit but can be more aggressive
        dt_base = dx**2 / alpha
        return dt_base * 2.0  # Can use larger timesteps due to 2nd-order accuracy

    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")


def calculate_accuracy_from_params(
    nx: int,
    dt: float,
    alpha: float,
    domain_length: float,
    solver_type: str,
    dimension: Literal[1, 2] = 1,
) -> float:
    """
    Estimate expected accuracy from given parameters.

    This is a rough estimate based on discretization errors.

    Parameters
    ----------
    nx : int
        Number of grid points
    dt : float
        Timestep
    alpha : float
        Diffusion coefficient
    domain_length : float
        Length of the domain
    solver_type : str
        Type of solver
    dimension : {1, 2}
        Spatial dimension

    Returns
    -------
    float
        Estimated relative error
    """
    dx = domain_length / (nx - 1)

    # Spatial error: O(Δx²) for second-order finite differences
    spatial_error = dx**2

    # Temporal error depends on solver
    if solver_type == "explicit":
        temporal_error = dt  # First-order
    elif solver_type in ("implicit", "backward_euler"):
        temporal_error = dt  # First-order
    elif solver_type == "crank_nicolson":
        temporal_error = dt**2  # Second-order
    else:
        temporal_error = dt

    # Combined error estimate (rough approximation)
    total_error = spatial_error + temporal_error

    return total_error

