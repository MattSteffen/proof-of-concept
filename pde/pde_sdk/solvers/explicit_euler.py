
from ..boundaries.dirichlet import DirichletBC
from ..utils.progress import ProgressTracker, create_progress_tracker


class ExplicitEuler1D:
    def __init__(self, dt: float):
        self.dt = dt

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 1D heat equation using explicit Euler method.

        Parameters
        ----------
        equation
            Heat equation instance
        t_final : float
            Final time
        progress : ProgressTracker, optional
            Progress tracker instance
        verbosity : {'none', 'summary', 'steps'}, optional
            Verbosity level (used if progress is None)

        Returns
        -------
        np.ndarray
            Solution array
        """
        grid = equation.grid
        dx = grid.dx
        alpha = equation.alpha
        r = alpha * self.dt / dx**2

        if r > 0.5:
            raise ValueError(f"Unstable timestep: r={r:.3f} > 0.5")

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Explicit Euler 1D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            new_values = grid.values.copy()
            for i in range(1, grid.nx - 1):
                new_values[i] = grid.values[i] + r * (
                    grid.values[i - 1] - 2 * grid.values[i] + grid.values[i + 1]
                )

            # Apply boundary conditions
            # Left boundary
            if isinstance(equation.left_bc, DirichletBC):
                new_values[0] = equation.left_bc.value
            else:
                # Neumann: ∂u/∂x = g, so u[0] = u[1] - g*dx
                new_values[0] = new_values[1] - equation.left_bc.value * dx

            # Right boundary
            if isinstance(equation.right_bc, DirichletBC):
                new_values[-1] = equation.right_bc.value
            else:
                # Neumann: ∂u/∂x = g, so u[-1] = u[-2] + g*dx
                new_values[-1] = new_values[-2] + equation.right_bc.value * dx

            grid.values = new_values
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values


class ExplicitEuler2D:
    def __init__(self, dt: float):
        self.dt = dt

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 2D heat equation using explicit Euler method.

        Parameters
        ----------
        equation
            Heat equation instance
        t_final : float
            Final time
        progress : ProgressTracker, optional
            Progress tracker instance
        verbosity : {'none', 'summary', 'steps'}, optional
            Verbosity level (used if progress is None)

        Returns
        -------
        np.ndarray
            Solution array
        """
        grid = equation.grid
        dx = grid.dx
        dy = grid.dy
        alpha = equation.alpha

        # Stability criteria for 2D heat equation
        r_x = alpha * self.dt / dx**2
        r_y = alpha * self.dt / dy**2
        r = max(r_x, r_y)

        if r > 0.25:  # More restrictive for 2D
            raise ValueError(f"Unstable timestep: r={r:.3f} > 0.25")

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Explicit Euler 2D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            new_values = grid.values.copy()

            # Interior points
            for i in range(1, grid.nx - 1):
                for j in range(1, grid.ny - 1):
                    laplacian = (
                        grid.values[i + 1, j]
                        - 2 * grid.values[i, j]
                        + grid.values[i - 1, j]
                    ) / dx**2 + (
                        grid.values[i, j + 1]
                        - 2 * grid.values[i, j]
                        + grid.values[i, j - 1]
                    ) / dy**2
                    new_values[i, j] = grid.values[i, j] + alpha * self.dt * laplacian

            # Apply boundary conditions
            # Left boundary (x=0)
            for j in range(grid.ny):
                if isinstance(equation.left_bc, DirichletBC):
                    new_values[0, j] = equation.left_bc.value
                else:
                    # Neumann: ∂u/∂x = g, so u[0,j] = u[1,j] - g*dx
                    new_values[0, j] = new_values[1, j] - equation.left_bc.value * dx

            # Right boundary (x=length_x)
            for j in range(grid.ny):
                if isinstance(equation.right_bc, DirichletBC):
                    new_values[-1, j] = equation.right_bc.value
                else:
                    # Neumann: ∂u/∂x = g, so u[-1,j] = u[-2,j] + g*dx
                    new_values[-1, j] = new_values[-2, j] + equation.right_bc.value * dx

            # Bottom boundary (y=0)
            for i in range(grid.nx):
                if isinstance(equation.bottom_bc, DirichletBC):
                    new_values[i, 0] = equation.bottom_bc.value
                else:
                    # Neumann: ∂u/∂y = g, so u[i,0] = u[i,1] - g*dy
                    new_values[i, 0] = new_values[i, 1] - equation.bottom_bc.value * dy

            # Top boundary (y=length_y)
            for i in range(grid.nx):
                if isinstance(equation.top_bc, DirichletBC):
                    new_values[i, -1] = equation.top_bc.value
                else:
                    # Neumann: ∂u/∂y = g, so u[i,-1] = u[i,-2] + g*dy
                    new_values[i, -1] = new_values[i, -2] + equation.top_bc.value * dy

            grid.values = new_values
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values
