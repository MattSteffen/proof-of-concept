
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from ..boundaries.dirichlet import DirichletBC
from ..utils.progress import ProgressTracker, create_progress_tracker


class BackwardEuler1D:
    """Backward Euler (implicit) solver for 1D heat equation.

    This solver uses an implicit time-stepping scheme that requires solving
    a linear system at each timestep. It's unconditionally stable, allowing
    larger timesteps than explicit methods.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 1D heat equation using backward Euler (implicit) method.

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

        # Build system matrix: (I - r*A)
        # A is the 1D Laplacian
        n = grid.nx
        diagonals = [
            (1 + 2 * r) * np.ones(n),  # main diagonal
            -r * np.ones(n - 1),  # lower diagonal
            -r * np.ones(n - 1),  # upper diagonal
        ]
        system_matrix = diags(diagonals, [0, -1, 1], format="csc")

        # Apply boundary conditions by modifying boundary rows
        system_matrix = system_matrix.tolil()

        # Left boundary
        if isinstance(equation.left_bc, DirichletBC):
            # Dirichlet: u[0] = value
            system_matrix[0, :] = 0
            system_matrix[0, 0] = 1
        else:
            # Neumann: (u[1] - u[0])/dx = g, so u[0] = u[1] - g*dx
            # Row 0: [-1, 1, 0, ...] scaled by 1/dx
            system_matrix[0, :] = 0
            system_matrix[0, 0] = -1.0 / dx
            system_matrix[0, 1] = 1.0 / dx

        # Right boundary
        if isinstance(equation.right_bc, DirichletBC):
            # Dirichlet: u[-1] = value
            system_matrix[-1, :] = 0
            system_matrix[-1, -1] = 1
        else:
            # Neumann: (u[-1] - u[-2])/dx = g, so u[-1] = u[-2] + g*dx
            # Row -1: [..., 0, -1/dx, 1/dx]
            system_matrix[-1, :] = 0
            system_matrix[-1, -2] = -1.0 / dx
            system_matrix[-1, -1] = 1.0 / dx

        system_matrix = system_matrix.tocsc()

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Backward Euler 1D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            b = grid.values.copy()

            # Enforce BCs in RHS
            if isinstance(equation.left_bc, DirichletBC):
                b[0] = equation.left_bc.value
            else:
                # Neumann: RHS = g (gradient value)
                b[0] = equation.left_bc.value

            if isinstance(equation.right_bc, DirichletBC):
                b[-1] = equation.right_bc.value
            else:
                # Neumann: RHS = g (gradient value)
                b[-1] = equation.right_bc.value

            # Solve linear system
            new_values = spsolve(system_matrix, b)

            grid.values = new_values
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values


class BackwardEuler2D:
    """Backward Euler (implicit) solver for 2D heat equation.

    This solver uses an implicit time-stepping scheme that requires solving
    a linear system at each timestep. It's unconditionally stable, allowing
    larger timesteps than explicit methods.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 2D heat equation using backward Euler (implicit) method.

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
        nx, ny = grid.nx, grid.ny
        dx, dy = grid.dx, grid.dy
        alpha = equation.alpha

        rx = alpha * self.dt / dx**2
        ry = alpha * self.dt / dy**2

        # 1D Laplacians
        ex = np.ones(nx)
        ey = np.ones(ny)

        tx = diags([ex * -rx, (1 + 2 * rx) * ex, ex * -rx], [-1, 0, 1], shape=(nx, nx))
        ty = diags([ey * -ry, (1 + 2 * ry) * ey, ey * -ry], [-1, 0, 1], shape=(ny, ny))

        # 2D Laplacian via Kronecker sum
        from scipy.sparse import identity, kron

        system_matrix = kron(identity(ny), tx) + kron(ty, identity(nx))

        # Convert to LIL for boundary modifications
        system_matrix = system_matrix.tolil()

        # Apply boundary conditions by modifying boundary rows
        # Left/right boundaries
        for j in range(ny):
            idx_left = j * nx
            idx_right = j * nx + (nx - 1)

            if isinstance(equation.left_bc, DirichletBC):
                # Dirichlet: u[0,j] = value
                system_matrix[idx_left, :] = 0
                system_matrix[idx_left, idx_left] = 1
            else:
                # Neumann: (u[1,j] - u[0,j])/dx = g
                system_matrix[idx_left, :] = 0
                system_matrix[idx_left, idx_left] = -1.0 / dx
                system_matrix[idx_left, idx_left + 1] = 1.0 / dx

            if isinstance(equation.right_bc, DirichletBC):
                # Dirichlet: u[-1,j] = value
                system_matrix[idx_right, :] = 0
                system_matrix[idx_right, idx_right] = 1
            else:
                # Neumann: (u[-1,j] - u[-2,j])/dx = g
                system_matrix[idx_right, :] = 0
                system_matrix[idx_right, idx_right - 1] = -1.0 / dx
                system_matrix[idx_right, idx_right] = 1.0 / dx

        # Bottom/top boundaries
        for i in range(nx):
            idx_bottom = i
            idx_top = (ny - 1) * nx + i

            if isinstance(equation.bottom_bc, DirichletBC):
                # Dirichlet: u[i,0] = value
                system_matrix[idx_bottom, :] = 0
                system_matrix[idx_bottom, idx_bottom] = 1
            else:
                # Neumann: (u[i,1] - u[i,0])/dy = g
                system_matrix[idx_bottom, :] = 0
                system_matrix[idx_bottom, idx_bottom] = -1.0 / dy
                system_matrix[idx_bottom, idx_bottom + nx] = 1.0 / dy

            if isinstance(equation.top_bc, DirichletBC):
                # Dirichlet: u[i,-1] = value
                system_matrix[idx_top, :] = 0
                system_matrix[idx_top, idx_top] = 1
            else:
                # Neumann: (u[i,-1] - u[i,-2])/dy = g
                system_matrix[idx_top, :] = 0
                system_matrix[idx_top, idx_top - nx] = -1.0 / dy
                system_matrix[idx_top, idx_top] = 1.0 / dy

        # Convert to CSC for solving
        system_matrix = system_matrix.tocsc()

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Backward Euler 2D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            b = grid.values.flatten()

            # Apply boundary conditions in RHS
            # Left/right boundaries
            for j in range(ny):
                idx_left = j * nx
                idx_right = j * nx + (nx - 1)

                if isinstance(equation.left_bc, DirichletBC):
                    b[idx_left] = equation.left_bc.value
                else:
                    b[idx_left] = equation.left_bc.value  # Neumann: gradient value

                if isinstance(equation.right_bc, DirichletBC):
                    b[idx_right] = equation.right_bc.value
                else:
                    b[idx_right] = equation.right_bc.value  # Neumann: gradient value

            # Bottom/top boundaries
            for i in range(nx):
                idx_bottom = i
                idx_top = (ny - 1) * nx + i

                if isinstance(equation.bottom_bc, DirichletBC):
                    b[idx_bottom] = equation.bottom_bc.value
                else:
                    b[idx_bottom] = equation.bottom_bc.value  # Neumann: gradient value

                if isinstance(equation.top_bc, DirichletBC):
                    b[idx_top] = equation.top_bc.value
                else:
                    b[idx_top] = equation.top_bc.value  # Neumann: gradient value

            # Solve system
            new_values = spsolve(system_matrix, b)
            grid.values = new_values.reshape((nx, ny))
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values
