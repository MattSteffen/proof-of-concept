
import numpy as np

from ..boundaries.dirichlet import DirichletBC
from ..utils.progress import ProgressTracker, create_progress_tracker


class JacobiPoisson2D:
    def __init__(self, max_iter=10000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self, equation, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 2D Poisson equation using Jacobi iterative method.

        Parameters
        ----------
        equation
            Poisson equation instance
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
        dx, dy = grid.dx, grid.dy
        rhs = equation.rhs

        u = np.zeros_like(rhs)  # initial guess

        # Precompute constants
        dx2, dy2 = dx**2, dy**2
        denom = 2 * (dx2 + dy2)

        # Setup progress tracking
        if progress is None and verbosity is not None:
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=self.max_iter,
                description="Jacobi Poisson 2D",
            )

        for it in range(self.max_iter):
            u_new = u.copy()

            # Jacobi update for interior points
            for i in range(1, grid.nx - 1):
                for j in range(1, grid.ny - 1):
                    u_new[i, j] = (
                        dy2 * (u[i - 1, j] + u[i + 1, j])
                        + dx2 * (u[i, j - 1] + u[i, j + 1])
                        + dx2 * dy2 * rhs[i, j]
                    ) / denom

            # Apply boundary conditions
            # Left boundary (x=0)
            for j in range(grid.ny):
                if isinstance(equation.left_bc, DirichletBC):
                    u_new[0, j] = equation.left_bc.value
                else:
                    # Neumann: (u[1,j] - u[0,j])/dx = g, so u[0,j] = u[1,j] - g*dx
                    u_new[0, j] = u_new[1, j] - equation.left_bc.value * dx

            # Right boundary (x=length_x)
            for j in range(grid.ny):
                if isinstance(equation.right_bc, DirichletBC):
                    u_new[-1, j] = equation.right_bc.value
                else:
                    # Neumann: (u[-1,j] - u[-2,j])/dx = g, so u[-1,j] = u[-2,j] + g*dx
                    u_new[-1, j] = u_new[-2, j] + equation.right_bc.value * dx

            # Bottom boundary (y=0)
            for i in range(grid.nx):
                if isinstance(equation.bottom_bc, DirichletBC):
                    u_new[i, 0] = equation.bottom_bc.value
                else:
                    # Neumann: (u[i,1] - u[i,0])/dy = g, so u[i,0] = u[i,1] - g*dy
                    u_new[i, 0] = u_new[i, 1] - equation.bottom_bc.value * dy

            # Top boundary (y=length_y)
            for i in range(grid.nx):
                if isinstance(equation.top_bc, DirichletBC):
                    u_new[i, -1] = equation.top_bc.value
                else:
                    # Neumann: (u[i,-1] - u[i,-2])/dy = g, so u[i,-1] = u[i,-2] + g*dy
                    u_new[i, -1] = u_new[i, -2] + equation.top_bc.value * dy

            # Convergence check
            error = np.linalg.norm(u_new - u, ord=np.inf)

            if progress is not None:
                progress.update(1, error=f"{error:.2e}")

            if error < self.tol:
                if progress is not None:
                    progress.print_summary(f"Converged in {it + 1} iterations with error {error:.2e}")
                    progress.close()
                else:
                    print(f"Converged in {it + 1} iterations with error {error:.2e}")
                break

            u = u_new

        if progress is not None and error >= self.tol:
            progress.print_summary(f"Reached max_iter={self.max_iter} with error {error:.2e}")
            progress.close()

        grid.values = u
        return u
