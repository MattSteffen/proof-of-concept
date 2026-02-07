
import numpy as np
from scipy.sparse import csc_matrix, diags, identity, kron
from scipy.sparse.linalg import factorized

from ..boundaries.dirichlet import DirichletBC
from ..utils.progress import ProgressTracker, create_progress_tracker


class CrankNicolson1D:
    """
    Crank–Nicolson solver for the 1D heat equation.

    - Second-order accurate in time
    - Unconditionally stable
    - Requires solving a tridiagonal system at each timestep
    - Uses matrix factorization reuse for efficiency
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._A = None  # Left-hand matrix (I - rA)
        self._B = None  # Right-hand matrix (I + rA)
        self._solveA = None  # Cached solver for A
        self._shape_key = None  # Cache key to detect when grid/params change

    def _build_matrices(self, nx: int, dx: float, alpha: float, left_bc, right_bc):
        """
        Build the left (A) and right (B) matrices for Crank–Nicolson.

        System: (I - rA) u^{n+1} = (I + rA) u^n
        where A is the 1D Laplacian operator with boundary conditions.
        """
        r = alpha * self.dt / (2.0 * dx**2)

        # Left-hand matrix: I - rA
        main_l = (1.0 + 2.0 * r) * np.ones(nx)
        off = -r * np.ones(nx - 1)
        left_matrix = diags([off, main_l, off], [-1, 0, 1], format="lil")

        # Right-hand matrix: I + rA
        main_r = (1.0 - 2.0 * r) * np.ones(nx)
        right_matrix = diags(
            [r * np.ones(nx - 1), main_r, r * np.ones(nx - 1)], [-1, 0, 1], format="lil"
        )

        # Apply boundary conditions
        # Left boundary
        if isinstance(left_bc, DirichletBC):
            # Dirichlet: u[0] = value
            left_matrix.rows[0] = [0]
            left_matrix.data[0] = [1.0]
            right_matrix.rows[0] = [0]
            right_matrix.data[0] = [1.0]
        else:
            # Neumann: (u[1] - u[0])/dx = g
            left_matrix.rows[0] = [0, 1]
            left_matrix.data[0] = [-1.0 / dx, 1.0 / dx]
            right_matrix.rows[0] = [0, 1]
            right_matrix.data[0] = [-1.0 / dx, 1.0 / dx]

        # Right boundary
        if isinstance(right_bc, DirichletBC):
            # Dirichlet: u[-1] = value
            left_matrix.rows[nx - 1] = [nx - 1]
            left_matrix.data[nx - 1] = [1.0]
            right_matrix.rows[nx - 1] = [nx - 1]
            right_matrix.data[nx - 1] = [1.0]
        else:
            # Neumann: (u[-1] - u[-2])/dx = g
            left_matrix.rows[nx - 1] = [nx - 2, nx - 1]
            left_matrix.data[nx - 1] = [-1.0 / dx, 1.0 / dx]
            right_matrix.rows[nx - 1] = [nx - 2, nx - 1]
            right_matrix.data[nx - 1] = [-1.0 / dx, 1.0 / dx]

        return left_matrix.tocsc(), right_matrix.tocsc()

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 1D heat equation up to time t_final.

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
        nx, dx = grid.nx, grid.dx
        alpha = equation.alpha
        # Include BC types in cache key
        left_bc_type = type(equation.left_bc).__name__
        right_bc_type = type(equation.right_bc).__name__
        key = (nx, dx, alpha, self.dt, left_bc_type, right_bc_type)

        # Build matrices and factorize if parameters changed
        if self._shape_key != key:
            left_matrix, right_matrix = self._build_matrices(
                nx, dx, alpha, equation.left_bc, equation.right_bc
            )
            self._A, self._B = left_matrix, right_matrix
            self._solveA = factorized(left_matrix)  # LU factorization for reuse
            self._shape_key = key

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Crank-Nicolson 1D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            # RHS: (I + rA) u^n
            b = self._B @ grid.values

            # Enforce boundary conditions on RHS
            if isinstance(equation.left_bc, DirichletBC):
                b[0] = equation.left_bc.value
            else:
                b[0] = equation.left_bc.value  # Neumann: gradient value

            if isinstance(equation.right_bc, DirichletBC):
                b[-1] = equation.right_bc.value
            else:
                b[-1] = equation.right_bc.value  # Neumann: gradient value

            # Solve (I - rA) u^{n+1} = b
            new_values = self._solveA(b)

            grid.values = new_values
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values


class CrankNicolson2D:
    """
    Crank–Nicolson solver for the 2D heat equation.

    - Second-order accurate in time
    - Unconditionally stable
    - Uses Kronecker products to build the 2D Laplacian
    - Factorizes the left-hand matrix once and reuses it
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._A = None
        self._B = None
        self._solveA = None
        self._shape_key = None  # Cache key for grid/params

    def _build_matrices(
        self, nx: int, ny: int, dx: float, dy: float, alpha: float, left_bc, right_bc, bottom_bc, top_bc
    ):
        """
        Build the left (A) and right (B) matrices for 2D Crank–Nicolson.

        System: (I - rx*Lx - ry*Ly) u^{n+1} = (I + rx*Lx + ry*Ly) u^n
        where Lx and Ly are 1D Laplacians applied in x and y directions.
        """
        rx = alpha * self.dt / (2.0 * dx**2)
        ry = alpha * self.dt / (2.0 * dy**2)

        ix = identity(nx, format="csc")
        iy = identity(ny, format="csc")

        lx = _lap1d(nx)  # Laplacian in x
        ly = _lap1d(ny)  # Laplacian in y

        # Left-hand matrix: I - rx*(Iy⊗Lx) - ry*(Ly⊗Ix)
        left_matrix = (
            kron(iy, ix, format="csc")
            - rx * kron(iy, lx, format="csc")
            - ry * kron(ly, ix, format="csc")
        )

        # Right-hand matrix: I + rx*(Iy⊗Lx) + ry*(Ly⊗Ix)
        right_matrix = (
            kron(iy, ix, format="csc")
            + rx * kron(iy, lx, format="csc")
            + ry * kron(ly, ix, format="csc")
        )

        # Apply boundary conditions
        left_matrix = left_matrix.tolil()
        right_matrix = right_matrix.tolil()

        # Left/right boundaries
        for j in range(ny):
            idx_left = j * nx
            idx_right = j * nx + (nx - 1)

            if isinstance(left_bc, DirichletBC):
                left_matrix.rows[idx_left] = [idx_left]
                left_matrix.data[idx_left] = [1.0]
                right_matrix.rows[idx_left] = [idx_left]
                right_matrix.data[idx_left] = [1.0]
            else:
                # Neumann: (u[1,j] - u[0,j])/dx = g
                left_matrix.rows[idx_left] = [idx_left, idx_left + 1]
                left_matrix.data[idx_left] = [-1.0 / dx, 1.0 / dx]
                right_matrix.rows[idx_left] = [idx_left, idx_left + 1]
                right_matrix.data[idx_left] = [-1.0 / dx, 1.0 / dx]

            if isinstance(right_bc, DirichletBC):
                left_matrix.rows[idx_right] = [idx_right]
                left_matrix.data[idx_right] = [1.0]
                right_matrix.rows[idx_right] = [idx_right]
                right_matrix.data[idx_right] = [1.0]
            else:
                # Neumann: (u[-1,j] - u[-2,j])/dx = g
                left_matrix.rows[idx_right] = [idx_right - 1, idx_right]
                left_matrix.data[idx_right] = [-1.0 / dx, 1.0 / dx]
                right_matrix.rows[idx_right] = [idx_right - 1, idx_right]
                right_matrix.data[idx_right] = [-1.0 / dx, 1.0 / dx]

        # Bottom/top boundaries
        for i in range(nx):
            idx_bottom = i
            idx_top = (ny - 1) * nx + i

            if isinstance(bottom_bc, DirichletBC):
                left_matrix.rows[idx_bottom] = [idx_bottom]
                left_matrix.data[idx_bottom] = [1.0]
                right_matrix.rows[idx_bottom] = [idx_bottom]
                right_matrix.data[idx_bottom] = [1.0]
            else:
                # Neumann: (u[i,1] - u[i,0])/dy = g
                left_matrix.rows[idx_bottom] = [idx_bottom, idx_bottom + nx]
                left_matrix.data[idx_bottom] = [-1.0 / dy, 1.0 / dy]
                right_matrix.rows[idx_bottom] = [idx_bottom, idx_bottom + nx]
                right_matrix.data[idx_bottom] = [-1.0 / dy, 1.0 / dy]

            if isinstance(top_bc, DirichletBC):
                left_matrix.rows[idx_top] = [idx_top]
                left_matrix.data[idx_top] = [1.0]
                right_matrix.rows[idx_top] = [idx_top]
                right_matrix.data[idx_top] = [1.0]
            else:
                # Neumann: (u[i,-1] - u[i,-2])/dy = g
                left_matrix.rows[idx_top] = [idx_top - nx, idx_top]
                left_matrix.data[idx_top] = [-1.0 / dy, 1.0 / dy]
                right_matrix.rows[idx_top] = [idx_top - nx, idx_top]
                right_matrix.data[idx_top] = [-1.0 / dy, 1.0 / dy]

        return left_matrix.tocsc(), right_matrix.tocsc()

    def solve(
        self, equation, t_final: float, progress: ProgressTracker | None = None, verbosity: str | None = None
    ):
        """
        Solve the 2D heat equation up to time t_final.

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
        # Include BC types in cache key
        left_bc_type = type(equation.left_bc).__name__
        right_bc_type = type(equation.right_bc).__name__
        bottom_bc_type = type(equation.bottom_bc).__name__
        top_bc_type = type(equation.top_bc).__name__
        key = (nx, ny, dx, dy, alpha, self.dt, left_bc_type, right_bc_type, bottom_bc_type, top_bc_type)

        # Build matrices and factorize if parameters changed
        if self._shape_key != key:
            left_matrix, right_matrix = self._build_matrices(
                nx, ny, dx, dy, alpha, equation.left_bc, equation.right_bc, equation.bottom_bc, equation.top_bc
            )
            self._A, self._B = left_matrix, right_matrix
            self._solveA = factorized(left_matrix)
            self._shape_key = key

        # Flatten grid values into 1D vector (C-order: x fastest)
        u = grid.values.reshape(nx * ny, order="C").copy()

        def apply_bc_rhs(b: np.ndarray):
            """
            Overwrite RHS entries at boundary nodes with boundary condition values.
            """
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

        # Setup progress tracking
        if progress is None and verbosity is not None:
            num_steps = int(t_final / self.dt) + 1
            progress = create_progress_tracker(
                verbosity=verbosity,
                total_steps=num_steps,
                description="Crank-Nicolson 2D",
            )

        t = 0.0
        step_count = 0
        while t < t_final:
            # RHS: (I + rA) u^n
            b = self._B @ u

            # Apply boundary conditions to RHS
            apply_bc_rhs(b)

            # Solve (I - rA) u^{n+1} = b
            u = self._solveA(b)
            t += self.dt
            step_count += 1

            if progress is not None:
                progress.update(1, t=f"{t:.4f}")

        # Reshape back to 2D grid
        grid.values = u.reshape((nx, ny), order="C")

        if progress is not None:
            progress.print_summary(f"Completed {step_count} steps")
            progress.close()

        return grid.values


def _lap1d(nx: int) -> csc_matrix:
    """
    Construct the 1D Laplacian stencil matrix (no scaling by dx^2).
    """
    return diags(
        [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1], format="csc"
    )
