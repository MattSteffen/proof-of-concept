import numpy as np
from scipy.sparse import csc_matrix, diags, identity, kron
from scipy.sparse.linalg import factorized


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

    def _build_matrices(self, nx: int, dx: float, alpha: float):
        """
        Build the left (A) and right (B) matrices for Crank–Nicolson.

        System: (I - rA) u^{n+1} = (I + rA) u^n
        where A is the 1D Laplacian operator with Dirichlet BCs.
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

        # Apply Dirichlet BCs: enforce u=BC at first and last nodes
        for row in (0, nx - 1):
            left_matrix.rows[row] = [row]
            left_matrix.data[row] = [1.0]
            right_matrix.rows[row] = [row]
            right_matrix.data[row] = [1.0]

        return left_matrix.tocsc(), right_matrix.tocsc()

    def solve(self, equation, t_final: float):
        """
        Solve the 1D heat equation up to time t_final.
        """
        grid = equation.grid
        nx, dx = grid.nx, grid.dx
        alpha = equation.alpha
        key = (nx, dx, alpha, self.dt)

        # Build matrices and factorize if parameters changed
        if self._shape_key != key:
            left_matrix, right_matrix = self._build_matrices(nx, dx, alpha)
            self._A, self._B = left_matrix, right_matrix
            self._solveA = factorized(left_matrix)  # LU factorization for reuse
            self._shape_key = key

        t = 0.0
        while t < t_final:
            # RHS: (I + rA) u^n
            b = self._B @ grid.values

            # Enforce Dirichlet BCs explicitly on RHS
            b[0] = equation.left_bc.value
            b[-1] = equation.right_bc.value

            # Solve (I - rA) u^{n+1} = b
            new_values = self._solveA(b)

            grid.values = new_values
            t += self.dt

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

    def _build_matrices(self, nx: int, ny: int, dx: float, dy: float, alpha: float):
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

        # Apply Dirichlet BCs: turn boundary rows into identity
        left_matrix = left_matrix.tolil()
        boundary_indices = set()

        # Left/right boundaries
        for j in range(ny):
            boundary_indices.add(j * nx)  # left
            boundary_indices.add(j * nx + (nx - 1))  # right
        # Bottom/top boundaries
        for i in range(nx):
            boundary_indices.add(i)  # bottom
            boundary_indices.add((ny - 1) * nx + i)  # top

        for idx in boundary_indices:
            left_matrix.rows[idx] = [idx]
            left_matrix.data[idx] = [1.0]

        return left_matrix.tocsc(), right_matrix

    def solve(self, equation, t_final: float):
        """
        Solve the 2D heat equation up to time t_final.
        """
        grid = equation.grid
        nx, ny = grid.nx, grid.ny
        dx, dy = grid.dx, grid.dy
        alpha = equation.alpha
        key = (nx, ny, dx, dy, alpha, self.dt)

        # Build matrices and factorize if parameters changed
        if self._shape_key != key:
            left_matrix, right_matrix = self._build_matrices(nx, ny, dx, dy, alpha)
            self._A, self._B = left_matrix, right_matrix
            self._solveA = factorized(left_matrix)
            self._shape_key = key

        # Flatten grid values into 1D vector (C-order: x fastest)
        u = grid.values.reshape(nx * ny, order="C").copy()

        def apply_dirichlet_rhs(b: np.ndarray):
            """
            Overwrite RHS entries at boundary nodes with Dirichlet values.
            """
            # Left/right boundaries
            for j in range(ny):
                b[j * nx] = equation.left_bc.value
                b[j * nx + (nx - 1)] = equation.right_bc.value
            # Bottom/top boundaries
            for i in range(nx):
                b[i] = equation.bottom_bc.value
                b[(ny - 1) * nx + i] = equation.top_bc.value

        t = 0.0
        while t < t_final:
            # RHS: (I + rA) u^n
            b = self._B @ u

            # Apply Dirichlet BCs to RHS
            apply_dirichlet_rhs(b)

            # Solve (I - rA) u^{n+1} = b
            u = self._solveA(b)
            t += self.dt

        # Reshape back to 2D grid
        grid.values = u.reshape((nx, ny), order="C")
        return grid.values


def _lap1d(nx: int) -> csc_matrix:
    """
    Construct the 1D Laplacian stencil matrix (no scaling by dx^2).
    """
    return diags(
        [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1], format="csc"
    )
