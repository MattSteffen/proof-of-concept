import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class BackwardEuler1D:
    """Backward Euler (implicit) solver for 1D heat equation.

    This solver uses an implicit time-stepping scheme that requires solving
    a linear system at each timestep. It's unconditionally stable, allowing
    larger timesteps than explicit methods.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def solve(self, equation, t_final: float):
        grid = equation.grid
        dx = grid.dx
        alpha = equation.alpha
        r = alpha * self.dt / dx**2

        # Build system matrix: (I - r*A)
        # A is the 1D Laplacian with Dirichlet BCs
        n = grid.nx
        diagonals = [
            (1 + 2 * r) * np.ones(n),  # main diagonal
            -r * np.ones(n - 1),  # lower diagonal
            -r * np.ones(n - 1),  # upper diagonal
        ]
        system_matrix = diags(diagonals, [0, -1, 1], format="csc")

        # Apply Dirichlet BCs by fixing first and last rows
        system_matrix = system_matrix.tolil()
        system_matrix[0, :] = 0
        system_matrix[0, 0] = 1
        system_matrix[-1, :] = 0
        system_matrix[-1, -1] = 1
        system_matrix = system_matrix.tocsc()

        t = 0.0
        while t < t_final:
            b = grid.values.copy()
            # Enforce BCs in RHS
            b[0] = equation.left_bc.value
            b[-1] = equation.right_bc.value

            # Solve linear system
            new_values = spsolve(system_matrix, b)

            grid.values = new_values
            t += self.dt

        return grid.values


class BackwardEuler2D:
    """Backward Euler (implicit) solver for 2D heat equation.

    This solver uses an implicit time-stepping scheme that requires solving
    a linear system at each timestep. It's unconditionally stable, allowing
    larger timesteps than explicit methods.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def solve(self, equation, t_final: float):
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

        # Convert to CSC for solving
        system_matrix = system_matrix.tocsc()

        t = 0.0
        while t < t_final:
            b = grid.values.flatten()

            # Apply Dirichlet BCs (set boundary rows in A and b)
            # For simplicity: enforce u=BC at boundary nodes
            # (more sophisticated handling can be added later)
            # Left/right boundaries
            for j in range(ny):
                b[j * nx] = equation.left_bc.value
                b[j * nx + (nx - 1)] = equation.right_bc.value
            # Bottom/top boundaries
            for i in range(nx):
                b[i] = equation.bottom_bc.value
                b[(ny - 1) * nx + i] = equation.top_bc.value

            # Solve system
            new_values = spsolve(system_matrix, b)
            grid.values = new_values.reshape((nx, ny))

            t += self.dt

        return grid.values
