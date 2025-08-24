import numpy as np


class JacobiPoisson2D:
    def __init__(self, max_iter=10000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, equation):
        grid = equation.grid
        dx, dy = grid.dx, grid.dy
        rhs = equation.rhs

        u = np.zeros_like(rhs)  # initial guess

        # Precompute constants
        dx2, dy2 = dx**2, dy**2
        denom = 2 * (dx2 + dy2)

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

            # Dirichlet BCs (assume 0 for now)
            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0

            # Convergence check
            error = np.linalg.norm(u_new - u, ord=np.inf)
            if error < self.tol:
                print(f"Converged in {it} iterations with error {error:.2e}")
                break

            u = u_new

        grid.values = u
        return u
