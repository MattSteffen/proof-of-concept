class ExplicitEuler1D:
    def __init__(self, dt: float):
        self.dt = dt

    def solve(self, equation, t_final: float):
        grid = equation.grid
        dx = grid.dx
        alpha = equation.alpha
        r = alpha * self.dt / dx**2

        if r > 0.5:
            raise ValueError(f"Unstable timestep: r={r:.3f} > 0.5")

        t = 0.0
        while t < t_final:
            new_values = grid.values.copy()
            for i in range(1, grid.nx - 1):
                new_values[i] = grid.values[i] + r * (
                    grid.values[i - 1] - 2 * grid.values[i] + grid.values[i + 1]
                )

            # Apply boundary conditions (left BC affects index 0, right BC affects index -1)
            new_values[0] = equation.left_bc.value
            new_values[-1] = equation.right_bc.value

            grid.values = new_values
            t += self.dt

        return grid.values


class ExplicitEuler2D:
    def __init__(self, dt: float):
        self.dt = dt

    def solve(self, equation, t_final: float):
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

        t = 0.0
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
                new_values[0, j] = equation.left_bc.value

            # Right boundary (x=length_x)
            for j in range(grid.ny):
                new_values[-1, j] = equation.right_bc.value

            # Bottom boundary (y=0)
            for i in range(grid.nx):
                new_values[i, 0] = equation.bottom_bc.value

            # Top boundary (y=length_y)
            for i in range(grid.nx):
                new_values[i, -1] = equation.top_bc.value

            grid.values = new_values
            t += self.dt

        return grid.values
