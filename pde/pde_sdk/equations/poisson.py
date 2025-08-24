import numpy as np

from ..domains.uniform2d import UniformGrid2D


class Poisson2D:
    def __init__(self, grid: UniformGrid2D, f):
        """
        Poisson equation: -∇²u = f(x,y)

        Parameters
        ----------
        grid : UniformGrid2D
            Computational grid
        f : callable
            Source term f(x,y), should accept arrays
        """
        self.grid = grid
        self.f = f

        # Initialize RHS values
        x_grid, y_grid = np.meshgrid(grid.x, grid.y, indexing="ij")
        self.rhs = f(x_grid, y_grid)
