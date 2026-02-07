import numpy as np

from ..boundaries.base import BoundaryCondition
from ..boundaries.dirichlet import DirichletBC
from ..domains.uniform2d import UniformGrid2D


class Poisson2D:
    def __init__(
        self,
        grid: UniformGrid2D,
        f,
        left_bc: BoundaryCondition | None = None,
        right_bc: BoundaryCondition | None = None,
        bottom_bc: BoundaryCondition | None = None,
        top_bc: BoundaryCondition | None = None,
    ):
        """
        Poisson equation: -∇²u = f(x,y)

        Parameters
        ----------
        grid : UniformGrid2D
            Computational grid
        f : callable
            Source term f(x,y), should accept arrays
        left_bc : BoundaryCondition, optional
            Left boundary condition (x=0). Defaults to DirichletBC(0.0)
        right_bc : BoundaryCondition, optional
            Right boundary condition (x=Lx). Defaults to DirichletBC(0.0)
        bottom_bc : BoundaryCondition, optional
            Bottom boundary condition (y=0). Defaults to DirichletBC(0.0)
        top_bc : BoundaryCondition, optional
            Top boundary condition (y=Ly). Defaults to DirichletBC(0.0)
        """
        self.grid = grid
        self.f = f

        # Initialize RHS values
        x_grid, y_grid = np.meshgrid(grid.x, grid.y, indexing="ij")
        self.rhs = f(x_grid, y_grid)

        # Set boundary conditions with defaults for backward compatibility
        self.left_bc = left_bc if left_bc is not None else DirichletBC(0.0)
        self.right_bc = right_bc if right_bc is not None else DirichletBC(0.0)
        self.bottom_bc = bottom_bc if bottom_bc is not None else DirichletBC(0.0)
        self.top_bc = top_bc if top_bc is not None else DirichletBC(0.0)
