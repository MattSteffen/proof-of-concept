from ..boundaries.base import BoundaryCondition
from ..domains.uniform1d import UniformGrid1D


class HeatEquation1D:
    """
    1D heat/diffusion equation: ∂u/∂t = α ∂²u/∂x²

    This class represents the 1D heat equation with constant diffusivity α.
    It requires a grid, boundary conditions, and an initial condition.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient (must be positive)
    grid : UniformGrid1D
        1D uniform grid for spatial discretization
    left_bc : BoundaryCondition
        Left boundary condition (at x=0)
    right_bc : BoundaryCondition
        Right boundary condition (at x=length)
    initial_condition : callable
        Function f(x) that returns initial values at each grid point

    Examples
    --------
    >>> import numpy as np
    >>> from pde_sdk.domains import UniformGrid1D
    >>> from pde_sdk.boundaries import DirichletBC
    >>> grid = UniformGrid1D(nx=101, length=1.0)
    >>> eq = HeatEquation1D(
    ...     alpha=0.01,
    ...     grid=grid,
    ...     left_bc=DirichletBC(0.0),
    ...     right_bc=DirichletBC(0.0),
    ...     initial_condition=lambda x: np.sin(np.pi * x)
    ... )
    """

    def __init__(
        self,
        alpha: float,
        grid: UniformGrid1D,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        initial_condition,
    ):
        self.alpha = alpha
        self.grid = grid
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.grid.values = initial_condition(self.grid.x)


class HeatEquation2D:
    """
    2D heat/diffusion equation: ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

    This class represents the 2D heat equation with constant diffusivity α.
    It requires a 2D grid, boundary conditions on all four sides, and an initial condition.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient (must be positive)
    grid : UniformGrid2D
        2D uniform grid for spatial discretization
    left_bc : BoundaryCondition
        Left boundary condition (at x=0)
    right_bc : BoundaryCondition
        Right boundary condition (at x=length_x)
    bottom_bc : BoundaryCondition
        Bottom boundary condition (at y=0)
    top_bc : BoundaryCondition
        Top boundary condition (at y=length_y)
    initial_condition : callable
        Function f(x, y) that returns initial values at each grid point

    Examples
    --------
    >>> import numpy as np
    >>> from pde_sdk.domains import UniformGrid2D
    >>> from pde_sdk.boundaries import DirichletBC
    >>> grid = UniformGrid2D(nx=51, ny=51, length_x=1.0, length_y=1.0)
    >>> eq = HeatEquation2D(
    ...     alpha=0.01,
    ...     grid=grid,
    ...     left_bc=DirichletBC(0.0),
    ...     right_bc=DirichletBC(0.0),
    ...     bottom_bc=DirichletBC(0.0),
    ...     top_bc=DirichletBC(0.0),
    ...     initial_condition=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    ... )
    """

    def __init__(
        self,
        alpha: float,
        grid,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        top_bc: BoundaryCondition,
        initial_condition,
    ):
        self.alpha = alpha
        self.grid = grid
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.top_bc = top_bc
        self.grid.values = initial_condition(self.grid.X, self.grid.Y)
