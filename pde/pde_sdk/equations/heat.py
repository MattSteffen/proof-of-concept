from ..boundaries.dirichlet import DirichletBC
from ..domains.uniform1d import UniformGrid1D


class HeatEquation1D:
    def __init__(
        self,
        alpha: float,
        grid: UniformGrid1D,
        left_bc: DirichletBC,
        right_bc: DirichletBC,
        initial_condition,
    ):
        self.alpha = alpha
        self.grid = grid
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.grid.values = initial_condition(self.grid.x)


class HeatEquation2D:
    def __init__(
        self,
        alpha: float,
        grid,
        left_bc: DirichletBC,
        right_bc: DirichletBC,
        bottom_bc: DirichletBC,
        top_bc: DirichletBC,
        initial_condition,
    ):
        self.alpha = alpha
        self.grid = grid
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.top_bc = top_bc
        self.grid.values = initial_condition(self.grid.X, self.grid.Y)
