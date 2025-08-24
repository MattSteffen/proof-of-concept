from .base import BoundaryCondition


class DirichletBC(BoundaryCondition):
    def __init__(self, value: float):
        self.value = value

    def apply(self, values):
        values[0] = self.value
        values[-1] = self.value
        return values
