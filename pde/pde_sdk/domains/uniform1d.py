import numpy as np


class UniformGrid1D:
    def __init__(self, nx: int, length: float):
        self.nx = nx
        self.length = length
        self.dx = length / (nx - 1)
        self.x = np.linspace(0, length, nx)
        self.values = np.zeros(nx)
