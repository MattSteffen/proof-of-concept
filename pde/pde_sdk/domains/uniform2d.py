import numpy as np


class UniformGrid2D:
    def __init__(self, nx: int, ny: int, length_x: float, length_y: float):
        self.nx = nx
        self.ny = ny
        self.length_x = length_x
        self.length_y = length_y
        self.dx = length_x / (nx - 1)
        self.dy = length_y / (ny - 1)
        self.x = np.linspace(0, length_x, nx)
        self.y = np.linspace(0, length_y, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.values = np.zeros((nx, ny))
