
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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

    def plot(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
        cmap: str = "viridis",
        levels: int | None = 20,
        **kwargs,
    ):
        """
        Plot the grid values as a contour plot.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        title : str, optional
            Plot title
        cmap : str
            Colormap name
        levels : int, optional
            Number of contour levels
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        plt.Axes
            The axes object
        """
        if plt is None:
            raise ImportError("matplotlib is required for plotting")

        from ..visualization.plot import plot_2d

        return plot_2d(
            self.X, self.Y, self.values, ax=ax, title=title, cmap=cmap, levels=levels, **kwargs
        )
