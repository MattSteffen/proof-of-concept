
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class UniformGrid1D:
    def __init__(self, nx: int, length: float):
        self.nx = nx
        self.length = length
        self.dx = length / (nx - 1)
        self.x = np.linspace(0, length, nx)
        self.values = np.zeros(nx)

    def plot(
        self,
        ax: plt.Axes | None = None,
        label: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Plot the grid values.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        label : str, optional
            Label for the plot
        title : str, optional
            Plot title
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        plt.Axes
            The axes object
        """
        if plt is None:
            raise ImportError("matplotlib is required for plotting")

        from ..visualization.plot import plot_1d

        return plot_1d(self.x, self.values, ax=ax, label=label, title=title, **kwargs)
