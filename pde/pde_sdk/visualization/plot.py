"""Plotting utilities for PDE solutions."""


import matplotlib.pyplot as plt
import numpy as np


def plot_1d(
    x: np.ndarray,
    u: np.ndarray,
    ax: plt.Axes | None = None,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "u(x)",
    **kwargs,
) -> plt.Axes:
    """
    Plot a 1D solution.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    u : np.ndarray
        Solution values
    ax : plt.Axes, optional
        Matplotlib axes to plot on (creates new figure if None)
    label : str, optional
        Label for the plot legend
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    **kwargs
        Additional arguments passed to plt.plot()

    Returns
    -------
    plt.Axes
        The axes object

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 101)
    >>> u = np.sin(np.pi * x)
    >>> plot_1d(x, u, title="1D Solution")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, u, label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_2d(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    u: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    cmap: str = "viridis",
    levels: int | None = 20,
    **kwargs,
) -> plt.Axes:
    """
    Plot a 2D solution as a contour plot.

    Parameters
    ----------
    X : np.ndarray
        X coordinate meshgrid
    Y : np.ndarray
        Y coordinate meshgrid
    u : np.ndarray
        2D solution array
    ax : plt.Axes, optional
        Matplotlib axes to plot on (creates new figure if None)
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    cmap : str
        Colormap name
    levels : int, optional
        Number of contour levels
    **kwargs
        Additional arguments passed to plt.contourf()

    Returns
    -------
    plt.Axes
        The axes object

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 51)
    >>> y = np.linspace(0, 1, 51)
    >>> x_coords, y_coords = np.meshgrid(x, y, indexing='ij')
    >>> u = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)
    >>> plot_2d(x_coords, y_coords, u, title="2D Solution")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if levels is not None:
        im = ax.contourf(x_coords, y_coords, u, levels=levels, cmap=cmap, **kwargs)
    else:
        im = ax.contourf(x_coords, y_coords, u, cmap=cmap, **kwargs)

    plt.colorbar(im, ax=ax, label="u(x,y)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.axis("equal")

    return ax


def plot_comparison(
    x: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot numerical solution and optionally compare with analytical solution.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    u_numerical : np.ndarray
        Numerical solution
    u_analytical : np.ndarray, optional
        Analytical solution for comparison
    ax : plt.Axes, optional
        Matplotlib axes to plot on (creates new figure if None)
    title : str, optional
        Plot title

    Returns
    -------
    plt.Axes
        The axes object

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 101)
    >>> u_num = np.sin(np.pi * x)
    >>> u_analytical = np.sin(np.pi * x)
    >>> plot_comparison(x, u_num, u_analytical)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, u_numerical, "b-", linewidth=2, label="Numerical")
    if u_analytical is not None:
        ax.plot(x, u_analytical, "r--", linewidth=2, label="Analytical")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_error(
    x: np.ndarray,
    error: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
    log_scale: bool = False,
) -> plt.Axes:
    """
    Plot error between numerical and analytical solutions.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    error : np.ndarray
        Error values (absolute or relative)
    ax : plt.Axes, optional
        Matplotlib axes to plot on (creates new figure if None)
    title : str, optional
        Plot title
    log_scale : bool
        Whether to use logarithmic scale for y-axis

    Returns
    -------
    plt.Axes
        The axes object

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 101)
    >>> error = np.abs(np.sin(np.pi * x) - np.sin(np.pi * x + 0.01))
    >>> plot_error(x, error, log_scale=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, error, "g-", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("|Error|")
    if title:
        ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return ax


def animate_solution(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    solutions: list[np.ndarray],
    times: np.ndarray | None = None,
    interval: int = 200,
    cmap: str = "viridis",
    levels: int = 20,
) -> None:
    """
    Create an animation of solution evolution over time.

    Parameters
    ----------
    x_coords : np.ndarray
        X coordinate meshgrid
    y_coords : np.ndarray
        Y coordinate meshgrid
    solutions : list[np.ndarray]
        List of 2D solution arrays at different times
    times : np.ndarray, optional
        Time values corresponding to each solution
    interval : int
        Animation frame interval in milliseconds
    cmap : str
        Colormap name
    levels : int
        Number of contour levels

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 51)
    >>> y = np.linspace(0, 1, 51)
    >>> x_coords, y_coords = np.meshgrid(x, y, indexing='ij')
    >>> solutions = [np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords) * np.exp(-t) for t in [0, 0.1, 0.2]]
    >>> animate_solution(x_coords, y_coords, solutions)
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        raise ImportError("matplotlib.animation is required for animations")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Initial plot
    im = ax.contourf(x_coords, y_coords, solutions[0], levels=levels, cmap=cmap)
    plt.colorbar(im, ax=ax, label="u(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    def animate(frame):
        ax.clear()
        im = ax.contourf(x_coords, y_coords, solutions[frame], levels=levels, cmap=cmap)
        plt.colorbar(im, ax=ax, label="u(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if times is not None:
            ax.set_title(f"t = {times[frame]:.3f}")
        else:
            ax.set_title(f"Frame {frame}")
        ax.axis("equal")
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(solutions), interval=interval, blit=False, repeat=True
    )

    plt.show()
    return ani

