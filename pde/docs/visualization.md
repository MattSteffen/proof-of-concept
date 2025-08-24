# Visualization & Plotting

This guide covers visualization techniques for PDE solutions using the PDE SDK. The SDK works with standard Python plotting libraries like matplotlib for creating publication-quality plots.

## Quick Start

### 1D Line Plots

```python
import numpy as np
import matplotlib.pyplot as plt

from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import ExplicitEuler1D

# Solve 1D heat equation
nx = 101
grid = UniformGrid1D(nx=nx, length=1.0)
ic = lambda x: np.sin(np.pi * x)
eq = HeatEquation1D(alpha=0.01, grid=grid,
                    left_bc=DirichletBC(0.0), right_bc=DirichletBC(0.0),
                    initial_condition=ic)
solver = ExplicitEuler1D(dt=1e-4)
solution = solver.solve(eq, t_final=0.1)

# Plot solution
plt.figure(figsize=(8, 6))
plt.plot(grid.x, solution, 'b-', linewidth=2, label='Numerical')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('1D Heat Equation Solution')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

### 2D Contour Plots

```python
import matplotlib.pyplot as plt

from pde_sdk.domains import UniformGrid2D
from pde_sdk.equations import HeatEquation2D
from pde_sdk.solvers import ExplicitEuler2D

# Solve 2D heat equation
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)
ic = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
bc = DirichletBC(0.0)
eq = HeatEquation2D(alpha=0.01, grid=grid,
                    left_bc=bc, right_bc=bc, bottom_bc=bc, top_bc=bc,
                    initial_condition=ic)
solver = ExplicitEuler2D(dt=5e-5)
solution = solver.solve(eq, t_final=0.05)

# Plot solution
plt.figure(figsize=(8, 6))
plt.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
plt.colorbar(label='u(x,y,t)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Heat Equation Solution')
plt.axis('equal')
plt.show()
```

## Advanced Visualization Techniques

### Comparison Plots

```python
# Compare numerical vs analytical solution
plt.figure(figsize=(12, 5))

# Analytical solution for 1D heat
x = grid.x
u_exact = np.exp(-np.pi**2 * 0.01 * 0.1) * np.sin(np.pi * x)

plt.subplot(1, 2, 1)
plt.plot(x, solution, 'b-', linewidth=2, label='Numerical')
plt.plot(x, u_exact, 'r--', linewidth=2, label='Analytical')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(solution - u_exact), 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('|Error|')
plt.title('Absolute Error')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

### 3D Surface Plots

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# Numerical solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(grid.X, grid.Y, solution, cmap='viridis', alpha=0.8)
ax1.set_title('Numerical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Analytical solution
ax2 = fig.add_subplot(132, projection='3d')
u_exact = np.exp(-2 * np.pi**2 * 0.01 * 0.05) * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
surf2 = ax2.plot_surface(grid.X, grid.Y, u_exact, cmap='plasma', alpha=0.8)
ax2.set_title('Analytical Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')

# Error surface
ax3 = fig.add_subplot(133, projection='3d')
error = np.abs(solution - u_exact)
surf3 = ax3.plot_surface(grid.X, grid.Y, error, cmap='hot', alpha=0.8)
ax3.set_title('Absolute Error')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('|error|')

plt.tight_layout()
plt.show()
```

### Animation and Time Evolution

```python
import matplotlib.animation as animation

# Solve at multiple time steps
times = np.linspace(0, 0.1, 11)
solutions = []

for t in times:
    solution = solver.solve(eq, dt)  # dt = times[1] - times[0]
    solutions.append(solution.copy())

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.contourf(grid.X, grid.Y, solutions[0], levels=20, cmap='viridis')
plt.colorbar(im, ax=ax, label='u')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D Heat Equation Evolution')

def animate(frame):
    im = ax.contourf(grid.X, grid.Y, solutions[frame], levels=20, cmap='viridis')
    ax.set_title('.3f')
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(solutions),
                             interval=500, blit=True)
plt.show()
```

## Visualization Best Practices

### Colormap Selection

```python
# Sequential colormaps (good for magnitude data)
cmaps_sequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

# Diverging colormaps (good for data with zero center)
cmaps_diverging = ['RdBu_r', 'RdYlBu_r', 'PiYG', 'BrBG']

# Example usage
plt.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
```

### Level/Contour Selection

```python
# Automatic level selection
plt.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')

# Manual level selection
levels = np.linspace(solution.min(), solution.max(), 21)
plt.contourf(grid.X, grid.Y, solution, levels=levels, cmap='viridis')

# Symmetric levels around zero
max_val = max(abs(solution.min()), abs(solution.max()))
levels = np.linspace(-max_val, max_val, 21)
plt.contourf(grid.X, grid.Y, solution, levels=levels, cmap='RdBu_r')
```

### Plot Formatting

```python
# High-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
im = ax.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('PDE Solution', fontsize=16)
ax.axis('equal')

# Colorbar with proper formatting
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('u(x,y)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()
```

## Specialized Plots

### Error Analysis Plots

```python
# Error convergence study
import matplotlib.pyplot as plt
import numpy as np

def analyze_convergence():
    resolutions = [21, 41, 81, 161]
    errors = []

    for nx in resolutions:
        grid = UniformGrid1D(nx=nx, length=1.0)
        # ... solve equation ...
        solution = solver.solve(eq, t_final=0.1)

        # Compare with fine reference solution
        x_ref = np.linspace(0, 1, 1001)
        u_ref = np.exp(-np.pi**2 * 0.01 * 0.1) * np.sin(np.pi * x_ref)

        # Interpolate to compare
        u_interp = np.interp(x_ref, grid.x, solution)
        error = np.max(np.abs(u_interp - u_ref))
        errors.append(error)

    # Plot convergence
    dx_values = [1.0/(nx-1) for nx in resolutions]

    plt.figure(figsize=(8, 6))
    plt.loglog(dx_values, errors, 'bo-', linewidth=2, markersize=8)
    plt.loglog(dx_values, [e**2 for e in errors], 'r--', linewidth=2, label='O(Δx²)')
    plt.xlabel('Grid spacing Δx')
    plt.ylabel('Maximum error')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

analyze_convergence()
```

### Vector Field Plots (for future velocity fields)

```python
# This would be relevant for Navier-Stokes or similar equations
# Currently not implemented in the SDK

# Example structure for velocity field visualization
def plot_velocity_field(u, v, grid):
    """Plot 2D vector field"""
    plt.figure(figsize=(10, 8))

    # Streamplot
    plt.streamplot(grid.X, grid.Y, u, v, density=2, linewidth=1, color='black')

    # Quiver plot (subset of points)
    skip = 4
    plt.quiver(grid.X[::skip, ::skip], grid.Y[::skip, ::skip],
               u[::skip, ::skip], v[::skip, ::skip],
               scale=20, color='red', width=0.005)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity Field')
    plt.axis('equal')
    plt.show()
```

## Exporting and Saving Plots

### High-Resolution Export

```python
# Save high-resolution plot
plt.figure(figsize=(12, 10), dpi=300)

# Create your plot
plt.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('PDE Solution')
plt.axis('equal')

# Save with high resolution
plt.savefig('pde_solution.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('pde_solution.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

### Batch Plotting

```python
# Generate multiple plots
def create_solution_plots():
    time_steps = [0.01, 0.05, 0.1, 0.5]

    for i, t_final in enumerate(time_steps):
        # Solve equation
        solution = solver.solve(eq, t_final)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
        ax.set_title('.3f')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='u')

        # Save plot
        plt.savefig('.3f')
        plt.close()

create_solution_plots()
```

## Interactive Visualization

### Jupyter Notebook Integration

```python
# For Jupyter notebooks
%matplotlib inline

# Interactive widgets (requires ipywidgets)
import ipywidgets as widgets
from IPython.display import display

def plot_solution_at_time(t):
    solution = solver.solve(eq, t)
    plt.figure(figsize=(8, 6))
    plt.contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('.3f')
    plt.axis('equal')
    plt.show()

# Create interactive slider
t_slider = widgets.FloatSlider(min=0, max=0.1, step=0.01, value=0.05)
widgets.interactive(plot_solution_at_time, t=t_slider)
```

## Troubleshooting

### Common Issues

1. **Plots not showing**
   - Add `plt.show()` at the end
   - Check if matplotlib backend is properly configured

2. **Poor aspect ratio**
   - Use `ax.axis('equal')` for 2D plots
   - Set appropriate figure size with `figsize=(width, height)`

3. **Colormap issues**
   - Use perceptually uniform colormaps like 'viridis'
   - Check data range with `print(solution.min(), solution.max())`

4. **Memory issues with large grids**
   - Reduce resolution for visualization
   - Use `plt.contour` instead of `plt.contourf` for large datasets

### Performance Tips

```python
# Fast plotting for large datasets
# Use contour instead of contourf for speed
plt.contour(grid.X, grid.Y, solution, levels=10, cmap='viridis')

# Reduce number of levels
plt.contourf(grid.X, grid.Y, solution, levels=15, cmap='viridis')

# Subsample for quiver/vector plots
skip = 4
plt.quiver(grid.X[::skip, ::skip], grid.Y[::skip, ::skip],
           u[::skip, ::skip], v[::skip, ::skip])
```

## Future Visualization Features

The SDK is designed to support advanced visualization capabilities:

- **Built-in plotting utilities**: Direct plotting methods on grid objects
- **Animation support**: Time evolution animations
- **Interactive plots**: Web-based interactive visualizations
- **Vector field plots**: For systems with velocity fields
- **Error visualization**: Built-in error analysis plots
- **Multi-physics visualization**: Multiple coupled PDEs
