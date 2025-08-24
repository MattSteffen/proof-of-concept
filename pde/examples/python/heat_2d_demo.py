import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.explicit_euler import ExplicitEuler2D

# Parameters
nx, ny = 31, 31
length_x, length_y = 1.0, 1.0
alpha = 0.01
dt = 5e-5  # Smaller timestep for 2D stability
t_final = 0.05

# Setup 2D grid
grid = UniformGrid2D(nx=nx, ny=ny, length_x=length_x, length_y=length_y)

# Initial condition: sin(πx) * sin(πy)
def initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Boundary conditions (all Dirichlet = 0)
left_bc = DirichletBC(0.0)
right_bc = DirichletBC(0.0)
bottom_bc = DirichletBC(0.0)
top_bc = DirichletBC(0.0)

# Create 2D heat equation
eq = HeatEquation2D(
    alpha=alpha,
    grid=grid,
    left_bc=left_bc,
    right_bc=right_bc,
    bottom_bc=bottom_bc,
    top_bc=top_bc,
    initial_condition=initial_condition,
)

# Solve
solver = ExplicitEuler2D(dt=dt)
u_final = solver.solve(eq, t_final=t_final)

# Analytical solution: e^{-2π²αt} sin(πx) sin(πy)
# The eigenvalues are π² + π² = 2π²
decay_rate = 2 * np.pi**2 * alpha * t_final
u_exact = np.exp(-decay_rate) * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)

# Calculate errors
error = np.abs(u_final - u_exact)
max_error = np.max(error)
rms_error = np.sqrt(np.mean(error**2))

print("2D Heat Equation Results:")
print(f"Grid size: {nx} x {ny}")
print(f"Alpha: {alpha}")
print(f"Timestep: {dt}")
print(f"Max numerical: {np.max(u_final):.6f}")
print(f"Max analytical: {np.max(u_exact):.6f}")
print(f"Max error: {max_error:.2e}")

# Create plots
fig = plt.figure(figsize=(15, 5))

# Plot 1: Numerical solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(grid.X, grid.Y, u_final, cmap='viridis', alpha=0.8)
ax1.set_title('Numerical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Plot 2: Analytical solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(grid.X, grid.Y, u_exact, cmap='plasma', alpha=0.8)
ax2.set_title('Analytical Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')

# Plot 3: Error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(grid.X, grid.Y, error, cmap='hot', alpha=0.8)
ax3.set_title('Absolute Error')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('|error|')

plt.tight_layout()
plt.show()

# Also create a 2D contour plot comparison
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

# Numerical
im1 = axes[0].contourf(grid.X, grid.Y, u_final, levels=20, cmap='viridis')
axes[0].set_title('Numerical Solution')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# Analytical
im2 = axes[1].contourf(grid.X, grid.Y, u_exact, levels=20, cmap='plasma')
axes[1].set_title('Analytical Solution')
axes[1].set_xlabel('x')
plt.colorbar(im2, ax=axes[1])

# Error
im3 = axes[2].contourf(grid.X, grid.Y, error, levels=20, cmap='hot')
axes[2].set_title('Absolute Error')
axes[2].set_xlabel('x')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
