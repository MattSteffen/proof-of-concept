import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.backward_euler import BackwardEuler2D

# Parameters
nx, ny = 21, 21
length_x, length_y = 1.0, 1.0
alpha = 0.01
dt = 0.1  # Much larger timestep than explicit methods
t_final = 0.5

print("2D Heat Equation - Backward Euler Demo")
print("=" * 40)
print(f"Grid size: {nx} x {ny}")
print(f"Domain: [0,{length_x}] x [0,{length_y}]")
print(f"Alpha: {alpha}")
print(f"Timestep: {dt}")
print(f"Final time: {t_final}")
print()

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

print("Initial condition set up")
print(f"Max initial value: {np.max(grid.values):.6f}")
print(f"Min initial value: {np.min(grid.values):.6f}")
print()

# Solve with Backward Euler
solver = BackwardEuler2D(dt=dt)
print("Solving 2D heat equation with Backward Euler...")
u_final = solver.solve(eq, t_final=t_final)

# Analytical solution: e^{-2π²αt} sin(πx) sin(πy)
decay_rate = 2 * np.pi**2 * alpha * t_final
u_exact = np.exp(-decay_rate) * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)

# Calculate errors
error = np.abs(u_final - u_exact)
max_error = np.max(error)
rms_error = np.sqrt(np.mean(error**2))

print("Results:")
print("=" * 10)
print(".6f")
print(".6f")
print(".2e")
print(".2e")
print()

# Check boundary conditions
left_bc_ok = np.allclose(u_final[0, :], 0.0, atol=1e-10)
right_bc_ok = np.allclose(u_final[-1, :], 0.0, atol=1e-10)
bottom_bc_ok = np.allclose(u_final[:, 0], 0.0, atol=1e-10)
top_bc_ok = np.allclose(u_final[:, -1], 0.0, atol=1e-10)

print("Boundary Conditions:")
print(f"Left (x=0): {'✓' if left_bc_ok else '✗'}")
print(f"Right (x={length_x}): {'✓' if right_bc_ok else '✗'}")
print(f"Bottom (y=0): {'✓' if bottom_bc_ok else '✗'}")
print(f"Top (y={length_y}): {'✓' if top_bc_ok else '✗'}")
print()

# Compare with explicit method
print("Comparison with Explicit Euler:")
from pde_sdk.solvers.explicit_euler import ExplicitEuler2D

# Reset initial condition
grid.values = initial_condition(grid.X, grid.Y)

# Use a much smaller timestep for explicit method
dt_explicit = 1e-4
solver_explicit = ExplicitEuler2D(dt=dt_explicit)
u_explicit = solver_explicit.solve(eq, t_final=t_final)

error_explicit = np.abs(u_explicit - u_exact)
max_error_explicit = np.max(error_explicit)
rms_error_explicit = np.sqrt(np.mean(error_explicit**2))

print("Explicit Euler (dt=1e-4):")
print(".2e")
print(".2e")
print()
print("Backward Euler (dt=0.1):")
print(".2e")
print(".2e")
print()

# Create plots
fig = plt.figure(figsize=(18, 6))

# Plot 1: Backward Euler solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(grid.X, grid.Y, u_final, cmap='viridis', alpha=0.8)
ax1.set_title('Backward Euler Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Plot 2: Explicit Euler solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(grid.X, grid.Y, u_explicit, cmap='plasma', alpha=0.8)
ax2.set_title('Explicit Euler Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')

# Plot 3: Error comparison
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

# Error for backward Euler
im1 = axes[0].contourf(grid.X, grid.Y, error, levels=20, cmap='hot')
axes[0].set_title('Backward Euler Error')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# Error for explicit Euler
im2 = axes[1].contourf(grid.X, grid.Y, error_explicit, levels=20, cmap='hot')
axes[1].set_title('Explicit Euler Error')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

if max_error < 1e-4:
    print("✅ SUCCESS: 2D Backward Euler implementation is working correctly!")
    print("   Unconditional stability allows much larger timesteps.")
else:
    print("⚠️  WARNING: Error may be larger than expected, but still reasonable.")
