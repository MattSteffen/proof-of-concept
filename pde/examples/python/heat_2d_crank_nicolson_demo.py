import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.crank_nicolson import CrankNicolson2D

# Parameters
nx, ny = 21, 21
length_x, length_y = 1.0, 1.0
alpha = 0.01
dt = 0.1  # Large timestep - unconditionally stable
t_final = 0.5

print("2D Heat Equation - Crank-Nicolson Demo")
print("=" * 45)
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

# Solve with Crank-Nicolson
solver = CrankNicolson2D(dt=dt)
print("Solving 2D heat equation with Crank-Nicolson...")
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
print(f"Max numerical: {np.max(u_final):.6f}")
print(f"Max analytical: {np.max(u_exact):.6f}")
print(f"Max error: {max_error:.2e}")
print(f"RMS error: {rms_error:.2e}")
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

# Compare with other methods
print("Comparison with other implicit methods:")
from pde_sdk.solvers.backward_euler import BackwardEuler2D

# Backward Euler comparison
grid.values = initial_condition(grid.X, grid.Y)
solver_backward = BackwardEuler2D(dt=dt)
u_backward = solver_backward.solve(eq, t_final)
error_backward = np.abs(u_backward - u_exact)

print(f"Backward Euler (dt={dt}): {np.max(error_backward):.2e} error")
print(f"Crank-Nicolson (dt={dt}): {max_error:.2e} error")
print(f"Improvement factor: {np.max(error_backward)/max_error:.1f}x")
print()

# Create plots
fig = plt.figure(figsize=(15, 5))

# Plot 1: Crank-Nicolson solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(grid.X, grid.Y, u_final, cmap='viridis', alpha=0.8)
ax1.set_title('Crank-Nicolson Solution')
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
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

# Error for Crank-Nicolson
im1 = axes[0].contourf(grid.X, grid.Y, error, levels=20, cmap='hot')
axes[0].set_title('Crank-Nicolson Error')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# Error comparison with backward Euler
error_comparison = np.abs(u_backward - u_exact)
im2 = axes[1].contourf(grid.X, grid.Y, error_comparison, levels=20, cmap='hot')
axes[1].set_title('Backward Euler Error')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

if max_error < 1e-4:
    print("✅ SUCCESS: 2D Crank-Nicolson implementation is working!")
    print("   Second-order accuracy provides excellent results.")
else:
    print("⚠️  Results may vary - check implementation")

print("\nCrank-Nicolson Advantages:")
print("• Second-order accurate in time (best accuracy)")
print("• Unconditionally stable (no timestep restrictions)")
print("• Superior accuracy compared to first-order methods")
print("• Best choice for high-accuracy 2D simulations")
