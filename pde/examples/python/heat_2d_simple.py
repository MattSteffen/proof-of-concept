import numpy as np
from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.explicit_euler import ExplicitEuler2D

# Parameters
nx, ny = 21, 21
length_x, length_y = 1.0, 1.0
alpha = 0.01
dt = 5e-5
t_final = 0.05

print("2D Heat Equation Demo")
print("=" * 30)
print(f"Grid: {nx} x {ny} points")
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

# Solve
solver = ExplicitEuler2D(dt=dt)
print("Solving 2D heat equation...")
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

# Show some sample values
print("Sample Values:")
print("-" * 15)
center_i, center_j = nx // 2, ny // 2
print(f"Center point ({grid.x[center_i]:.3f}, {grid.y[center_j]:.3f}):")
print(f"Numerical: {u_final[center_i, center_j]:.6f}")
print(f"Analytical: {u_exact[center_i, center_j]:.6f}")
print(f"Error: {error[center_i, center_j]:.2e}")
print()

if max_error < 1e-4:
    print("✅ SUCCESS: 2D heat equation implementation is working correctly!")
else:
    print("⚠️  WARNING: Error may be larger than expected, but still reasonable.")
