import numpy as np
import matplotlib.pyplot as plt

from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.backward_euler import BackwardEuler1D

# Parameters
nx = 51
length = 1.0
alpha = 0.01
dt = 0.1  # Much larger timestep than explicit methods (dt=0.001 needed for explicit)
t_final = 1.0

print("1D Heat Equation - Backward Euler Demo")
print("=" * 40)
print(f"Grid points: {nx}")
print(f"Domain: [0, {length}]")
print(f"Alpha: {alpha}")
print(f"Timestep: {dt}")
print(f"Final time: {t_final}")
print()

# Setup
grid = UniformGrid1D(nx=nx, length=length)
ic = lambda x: np.sin(np.pi * x)

eq = HeatEquation1D(
    alpha=alpha,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=ic,
)

# Solve with Backward Euler
solver = BackwardEuler1D(dt=dt)
print("Solving with Backward Euler...")
u_final = solver.solve(eq, t_final=t_final)

# Analytical solution
x = grid.x
u_exact = np.exp(-np.pi**2 * alpha * t_final) * np.sin(np.pi * x)

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
left_bc_ok = np.allclose(u_final[0], 0.0, atol=1e-10)
right_bc_ok = np.allclose(u_final[-1], 0.0, atol=1e-10)

print("Boundary Conditions:")
print(f"Left (x=0): {'✓' if left_bc_ok else '✗'}")
print(f"Right (x={length}): {'✓' if right_bc_ok else '✗'}")
print()

# Compare with explicit method for reference
print("Comparison with Explicit Euler:")
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

# Reset initial condition
grid.values = ic(grid.x)

# Use a much smaller timestep for explicit method
dt_explicit = 1e-4
solver_explicit = ExplicitEuler1D(dt=dt_explicit)
u_explicit = solver_explicit.solve(eq, t_final=t_final)

error_explicit = np.abs(u_explicit - u_exact)
max_error_explicit = np.max(error_explicit)
rms_error_explicit = np.sqrt(np.mean(error_explicit**2))

print("Explicit Euler (dt=1e-4):")
print(f"Max error: {max_error_explicit:.2e}")
print(f"Mean error: {np.mean(error_explicit):.2e}")
print()
print("Backward Euler (dt=0.1):")
print(f"Max error: {max_error:.2e}")
print(f"Mean error: {np.mean(error):.2e}")
print()

# Plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, u_final, 'b-', label='Backward Euler', linewidth=2)
plt.plot(x, u_exact, 'r--', label='Analytical', linewidth=2)
plt.title('Backward Euler Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, u_explicit, 'g-', label='Explicit Euler', linewidth=2)
plt.plot(x, u_exact, 'r--', label='Analytical', linewidth=2)
plt.title('Explicit Euler Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x, error, 'b-', label='Backward Error', linewidth=2)
plt.plot(x, error_explicit, 'g-', label='Explicit Error', linewidth=2)
plt.title('Error Comparison')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

if max_error < 1e-4:
    print("✅ SUCCESS: Backward Euler implementation is working correctly!")
    print("   Unconditional stability allows much larger timesteps.")
else:
    print("⚠️  WARNING: Error may be larger than expected.")
