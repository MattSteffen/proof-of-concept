import numpy as np
import matplotlib.pyplot as plt

from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.crank_nicolson import CrankNicolson1D

# Parameters
nx = 51
length = 1.0
alpha = 0.01
dt = 0.01  # Much larger timestep than explicit methods
t_final = 1.0

print("1D Heat Equation - Crank-Nicolson Demo")
print("=" * 45)
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

# Solve with Crank-Nicolson
solver = CrankNicolson1D(dt=dt)
print("Solving with Crank-Nicolson...")
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

# Compare with other methods
print("Comparison with other methods:")
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D
from pde_sdk.solvers.backward_euler import BackwardEuler1D

# Explicit Euler (small timestep for stability)
grid.values = ic(grid.x)
solver_explicit = ExplicitEuler1D(dt=1e-4)
u_explicit = solver_explicit.solve(eq, t_final)
error_explicit = np.abs(u_explicit - u_exact)
steps_explicit = int(t_final / 1e-4)

# Backward Euler
grid.values = ic(grid.x)
solver_backward = BackwardEuler1D(dt=0.005)
u_backward = solver_backward.solve(eq, t_final)
error_backward = np.abs(u_backward - u_exact)
steps_backward = int(t_final / 0.005)

print(f"Explicit Euler (dt=1e-4):  {np.max(error_explicit):.2e} error, {steps_explicit:4d} steps")
print(f"Backward Euler (dt=0.005): {np.max(error_backward):.2e} error, {steps_backward:4d} steps")
print(f"Crank-Nicolson (dt=0.01):  {max_error:.2e} error, {int(t_final/dt):4d} steps")
print()

if max_error < 1e-5:
    print("✅ SUCCESS: Crank-Nicolson shows superior accuracy!")
    print("   Second-order time accuracy gives much better results.")
else:
    print("⚠️  Results may vary - check implementation")

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, u_final, 'r-', label='Crank-Nicolson', linewidth=2)
plt.plot(x, u_exact, 'k--', label='Analytical', linewidth=2)
plt.title('Crank-Nicolson Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, u_explicit, 'b-', label='Explicit Euler', linewidth=2)
plt.plot(x, u_exact, 'k--', label='Analytical', linewidth=2)
plt.title('Explicit Euler Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x, error, 'r-', label='Crank-Nicolson Error', linewidth=2)
plt.plot(x, error_explicit, 'b-', label='Explicit Error', linewidth=2)
plt.title('Error Comparison')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Crank-Nicolson Advantages:")
print("• Second-order accurate in time (vs first-order for Euler methods)")
print("• Unconditionally stable (like backward Euler)")
print("• Much higher accuracy per timestep")
print("• Ideal for high-accuracy simulations")
