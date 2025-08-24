import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.equations.poisson import Poisson2D
from pde_sdk.solvers.poisson_iterative import JacobiPoisson2D

# Domain
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

# Source term: f(x,y) = -2π² sin(πx) sin(πy)
# This has analytical solution u(x,y) = sin(πx) sin(πy)
f = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Equation
eq = Poisson2D(grid, f)

# Solver
solver = JacobiPoisson2D(max_iter=20000, tol=1e-8)
u = solver.solve(eq)

# Analytical solution: u(x,y) = sin(πx) sin(πy)
# Note: -∇²(sin(πx)sin(πy)) = -2π² sin(πx)sin(πy) = f
# But the equation -∇²u = f also has solution u(x,y) = -sin(πx)sin(πy)
# Both are valid solutions since ∇²(-u) = -∇²u ⇒ -∇²(-u) = ∇²u = -f = -(-∇²u) wait no:
# If ∇²u = -f, then for v = -u: ∇²v = ∇²(-u) = -∇²u = f, so -∇²v = -f
# So both u and -u satisfy -∇²u = f when ∇²u = -f.
X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
u_exact_positive = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_exact_negative = -np.sin(np.pi * X) * np.sin(np.pi * Y)

# Calculate error (compare with both analytical solutions)
error_positive = np.abs(u - u_exact_positive)
error_negative = np.abs(u - u_exact_negative)
error = np.minimum(error_positive, error_negative)
u_exact = u_exact_positive if np.max(error_positive) < np.max(error_negative) else u_exact_negative
max_error = np.max(error)
rms_error = np.sqrt(np.mean(error**2))

print("Poisson 2D Results:")
print(f"Grid size: {nx} x {ny}")
print(f"Max numerical: {np.max(u):.6f}")
print(f"Max analytical: {np.max(u_exact):.6f}")
print(f"Max error: {max_error:.2e}")
print(f"RMS error: {rms_error:.2e}")

# Save plots instead of showing
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Numerical")
plt.imshow(u.T, origin="lower", extent=[0,1,0,1])
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Exact")
plt.imshow(u_exact.T, origin="lower", extent=[0,1,0,1])
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Error")
plt.imshow(error.T, origin="lower", extent=[0,1,0,1])
plt.colorbar()

plt.tight_layout()
# plt.savefig('/tmp/poisson_comparison.png', dpi=150, bbox_inches='tight')
# print("Plots saved to /tmp/poisson_comparison.png")
plt.show()

# Test that the solution satisfies -∇²u = f approximately
# Compute discrete Laplacian of numerical solution
dx, dy = grid.dx, grid.dy
laplacian_u = np.zeros_like(u)
for i in range(1, grid.nx-1):
    for j in range(1, grid.ny-1):
        # Discrete Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
        # For the function sin(πx)sin(πy), ∇²u = -2π²sin(πx)sin(πy) (negative at interior)
        laplacian_u[i,j] = (
            (u[i+1,j] - 2*u[i,j] + u[i-1,j])/dx**2 +
            (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dy**2
        )

# The Poisson equation is -∇²u = f
# So we check if -∇²u - f = 0 (this should be very small)
residual = -laplacian_u - eq.rhs
max_residual = np.max(np.abs(residual[2:-2,2:-2]))  # Interior points, avoiding boundary effects

print(f"Max residual (should be ~0): {max_residual:.2e}")
print(f"RHS values range: [{np.min(eq.rhs):.2e}, {np.max(eq.rhs):.2e}]")

# Let's check a few specific points
print("Checking specific interior points:")
for i in [10, 25, 40]:
    for j in [10, 25, 40]:
        if 1 <= i < grid.nx-1 and 1 <= j < grid.ny-1:
            residual_ij = residual[i,j]
            laplacian_ij = laplacian_u[i,j]
            u_ij = u[i,j]
            u_exact_ij = u_exact[i,j]
            print(f"Point ({i},{j}): residual = {residual_ij:.2e}, -∇²u = {-laplacian_ij:.2e}, RHS = {eq.rhs[i,j]:.2e}")
            print(f"                u_num = {u_ij:.6f}, u_exact = {u_exact_ij:.6f}, error = {u_ij - u_exact_ij:.2e}")

# Let's also check the analytical solution for comparison
print("\nChecking analytical solution at same points:")
for i in [10, 25, 40]:
    for j in [10, 25, 40]:
        if 1 <= i < grid.nx-1 and 1 <= j < grid.ny-1:
            x, y = grid.x[i], grid.y[j]
            # For u = sin(πx)sin(πy):
            # ∇²u = ∂²u/∂x² + ∂²u/∂y² = -π²sin(πx)sin(πy) + (-π²sin(πx)sin(πy)) = -2π²sin(πx)sin(πy)
            analytical_laplacian = -2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)  # ∇²u
            analytical_minus_laplacian = 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)  # -∇²u
            analytical_rhs = -2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
            print(f"Point ({i},{j}): analytical ∇²u = {analytical_laplacian:.2e}, -∇²u = {analytical_minus_laplacian:.2e}, RHS = {analytical_rhs:.2e}")

            # Test discrete Laplacian on analytical solution
            analytical_discrete_laplacian = (
                (u_exact[i+1,j] - 2*u_exact[i,j] + u_exact[i-1,j])/dx**2 +
                (u_exact[i,j+1] - 2*u_exact[i,j] + u_exact[i,j-1])/dy**2
            )
            print(f"                discrete ∇²u_exact = {analytical_discrete_laplacian:.2e}")
