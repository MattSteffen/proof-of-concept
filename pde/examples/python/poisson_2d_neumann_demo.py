"""
2D Poisson Equation with Neumann Boundary Conditions

This example solves the Poisson equation with Neumann BCs (zero gradient)
on all boundaries. Note: For Neumann BCs, the solution is only determined
up to an additive constant.
"""

import matplotlib.pyplot as plt
import numpy as np

from pde_sdk.boundaries.neumann import NeumannBC
from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.equations.poisson import Poisson2D
from pde_sdk.solvers.poisson_iterative import JacobiPoisson2D

# Parameters
nx, ny = 51, 51
length_x, length_y = 1.0, 1.0

# Setup grid
grid = UniformGrid2D(nx=nx, ny=ny, length_x=length_x, length_y=length_y)

# Source term: constant heat source
def source_term(x, y):
    return np.ones_like(x) * 2.0

# Neumann BCs: zero gradient on all boundaries
left_bc = NeumannBC(0.0)
right_bc = NeumannBC(0.0)
bottom_bc = NeumannBC(0.0)
top_bc = NeumannBC(0.0)

eq = Poisson2D(
    grid=grid,
    f=source_term,
    left_bc=left_bc,
    right_bc=right_bc,
    bottom_bc=bottom_bc,
    top_bc=top_bc,
)

solver = JacobiPoisson2D(max_iter=5000, tol=1e-6)
u_final = solver.solve(eq)

# Plot results
X, Y = grid.X, grid.Y
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Solution
im1 = axes[0].contourf(X, Y, u_final, levels=20, cmap="viridis")
axes[0].set_title("Poisson Solution (Neumann BCs)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im1, ax=axes[0])

# Source term
source_vals = source_term(X, Y)
im2 = axes[1].contourf(X, Y, source_vals, levels=20, cmap="Reds")
axes[1].set_title("Source Term f(x,y)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im2, ax=axes[1])

plt.suptitle("2D Poisson Equation with Neumann BCs (Zero Gradient)")
plt.tight_layout()
plt.show()

# Verify Neumann BCs
dx, dy = grid.dx, grid.dy
print("Boundary gradient checks (should be ~0):")
print(f"  Left boundary: max = {np.max(np.abs((u_final[1, :] - u_final[0, :]) / dx)):.6f}")
print(f"  Right boundary: max = {np.max(np.abs((u_final[-1, :] - u_final[-2, :]) / dx)):.6f}")
print(f"  Bottom boundary: max = {np.max(np.abs((u_final[:, 1] - u_final[:, 0]) / dy)):.6f}")
print(f"  Top boundary: max = {np.max(np.abs((u_final[:, -1] - u_final[:, -2]) / dy)):.6f}")

