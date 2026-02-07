"""
2D Heat Equation with Neumann Boundary Conditions

This example demonstrates solving the 2D heat equation with Neumann BCs
(insulated boundaries) using explicit Euler method.
"""

import matplotlib.pyplot as plt
import numpy as np

from pde_sdk.boundaries.neumann import NeumannBC
from pde_sdk.domains.uniform2d import UniformGrid2D
from pde_sdk.equations.heat import HeatEquation2D
from pde_sdk.solvers.explicit_euler import ExplicitEuler2D

# Parameters
nx, ny = 51, 51
length_x, length_y = 1.0, 1.0
alpha = 0.01
dt = 1e-6
t_final = 0.05

# Setup grid
grid = UniformGrid2D(nx=nx, ny=ny, length_x=length_x, length_y=length_y)

# Initial condition: Gaussian pulse in center
def initial_condition(x, y):
    center_x, center_y = length_x / 2, length_y / 2
    width = 0.15
    return np.exp(
        -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * width**2)
    )

# Neumann BCs: zero gradient on all boundaries (insulated)
left_bc = NeumannBC(0.0)  # ∂u/∂x = 0 at x=0
right_bc = NeumannBC(0.0)  # ∂u/∂x = 0 at x=Lx
bottom_bc = NeumannBC(0.0)  # ∂u/∂y = 0 at y=0
top_bc = NeumannBC(0.0)  # ∂u/∂y = 0 at y=Ly

eq = HeatEquation2D(
    alpha=alpha,
    grid=grid,
    left_bc=left_bc,
    right_bc=right_bc,
    bottom_bc=bottom_bc,
    top_bc=top_bc,
    initial_condition=initial_condition,
)

solver = ExplicitEuler2D(dt=dt)
u_final = solver.solve(eq, t_final=t_final)

# Plot results
X, Y = grid.X, grid.Y
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Initial condition
u_initial = initial_condition(X, Y)
im1 = axes[0].contourf(X, Y, u_initial, levels=20, cmap="hot")
axes[0].set_title("Initial Condition")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im1, ax=axes[0])

# Final solution
im2 = axes[1].contourf(X, Y, u_final, levels=20, cmap="hot")
axes[1].set_title(f"Solution at t={t_final}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im2, ax=axes[1])

plt.suptitle("2D Heat Equation with Neumann BCs (Insulated Boundaries)")
plt.tight_layout()
plt.show()

# Verify Neumann BCs
dx, dy = grid.dx, grid.dy
print("Boundary gradient checks:")
print(f"  Left boundary (x=0): {(u_final[1, :] - u_final[0, :]) / dx}")
print(f"  Right boundary (x=Lx): {(u_final[-1, :] - u_final[-2, :]) / dx}")
print(f"  Bottom boundary (y=0): {(u_final[:, 1] - u_final[:, 0]) / dy}")
print(f"  Top boundary (y=Ly): {(u_final[:, -1] - u_final[:, -2]) / dy}")

