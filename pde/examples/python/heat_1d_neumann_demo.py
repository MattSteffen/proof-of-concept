"""
1D Heat Equation with Neumann Boundary Conditions

This example demonstrates solving the 1D heat equation with Neumann BCs
(insulated boundaries) using explicit Euler method.
"""

import matplotlib.pyplot as plt
import numpy as np

from pde_sdk.boundaries.neumann import NeumannBC
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

# Parameters
nx = 51
length = 1.0
alpha = 0.01
dt = 1e-4
t_final = 0.1

# Setup grid
grid = UniformGrid1D(nx=nx, length=length)

# Initial condition: Gaussian pulse
def initial_condition(x):
    center = length / 2
    width = 0.1
    return np.exp(-((x - center) ** 2) / (2 * width**2))

# Neumann BCs: zero gradient (insulated boundaries)
# This means no heat flux through the boundaries
left_bc = NeumannBC(0.0)  # ∂u/∂x = 0 at x=0
right_bc = NeumannBC(0.0)  # ∂u/∂x = 0 at x=L

eq = HeatEquation1D(
    alpha=alpha,
    grid=grid,
    left_bc=left_bc,
    right_bc=right_bc,
    initial_condition=initial_condition,
)

solver = ExplicitEuler1D(dt=dt)
u_final = solver.solve(eq, t_final=t_final)

# Plot results
x = grid.x
plt.figure(figsize=(10, 6))
plt.plot(x, initial_condition(x), "k--", label="Initial condition", linewidth=2)
plt.plot(x, u_final, "b-", label=f"Solution at t={t_final}", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("1D Heat Equation with Neumann BCs (Insulated Boundaries)")
plt.legend()
plt.grid(True, alpha=0.3)

# Verify Neumann BCs are satisfied
dx = grid.dx
gradient_left = (u_final[1] - u_final[0]) / dx
gradient_right = (u_final[-1] - u_final[-2]) / dx
print(f"Left boundary gradient: {gradient_left:.6f} (should be ~0)")
print(f"Right boundary gradient: {gradient_right:.6f} (should be ~0)")

plt.tight_layout()
plt.show()

