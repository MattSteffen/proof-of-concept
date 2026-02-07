"""
1D Heat Equation with Mixed Dirichlet and Neumann Boundary Conditions

This example shows a heat conduction problem where:
- Left boundary: Fixed temperature (Dirichlet)
- Right boundary: Insulated (Neumann, zero gradient)
"""

import matplotlib.pyplot as plt
import numpy as np

from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.boundaries.neumann import NeumannBC
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

# Parameters
nx = 51
length = 1.0
alpha = 0.01
dt = 1e-4
t_final = 0.2

# Setup grid
grid = UniformGrid1D(nx=nx, length=length)

# Initial condition: cold rod
def initial_condition(x):
    return np.zeros_like(x)

# Mixed BCs:
# Left: Fixed temperature (hot wall)
# Right: Insulated (zero gradient)
left_bc = DirichletBC(1.0)  # u = 1.0 at x=0
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
plt.plot(x, initial_condition(x), "k--", label="Initial condition (cold)", linewidth=2)
plt.plot(x, u_final, "r-", label=f"Solution at t={t_final}", linewidth=2)
plt.axhline(y=1.0, color="orange", linestyle=":", label="Left BC (fixed temp)", alpha=0.7)
plt.xlabel("x")
plt.ylabel("Temperature u(x,t)")
plt.title("1D Heat Equation: Hot Left Wall, Insulated Right")
plt.legend()
plt.grid(True, alpha=0.3)

# Verify boundary conditions
print(f"Left boundary value: {u_final[0]:.6f} (should be 1.0)")
dx = grid.dx
gradient_right = (u_final[-1] - u_final[-2]) / dx
print(f"Right boundary gradient: {gradient_right:.6f} (should be ~0)")

plt.tight_layout()
plt.show()

