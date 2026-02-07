"""
Boundary Condition Comparison: Dirichlet vs Neumann

This example compares solutions of the 1D heat equation with:
1. Dirichlet BCs (fixed temperatures)
2. Neumann BCs (insulated boundaries)
3. Mixed BCs (one Dirichlet, one Neumann)
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
t_final = 0.1

# Setup grid
grid = UniformGrid1D(nx=nx, length=length)
x = grid.x

# Initial condition: Gaussian pulse
def initial_condition(x):
    center = length / 2
    width = 0.1
    return np.exp(-((x - center) ** 2) / (2 * width**2))

# Case 1: Dirichlet BCs (fixed temperatures)
grid1 = UniformGrid1D(nx=nx, length=length)
eq1 = HeatEquation1D(
    alpha=alpha,
    grid=grid1,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=initial_condition,
)
solver1 = ExplicitEuler1D(dt=dt)
u1 = solver1.solve(eq1, t_final=t_final)

# Case 2: Neumann BCs (insulated boundaries)
grid2 = UniformGrid1D(nx=nx, length=length)
eq2 = HeatEquation1D(
    alpha=alpha,
    grid=grid2,
    left_bc=NeumannBC(0.0),
    right_bc=NeumannBC(0.0),
    initial_condition=initial_condition,
)
solver2 = ExplicitEuler1D(dt=dt)
u2 = solver2.solve(eq2, t_final=t_final)

# Case 3: Mixed BCs (Dirichlet left, Neumann right)
grid3 = UniformGrid1D(nx=nx, length=length)
eq3 = HeatEquation1D(
    alpha=alpha,
    grid=grid3,
    left_bc=DirichletBC(0.0),
    right_bc=NeumannBC(0.0),
    initial_condition=initial_condition,
)
solver3 = ExplicitEuler1D(dt=dt)
u3 = solver3.solve(eq3, t_final=t_final)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(x, initial_condition(x), "k--", label="Initial condition", linewidth=2, alpha=0.7)
plt.plot(x, u1, "b-", label="Dirichlet BCs (u=0 at both ends)", linewidth=2)
plt.plot(x, u2, "r-", label="Neumann BCs (insulated)", linewidth=2)
plt.plot(x, u3, "g-", label="Mixed (Dirichlet left, Neumann right)", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"1D Heat Equation: Boundary Condition Comparison (t={t_final})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print boundary condition information
dx = grid.dx
print("Boundary Condition Summary:")
print("\n1. Dirichlet BCs:")
print(f"   Left: u(0) = {u1[0]:.6f}")
print(f"   Right: u(L) = {u1[-1]:.6f}")

print("\n2. Neumann BCs:")
print(f"   Left gradient: {(u2[1] - u2[0]) / dx:.6f} (should be ~0)")
print(f"   Right gradient: {(u2[-1] - u2[-2]) / dx:.6f} (should be ~0)")

print("\n3. Mixed BCs:")
print(f"   Left (Dirichlet): u(0) = {u3[0]:.6f}")
print(f"   Right (Neumann): gradient = {(u3[-1] - u3[-2]) / dx:.6f} (should be ~0)")

