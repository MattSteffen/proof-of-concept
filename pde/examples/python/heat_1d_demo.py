import numpy as np
import matplotlib.pyplot as plt

from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

# Parameters
nx = 51
length = 1.0
alpha = 0.01
dt = 1e-4
t_final = 0.1

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

solver = ExplicitEuler1D(dt=dt)
u_final = solver.solve(eq, t_final=t_final)

# Analytical solution
x = grid.x
u_exact = np.exp(-np.pi**2 * alpha * t_final) * np.sin(np.pi * x)

# Plot
plt.plot(x, u_final, label="Numerical")
plt.plot(x, u_exact, "--", label="Analytical")
plt.legend()
plt.title("1D Heat Equation (Explicit Euler)")
plt.show()
