# LLM Guide: Solving PDEs with the PDE SDK

This guide provides step-by-step instructions for AI assistants (LLMs) to help users solve PDEs using the PDE SDK. Follow these patterns to provide accurate, helpful guidance.

## Quick Decision Tree

```
1. What type of PDE?
   ├─ Heat/Diffusion (time-dependent) → Use HeatEquation1D/2D
   └─ Poisson (stationary) → Use Poisson2D

2. What dimension?
   ├─ 1D → UniformGrid1D
   └─ 2D → UniformGrid2D

3. What boundary conditions?
   ├─ Fixed values → DirichletBC
   └─ Fixed gradients → NeumannBC

4. What solver?
   ├─ Fast, simple → ExplicitEuler
   ├─ Balanced → CrankNicolson (recommended)
   └─ Stable, slower → BackwardEuler
```

## Step-by-Step Pattern

### Pattern 1: Basic 1D Heat Equation

```python
# Step 1: Import required modules
import numpy as np
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import ExplicitEuler1D

# Step 2: Create grid
nx = 101  # Number of grid points
length = 1.0  # Domain length
grid = UniformGrid1D(nx=nx, length=length)

# Step 3: Define initial condition
def initial_condition(x):
    return np.sin(np.pi * x)

# Step 4: Set up equation with boundary conditions
alpha = 0.01  # Diffusion coefficient
equation = HeatEquation1D(
    alpha=alpha,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=initial_condition
)

# Step 5: Choose solver
dt = 1e-4  # Timestep (must satisfy stability: dt <= dx²/(2*alpha))
solver = ExplicitEuler1D(dt=dt)

# Step 6: Solve
t_final = 0.1
solution = solver.solve(equation, t_final=t_final, verbosity='summary')

# Step 7: Visualize
grid.plot(title="1D Heat Equation Solution")
```

### Pattern 2: Using Configuration System

```python
from pde_sdk.config import SolverConfig

# Step 1: Create configuration based on accuracy/speed preference
config = SolverConfig(
    target_accuracy=0.001,  # 0.1% accuracy
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# Step 2: Get recommended parameters
grid_params = config.get_grid_params()
solver_params = config.get_solver_params()

# Step 3: Create grid and solver
grid = UniformGrid1D(**grid_params)
solver_class = getattr(pde_sdk.solvers, config.solver_type)
solver = solver_class(**solver_params)

# Step 4: Set up equation and solve
equation = HeatEquation1D(
    alpha=config.alpha,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=lambda x: np.sin(np.pi * x)
)

solution = solver.solve(equation, t_final=0.1, verbosity='steps')
```

### Pattern 3: 2D Poisson Equation

```python
import numpy as np
from pde_sdk.domains import UniformGrid2D
from pde_sdk.equations import Poisson2D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import JacobiPoisson2D

# Step 1: Create 2D grid
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

# Step 2: Define source term
def source_term(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Step 3: Set up equation
equation = Poisson2D(
    grid=grid,
    f=source_term,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    bottom_bc=DirichletBC(0.0),
    top_bc=DirichletBC(0.0)
)

# Step 4: Solve
solver = JacobiPoisson2D(max_iter=10000, tol=1e-6)
solution = solver.solve(equation, verbosity='summary')

# Step 5: Visualize
grid.plot(title="2D Poisson Solution")
```

## Common Patterns

### Pattern: Comparing Solvers

```python
from pde_sdk.solvers import ExplicitEuler1D, CrankNicolson1D, BackwardEuler1D

# Set up same equation for all solvers
# ... (equation setup) ...

solvers = {
    'Explicit': ExplicitEuler1D(dt=1e-4),
    'Crank-Nicolson': CrankNicolson1D(dt=1e-3),
    'Backward': BackwardEuler1D(dt=1e-3)
}

solutions = {}
for name, solver in solvers.items():
    solutions[name] = solver.solve(equation, t_final=0.1, verbosity='none')
```

### Pattern: Error Analysis

```python
# Analytical solution (if known)
def analytical_solution(x, t):
    return np.exp(-np.pi**2 * 0.01 * t) * np.sin(np.pi * x)

# Compute error
u_exact = analytical_solution(grid.x, t_final)
error = np.abs(solution - u_exact)
max_error = np.max(error)

print(f"Maximum error: {max_error:.2e}")

# Plot error
from pde_sdk.visualization import plot_error
plot_error(grid.x, error, log_scale=True, title="Error Analysis")
```

### Pattern: Progress Tracking

```python
# Option 1: Use verbosity parameter
solution = solver.solve(equation, t_final=0.1, verbosity='steps')

# Option 2: Create custom progress tracker
from pde_sdk.utils import ProgressTracker

progress = ProgressTracker(verbosity='steps', total_steps=1000, description="Solving")
solution = solver.solve(equation, t_final=0.1, progress=progress)
```

## Troubleshooting Guide

### Issue: "Unstable timestep" error

**Problem**: CFL condition violated for explicit methods.

**Solution**:
1. Reduce timestep: `dt = 0.5 * dx**2 / (2 * alpha)` for 1D
2. Or switch to implicit solver: `CrankNicolson1D` or `BackwardEuler1D`

```python
# Check stability
r = alpha * dt / dx**2
if r > 0.5:  # For 1D explicit
    print(f"Unstable! r={r:.3f} > 0.5")
    dt_safe = 0.5 * dx**2 / (2 * alpha)
    print(f"Use dt <= {dt_safe:.2e}")
```

### Issue: Slow convergence for Poisson

**Problem**: Jacobi method converges slowly.

**Solution**:
1. Increase `max_iter`
2. Relax `tol` if appropriate
3. Use better initial guess (if possible)

```python
solver = JacobiPoisson2D(max_iter=50000, tol=1e-8)
```

### Issue: Poor accuracy

**Problem**: Solution doesn't match expected accuracy.

**Solution**:
1. Increase grid resolution (`nx`, `ny`)
2. Reduce timestep (`dt`)
3. Use higher-order solver (`CrankNicolson`)

```python
# Use configuration system to get optimal parameters
config = SolverConfig(target_accuracy=0.0001, problem_type='heat_1d', alpha=0.01)
```

### Issue: Memory issues with large grids

**Problem**: 2D grids consume too much memory.

**Solution**:
1. Reduce resolution
2. Use float32 instead of float64 (if accuracy allows)
3. Process in chunks (advanced)

```python
# For 2D, be conservative with resolution
nx, ny = 101, 101  # Reasonable for most cases
# Avoid: nx, ny = 1001, 1001  # Too large
```

## Parameter Selection Guidelines

### Grid Resolution

- **Quick testing**: nx = 21-51
- **Engineering analysis**: nx = 51-101
- **High accuracy**: nx = 101-201
- **Reference solutions**: nx = 201-501

### Timestep Selection

**Explicit methods** (must satisfy CFL):
- 1D: `dt <= 0.5 * dx² / (2*alpha)`
- 2D: `dt <= 0.25 * min(dx,dy)² / (4*alpha)`

**Implicit methods** (stability not limiting):
- Base on accuracy: `dt ≈ dx² / alpha`
- Can use larger timesteps

**Crank-Nicolson** (best balance):
- Can use `dt ≈ 2 * dx² / alpha` due to 2nd-order accuracy

### Solver Selection

| Requirement | 1D Recommendation | 2D Recommendation |
|-------------|-------------------|-------------------|
| Fast, simple | ExplicitEuler | ExplicitEuler |
| Balanced | CrankNicolson | CrankNicolson |
| Maximum stability | BackwardEuler | BackwardEuler |
| Stationary problems | N/A | JacobiPoisson |

## Best Practices for LLMs

1. **Always check stability** for explicit methods before solving
2. **Recommend Crank-Nicolson** as default for most cases (good balance)
3. **Use configuration system** when user specifies accuracy/speed preferences
4. **Suggest progress tracking** for long-running simulations (`verbosity='steps'`)
5. **Include visualization** in examples to help users understand results
6. **Provide error analysis** when analytical solutions are available
7. **Warn about memory** for large 2D grids (>500x500)

## Example: Complete Workflow

```python
"""
Complete workflow for solving 1D heat equation with error analysis.
"""

import numpy as np
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import CrankNicolson1D
from pde_sdk.visualization import plot_comparison, plot_error

# Configuration
nx = 101
length = 1.0
alpha = 0.01
dt = 1e-3
t_final = 0.1

# Setup
grid = UniformGrid1D(nx=nx, length=length)
ic = lambda x: np.sin(np.pi * x)
equation = HeatEquation1D(
    alpha=alpha,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=ic
)

# Solve
solver = CrankNicolson1D(dt=dt)
solution = solver.solve(equation, t_final=t_final, verbosity='summary')

# Analytical solution
u_exact = np.exp(-np.pi**2 * alpha * t_final) * np.sin(np.pi * grid.x)

# Compare
plot_comparison(grid.x, solution, u_exact, title="Numerical vs Analytical")
error = np.abs(solution - u_exact)
plot_error(grid.x, error, log_scale=True, title="Error")
print(f"Max error: {np.max(error):.2e}")
```

This guide should enable LLMs to provide accurate, helpful guidance for solving PDEs with the SDK.

