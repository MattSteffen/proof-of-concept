# Configuration System

The PDE SDK provides an automatic configuration system that calculates optimal solver parameters based on your accuracy and speed preferences.

## Overview

Instead of manually selecting grid resolution (`nx`), timestep (`dt`), and solver type, you can specify your desired accuracy or speed preference, and the configuration system will automatically determine the best parameters.

## Basic Usage

### Using Target Accuracy

```python
from pde_sdk.config import SolverConfig

# Specify target accuracy (e.g., 0.001 for 0.1% accuracy)
config = SolverConfig(
    target_accuracy=0.001,
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# Get recommended parameters
grid_params = config.get_grid_params()
solver_params = config.get_solver_params()

print(f"Recommended nx: {grid_params['nx']}")
print(f"Recommended dt: {solver_params['dt']:.2e}")
print(f"Recommended solver: {config.solver_type}")
```

### Using Speed Preference

```python
# Specify speed preference (0.0 = prioritize accuracy, 1.0 = prioritize speed)
config = SolverConfig(
    speed_preference=0.7,  # 70% speed preference
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# Use the configuration
from pde_sdk.domains import UniformGrid1D
from pde_sdk.solvers import ExplicitEuler1D

grid = UniformGrid1D(**config.get_grid_params())
solver = ExplicitEuler1D(**config.get_solver_params())
```

## Accuracy vs Speed Trade-offs

The configuration system uses a sliding scale between accuracy and speed:

- **High accuracy** (low `target_accuracy` or low `speed_preference`):
  - More grid points (`nx` larger)
  - Smaller timesteps (`dt` smaller)
  - May use implicit methods for stability

- **High speed** (high `speed_preference`):
  - Fewer grid points (`nx` smaller)
  - Larger timesteps (`dt` larger)
  - Prefers explicit methods

## Parameter Calculation

### Grid Resolution

The system uses the rule of thumb: `Δx ≈ √(target_accuracy)` for second-order spatial discretization.

```python
from pde_sdk.config import calculate_optimal_nx

nx = calculate_optimal_nx(
    target_accuracy=0.001,
    domain_length=1.0,
    dimension=1
)
```

### Timestep Selection

Timesteps are calculated based on:
- **Explicit methods**: Must satisfy CFL condition
- **Implicit methods**: Based on accuracy requirements
- **Crank-Nicolson**: Can use larger timesteps due to 2nd-order accuracy

```python
from pde_sdk.config import calculate_optimal_dt

dt = calculate_optimal_dt(
    alpha=0.01,
    dx=0.01,
    solver_type='explicit',
    dimension=1
)
```

## Solver Recommendations

The system automatically selects the best solver based on your preferences:

```python
from pde_sdk.config import recommend_solver

solver_name, reason = recommend_solver(
    problem_type='heat_1d',
    accuracy_requirement='high',
    problem_size='medium'
)

print(f"Recommended: {solver_name} - {reason}")
```

## Supported Problem Types

- `'heat_1d'`: 1D heat/diffusion equation
- `'heat_2d'`: 2D heat/diffusion equation
- `'poisson_2d'`: 2D Poisson equation

## Examples

### Example 1: High Accuracy Configuration

```python
config = SolverConfig(
    target_accuracy=0.0001,  # 0.01% accuracy
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# This will result in:
# - High grid resolution (nx ~ 101-201)
# - Small timestep
# - Likely Crank-Nicolson or Backward Euler solver
```

### Example 2: Fast Configuration

```python
config = SolverConfig(
    speed_preference=0.9,  # Prioritize speed
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# This will result in:
# - Lower grid resolution (nx ~ 51-101)
# - Larger timestep (if stable)
# - Explicit Euler solver
```

### Example 3: Complete Workflow

```python
import numpy as np
from pde_sdk.config import SolverConfig
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
import pde_sdk.solvers as solvers

# Step 1: Configure
config = SolverConfig(
    target_accuracy=0.001,
    problem_type='heat_1d',
    alpha=0.01,
    domain_length=1.0
)

# Step 2: Create grid and solver
grid = UniformGrid1D(**config.get_grid_params())
solver_class = getattr(solvers, config.solver_type)
solver = solver_class(**config.get_solver_params())

# Step 3: Set up equation
equation = HeatEquation1D(
    alpha=config.alpha,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=lambda x: np.sin(np.pi * x)
)

# Step 4: Solve
solution = solver.solve(equation, t_final=0.1, verbosity='summary')
```

## Advanced Usage

### Custom Domain Lengths

For 2D problems, you can specify different lengths:

```python
config = SolverConfig(
    target_accuracy=0.001,
    problem_type='heat_2d',
    alpha=0.01,
    domain_length=2.0,      # x-direction
    domain_length_y=1.0     # y-direction
)
```

### Accessing All Parameters

```python
config = SolverConfig(...)

# Grid parameters
grid_params = config.get_grid_params()
# Returns: {'nx': 101, 'length': 1.0} for 1D
# Returns: {'nx': 101, 'ny': 101, 'length_x': 1.0, 'length_y': 1.0} for 2D

# Solver parameters
solver_params = config.get_solver_params()
# Returns: {'dt': 1e-3} for time-stepping solvers
# Returns: {'max_iter': 10000, 'tol': 1e-6} for Poisson solvers

# Direct access
print(config.nx)
print(config.dt)
print(config.solver_type)
```

## Best Practices

1. **Start with accuracy**: Specify `target_accuracy` if you know your requirements
2. **Use speed preference for exploration**: Use `speed_preference` when exploring different configurations
3. **Verify stability**: For explicit methods, the system ensures stability, but always verify results
4. **Adjust as needed**: The recommendations are starting points - adjust based on your specific problem

## Limitations

- The configuration system provides reasonable defaults but may not be optimal for all problems
- For highly specialized problems, manual parameter tuning may be necessary
- The accuracy estimates are approximate and based on standard discretization errors

