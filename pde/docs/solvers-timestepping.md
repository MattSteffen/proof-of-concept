# Time-Stepping Solvers

Time-stepping solvers are used for time-dependent (parabolic) PDEs like the heat equation. The PDE SDK provides three main time-stepping methods: explicit Euler, backward Euler, and Crank-Nicolson.

## Overview

### Explicit Euler (Forward Euler)
- **Accuracy**: First-order in time (O(Δt))
- **Stability**: Conditionally stable (CFL condition required)
- **Advantages**: Simple, fast per step, minimal memory
- **Disadvantages**: Small timesteps required for stability

### Backward Euler
- **Accuracy**: First-order in time (O(Δt))
- **Stability**: Unconditionally stable
- **Advantages**: Large timesteps allowed, stable
- **Disadvantages**: Requires solving linear system each step

### Crank-Nicolson (Trapezoidal)
- **Accuracy**: Second-order in time (O(Δt²))
- **Stability**: Unconditionally stable
- **Advantages**: Best accuracy, large timesteps allowed
- **Disadvantages**: More complex, linear system required

## Stability Analysis

### CFL Condition (Explicit Methods)

For explicit Euler, the timestep must satisfy:
```
Δt ≤ Δx² / (2α)    (1D heat equation)
Δt ≤ 1/(2α(1/Δx² + 1/Δy²))    (2D heat equation)
```

The SDK automatically checks these conditions and raises informative errors if violated.

### Example Stability Limits
- 1D heat: α=0.01, Δx=0.01, Δt ≤ 5e-4
- 2D heat: α=0.01, Δx=Δy=0.01, Δt ≤ 2.5e-4

## Usage Examples

### 1D Heat Equation - Explicit Euler

```python
import numpy as np
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import ExplicitEuler1D

# Setup
nx = 101
grid = UniformGrid1D(nx=nx, length=1.0)
initial_condition = lambda x: np.sin(np.pi * x)

equation = HeatEquation1D(
    alpha=0.01,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=initial_condition
)

# Solve with explicit Euler
solver = ExplicitEuler1D(dt=1e-4)  # Must satisfy CFL condition
solution = solver.solve(equation, t_final=0.1)
```

### 1D Heat Equation - Crank-Nicolson

```python
from pde_sdk.solvers import CrankNicolson1D

# Same setup as above, but can use larger timesteps
solver = CrankNicolson1D(dt=1e-3)  # Larger dt possible due to stability
solution = solver.solve(equation, t_final=0.1)
```

### 2D Heat Equation - Explicit Euler

```python
from pde_sdk.domains import UniformGrid2D
from pde_sdk.equations import HeatEquation2D
from pde_sdk.solvers import ExplicitEuler2D

# Setup 2D grid
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

def initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# All boundaries at 0
bc = DirichletBC(0.0)
equation = HeatEquation2D(
    alpha=0.01,
    grid=grid,
    left_bc=bc, right_bc=bc, bottom_bc=bc, top_bc=bc,
    initial_condition=initial_condition
)

# Solve (smaller dt needed for 2D stability)
solver = ExplicitEuler2D(dt=5e-5)
solution = solver.solve(equation, t_final=0.05)
```

### 2D Heat Equation - Crank-Nicolson

```python
from pde_sdk.solvers import CrankNicolson2D

# Same 2D setup, but can use larger timesteps
solver = CrankNicolson2D(dt=1e-3)  # Much larger dt possible
solution = solver.solve(equation, t_final=0.05)
```

## Choosing the Right Solver

### When to Use Explicit Euler
- Simple problems with known stability requirements
- When you need maximum performance per timestep
- For educational purposes (conceptually simplest)
- When timestep is constrained by other considerations

### When to Use Backward Euler
- When stability is the primary concern
- For stiff problems requiring large timesteps
- When you need guaranteed convergence
- Good compromise between complexity and stability

### When to Use Crank-Nicolson
- When accuracy is more important than simplicity
- For production simulations requiring high precision
- When you want the best possible convergence rate
- When computational cost of linear systems is acceptable

## Performance Considerations

### Computational Cost per Timestep
- **Explicit Euler**: O(N) - just array operations
- **Backward Euler**: O(N) - sparse matrix-vector multiplication + solve
- **Crank-Nicolson**: O(N) - similar to backward Euler

### Total Cost for Same Accuracy
- **Explicit Euler**: Many small steps needed
- **Backward Euler**: Fewer steps, but each more expensive
- **Crank-Nicolson**: Fewest steps needed due to higher accuracy

### Memory Usage
- **Explicit Euler**: Minimal (just grid storage)
- **Backward Euler/Crank-Nicolson**: Matrix storage + factorization

## Error Analysis

### Convergence Rates
- **Explicit Euler**: O(Δt) temporal error + O(Δx²) spatial error
- **Backward Euler**: O(Δt) temporal error + O(Δx²) spatial error
- **Crank-Nicolson**: O(Δt²) temporal error + O(Δx²) spatial error

### Practical Accuracy Comparison
For the same computational effort:
- Crank-Nicolson typically gives best accuracy
- Backward Euler is more accurate than explicit for same stability
- Explicit Euler requires many steps for comparable accuracy

## Advanced Usage

### Custom Timestepping
```python
# Variable timestep (not built-in, but you can implement)
solver = ExplicitEuler1D(dt=1e-4)
t = 0.0
while t < t_final:
    # Adapt timestep based on some criterion
    dt = adapt_timestep(t, solution)
    solver.dt = dt
    solution = solver.solve(equation, dt)
    t += dt
```

### Monitoring Solution
```python
# Track solution evolution
solver = CrankNicolson1D(dt=1e-3)
solutions = []
times = []
t = 0.0
while t < t_final:
    solution = solver.solve(equation, solver.dt)
    solutions.append(solution.copy())
    times.append(t)
    t += solver.dt
```

## Troubleshooting

### Common Issues

1. **Stability violations (explicit methods)**
   - Error: "Unstable timestep"
   - Solution: Reduce Δt or increase grid resolution

2. **Poor convergence (implicit methods)**
   - Check boundary conditions
   - Verify equation setup
   - Ensure proper initial conditions

3. **Memory issues (large 2D problems)**
   - Use smaller grids for testing
   - Consider domain decomposition (future feature)

### Debugging Tips
- Start with small test problems
- Compare against analytical solutions
- Use coarse grids initially
- Verify boundary condition implementation
