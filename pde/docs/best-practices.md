# Best Practices & Configuration

This guide provides best practices, configuration recommendations, and troubleshooting tips for using the PDE SDK effectively.

## Configuration Guidelines

### Grid Resolution Selection

#### Accuracy vs Performance Trade-offs

```python
# Rule of thumb: Δx ≈ √(target_accuracy)
# For 1% accuracy, Δx ≈ 0.1
# For 0.1% accuracy, Δx ≈ 0.01

def select_grid_resolution(target_accuracy=0.01, domain_length=1.0):
    """Select appropriate grid resolution for target accuracy"""
    dx = target_accuracy ** 0.5  # Square root rule
    nx = int(domain_length / dx) + 1

    # Round to reasonable numbers
    if nx < 50:
        nx = 50  # Minimum for reasonable accuracy
    elif nx > 1000:
        nx = 1000  # Maximum for computational feasibility

    return nx, dx

# Example usage
nx, dx = select_grid_resolution(target_accuracy=0.001, domain_length=1.0)
grid = UniformGrid1D(nx=nx, length=1.0)
print(f"Using {nx} points, dx={dx:.4f}")
```

#### Common Resolution Guidelines

| Application | Target Accuracy | Recommended nx | Δx |
|-------------|----------------|----------------|-----|
| Quick testing | 10% | 21-51 | 0.05-0.02 |
| Engineering analysis | 1% | 51-101 | 0.02-0.01 |
| Scientific publication | 0.1% | 101-201 | 0.01-0.005 |
| Reference solutions | 0.01% | 201-501 | 0.005-0.002 |

### Timestep Selection

#### Explicit Methods - CFL Condition

```python
def optimal_timestep_explicit(alpha, dx, safety_factor=0.5):
    """Calculate optimal timestep for explicit methods"""
    # 1D: dt ≤ dx²/(2α)
    # 2D: dt ≤ dx²/(4α) for square grids
    dt_max = safety_factor * dx**2 / (2 * alpha)
    return dt_max

# Example
alpha = 0.01
dx = 0.01
dt_optimal = optimal_timestep_explicit(alpha, dx)
print(f"Optimal dt: {dt_optimal:.2e}s")

# For 2D problems, reduce by another factor of 2
dt_2d = dt_optimal / 2
```

#### Implicit Methods - Accuracy Considerations

```python
def optimal_timestep_implicit(alpha, dx, target_accuracy=0.01):
    """Select timestep for implicit methods (stability not limiting)"""
    # Base on accuracy: dt ≈ dx²/α for diffusion problems
    dt_base = dx**2 / alpha

    # Adjust for target accuracy
    # Higher accuracy requires smaller dt
    accuracy_factor = target_accuracy / 0.01
    dt_optimal = dt_base * accuracy_factor

    return dt_optimal

# Example
dt_implicit = optimal_timestep_implicit(alpha=0.01, dx=0.01)
print(f"Implicit dt: {dt_implicit:.2e}s")
```

## Solver Selection Strategy

### Decision Tree

```python
def recommend_solver(equation_type, problem_size, accuracy_requirement):
    """
    Recommend appropriate solver based on problem characteristics
    """

    if equation_type == "heat_1d":
        if accuracy_requirement == "low":
            return "ExplicitEuler1D", "Fast, simple"
        elif accuracy_requirement == "medium":
            return "CrankNicolson1D", "Good accuracy, reasonable cost"
        else:  # high
            return "BackwardEuler1D", "Stable, good convergence"

    elif equation_type == "heat_2d":
        if problem_size == "small" and accuracy_requirement == "low":
            return "ExplicitEuler2D", "Fast for small problems"
        else:
            return "CrankNicolson2D", "Best balance of speed and accuracy"

    elif equation_type == "poisson":
        if problem_size == "large":
            return "JacobiPoisson2D", "Simple, but may be slow"
        else:
            return "JacobiPoisson2D", "Works well for medium problems"

    return "CrankNicolson1D", "Default choice"
```

### Performance Benchmarks

```python
import time

def benchmark_solvers():
    """Compare solver performance"""

    solvers = {
        'ExplicitEuler1D': ExplicitEuler1D(dt=1e-4),
        'CrankNicolson1D': CrankNicolson1D(dt=1e-3),
        'BackwardEuler1D': BackwardEuler1D(dt=1e-3)
    }

    results = {}

    for name, solver in solvers.items():
        start_time = time.time()
        solution = solver.solve(equation, t_final=0.1)
        elapsed = time.time() - start_time

        # Compute error
        u_exact = analytical_solution(grid.x, 0.1)
        error = np.max(np.abs(solution - u_exact))

        results[name] = {
            'time': elapsed,
            'error': error,
            'efficiency': error / elapsed  # Error per second
        }

        print(".3f")

    return results
```

## Error Analysis & Verification

### Convergence Testing

```python
def test_convergence():
    """Test spatial convergence of numerical method"""

    resolutions = [21, 41, 81, 161]
    errors = []

    for nx in resolutions:
        grid = UniformGrid1D(nx=nx, length=1.0)
        # Setup equation with manufactured solution
        # ... solve ...

        # Compute error
        x_fine = np.linspace(0, 1, 1001)
        u_exact_fine = analytical_solution(x_fine, t_final)
        u_numerical_interp = np.interp(x_fine, grid.x, solution)

        error = np.max(np.abs(u_numerical_interp - u_exact_fine))
        errors.append(error)
        print("4d")

    # Check convergence rate
    dx_values = [1.0/(nx-1) for nx in resolutions[:-1]]
    error_ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]

    print("
Convergence analysis:")
    print(f"Error ratios: {error_ratios}")
    print(f"Expected for O(Δx²): {2.0**2:.1f}")
    print(f"Achieved: {np.mean(error_ratios):.2f}")

test_convergence()
```

### Analytical Solution Verification

```python
def verify_analytical_solution():
    """Compare numerical solution with analytical solution"""

    # Define manufactured solution
    def manufactured_solution(x, y, t):
        return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    # Create appropriate source term
    def source_term(x, y):
        # -∇²u + ∂u/∂t = f for heat equation
        # For u = e^{-t} sin(πx) sin(πy):
        # ∇²u = -2π² sin(πx) sin(πy)
        # ∂u/∂t = -e^{-t} sin(πx) sin(πy)
        # So -∇²u + ∂u/∂t = 2π² e^{-t} sin(πx) sin(πy) - e^{-t} sin(πx) sin(πy)
        # = e^{-t} sin(πx) sin(πy) (2π² - 1)
        return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) * (2 * np.pi**2 - 1)

    # Setup and solve
    # ... solve equation with manufactured source ...

    # Compare solutions
    u_exact = manufactured_solution(grid.X, grid.Y, t_final)
    error = np.abs(solution - u_exact)

    print(f"Max error: {np.max(error):.2e}")
    print(f"RMS error: {np.sqrt(np.mean(error**2)):.2e}")

    return error
```

## Debugging Common Issues

### Stability Problems

```python
# Check CFL condition for explicit methods
def check_stability_explicit(solver, equation):
    """Check if explicit solver meets stability requirements"""

    if hasattr(solver, 'dt') and hasattr(equation, 'alpha'):
        grid = equation.grid
        alpha = equation.alpha
        dt = solver.dt

        if hasattr(grid, 'dx'):  # 1D
            r = alpha * dt / grid.dx**2
            stable = r <= 0.5
            print(f"1D stability: r={r:.3f}, stable={stable}")
            if not stable:
                dt_max = 0.5 * grid.dx**2 / alpha
                print(f"Recommended dt: {dt_max:.2e}")
        else:  # 2D
            rx = alpha * dt / grid.dx**2
            ry = alpha * dt / grid.dy**2
            r = max(rx, ry)
            stable = r <= 0.25
            print(f"2D stability: r={r:.3f}, stable={stable}")
            if not stable:
                dt_max = 0.25 * min(grid.dx, grid.dy)**2 / alpha
                print(f"Recommended dt: {dt_max:.2e}")

check_stability_explicit(solver, equation)
```

### Boundary Condition Issues

```python
def debug_boundary_conditions(equation):
    """Debug boundary condition setup"""

    print("Boundary condition debug info:")
    print(f"Equation type: {type(equation).__name__}")

    if hasattr(equation, 'left_bc'):
        print(f"Left BC: {equation.left_bc.value}")
        print(f"Right BC: {equation.right_bc.value}")

    if hasattr(equation, 'bottom_bc'):
        print(f"Bottom BC: {equation.bottom_bc.value}")
        print(f"Top BC: {equation.top_bc.value}")

    # Check if BCs are consistent with initial condition
    grid = equation.grid
    ic_at_boundaries = {}

    if hasattr(grid, 'x'):  # 1D
        ic_left = equation.initial_condition(grid.x[0])
        ic_right = equation.initial_condition(grid.x[-1])
        print(f"IC at left: {ic_left:.3f}, BC: {equation.left_bc.value}")
        print(f"IC at right: {ic_right:.3f}, BC: {equation.right_bc.value}")
    else:  # 2D
        ic_func = equation.initial_condition
        ic_left = np.mean([ic_func(grid.x[0], y) for y in grid.y])
        ic_right = np.mean([ic_func(grid.x[-1], y) for y in grid.y])
        print(f"Average IC at left: {ic_left:.3f}, BC: {equation.left_bc.value}")
        print(f"Average IC at right: {ic_right:.3f}, BC: {equation.right_bc.value}")

debug_boundary_conditions(equation)
```

### Convergence Issues

```python
def debug_convergence(solver, equation, t_final):
    """Debug solver convergence issues"""

    print("Convergence debug:")

    # Try with different timesteps
    timesteps = [1e-3, 1e-4, 1e-5]
    solutions = []

    for dt in timesteps:
        solver_copy = type(solver)(dt=dt)
        try:
            solution = solver_copy.solve(equation, t_final)
            max_val = np.max(np.abs(solution))
            solutions.append((dt, max_val))
            print("2e")
        except Exception as e:
            print("2e")

    # Check for timestep convergence
    if len(solutions) >= 2:
        dt1, val1 = solutions[0]
        dt2, val2 = solutions[1]
        ratio = abs(val1 - val2) / abs(val1)
        print(f"Solution change with 10x smaller dt: {ratio:.1e}")
        if ratio > 0.1:
            print("WARNING: Large change suggests timestep too large")
        elif ratio < 1e-6:
            print("Good convergence")

debug_convergence(solver, equation, t_final=0.1)
```

## Memory Management

### Large Grid Handling

```python
def optimize_memory_usage(nx, ny=None):
    """Tips for handling large grids"""

    if ny is None:  # 1D
        memory_mb = (nx * 8) / (1024**2)  # 8 bytes per float64
        print(f"1D grid memory: {memory_mb:.2f} MB")

        if memory_mb > 1000:  # > 1GB
            print("WARNING: Large memory usage")
            print("Consider:")
            print("- Reducing resolution")
            print("- Using float32 instead of float64")
            print("- Processing in chunks")
    else:  # 2D
        memory_mb = (nx * ny * 8) / (1024**2)
        print(f"2D grid memory: {memory_mb:.2f} MB")

        if memory_mb > 1000:
            print("WARNING: Large memory usage")
            print("Consider:")
            print("- Reducing resolution")
            print("- Using sparse methods for certain problems")
            print("- Domain decomposition")

    return memory_mb

# Check memory usage
memory = optimize_memory_usage(nx=501, ny=501)
```

### Efficient Computation

```python
# Use vectorized operations
import numpy as np

def efficient_computation(grid):
    """Use vectorized operations for better performance"""

    # Good: vectorized
    result = np.sin(np.pi * grid.x) * np.exp(-grid.x)

    # Avoid: loops
    # result = np.zeros_like(grid.x)
    # for i in range(len(grid.x)):
    #     result[i] = np.sin(np.pi * grid.x[i]) * np.exp(-grid.x[i])

    return result

# Profile performance
import time
start = time.time()
result = efficient_computation(grid)
elapsed = time.time() - start
print(f"Computation time: {elapsed:.3f}s")
```

## Code Organization

### Project Structure

```python
# Recommended file organization
pde_simulation/
├── config.py          # Simulation parameters
├── setup_equation.py  # Equation setup functions
├── run_simulation.py  # Main simulation script
├── analyze_results.py # Post-processing and analysis
└── visualize.py       # Plotting functions
```

### Configuration Management

```python
# config.py
class SimulationConfig:
    """Central configuration for PDE simulations"""

    def __init__(self):
        # Domain
        self.length_x = 1.0
        self.length_y = 1.0
        self.nx = 101
        self.ny = 101

        # Physical parameters
        self.alpha = 0.01
        self.t_final = 0.1

        # Solver settings
        self.solver_type = 'CrankNicolson2D'
        self.dt = 1e-3

        # Boundary conditions
        self.bc_left = 0.0
        self.bc_right = 0.0
        self.bc_bottom = 0.0
        self.bc_top = 0.0

    def get_grid(self):
        return UniformGrid2D(self.nx, self.ny, self.length_x, self.length_y)

    def get_timestep(self):
        if 'Explicit' in self.solver_type:
            return optimal_timestep_explicit(self.alpha, 1.0/self.nx)
        else:
            return self.dt

# Usage
config = SimulationConfig()
grid = config.get_grid()
dt = config.get_timestep()
```

### Reusable Components

```python
def create_heat_equation(config):
    """Factory function for heat equations"""

    grid = config.get_grid()

    # Initial condition
    def ic(x, y=None):
        if y is None:  # 1D
            return np.sin(np.pi * x)
        else:  # 2D
            return np.sin(np.pi * x) * np.sin(np.pi * y)

    # Boundary conditions
    bc = DirichletBC(config.bc_left)

    if hasattr(config, 'length_y'):  # 2D
        equation = HeatEquation2D(
            alpha=config.alpha,
            grid=grid,
            left_bc=bc, right_bc=bc,
            bottom_bc=bc, top_bc=bc,
            initial_condition=ic
        )
    else:  # 1D
        equation = HeatEquation1D(
            alpha=config.alpha,
            grid=grid,
            left_bc=bc, right_bc=bc,
            initial_condition=lambda x: ic(x)
        )

    return equation
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_simulation():
    """Profile simulation performance"""

    profiler = cProfile.Profile()
    profiler.enable()

    # Run simulation
    solution = solver.solve(equation, t_final=0.1)

    profiler.disable()

    # Print results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)

# Usage
profile_simulation()
```

### Optimization Tips

1. **Use appropriate data types**
```python
# Use float32 for visualization (saves memory)
grid.values = grid.values.astype(np.float32)

# Use float64 for computation (better accuracy)
grid.values = grid.values.astype(np.float64)
```

2. **Pre-allocate arrays**
```python
# Good
solutions = np.zeros((n_steps, grid.nx))

# Avoid
solutions = []
for step in range(n_steps):
    solutions.append(solver.solve(...))
```

3. **Avoid unnecessary computations**
```python
# Cache expensive operations
if not hasattr(self, '_cached_matrix'):
    self._cached_matrix = self.compute_matrix()

# Use in-place operations when possible
grid.values += dt * rhs
```

## Future-Proofing

### Modular Design

```python
# Design for extensibility
class PDESolver:
    """Base class for PDE solvers"""

    def __init__(self, config):
        self.config = config

    def solve(self, equation, t_final):
        """Main solve method - implement in subclasses"""
        raise NotImplementedError

    def validate_setup(self, equation):
        """Validate equation setup"""
        # Check compatibility
        return True

# Easy to extend
class CustomSolver(PDESolver):
    def solve(self, equation, t_final):
        # Custom implementation
        pass
```

### Version Control for Results

```python
def save_simulation_results(solver, equation, solution, metadata=None):
    """Save simulation results with metadata"""

    import json
    from datetime import datetime

    # Create result structure
    results = {
        'timestamp': datetime.now().isoformat(),
        'solver_type': type(solver).__name__,
        'equation_type': type(equation).__name__,
        'grid_size': list(solution.shape),
        'parameters': {
            'alpha': equation.alpha if hasattr(equation, 'alpha') else None,
            'dt': solver.dt if hasattr(solver, 'dt') else None,
        },
        'metadata': metadata or {},
        'solution': solution.tolist()  # Convert to list for JSON
    }

    # Save to file
    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filename}")
```

This comprehensive guide covers the essential best practices for using the PDE SDK effectively, from basic configuration to advanced optimization techniques.
