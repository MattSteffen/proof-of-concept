# Iterative Solvers

Iterative solvers are used for stationary (elliptic) PDEs like the Poisson equation. The PDE SDK currently provides the Jacobi iterative method for solving elliptic problems.

## Overview

### Jacobi Method for Poisson Equation
- **Equation**: -∇²u = f(x,y)
- **Method**: Stationary iterative method
- **Convergence**: Depends on problem properties
- **Advantages**: Simple, memory efficient, easy to parallelize
- **Limitations**: May be slow for large problems

## Mathematical Background

### Poisson Equation
```
-∇²u = f(x,y)  in domain Ω
     u = g(x,y)  on boundary ∂Ω
```

### Discrete Form
For a uniform grid, the discrete Laplacian becomes:
```
∇²u ≈ (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4u[i,j]) / h²
```

### Jacobi Iteration
The Jacobi method solves the discrete system:
```
u_new[i,j] = (h²/4) * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h² f[i,j]) / (2(hx² + hy²))
```

For unit grid spacing (hx = hy = 1):
```
u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + f[i,j]) / 4
```

## Usage Example

### Basic Poisson Problem

```python
import numpy as np
from pde_sdk.domains import UniformGrid2D
from pde_sdk.equations import Poisson2D
from pde_sdk.solvers import JacobiPoisson2D

# Setup domain
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

# Define source term: f(x,y) = -2π² sin(πx) sin(πy)
# This has analytical solution u(x,y) = sin(πx) sin(πy)
f = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Create equation
equation = Poisson2D(grid, f)

# Solve with Jacobi iteration
solver = JacobiPoisson2D(max_iter=10000, tol=1e-6)
solution = solver.solve(equation)
```

### Advanced Configuration

```python
# More restrictive tolerance for higher accuracy
solver_strict = JacobiPoisson2D(max_iter=20000, tol=1e-8)

# Allow more iterations for difficult problems
solver_relaxed = JacobiPoisson2D(max_iter=50000, tol=1e-4)

# Quick solution for testing
solver_quick = JacobiPoisson2D(max_iter=1000, tol=1e-3)
```

## Convergence Analysis

### Convergence Condition
The Jacobi method converges when the spectral radius of the iteration matrix is < 1. For the Poisson equation with Dirichlet boundaries, this is generally true, but convergence can be slow for:

- High grid resolution (many points)
- Irregular domains
- Poor initial guesses

### Convergence Rate
The error typically decreases as:
```
||error||_{k+1} ≈ ρ ||error||_k
```
where ρ < 1 is the convergence rate.

### Typical Convergence Behavior
- **Good cases**: ρ ≈ 0.5-0.8, converges in hundreds of iterations
- **Poor cases**: ρ ≈ 0.95-0.99, may need thousands of iterations
- **Divergent**: ρ ≥ 1.0 (rare for Poisson with Dirichlet BCs)

## Performance Optimization

### Grid Resolution Impact
```python
# Fine grid - may need more iterations
nx, ny = 201, 201  # 40k points
solver = JacobiPoisson2D(max_iter=50000, tol=1e-6)

# Coarse grid - converges faster
nx, ny = 51, 51   # 2.5k points
solver = JacobiPoisson2D(max_iter=5000, tol=1e-6)
```

### Tolerance Selection
```python
# For visualization: 1e-3 relative error is often sufficient
solver_visual = JacobiPoisson2D(max_iter=10000, tol=1e-3)

# For quantitative analysis: 1e-6 or better
solver_analysis = JacobiPoisson2D(max_iter=50000, tol=1e-6)

# For production: 1e-8 for high precision
solver_production = JacobiPoisson2D(max_iter=100000, tol=1e-8)
```

### Initial Guess
The current implementation uses `u = 0` as initial guess. For better performance with multiple solves:

```python
# For time-dependent problems, use previous timestep as initial guess
# (would require custom implementation)
```

## Alternative Iterative Methods (Future)

The SDK could be extended with other iterative methods:

### Gauss-Seidel
- **Advantage**: Typically converges faster than Jacobi
- **Implementation**: Update values immediately as computed
- **Convergence**: 2-3x faster than Jacobi for typical problems

### Successive Over-Relaxation (SOR)
- **Advantage**: Even faster convergence with optimal ω
- **Parameter**: Relaxation factor ω (typically 1.5-1.9)
- **Tuning**: Requires problem-specific optimization

### Conjugate Gradient
- **Advantage**: Superior for large sparse systems
- **Method**: Krylov subspace method
- **Preconditioning**: Can dramatically improve convergence

## Troubleshooting

### Common Issues

1. **Non-convergence (too few iterations)**
   - Increase `max_iter` parameter
   - Check if problem is well-posed
   - Verify boundary conditions

2. **Slow convergence**
   - Reduce grid spacing for faster relative convergence
   - Use more sophisticated iterative method (future feature)
   - Consider if problem has high-frequency components

3. **Poor accuracy**
   - Decrease tolerance parameter
   - Increase maximum iterations
   - Verify source term and boundary conditions

### Debugging Tips

```python
# Monitor convergence
solver = JacobiPoisson2D(max_iter=10000, tol=1e-6)
solution = solver.solve(equation)
# Method prints convergence info when tol is reached

# Check residual manually
import numpy as np
dx, dy = grid.dx, grid.dy
residual = np.zeros_like(solution)
for i in range(1, nx-1):
    for j in range(1, ny-1):
        laplacian = (solution[i+1,j] - 2*solution[i,j] + solution[i-1,j])/dx**2 + \
                   (solution[i,j+1] - 2*solution[i,j] + solution[i,j-1])/dy**2
        residual[i,j] = -laplacian - equation.rhs[i,j]
max_residual = np.max(np.abs(residual))
print(f"Max residual: {max_residual}")
```

### Error Analysis

```python
# Compare with analytical solution
u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
error = np.abs(solution - u_exact)
print(f"Max error: {np.max(error)}")
print(f"RMS error: {np.sqrt(np.mean(error**2))}")
```

## Applications

### Common Poisson Problems

1. **Electrostatics**: -∇²φ = ρ/ε₀
2. **Steady-state heat**: -∇²T = Q/k
3. **Fluid flow**: -∇²ψ = ω (stream function)
4. **Gravitational potential**: -∇²Φ = 4πGρ

### Example: Electrostatic Potential

```python
# Point charge in 2D (approximated)
def point_charge_source(x, y, charge_x=0.5, charge_y=0.5, strength=1.0):
    # Approximate delta function with narrow Gaussian
    sigma = grid.dx * 2  # Small width
    r2 = (x - charge_x)**2 + (y - charge_y)**2
    return strength * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)

# Note: This is approximate - true point charge requires special handling
f = lambda x, y: point_charge_source(x, y)
```

## Future Enhancements

The iterative solver module could be extended with:

1. **Additional methods**: Gauss-Seidel, SOR, CG
2. **Preconditioning**: Improve convergence rate
3. **Adaptive tolerance**: Adjust based on error estimates
4. **Multigrid**: For large-scale problems
5. **Parallel implementation**: For distributed computing
