# PDE SDK Overview

The PDE SDK is a finite-difference library for solving partial differential equations on uniform grids. It provides a clean, composable Python API with a modular design that supports both pure Python development and future Rust acceleration.

## Architecture

The SDK follows a modular, component-based architecture where PDE problems are composed from four main elements:

```
Equation + Domain + Boundary Conditions + Solver = Solution
```

### Core Components

#### 1. Domains (Grids)
- **UniformGrid1D**: 1D uniform finite difference grid
- **UniformGrid2D**: 2D uniform finite difference grid
- Handle spatial discretization and coordinate management

#### 2. Equations
- **HeatEquation1D/2D**: Time-dependent heat/diffusion equation
- **Poisson2D**: Elliptic Poisson equation
- Define the mathematical problem to be solved

#### 3. Boundary Conditions
- **DirichletBC**: Fixed boundary values
- **BoundaryCondition**: Base class for extensibility
- Applied at domain boundaries during solving

#### 4. Solvers
- **Time-stepping solvers**: Explicit/Implicit methods for time-dependent problems
- **Iterative solvers**: Stationary methods for elliptic problems
- Handle numerical solution algorithms

## Supported PDEs

### Parabolic (Time-Dependent)
**Heat/Diffusion Equation:**
```
∂u/∂t = α ∇²u
```
- 1D and 2D implementations
- Constant diffusivity α
- Manufactured solutions for verification

### Elliptic (Stationary)
**Poisson Equation:**
```
-∇²u = f(x,y)
```
- 2D implementation
- Arbitrary source term f(x,y)
- Iterative solution methods

## Solver Types

### Time-Stepping Solvers
- **ExplicitEuler1D/2D**: Forward Euler, first-order accurate, conditionally stable
- **CrankNicolson1D/2D**: Trapezoidal rule, second-order accurate, unconditionally stable
- **BackwardEuler1D/2D**: Backward Euler, first-order accurate, unconditionally stable

### Iterative Solvers
- **JacobiPoisson2D**: Stationary iterative method for Poisson equation

## Key Features

### Stability & Robustness
- Automatic CFL condition checking for explicit methods
- Clear error messages for stability violations
- Convergence monitoring for iterative methods

### Composability
- Mix and match components: any equation + any compatible domain + solver
- Extensible design for adding new equations, boundaries, or solvers
- Clean separation of mathematical formulation from numerical method

### Development Focus
- Pure Python implementation for clarity and ease of modification
- Comprehensive test suite with analytical solutions
- Ready for modular Rust acceleration in performance-critical components

## Usage Pattern

```python
import numpy as np
from pde_sdk import domains, equations, boundaries, solvers

# 1. Define spatial domain
grid = domains.UniformGrid1D(nx=101, length=1.0)

# 2. Set up equation with boundary conditions
equation = equations.HeatEquation1D(
    alpha=0.01,
    grid=grid,
    left_bc=boundaries.DirichletBC(0.0),
    right_bc=boundaries.DirichletBC(0.0),
    initial_condition=lambda x: np.sin(np.pi * x)
)

# 3. Choose and configure solver
solver = solvers.ExplicitEuler1D(dt=1e-4)

# 4. Solve
solution = solver.solve(equation, t_final=0.1)
```

## Project Status

**Phase 1: Pure Python SDK** ✅
- Core functionality implemented
- Test suite with analytical validation
- Working examples and documentation

**Phase 2: Rust Integration** 🔄 (Future)
- High-performance Rust backends
- PyO3 bindings for seamless integration
- Performance benchmarking

## Getting Started

Dive deeper with our comprehensive user guide:

<div class="grid cards" markdown>
- :material-function: **Equations & Boundaries**
- :material-grid: **Domains & Grids**
- :material-calculator: **Time-Stepping Solvers**
- :material-iteration: **Iterative Solvers**
- :material-chart-line: **Visualization**
- :material-lightbulb: **Best Practices**
</div>
