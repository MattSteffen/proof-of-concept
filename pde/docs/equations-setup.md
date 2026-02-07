# Equations & Boundary Conditions

This guide covers how to set up PDE equations and boundary conditions in the PDE SDK. The SDK supports both time-dependent (parabolic) and stationary (elliptic) equations.

## Supported Equations

### Heat/Diffusion Equation (Parabolic)

#### 1D Heat Equation
```
∂u/∂t = α ∂²u/∂x²    for x ∈ [0, L], t > 0
```

**Parameters:**
- `α`: Thermal diffusivity (positive constant)
- `L`: Domain length
- `u(x,0)`: Initial condition
- Boundary conditions at x=0 and x=L

#### 2D Heat Equation
```
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)    for (x,y) ∈ [0, Lx]×[0, Ly], t > 0
```

**Parameters:**
- `α`: Thermal diffusivity
- `Lx`, `Ly`: Domain dimensions
- `u(x,y,0)`: Initial condition
- Boundary conditions on all four sides

### Poisson Equation (Elliptic)

#### 2D Poisson Equation
```
-∇²u = f(x,y)    for (x,y) ∈ [0, Lx]×[0, Ly]
```

**Parameters:**
- `f(x,y)`: Source term (right-hand side function)
- Boundary conditions on all boundaries

## Setting Up Equations

### 1D Heat Equation

```python
import numpy as np
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC

# 1. Define spatial domain
nx = 101
grid = UniformGrid1D(nx=nx, length=1.0)

# 2. Define initial condition
def initial_condition(x):
    return np.sin(np.pi * x)  # Sine wave

# 3. Set boundary conditions
left_bc = DirichletBC(0.0)   # u(0,t) = 0
right_bc = DirichletBC(0.0)  # u(L,t) = 0

# 4. Create equation
equation = HeatEquation1D(
    alpha=0.01,           # Diffusivity
    grid=grid,
    left_bc=left_bc,
    right_bc=right_bc,
    initial_condition=initial_condition
)
```

### 2D Heat Equation

```python
from pde_sdk.domains import UniformGrid2D
from pde_sdk.equations import HeatEquation2D

# 1. Define 2D spatial domain
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

# 2. Define initial condition
def initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# 3. Set boundary conditions (all zero)
bc_zero = DirichletBC(0.0)
equation = HeatEquation2D(
    alpha=0.01,
    grid=grid,
    left_bc=bc_zero,
    right_bc=bc_zero,
    bottom_bc=bc_zero,
    top_bc=bc_zero,
    initial_condition=initial_condition
)
```

### 2D Poisson Equation

```python
from pde_sdk.equations import Poisson2D

# 1. Define 2D domain (same as above)
nx, ny = 51, 51
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

# 2. Define source term f(x,y)
# For manufactured solution u = sin(πx)sin(πy)
# We have -∇²u = -2π² sin(πx)sin(πy) = f(x,y)
def source_term(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# 3. Create equation (defaults to Dirichlet BCs with value 0.0)
equation = Poisson2D(grid, source_term)

# Or specify boundary conditions explicitly
from pde_sdk.boundaries import DirichletBC, NeumannBC

equation = Poisson2D(
    grid=grid,
    f=source_term,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    bottom_bc=DirichletBC(0.0),
    top_bc=DirichletBC(0.0)
)

# Example with Neumann BCs (insulated boundaries)
equation_neumann = Poisson2D(
    grid=grid,
    f=source_term,
    left_bc=NeumannBC(0.0),   # Zero gradient
    right_bc=NeumannBC(0.0),
    bottom_bc=NeumannBC(0.0),
    top_bc=NeumannBC(0.0)
)
```

## Boundary Conditions

### Supported Boundary Conditions

#### Dirichlet Boundary Conditions
Fixed boundary values:
```
u = g(x,y)    on boundary
```

**Usage:**
```python
from pde_sdk.boundaries import DirichletBC

# Constant value
bc = DirichletBC(0.0)

# The value can be any float
bc = DirichletBC(1.5)
```

#### Neumann Boundary Conditions
Fixed normal derivative (gradient) at boundaries:
```
∂u/∂n = g    on boundary
```

For 1D: `∂u/∂x = g`  
For 2D: `∂u/∂n = g` (normal derivative)

**Usage:**
```python
from pde_sdk.boundaries import NeumannBC

# Zero gradient (insulated boundary)
bc = NeumannBC(0.0)

# Non-zero gradient
bc = NeumannBC(1.5)  # ∂u/∂n = 1.5
```

**Example with Neumann BCs:**
```python
from pde_sdk.boundaries import DirichletBC, NeumannBC
from pde_sdk.equations.heat import HeatEquation1D

grid = UniformGrid1D(nx=101, length=1.0)

# Zero gradient on left, fixed value on right
left_bc = NeumannBC(0.0)   # Insulated
right_bc = DirichletBC(0.0)  # Fixed temperature

eq = HeatEquation1D(
    alpha=0.01,
    grid=grid,
    left_bc=left_bc,
    right_bc=right_bc,
    initial_condition=lambda x: np.sin(np.pi * x)
)
```

### Boundary Condition Application

#### 1D Boundaries
- `left_bc`: Applied at x = 0
- `right_bc`: Applied at x = L

#### 2D Boundaries
- `left_bc`: Applied at x = 0 (all y)
- `right_bc`: Applied at x = Lx (all y)
- `bottom_bc`: Applied at y = 0 (all x)
- `top_bc`: Applied at y = Ly (all x)

### Common Boundary Patterns

```python
# All boundaries zero (most common for eigenproblems)
bc_zero = DirichletBC(0.0)
equation = HeatEquation2D(
    alpha=0.01, grid=grid,
    left_bc=bc_zero, right_bc=bc_zero,
    bottom_bc=bc_zero, top_bc=bc_zero,
    initial_condition=ic
)

# Different values on different boundaries
equation = HeatEquation2D(
    alpha=0.01, grid=grid,
    left_bc=DirichletBC(0.0),    # Cold wall
    right_bc=DirichletBC(1.0),   # Hot wall
    bottom_bc=DirichletBC(0.5),  # Intermediate
    top_bc=DirichletBC(0.5),     # Intermediate
    initial_condition=ic
)
```

## Initial Conditions

### For Heat Equations

Initial conditions can be:
- **Callable functions**: `lambda x: np.sin(np.pi * x)`
- **NumPy arrays**: Pre-computed values matching grid size
- **Constants**: `lambda x, y: 1.0` (for 2D)

### Examples

```python
# 1D examples
ic_sine = lambda x: np.sin(np.pi * x)
ic_gaussian = lambda x: np.exp(-((x - 0.5) / 0.1)**2)
ic_step = lambda x: 1.0 if x > 0.5 else 0.0
ic_random = lambda x: np.random.random(len(x))

# 2D examples
def ic_product(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def ic_gaussian_2d(x, y):
    return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01)

def ic_step_2d(x, y):
    return 1.0 if (x > 0.3) and (x < 0.7) and (y > 0.3) and (y < 0.7) else 0.0
```

### Initial Condition Validation

The SDK automatically:
- Checks that initial conditions match grid dimensions
- Handles callable functions by evaluating on grid coordinates
- Converts arrays to the proper format

## Source Terms (for Poisson)

### For Poisson Equations

Source terms should be:
- **Callable functions**: `lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)`
- **Accept arrays**: Function should work with numpy arrays

### Common Source Patterns

```python
# Constant source
f_constant = lambda x, y: 1.0

# Linear source
f_linear = lambda x, y: x + y

# Gaussian source
def f_gaussian(x, y):
    cx, cy = 0.5, 0.5  # Center
    sigma = 0.1
    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

# Manufactured solution source
# For u = x² + y², we have -∇²u = -2 - 2 = -4
f_manufactured = lambda x, y: -4.0
```

## Parameter Selection

### Diffusivity (α) Selection

```python
# Physical values
alpha_air = 2.2e-5      # Air at room temperature (m²/s)
alpha_water = 1.4e-7    # Water (m²/s)
alpha_copper = 1.1e-4   # Copper (m²/s)

# Normalized values (common in math problems)
alpha_unit = 1.0        # Normalized diffusivity
alpha_small = 0.01      # Slow diffusion
alpha_large = 10.0      # Fast diffusion
```

### Dimension Selection

```python
# Square domain
grid = UniformGrid2D(nx=51, ny=51, length_x=1.0, length_y=1.0)

# Rectangular domain
grid = UniformGrid2D(nx=101, ny=51, length_x=2.0, length_y=1.0)

# High resolution
grid = UniformGrid2D(nx=201, ny=201, length_x=1.0, length_y=1.0)
```

## Equation Validation

### Common Issues

1. **Wrong initial condition dimensions**
   - Ensure callable returns correct size for grid
   - Check that arrays match `grid.nx` or `(grid.nx, grid.ny)`

2. **Boundary condition mismatches**
   - Verify all boundaries are specified for 2D problems
   - Check that boundary values are reasonable

3. **Source term issues (Poisson)**
   - Ensure function works with array inputs
   - Check mathematical correctness

### Validation Examples

```python
# Test initial condition
x_test = np.linspace(0, 1, 10)
ic_values = initial_condition(x_test)
print(f"IC shape: {ic_values.shape}")  # Should be (10,)

# Test 2D initial condition
x_test = np.linspace(0, 1, 10)
y_test = np.linspace(0, 1, 10)
X_test, Y_test = np.meshgrid(x_test, y_test)
ic_values = initial_condition(X_test, Y_test)
print(f"IC shape: {ic_values.shape}")  # Should be (10, 10)

# Test source term
f_values = source_term(X_test, Y_test)
print(f"Source shape: {f_values.shape}")  # Should be (10, 10)
```

## Advanced Usage

### Custom Equation Setup

```python
# Time-dependent source (would require custom equation class)
class HeatEquationWithSource(HeatEquation1D):
    def __init__(self, alpha, grid, left_bc, right_bc, ic, source_func):
        super().__init__(alpha, grid, left_bc, right_bc, ic)
        self.source = source_func
```

### Multiple Equations

```python
# Set up multiple equations for comparison
equations = []
alphas = [0.01, 0.1, 1.0]

for alpha in alphas:
    eq = HeatEquation1D(
        alpha=alpha,
        grid=grid,
        left_bc=DirichletBC(0.0),
        right_bc=DirichletBC(0.0),
        initial_condition=ic
    )
    equations.append(eq)
```

### Parameter Studies

```python
# Grid refinement study
grids = []
n_values = [21, 41, 81, 161]

for n in n_values:
    grid = UniformGrid1D(nx=n, length=1.0)
    grids.append(grid)
```

## Future Extensions

The equation system is designed to be extensible:

### Planned Features
- **Robin boundary conditions**: ∂u/∂n + κu = g
- **Time-dependent boundary conditions**: u = g(t)
- **Variable coefficients**: α(x,y) instead of constant
- **Non-rectangular domains**: Complex geometries

### Custom Equations
Users can create custom equation classes by inheriting from base classes and implementing the required interface.
