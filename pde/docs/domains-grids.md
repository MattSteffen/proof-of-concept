# Domains & Grids

This guide covers the spatial discretization used in the PDE SDK. The SDK uses uniform finite difference grids for both 1D and 2D problems.

## Grid Architecture

### UniformGrid1D

#### Properties
- **nx**: Number of grid points
- **length**: Domain length
- **dx**: Grid spacing (length / (nx - 1))
- **x**: Coordinate array of shape (nx,)
- **values**: Solution array of shape (nx,)

#### Coordinate System
```
x = [0, dx, 2*dx, ..., (nx-1)*dx]
```
where `dx = length / (nx - 1)`.

#### Usage
```python
from pde_sdk.domains import UniformGrid1D

# Basic setup
grid = UniformGrid1D(nx=101, length=1.0)
print(f"Grid spacing: {grid.dx}")
print(f"Coordinates: {grid.x[:5]}...")  # First 5 points
```

### UniformGrid2D

#### Properties
- **nx, ny**: Number of grid points in x and y
- **length_x, length_y**: Domain dimensions
- **dx, dy**: Grid spacing in x and y
- **x**: x-coordinate array of shape (nx,)
- **y**: y-coordinate array of shape (ny,)
- **X, Y**: 2D coordinate meshes of shape (nx, ny)
- **values**: Solution array of shape (nx, ny)

#### Coordinate System
```
x = [0, dx, 2*dx, ..., (nx-1)*dx]
y = [0, dy, 2*dy, ..., (ny-1)*dy]
X, Y = meshgrid(x, y)  # Using indexing='ij'
```

#### Usage
```python
from pde_sdk.domains import UniformGrid2D

# Square domain
grid = UniformGrid2D(nx=51, ny=51, length_x=1.0, length_y=1.0)

# Rectangular domain
grid = UniformGrid2D(nx=101, ny=51, length_x=2.0, length_y=1.0)

print(f"Grid shape: {grid.values.shape}")
print(f"X coordinates: {grid.x[:3]}")
print(f"Y coordinates: {grid.y[:3]}")
```

## Grid Resolution Guidelines

### Resolution Selection

#### Rule of Thumb
- **Coarse**: nx = 21-51, for quick testing
- **Medium**: nx = 51-101, for most problems
- **Fine**: nx = 101-201, for high accuracy
- **Very Fine**: nx = 201-501, for convergence studies

#### Memory Usage
```python
# Memory scales with number of points
nx, ny = 101, 101
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)
memory_bytes = grid.values.nbytes  # 8 bytes per float64
print(f"Memory usage: {memory_bytes / 1024**2:.1f} MB")
```

### Accuracy Considerations

#### Spatial Error
Finite difference methods have O(Δx²) spatial error. For convergence:
- **Δx = 0.01**: Good for most engineering problems
- **Δx = 0.005**: High accuracy applications
- **Δx = 0.001**: Reference solutions

#### Example Grid Selection
```python
# For accuracy ~1%
length = 1.0
target_accuracy = 0.01  # 1% relative error
nx = int(length / target_accuracy) + 1
grid = UniformGrid1D(nx=nx, length=length)
print(f"dx = {grid.dx:.4f} for {nx} points")
```

## Grid Initialization

### Default Values
```python
# All grids initialize with zeros
grid = UniformGrid1D(nx=10, length=1.0)
print(grid.values)  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

grid = UniformGrid2D(nx=3, ny=3, length_x=1.0, length_y=1.0)
print(grid.values)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]
```

### Custom Initial Values
```python
import numpy as np

# 1D custom initialization
grid = UniformGrid1D(nx=101, length=1.0)
grid.values = np.sin(np.pi * grid.x)  # Sine wave

# 2D custom initialization
grid = UniformGrid2D(nx=51, ny=51, length_x=1.0, length_y=1.0)
X, Y = grid.X, grid.Y
grid.values = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)  # Gaussian
```

## Boundary Points

### 1D Boundary Indexing
```python
grid = UniformGrid1D(nx=5, length=1.0)
print("x coordinates:", grid.x)  # [0.   0.25 0.5  0.75 1.  ]

# Boundary indices
left_boundary = 0      # x = 0
right_boundary = -1    # x = 1.0

# Interior points
interior = slice(1, -1)  # x = [0.25, 0.5, 0.75]
```

### 2D Boundary Indexing
```python
grid = UniformGrid2D(nx=5, ny=5, length_x=1.0, length_y=1.0)

# Corner points
bottom_left = (0, 0)      # (x,y) = (0,0)
bottom_right = (-1, 0)    # (x,y) = (1,0)
top_left = (0, -1)        # (x,y) = (0,1)
top_right = (-1, -1)      # (x,y) = (1,1)

# Edge boundaries (excluding corners)
left_edge = (0, slice(1, -1))     # x=0, y=interior
right_edge = (-1, slice(1, -1))   # x=1, y=interior
bottom_edge = (slice(1, -1), 0)   # y=0, x=interior
top_edge = (slice(1, -1), -1)     # y=1, x=interior

# Interior points
interior = (slice(1, -1), slice(1, -1))
```

## Grid Operations

### Accessing Values
```python
# 1D
grid = UniformGrid1D(nx=101, length=1.0)
center_value = grid.values[50]      # Value at center
boundary_values = grid.values[[0, -1]]  # Both boundaries

# 2D
grid = UniformGrid2D(nx=51, ny=51, length_x=1.0, length_y=1.0)
center_value = grid.values[25, 25]  # Center point
corner_values = grid.values[[0, -1], [0, -1]]  # All corners
```

### Modifying Values
```python
# Set specific points
grid.values[10:20] = 1.0  # Set range in 1D

# Apply function
grid.values = np.sin(np.pi * grid.x)  # 1D
grid.values = np.sin(np.pi * grid.X) * np.cos(np.pi * grid.Y)  # 2D
```

### Grid Information
```python
# Grid properties
print(f"1D: {grid.nx} points, dx={grid.dx:.3f}")
print(f"2D: {grid.nx}x{grid.ny} points, dx={grid.dx:.3f}, dy={grid.dy:.3f}")

# Coordinate ranges
print(f"X range: [{grid.x[0]:.3f}, {grid.x[-1]:.3f}]")
print(f"Y range: [{grid.y[0]:.3f}, {grid.y[-1]:.3f}]")
```

## Advanced Grid Usage

### Non-Uniform Spacing (Future)
The current SDK uses uniform grids, but the architecture supports extension to non-uniform grids.

### Multiple Grids
```python
# Different resolutions for convergence study
grids = []
resolutions = [21, 41, 81, 161]

for nx in resolutions:
    grid = UniformGrid1D(nx=nx, length=1.0)
    grids.append(grid)
```

### Grid Transformations
```python
# Transform coordinates
grid = UniformGrid1D(nx=101, length=1.0)

# Scale coordinates
x_scaled = grid.x * 2.0 - 1.0  # Transform [0,1] -> [-1,1]

# Apply coordinate transformation
grid.values = np.sin(np.pi * x_scaled)
```

### Grid Export/Import
```python
# Save grid data
np.save('solution.npy', grid.values)
np.save('coordinates.npy', grid.x)

# Load grid data
loaded_values = np.load('solution.npy')
loaded_grid = UniformGrid1D(nx=len(loaded_values), length=1.0)
loaded_grid.values = loaded_values
```

## Grid Validation

### Sanity Checks
```python
def validate_grid(grid):
    """Basic grid validation"""
    if hasattr(grid, 'nx'):  # 1D
        assert len(grid.x) == grid.nx
        assert len(grid.values) == grid.nx
        assert np.isclose(grid.dx, grid.length / (grid.nx - 1))
        print("1D grid validation passed")
    else:  # 2D
        assert grid.X.shape == (grid.nx, grid.ny)
        assert grid.Y.shape == (grid.nx, grid.ny)
        assert grid.values.shape == (grid.nx, grid.ny)
        assert np.isclose(grid.dx, grid.length_x / (grid.nx - 1))
        assert np.isclose(grid.dy, grid.length_y / (grid.ny - 1))
        print("2D grid validation passed")
```

### Common Issues

1. **Off-by-one errors**
   - Remember: `nx` points gives `nx-1` intervals
   - Index 0 and -1 are boundaries

2. **Coordinate confusion**
   - `grid.x` is 1D array
   - `grid.X`, `grid.Y` are 2D meshes

3. **Memory issues**
   - Large grids consume significant memory
   - Consider using smaller grids for testing

### Debugging Tips
```python
# Check grid properties
print(f"Grid shape: {grid.values.shape}")
print(f"X range: [{grid.x.min():.3f}, {grid.x.max():.3f}]")
print(f"Min/Max values: [{grid.values.min():.3f}, {grid.values.max():.3f}]")

# Visualize grid points
import matplotlib.pyplot as plt
if hasattr(grid, 'nx') and hasattr(grid, 'ny'):  # 2D
    plt.scatter(grid.X.flatten(), grid.Y.flatten(), s=1)
    plt.title("Grid Points")
    plt.show()
```

## Performance Considerations

### Memory Usage
```python
# Memory scales with N² for 2D problems
nx, ny = 100, 100
grid = UniformGrid2D(nx=nx, ny=ny, length_x=1.0, length_y=1.0)

memory_mb = grid.values.nbytes / (1024**2)
print(f"Grid memory usage: {memory_mb:.1f} MB")

# Coordinate arrays also consume memory
coord_memory_mb = (grid.X.nbytes + grid.Y.nbytes) / (1024**2)
print(f"Coordinate memory: {coord_memory_mb:.1f} MB")
```

### Computation Time
```python
# Operation scaling
import time

grid_sizes = [50, 100, 200]
for n in grid_sizes:
    grid = UniformGrid2D(nx=n, ny=n, length_x=1.0, length_y=1.0)

    start = time.time()
    result = np.sin(grid.X) * np.cos(grid.Y)  # Sample operation
    elapsed = time.time() - start

    print(f"Size {n}x{n}: {elapsed:.3f}s")
```

## Future Extensions

The grid system is designed for extensibility:

### Planned Features
- **Non-uniform grids**: Adaptive mesh refinement
- **Curvilinear coordinates**: Complex geometries
- **3D grids**: Volume discretization
- **Parallel grids**: Domain decomposition
- **Adaptive grids**: Error-based refinement

### Custom Grids
Users can potentially create custom grid classes by implementing the required interface with properties like `values`, coordinate arrays, and spacing information.
