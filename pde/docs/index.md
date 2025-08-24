# PDE SDK Documentation

<div class="grid cards" markdown>

- :material-rocket:{ .lg .middle } **Get Started**

    ---

    New to the PDE SDK? Start here to learn the basics and get up and running quickly.

    [:octicons-arrow-right-24: Quick Start](overview.md)

- :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Comprehensive guides for using all features of the PDE SDK.

    [:octicons-arrow-right-24: User Guide](equations-setup.md)

- :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Detailed API documentation for all classes and functions.

    [:octicons-arrow-right-24: API Reference](overview.md)

- :material-roadmap:{ .lg .middle } **Roadmap**

    ---

    Learn about planned features and future development.

    [:octicons-arrow-right-24: Roadmap](Roadmap.md)

</div>

## What is the PDE SDK?

The **PDE SDK** is a modern, composable Python library for solving partial differential equations using finite difference methods. Built with a focus on usability, performance, and extensibility, it provides:

- :material-check-circle: **Simple API**: Get started with just a few lines of code
- :material-check-circle: **Multiple solvers**: From explicit to implicit methods
- :material-check-circle: **Rich visualization**: Built-in plotting and analysis tools
- :material-check-circle: **Future-proof**: Designed for modular Rust acceleration

## Key Features

### Supported Equations

| Equation Type | 1D Support | 2D Support | Solvers |
|---------------|------------|------------|---------|
| Heat/Diffusion | ✅ | ✅ | Explicit Euler, Crank-Nicolson, Backward Euler |
| Poisson | ❌ | ✅ | Jacobi Iterative |

### Solver Comparison

| Solver | Accuracy | Stability | Performance | Complexity |
|--------|----------|-----------|-------------|------------|
| Explicit Euler | O(Δt) | CFL Required | Fastest | Simple |
| Crank-Nicolson | O(Δt²) | Unconditional | Fast | Medium |
| Backward Euler | O(Δt) | Unconditional | Medium | Medium |
| Jacobi | - | Iterative | Depends on tolerance | Simple |

## Quick Example

Here's a complete example solving the 1D heat equation:

```python
import numpy as np
from pde_sdk.domains import UniformGrid1D
from pde_sdk.equations import HeatEquation1D
from pde_sdk.boundaries import DirichletBC
from pde_sdk.solvers import ExplicitEuler1D

# Setup domain and equation
grid = UniformGrid1D(nx=101, length=1.0)
equation = HeatEquation1D(
    alpha=0.01,
    grid=grid,
    left_bc=DirichletBC(0.0),
    right_bc=DirichletBC(0.0),
    initial_condition=lambda x: np.sin(np.pi * x)
)

# Solve
solver = ExplicitEuler1D(dt=1e-4)
solution = solver.solve(equation, t_final=0.1)

# The solution is now available in grid.values
print(f"Solution shape: {solution.shape}")
print(f"Max value: {np.max(solution):.4f}")
```

## Installation

The PDE SDK is designed to be easy to install and use:

```bash
# Using uv (recommended)
uv add pde-sdk

# Using pip
pip install pde-sdk
```

## Development

The PDE SDK is currently in **Phase 1** (Pure Python implementation) with **Phase 2** (Rust acceleration) planned for the future.

### Current Status
- ✅ Core finite difference algorithms
- ✅ Multiple solver implementations
- ✅ Comprehensive test suite
- ✅ Documentation and examples
- ✅ Python-first design philosophy

### Future Roadmap
- 🚧 Rust backend integration
- 🚧 GPU acceleration support
- 🚧 Advanced boundary conditions
- 🚧 3D domain support

## Community & Support

- :material-github: [GitHub Repository](https://github.com/MattSteffen/proof-of-concept/tree/main/pde)
<!-- - :material-book-open: [Documentation](https://pde-sdk.readthedocs.io/) -->
<!-- - :material-help-circle: [Issue Tracker](https://github.com/your-org/pde-sdk/issues) -->

---

*Built with ❤️ for scientific computing and PDE solving*
