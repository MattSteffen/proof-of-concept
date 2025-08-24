# README.md

# PDE SDK (Python-First, Rust-Ready)

Finite-difference SDK for solving common PDEs on uniform 1D/2D grids.
**Phase 1: Pure Python development** - Building a solid foundation with clean,
testable code. **Phase 2: Modular Rust integration** - Replace performance-critical
components piece by piece via PyO3/maturin.

Status: early-alpha (API may change)

- **Current Phase**: Pure Python SDK (`pde_sdk`)
- **Future Phase**: Rust acceleration (`pde-sdk` → `pde_sdk_rust`)
- Targets: 1D/2D heat (parabolic) and Poisson (elliptic) first. Add
  advection-diffusion and basic hyperbolic later.

Highlights
- Simple, composable API (equation, domain, boundary, solver)
- Pure-Python reference solvers for clarity and testing
- **Modular Rust integration**: Add Rust components when ready
- Clear path for growth: start 1D, then 2D, keep grids uniform
- **Development-first**: Focus on Python until core SDK is mature

Repository layout
```text
proof-of-concept/
└── pde/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml                # Python packaging
    ├── Makefile                      # Main development orchestrator
    │
    ├── pde_sdk/                      # Python package (Phase 1 focus)
    │   ├── __init__.py
    │   ├── Makefile                  # Python development commands
    │   ├── equations/
    │   │   ├── __init__.py
    │   │   ├── heat.py
    │   │   └── poisson.py
    │   ├── domains/
    │   │   ├── __init__.py
    │   │   ├── uniform1d.py
    │   │   └── uniform2d.py
    │   ├── boundaries/
    │   │   ├── __init__.py
    │   │   ├── dirichlet.py
    │   │   └── neumann.py
    │   ├── solvers/
    │   │   ├── __init__.py
    │   │   ├── explicit_euler.py
    │   │   └── crank_nicolson.py
    │   ├── visualization/
    │   │   ├── __init__.py
    │   │   └── plot.py
    │   └── rust_backend/
    │       └── __init__.py          # Future: Imports from `pde_sdk_rust`
    │
    ├── rust/                         # Phase 2: Rust components (ready for integration)
    │   └── pde_sdk_rs/
    │       ├── Makefile              # Rust development commands
    │       ├── Cargo.toml            # Rust crate configuration
    │       └── src/
    │           ├── lib.rs            # PyO3 bindings (for future integration)
    │           ├── grid.rs
    │           ├── solver.rs
    │           └── equations.rs
    │
    ├── tests/                        # Python tests (pytest)
    │   ├── test_heat.py
    │   ├── test_poisson.py
    │   └── test_integration.py
    │
    ├── examples/                     # Python examples
    │   ├── python/
    │   │   ├── heat_1d_demo.py
    │   │   └── poisson_2d_demo.py
    │   └── notebooks/
    │       └── intro.ipynb
    │
    └── docs/
        ├── ROADMAP.md
        ├── index.md
        └── api_reference.md
```

Install

Prereqs
- Python 3.11+
- uv (recommended) or pip for package management

Pure Python Development (Current Phase)
```bash
cd proof-of-concept/pde

# Option 1: Using uv (recommended)
uv sync --dev
uv pip install -e .

# Option 2: Using pip
pip install -e .
```

Future: Rust Integration (Phase 2)
When ready to integrate Rust components:
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
uv pip install maturin

# Build and install Rust extensions
maturin develop -m rust/pde_sdk_rs/Cargo.toml
```

Quickstart

Pure-Python 1D heat (Explicit Euler)
```python
import numpy as np
from pde_sdk.domains.uniform1d import UniformGrid1D
from pde_sdk.boundaries.dirichlet import DirichletBC
from pde_sdk.equations.heat import HeatEquation1D
from pde_sdk.solvers.explicit_euler import ExplicitEuler1D

nx = 101
grid = UniformGrid1D(nx=nx, length=1.0)
ic = lambda x: np.sin(np.pi * x)

eq = HeatEquation1D(alpha=0.01, grid=grid, left=DirichletBC(0.0),
                    right=DirichletBC(0.0), initial=ic)

solver = ExplicitEuler1D(dt=1e-4)
u_final = solver.solve(eq, t_final=0.1)
```

Future: Rust-Accelerated Version (Phase 2)
```python
# This will be available after Rust integration
# from pde_sdk.rust_backend import Heat1DSolver
# solver = Heat1DSolver(nx=nx, length=1.0, alpha=0.01, ...)
```

Rust usage (crate as a native Rust lib)
```rust
use pde_sdk::solver::ExplicitEuler1D;
use pde_sdk::grid::UniformGrid1D;

fn main() {
    let mut grid = UniformGrid1D::new(101, 1.0);
    grid.fill_with_sin_pi_x();
    let solver = ExplicitEuler1D::new(1e-4, 0.01);
    let u = solver.solve(&mut grid, 0.1);
    println!("u[50] = {}", u[50]);
}
```

Development

Python (Current Phase)
- Run tests: `make test` or `cd pde_sdk && make test`
- Format: `make format` or `cd pde_sdk && make format`
- Lint: `make lint` or `cd pde_sdk && make lint`
- Full check: `make py-check` or `cd pde_sdk && make cycle`

Future: Rust Development (Phase 2)
- Tests: `cd rust/pde_sdk_rs && make test`
- Lint: `cd rust/pde_sdk_rs && make lint`
- Format: `cd rust/pde_sdk_rs && make format`
- Build: `cd rust/pde_sdk_rs && make build`

Documentation
- **MkDocs Documentation**: Complete user guide with examples
  - Build docs: `make docs` or `uv run mkdocs build`
  - Serve locally: `make docs-serve` or `uv run mkdocs serve`
  - Deploy to GitHub Pages: `make docs-deploy` or `uv run mkdocs gh-deploy --force`
- **Roadmap**: See docs/Roadmap.md for development plans
- **Rust docs**: cargo doc --open (from rust/pde_sdk_rs)

Design principles
- Composition over inheritance: equation + domain + BC + solver
- Uniform grids (1D/2D) first; finite differences (second-order)
- Clean error messages, stability checks (CFL), and progress hooks
- Python API first, Rust for hot loops, with parity in behavior

Roadmap
See ROADMAP.md

Desired Features
- Perfect instructions for LLM how to solve any given pde (that can be solved with the latest versions)
- Configuration calculation
    - If you want accuracy < .001, then you need time step at dt=.0005 
    - Provide a sliding scale for what acuracy you want, what speed you want etc. Then the rest of the design decisions for the solution will be fixed from that.
    - You can manipulate variables, and the ones that have to be set to achieve your desired state are set for you.
- Progress bar for solutions
    - Dynamically show individual steps or whole solution based on log level or some equivalent