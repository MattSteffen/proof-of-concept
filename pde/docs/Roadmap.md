# PDE SDK Roadmap

**Phase 1: Pure Python SDK** - Build a solid, testable foundation
**Phase 2: Modular Rust Integration** - Add Rust components piece by piece

This roadmap keeps scope contained while delivering a useful SDK with a
clear migration path. Milestones are small and test-driven.

Guiding principles
- **Phase 1 focus**: Deliver value early with pure Python
- **Phase 2 ready**: Design for modular Rust integration
- Keep grids uniform (1D/2D) and finite differences initially
- Composition over inheritance; simple APIs for simple cases
- Python reference first; modular Rust integration; consistent tests
- Bake in DX: examples, docs, helpful errors, progress hooks

Out of scope (for now)
- Finite elements and unstructured meshes
- Adaptive mesh refinement (AMR)
- 3D domains (consider later)
- GPUs; MPI-scale distributed parallelism
- Full Rust rewrite (modular integration instead)

Phase 0 — Project setup (Day 0–1)
- Skeleton repo structure (see README)
- Tooling
  - Python: uv, ruff, black, pytest, mypy
  - Rust: rustfmt, clippy ready for Phase 2
- CI (optional initial)
  - Lint + test Python
  - Basic Makefile structure
- Definition of Done (DoD)
  - Tests run locally; development workflow established
  - Ready for Python development
  - Rust structure in place for future integration

Phase 1 — Python MVP: 1D Heat (Week 1)
- Features
  - UniformGrid1D
  - DirichletBC, NeumannBC
  - HeatEquation1D with constant alpha
  - ExplicitEuler1D solver with stability check (r <= 0.5)
  - Basic visualization helper (line plot)
- Tests
  - Compare against analytical solution u(x,t) = sin(pi x) e^{-pi^2 alpha t}
  - CFL violation raises informative error
- Examples
  - examples/python/heat_1d_demo.py
- DoD
  - API stable for 1D heat; documented in README

Phase 2 — Python: 2D Heat + Poisson (Weeks 2–3)
- Features
  - UniformGrid2D
  - HeatEquation2D (Explicit Euler; Crank-Nicolson shell)
  - Poisson2D with source term f(x,y)
  - Linear solvers (Jacobi, Gauss-Seidel) for Poisson
  - Variable coefficients (scalars or callables) supported
- Tests
  - 2D Poisson manufactured solution (sine modes)
  - 2D heat separable solution (sine product)
- Examples
  - poisson_2d_demo.py
- DoD
  - 1D/2D basic problems solved reliably; docstrings + examples

Phase 3 — Python: Enhanced Solvers (Week 4)
- Features
  - Crank-Nicolson1D/2D (implicit time stepping)
  - Improved solver stability and performance
  - Better error handling and diagnostics
- Tests
  - Convergence rate verification
  - Stability analysis improvements
- DoD
  - All Python solvers working reliably
  - Ready for potential Rust acceleration

Phase 4 — Python: Documentation & Examples (Weeks 5–6)
- Features
  - Comprehensive docstrings
  - Jupyter notebook examples
  - API documentation
  - Performance benchmarks
- Tests
  - Example verification
  - Documentation tests
- DoD
  - Clear usage examples
  - Complete API documentation

**Phase 2: Modular Rust Integration** (Future - when Python SDK is mature)

Phase 5 — First Rust Component: 1D Heat Solver (Future)
- Features
  - Rust implementation of ExplicitEuler1D
  - PyO3 bindings for seamless Python integration
  - Performance benchmarking vs Python
- Tests
  - Equivalence testing with Python reference
  - Performance benchmarks
- DoD
  - Drop-in Rust replacement for 1D heat solver

Phase 6 — Rust Components: 2D & Poisson (Future)
- Features
  - Grid2D with efficient memory layout
  - 2D Laplacian stencils
  - Iterative Poisson solvers
  - Optional parallel processing with Rayon
- Tests
  - Parity with Python implementations
- DoD
  - Performance-critical components available in Rust

Phase 7 — Rust Integration: Advanced Features (Future)
- Features
  - Implicit time stepping (Crank-Nicolson)
  - Sparse linear system solvers
  - Advanced boundary conditions
- Tests
  - Convergence verification
  - Performance optimization
- DoD
  - Full feature parity with Python implementation

Phase 8 — Python: Advanced Features (Future)
- Features
  - 1D/2D advection-diffusion with upwind (1st order)
  - CFL diagnostics and auto-suggested dt
  - Advanced boundary conditions
- Tests
  - Advection of Gaussian pulse; diffusion smoothing
- DoD
  - Clear stability errors; documented examples

Phase 9 — Python: Production Release (Future)
- Features
  - Comprehensive test suite
  - Performance optimizations
  - Production-ready error handling
- Docs
  - Complete usage documentation
  - API reference
- Packaging
  - PyPI release
  - Wheel building
- DoD
  - v0.1 tag; changelog; installation instructions verified

Phase 10 — Rust: Complete Integration (Future)
- Features
  - All Python functionality available in Rust
  - Seamless Python/Rust interoperability
  - Performance benchmarking suite
- Tests
  - Full compatibility testing
  - Performance regression tests
- DoD
  - Users can choose Python or Rust implementations
  - Performance benefits clearly demonstrated

Cross-cutting concerns

API design
- Simple case trivial
  - solve_heat_1d(initial, alpha, length, dt, t_final, nx)
- Complex case possible
  - Compose Equation + Domain + BC + Solver
- Modular design for future Rust integration

Error handling
- Clear messages for CFL/stability, BC mismatches, size mismatches
- Python: raise ValueError/RuntimeError with hints
- Future: Rust errors mapped to Python exceptions

Performance
- Start correct; then optimize
- Python: NumPy-optimized implementations
- Future: Rust acceleration for performance-critical components
- Benchmarks: Python timing utilities, future Rust criterion

Testing strategy
- Unit tests per module (equations, solvers, boundaries)
- Analytical solutions where possible
- Property tests (e.g., mass conservation with Neumann)
- Future: Parity tests Python <-> Rust implementations

Versioning
- SemVer
  - 0.x until API stabilizes
  - Breaking changes noted in CHANGELOG

Open decisions (to resolve as we progress)
- Which sparse backend in Rust first: sprs vs. custom CRS
- Expose ndarray types in Rust API or stick to Vec<T> internally
- Add adaptive time stepping in v0.1 or defer to v0.2

Nice-to-haves (post v0.1)
- WebAssembly demo with a small Next.js frontend
- CLI: pde run --problem heat1d --nx 201 --dt 1e-4 --t 0.1
- More schemes: RK4, TVD for advection, higher-order stencils
- 3D support if demand justifies

Acceptance checklist per feature
- Reference (Python) implementation with tests
- Rust implementation with identical behavior
- Documentation and a runnable example
- Benchmarks (if performance-sensitive)
- API reviewed for clarity and consistency

References
- LeVeque, Finite Difference Methods for Ordinary and Partial Differential
  Equations
- SciPy, PETSc, FEniCS, deal.II, SciML