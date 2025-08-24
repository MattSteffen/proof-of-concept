//! # PDE SDK - Rust Backend
//!
//! High-performance Rust backend for solving PDEs using finite differences.
//! This crate provides PyO3 bindings for use from Python.
//!
//! ## Features
//!
//! - 1D and 2D uniform grid solvers
//! - Heat equation with explicit Euler method
//! - PyO3 integration with Python
//! - Optional parallel computation with Rayon
//!
//! ## Usage
//!
//! ```rust
//! use pde_sdk::solver::ExplicitEuler1D;
//! use pde_sdk::grid::UniformGrid1D;
//!
//! let mut grid = UniformGrid1D::new(101, 1.0);
//! grid.fill_with_sin_pi_x();
//! let solver = ExplicitEuler1D::new(1e-4, 0.01);
//! let result = solver.solve(&mut grid, 0.1);
//! ```

use pyo3::prelude::*;

/// PDE SDK Python module
#[pymodule]
fn pde_sdk_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<grid::UniformGrid1D>()?;
    m.add_class::<solver::ExplicitEuler1D>()?;
    m.add_class::<Heat1DSolver>()?;
    Ok(())
}

// Re-export modules for internal use
pub mod grid;
pub mod solver;
pub mod equations;

// Python-facing API
use pyo3::prelude::*;

/// 1D Heat Equation Solver for Python
#[pyclass]
#[derive(Clone)]
pub struct Heat1DSolver {
    nx: usize,
    length: f64,
    alpha: f64,
    left_value: f64,
    right_value: f64,
}

#[pymethods]
impl Heat1DSolver {
    #[new]
    fn new(nx: usize, length: f64, alpha: f64, left_value: f64, right_value: f64) -> Self {
        Self {
            nx,
            length,
            alpha,
            left_value,
            right_value,
        }
    }

    fn solve(&self, initial: Vec<f64>, dt: f64, t_final: f64) -> PyResult<Vec<f64>> {
        if initial.len() != self.nx {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Initial condition length must match nx"
            ));
        }

        let mut u = initial;
        let dx = self.length / (self.nx - 1) as f64;
        let r = self.alpha * dt / (dx * dx);

        if r >= 0.5 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("CFL condition violated: r = {:.3} >= 0.5. Reduce dt.", r)
            ));
        }

        let steps = (t_final / dt) as usize;

        for _step in 0..steps {
            let mut u_new = u.clone();

            for i in 1..(self.nx - 1) {
                u_new[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
            }

            // Apply boundary conditions
            u_new[0] = self.left_value;
            u_new[self.nx - 1] = self.right_value;

            u = u_new;
        }

        Ok(u)
    }
}
