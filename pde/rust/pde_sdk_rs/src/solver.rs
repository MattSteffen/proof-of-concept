//! Solver module for PDE SDK
//!
//! Provides numerical solvers for PDE equations.

use super::grid::UniformGrid1D;
use pyo3::prelude::*;

/// Explicit Euler solver for 1D heat equation
#[pyclass]
#[derive(Clone)]
pub struct ExplicitEuler1D {
    pub dt: f64,
    pub alpha: f64,
}

#[pymethods]
impl ExplicitEuler1D {
    #[new]
    pub fn new(dt: f64, alpha: f64) -> Self {
        Self { dt, alpha }
    }

    pub fn solve(&self, grid: &mut UniformGrid1D, t_final: f64) -> PyResult<Vec<f64>> {
        let dx = grid.dx();
        let r = self.alpha * self.dt / (dx * dx);

        if r >= 0.5 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("CFL condition violated: r = {:.3} >= 0.5. Reduce dt.", r)
            ));
        }

        let steps = (t_final / self.dt) as usize;
        let mut u = grid.data.clone();

        for _step in 0..steps {
            let mut u_new = u.clone();

            for i in 1..(grid.nx - 1) {
                u_new[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
            }

            u = u_new;
        }

        Ok(u)
    }
}
