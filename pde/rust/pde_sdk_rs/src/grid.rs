//! Grid module for PDE SDK
//!
//! Provides uniform grid implementations for 1D and 2D domains.

use pyo3::prelude::*;

/// 1D Uniform Grid
#[pyclass]
#[derive(Clone)]
pub struct UniformGrid1D {
    pub nx: usize,
    pub length: f64,
    pub data: Vec<f64>,
}

#[pymethods]
impl UniformGrid1D {
    #[new]
    pub fn new(nx: usize, length: f64) -> Self {
        Self {
            nx,
            length,
            data: vec![0.0; nx],
        }
    }

    pub fn fill_with_sin_pi_x(&mut self) {
        let dx = self.length / (self.nx - 1) as f64;
        for i in 0..self.nx {
            let x = i as f64 * dx;
            self.data[i] = (std::f64::consts::PI * x).sin();
        }
    }

    pub fn dx(&self) -> f64 {
        self.length / (self.nx - 1) as f64
    }

    #[getter]
    pub fn get_data(&self) -> Vec<f64> {
        self.data.clone()
    }
}
