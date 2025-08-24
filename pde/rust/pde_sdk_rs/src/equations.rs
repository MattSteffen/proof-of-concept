//! Equations module for PDE SDK
//!
//! Provides implementations of common PDE equations.

/// Heat equation parameters and utilities
pub struct HeatEquation {
    pub alpha: f64,
}

impl HeatEquation {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Analytical solution for heat equation with sinusoidal initial condition
    pub fn analytical_solution(&self, x: f64, t: f64) -> f64 {
        (std::f64::consts::PI * x).sin() * (-self.alpha * std::f64::consts::PI * std::f64::consts::PI * t).exp()
    }
}

/// Poisson equation utilities
pub struct PoissonEquation;

impl PoissonEquation {
    /// Right-hand side for manufactured solution
    pub fn manufactured_rhs(x: f64, y: f64) -> f64 {
        -2.0 * std::f64::consts::PI * std::f64::consts::PI * (x * x + y * y).sin()
    }

    /// Manufactured analytical solution
    pub fn manufactured_solution(x: f64, y: f64) -> f64 {
        (x * x + y * y).sin()
    }
}
