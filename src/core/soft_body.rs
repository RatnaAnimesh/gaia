use macroquad::math::{Mat3, Vec3};
use rayon::prelude::*;

/// Represents a single tetrahedral element in a soft body mesh.
pub struct Tetrahedron {
    pub v0: usize,
    pub v1: usize,
    pub v2: usize,
    pub v3: usize,
    /// Inverse of the reference shape matrix $\mathbf{D}_m^{-1}$
    pub inv_rest_shape: Mat3,
    pub volume: f32,
}

/// A Spectral Matrix-Free Soft Body solver.
/// Evaluates Neo-Hookean hyperelastic forces without assembling a global $O(N^2)$ Jacobian matrix.
pub struct MatrixFreeSoftBody {
    pub particles: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub masses: Vec<f32>,
    pub elements: Vec<Tetrahedron>,
    
    // Hyperelastic material parameters
    pub mu: f32,     // Shear modulus
    pub lambda: f32, // Lame's first parameter (bulk modulus)
}

impl MatrixFreeSoftBody {
    pub fn new(mu: f32, lambda: f32) -> Self {
        Self {
            particles: Vec::new(),
            velocities: Vec::new(),
            masses: Vec::new(),
            elements: Vec::new(),
            mu,
            lambda,
        }
    }

    /// PARALLEL force computation via rayon parallel fold.
    /// Each tet's contribution is independent  no data races.
    /// Forces are reduced into a per-particle accumulator using atomic-free fold.
    pub fn compute_forces(&self) -> Vec<Vec3> {
        let n = self.particles.len();

        // Each rayon thread produces its own local force Vec, then we sum them.
        let partial_forces: Vec<Vec<Vec3>> = self.elements
            .par_iter()
            .map(|tet| {
                let mut local = vec![Vec3::ZERO; n];
                let x0 = self.particles[tet.v0];
                let x1 = self.particles[tet.v1];
                let x2 = self.particles[tet.v2];
                let x3 = self.particles[tet.v3];

                let ds = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
                let f  = ds * tet.inv_rest_shape;
                let j  = f.determinant();
                if j <= 0.0 { return local; }

                let f_inv_t  = f.inverse().transpose();
                let p_stress = f * self.mu - f_inv_t * (self.mu - self.lambda * j.ln());
                let h        = p_stress * tet.inv_rest_shape.transpose() * (-tet.volume);

                let f1 = h.col(0); let f2 = h.col(1); let f3 = h.col(2);
                let f0 = -(f1 + f2 + f3);

                local[tet.v0] = f0;
                local[tet.v1] = f1;
                local[tet.v2] = f2;
                local[tet.v3] = f3;
                local
            })
            .collect();

        // Sum all partial force vecs into one
        let mut forces = vec![Vec3::ZERO; n];
        for pf in &partial_forces {
            for (i, &f) in pf.iter().enumerate() {
                forces[i] += f;
            }
        }
        forces
    }
    
    /// Basic explicit Euler integration (scaffold for full implicit solver).
    pub fn step(&mut self, dt: f32) {
        let forces = self.compute_forces();
        
        for i in 0..self.particles.len() {
            if self.masses[i] > 0.0 {
                let inv_m = 1.0 / self.masses[i];
                let total_accel = forces[i] * inv_m + Vec3::new(0.0, -9.81, 0.0);
                self.velocities[i] += total_accel * dt;
                self.particles[i] += self.velocities[i] * dt;
                
                // Floor collision
                if self.particles[i].y < 0.0 {
                    self.particles[i].y = 0.0;
                    self.velocities[i].y *= -0.4; // damped bounce
                }
            }
        }
    }
}
