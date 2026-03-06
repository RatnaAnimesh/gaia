use macroquad::math::{Mat3, Vec3};

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

    /// Computes $\mathbf{K} \Delta \mathbf{v}$ inherently on the fly via tensor contractions
    /// rather than assembling a sparse matrix $\mathbf{K}$.
    pub fn compute_forces(&self) -> Vec<Vec3> {
        let mut forces = vec![Vec3::ZERO; self.particles.len()];

        for tet in &self.elements {
            let x0 = self.particles[tet.v0];
            let x1 = self.particles[tet.v1];
            let x2 = self.particles[tet.v2];
            let x3 = self.particles[tet.v3];

            // Current shape matrix $\mathbf{D}_s$
            let ds = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
            
            // Deformation Gradient $\mathbf{F} = \mathbf{D}_s \mathbf{D}_m^{-1}$
            let f = ds * tet.inv_rest_shape;
            
            // J = det(F) represents volume change
            let j = f.determinant();
            if j <= 0.0 { continue; } // Skip inverted elements
            
            // Neo-Hookean First Piola-Kirchhoff Stress Tensor $\mathbf{P}$
            // P = mu (F - F^{-T}) + lambda * ln(J) * F^{-T}
            let f_inv_t = f.inverse().transpose();
            let p_stress = f * self.mu - f_inv_t * (self.mu - self.lambda * j.ln());

            // Tensor contraction $\mathbf{H} = -V_0 \mathbf{P} \mathbf{D}_m^{-T}$
            let h = p_stress * tet.inv_rest_shape.transpose() * (-tet.volume);
            
            let f1 = h.col(0);
            let f2 = h.col(1);
            let f3 = h.col(2);
            let f0 = -(f1 + f2 + f3);

            // Matrix-Free Force Accumulation
            forces[tet.v0] += f0;
            forces[tet.v1] += f1;
            forces[tet.v2] += f2;
            forces[tet.v3] += f3;
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
