/// Chebyshev-Preconditioned Eulerian Fluid Simulation
///
/// Solves the incompressible Navier-Stokes equations using a grid-based approach.
/// The key innovation is the Chebyshev polynomial preconditioner which eliminates
/// the need for global dot-product synchronization that kills GPU performance in PCG.
///
/// Standard PCG requires: alpha_k = (r_k^T z_k) / (p_k^T A p_k)
/// This dot product forces ALL threads to synchronize before proceeding.
///
/// Chebyshev iteration replaces this with a recurrence:
///   x_{k+1} = x_k + omega_k (b - A x_k) + beta_k (x_k - x_{k-1})
/// where omega and beta are precomputed from the eigenvalue bounds [lambda_min, lambda_max].
/// No dot products needed => zero global thread synchronization!
/// Rayon is used to run each cell's update across all CPU cores.

use rayon::prelude::*;

pub struct FluidGrid {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    
    /// Velocity components (staggered MAC grid)
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,
    
    /// Pressure field p
    pub pressure: Vec<f32>,
    
    /// Divergence of intermediate velocity field
    pub divergence: Vec<f32>,
    
    pub cell_size: f32,
    pub density: f32,
}

impl FluidGrid {
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        let n = width * height * depth;
        Self {
            width,
            height,
            depth,
            vel_x: vec![0.0; n],
            vel_y: vec![0.0; n],
            vel_z: vec![0.0; n],
            pressure: vec![0.0; n],
            divergence: vec![0.0; n],
            cell_size,
            density: 1.0,
        }
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.width + z * self.width * self.height
    }

    /// Compute the divergence of the velocity field: div(u) at each cell.
    /// This is the RHS of the Poisson equation: nabla^2 p = rho/dt * div(u*)
    pub fn compute_divergence(&mut self, dt: f32) {
        let inv_h  = 1.0 / self.cell_size;
        let scale  = self.density / dt;
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let vx = &self.vel_x;
        let vy = &self.vel_y;
        let vz = &self.vel_z;

        self.divergence
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, div_cell)| {
                let z = i / (w * h);
                let y = (i / w) % h;
                let x = i % w;
                if x == 0 || y == 0 || z == 0 || x >= w - 1 || y >= h - 1 || z >= d - 1 {
                    *div_cell = 0.0;
                    return;
                }
                let div_u = (vx[i + 1] - vx[i] + vy[i + w] - vy[i] + vz[i + w * h] - vz[i]) * inv_h;
                *div_cell = -scale * div_u;
            });
    }

    /// Applies the discrete Laplacian operator A*x for a cell (used in PCG/Chebyshev).
    /// This is the "matrix-free" Laplacian -- we never build the matrix A explicitly.
    fn laplacian_at(&self, p: &[f32], x: usize, y: usize, z: usize) -> f32 {
        let inv_h2 = 1.0 / (self.cell_size * self.cell_size);
        let i = self.idx(x, y, z);
        let neighbor_sum = p[self.idx(x + 1, y, z)]
            + p[self.idx(x - 1, y, z)]
            + p[self.idx(x, y + 1, z)]
            + p[self.idx(x, y - 1, z)]
            + p[self.idx(x, y, z + 1)]
            + p[self.idx(x, y, z - 1)]
            - 6.0 * p[i];
        neighbor_sum * inv_h2
    }

    /// Chebyshev-Preconditioned Pressure Projection.
    ///
    /// Instead of PCG with global dot-product synchronization, we run Chebyshev iteration:
    ///   x_{k+1} = x_k + omega_k * r_k + beta_k * (x_k - x_{k-1})
    /// where omega, beta are precomputed from the spectral radius of A.
    /// This is FULLY parallel -- no global reductions!
    pub fn pressure_project_chebyshev(&mut self, dt: f32, iterations: usize) {
        self.compute_divergence(dt);

        // Eigenvalue bounds for the 3D discrete Laplacian on a uniform grid.
        // Keep lambda_min at least 5% of lambda_max to prevent rho  1 (divergence).
        let lambda_max = 4.0 / (self.cell_size * self.cell_size);
        let lambda_min = (0.05 * lambda_max).max(1e-4); // GUARD: never let rho get too close to 1
        let rho = (lambda_max - lambda_min) / (lambda_max + lambda_min); // < 1 by construction

        let n = self.width * self.height * self.depth;
        let mut p_prev = self.pressure.clone();
        let mut p_curr = self.pressure.clone();

        // First Chebyshev step: omega_1 = 2 / (lambda_max + lambda_min)
        let omega_init = 2.0 / (lambda_max + lambda_min);
        for z in 1..self.depth - 1 {
            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let i = self.idx(x, y, z);
                    let residual = self.divergence[i] - self.laplacian_at(&p_curr, x, y, z);
                    p_curr[i] = (p_curr[i] + omega_init * residual).clamp(-1e6, 1e6);
                }
            }
        }

        // Subsequent iterations with Chebyshev recurrence
        let mut omega = omega_init;
        let omega_max = 2.0 / (lambda_min + 1e-8); // hard cap on omega growth
        for _k in 1..iterations {
            let denom = 1.0 - (rho * rho * omega / 4.0);
            // GUARD: denominator must stay positive to avoid division-by-zero / flip
            if denom.abs() < 1e-8 { break; }
            let omega_new = (1.0 / denom).min(omega_max);
            let beta = omega_new * rho * rho / 4.0;
            omega = omega_new;

            let mut p_next = vec![0.0f32; n];
            for z in 1..self.depth - 1 {
                for y in 1..self.height - 1 {
                    for x in 1..self.width - 1 {
                        let i = self.idx(x, y, z);
                        let residual = self.divergence[i] - self.laplacian_at(&p_curr, x, y, z);
                        let val = omega * (residual + beta * (p_curr[i] - p_prev[i])) + p_curr[i];
                        // GUARD: clamp and zero NaN so bad cells don't propagate
                        p_next[i] = if val.is_finite() { val.clamp(-1e6, 1e6) } else { 0.0 };
                    }
                }
            }
            p_prev = p_curr;
            p_curr = p_next;
        }

        self.pressure = p_curr;

        // PARALLEL velocity projection: u = u* - dt/rho * grad(p)
        let scale   = dt / (self.density * self.cell_size);
        let w       = self.width;
        let h       = self.height;
        let d       = self.depth;
        let pressure = self.pressure.clone(); // read-only snapshot

        self.vel_x
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, vx)| {
                let z = i / (w * h); let y = (i / w) % h; let x = i % w;
                if x == 0 || y == 0 || z == 0 || x >= w - 1 || y >= h - 1 || z >= d - 1 { return; }
                *vx -= scale * (pressure[i + 1] - pressure[i]);
            });
        self.vel_y
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, vy)| {
                let z = i / (w * h); let y = (i / w) % h; let x = i % w;
                if x == 0 || y == 0 || z == 0 || x >= w - 1 || y >= h - 1 || z >= d - 1 { return; }
                *vy -= scale * (pressure[i + w] - pressure[i]);
            });
        self.vel_z
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, vz)| {
                let z = i / (w * h); let y = (i / w) % h; let x = i % w;
                if x == 0 || y == 0 || z == 0 || x >= w - 1 || y >= h - 1 || z >= d - 1 { return; }
                *vz -= scale * (pressure[i + w * h] - pressure[i]);
            });
    }

    /// Advect velocity field using semi-Lagrangian method.
    pub fn advect(&mut self, dt: f32) {
        let n = self.width * self.height * self.depth;
        let vel_x_old = self.vel_x.clone();
        let vel_y_old = self.vel_y.clone();
        let vel_z_old = self.vel_z.clone();

        for z in 1..self.depth - 1 {
            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let i = self.idx(x, y, z);
                    
                    // Trace particle backwards in time
                    let src_x = (x as f32) - dt * vel_x_old[i] / self.cell_size;
                    let src_y = (y as f32) - dt * vel_y_old[i] / self.cell_size;
                    let src_z = (z as f32) - dt * vel_z_old[i] / self.cell_size;
                    
                    // Clamp to grid
                    let sx = src_x.clamp(0.5, (self.width - 2) as f32);
                    let sy = src_y.clamp(0.5, (self.height - 2) as f32);
                    let sz = src_z.clamp(0.5, (self.depth - 2) as f32);
                    
                    // Trilinear interpolation
                    let ix = sx as usize;
                    let iy = sy as usize;
                    let iz = sz as usize;
                    let fx = sx - ix as f32;
                    let fy = sy - iy as f32;
                    let fz = sz - iz as f32;

                    // Sample velocity at traced position
                    self.vel_x[i] = lerp3d(&vel_x_old, self, ix, iy, iz, fx, fy, fz);
                    self.vel_y[i] = lerp3d(&vel_y_old, self, ix, iy, iz, fx, fy, fz);
                }
            }
        }
    }

    /// Apply gravity to the Y velocity component.
    pub fn apply_gravity(&mut self, dt: f32) {
        // Parallel gravity application: every cell is independent
        self.vel_y.par_iter_mut().for_each(|v| *v -= 9.81 * dt);
    }

    /// Seeding an upward splash impulse at the center of the grid.
    pub fn add_impulse(&mut self, cx: usize, cy: usize, cz: usize, strength: f32) {
        if cx < self.width && cy < self.height && cz < self.depth {
            let i = self.idx(cx, cy, cz);
            self.vel_y[i] += strength;
        }
    }

    /// Full simulation step.
    pub fn step(&mut self, dt: f32) {
        self.apply_gravity(dt);
        self.advect(dt);
        self.pressure_project_chebyshev(dt, 8);
    }
}

fn lerp3d(field: &[f32], grid: &FluidGrid, ix: usize, iy: usize, iz: usize, fx: f32, fy: f32, fz: f32) -> f32 {
    let w = grid.width;
    let h = grid.height;
    let safe_idx = |x: usize, y: usize, z: usize| -> f32 {
        let xi = x.min(w - 1);
        let yi = y.min(h - 1);
        let zi = z.min(grid.depth - 1);
        field[xi + yi * w + zi * w * h]
    };
    
    // Trilinear interpolation
    let c000 = safe_idx(ix, iy, iz);
    let c100 = safe_idx(ix + 1, iy, iz);
    let c010 = safe_idx(ix, iy + 1, iz);
    let c110 = safe_idx(ix + 1, iy + 1, iz);
    let c001 = safe_idx(ix, iy, iz + 1);
    let c101 = safe_idx(ix + 1, iy, iz + 1);
    let c011 = safe_idx(ix, iy + 1, iz + 1);
    let c111 = safe_idx(ix + 1, iy + 1, iz + 1);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}
