/// Phase 16: Position-Based Dynamics (PBD) Cloth Simulation
///
/// PBD is inherently stable for cloth because constraints are solved
/// in position-space rather than force-space, avoiding numerical blow-up.
///
/// A cloth mesh is a grid of particles connected by:
///   - Stretch constraints (keep edge lengths near rest)
///   - Bend constraints (resist folding between adjacent triangles)
///
/// Constraint projection: for each constraint C(x,x) = |x-x| - d_rest
///   x = -w/(w+w) * (|p-p| - d_rest) * (p-p)/|p-p|
///   x = +w/(w+w) * (|p-p| - d_rest) * (p-p)/|p-p|

use macroquad::prelude::Vec3;

pub struct ClothParticle {
    pub position:      Vec3,
    pub prev_position: Vec3, // For Verlet integration
    pub inv_mass:      f32,  // 0 = pinned
    pub velocity:      Vec3,
}

impl ClothParticle {
    pub fn new(pos: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self { position: pos, prev_position: pos, inv_mass, velocity: Vec3::ZERO }
    }
}

pub struct StretchConstraint {
    pub a:         usize,
    pub b:         usize,
    pub rest_len:  f32,
    pub stiffness: f32, // [0..1] in PBD (1 = rigid)
}

pub struct BendConstraint {
    pub a:         usize,
    pub b:         usize,
    pub c:         usize,
    pub d:         usize,
    pub rest_angle: f32,
    pub stiffness:  f32,
}

pub struct Cloth {
    pub particles:  Vec<ClothParticle>,
    pub stretches:  Vec<StretchConstraint>,
    pub bends:      Vec<BendConstraint>,
    pub gravity:    Vec3,
    pub wind:       Vec3,
    pub iterations: usize,
}

impl Cloth {
    /// Create a rectangular cloth grid
    pub fn grid(rows: usize, cols: usize, spacing: f32, origin: Vec3) -> Self {
        let mut particles = Vec::new();
        let mut stretches = Vec::new();
        let mut bends = Vec::new();

        for r in 0..rows {
            for c in 0..cols {
                let pos = origin + Vec3::new(c as f32 * spacing, 0.0, r as f32 * spacing);
                let inv_mass = if r == 0 { 0.0 } else { 1.0 }; // Pin top row
                particles.push(ClothParticle::new(pos, if inv_mass > 0.0 { 1.0 } else { 0.0 }));
            }
        }

        let idx = |r: usize, c: usize| r * cols + c;

        // Structural stretch constraints
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    stretches.push(StretchConstraint { a: idx(r, c), b: idx(r, c + 1), rest_len: spacing, stiffness: 0.95 });
                }
                if r + 1 < rows {
                    stretches.push(StretchConstraint { a: idx(r, c), b: idx(r + 1, c), rest_len: spacing, stiffness: 0.95 });
                }
                // Shear
                if c + 1 < cols && r + 1 < rows {
                    let diag = (spacing * spacing + spacing * spacing).sqrt();
                    stretches.push(StretchConstraint { a: idx(r, c), b: idx(r + 1, c + 1), rest_len: diag, stiffness: 0.7 });
                    stretches.push(StretchConstraint { a: idx(r, c + 1), b: idx(r + 1, c), rest_len: diag, stiffness: 0.7 });
                }
            }
        }

        Self { particles, stretches, bends, gravity: Vec3::new(0.0, -9.81, 0.0), wind: Vec3::new(0.3, 0.0, 0.1), iterations: 8 }
    }

    pub fn step(&mut self, dt: f32) {
        // 1. Apply external forces (Verlet-style)
        for p in &mut self.particles {
            if p.inv_mass == 0.0 { continue; }
            let accel = self.gravity + self.wind;
            let velocity = p.position - p.prev_position;
            p.prev_position = p.position;
            p.position += velocity + accel * dt * dt;

            // Floor collision
            if p.position.y < 0.05 {
                p.position.y = 0.05;
            }
        }

        // 2. Iterative constraint resolution (PBD)
        for _ in 0..self.iterations {
            for c in &self.stretches {
                let pa = self.particles[c.a].position;
                let pb = self.particles[c.b].position;
                let wa = self.particles[c.a].inv_mass;
                let wb = self.particles[c.b].inv_mass;

                let delta = pb - pa;
                let length = delta.length();
                if length < 1e-6 { continue; }

                let correction = delta / length * ((length - c.rest_len) * c.stiffness);
                let total_w = wa + wb;
                if total_w < 1e-10 { continue; }

                self.particles[c.a].position += correction * (wa / total_w);
                self.particles[c.b].position -= correction * (wb / total_w);
            }
        }
    }
}
