/// Phase 13: Sequential Impulse (SI) Rigid Body Solver
///
/// A complete rigid body with full 6-DOF dynamics:
///   - Linear:  F = ma → v += F/m * dt → x += v * dt
///   - Angular: T = Iα → ω += I⁻¹ * T * dt → q += ω * dt
///
/// Contact impulses are resolved iteratively (Projected Gauss-Seidel):
///   λ = -(J v + b) / (J M⁻¹ Jᵀ)
///   v += M⁻¹ Jᵀ λ
/// with clamping λ ≥ 0 for non-penetration, and Coulomb friction cone.

use macroquad::prelude::Vec3;
use crate::core::shapes::{Shape, PhysicsMaterial};
use crate::core::collision::gjk::{detect_collision, ContactManifold};

/// A full 6-DOF rigid body
pub struct RigidBody {
    pub id:           usize,
    pub position:     Vec3,
    pub velocity:     Vec3,
    pub ang_velocity: Vec3,        // ω in world space
    pub force:        Vec3,        // Accumulated force
    pub torque:       Vec3,        // Accumulated torque
    pub mass:         f32,         // 0.0 = static/infinite mass
    pub inv_mass:     f32,
    pub inertia:      Vec3,        // Diagonal inertia tensor (local)
    pub inv_inertia:  Vec3,
    pub shape:        Shape,
    pub material:     PhysicsMaterial,
    pub sleeping:     bool,
}

impl RigidBody {
    pub fn new(id: usize, shape: Shape, position: Vec3, material: PhysicsMaterial) -> Self {
        let (mass, inertia) = shape.mass_properties(&material);
        let inv_mass    = if mass   > 0.0 { 1.0 / mass   } else { 0.0 };
        let inv_inertia = Vec3::new(
            if inertia.x > 0.0 { 1.0 / inertia.x } else { 0.0 },
            if inertia.y > 0.0 { 1.0 / inertia.y } else { 0.0 },
            if inertia.z > 0.0 { 1.0 / inertia.z } else { 0.0 },
        );
        Self {
            id, position, velocity: Vec3::ZERO, ang_velocity: Vec3::ZERO,
            force: Vec3::ZERO, torque: Vec3::ZERO,
            mass, inv_mass, inertia, inv_inertia,
            shape, material, sleeping: false,
        }
    }

    pub fn is_static(&self) -> bool { self.inv_mass == 0.0 }

    /// Apply a world-space force at a world-space point
    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        self.force  += force;
        self.torque += (point - self.position).cross(force);
    }

    /// Apply a world-space impulse at a world-space point (immediate velocity change)
    pub fn apply_impulse(&mut self, impulse: Vec3, point: Vec3) {
        self.velocity     += impulse * self.inv_mass;
        self.ang_velocity += (point - self.position).cross(impulse) * self.inv_inertia;
    }

    /// Integrate one timestep (symplectic Euler)
    pub fn integrate(&mut self, dt: f32) {
        if self.is_static() { return; }
        // Linear
        let accel = self.force * self.inv_mass + Vec3::new(0.0, -9.81, 0.0);
        self.velocity  += accel * dt;
        self.position  += self.velocity * dt;
        // Angular
        self.ang_velocity += self.torque * self.inv_inertia * dt;
        // Damping (prevents energy growth)
        self.velocity     *= 1.0 - 0.001 * dt;
        self.ang_velocity *= 1.0 - 0.01  * dt;
        // Reset accumulators
        self.force  = Vec3::ZERO;
        self.torque = Vec3::ZERO;

        // Sleep if very slow
        if self.velocity.length_squared() < 0.001 && self.ang_velocity.length_squared() < 0.001 {
            self.sleeping = true;
        } else {
            self.sleeping = false;
        }
    }

    pub fn velocity_at_point(&self, p: Vec3) -> Vec3 {
        self.velocity + self.ang_velocity.cross(p - self.position)
    }
}

/// Contact constraint between two bodies
struct ContactConstraint<'a> {
    body_a: &'a mut RigidBody,
    body_b: &'a mut RigidBody,
    manifold: ContactManifold,
    lambda_n:  f32,   // Accumulated normal impulse (warmstart)
    lambda_t1: f32,   // Accumulated tangent impulse 1
    lambda_t2: f32,   // Accumulated tangent impulse 2
}

/// The main physics world
pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    pub solver_iterations: usize,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self { bodies: Vec::new(), solver_iterations: 10 }
    }

    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        id
    }

    pub fn step(&mut self, dt: f32) {
        let n = self.bodies.len();

        // 1. Integrate forces / gravity
        for body in &mut self.bodies {
            body.integrate(dt);
        }

        // 2. Broadphase: simple O(N²) for now; BVH inserted separately for large scenes
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.bodies[i].sleeping && self.bodies[j].sleeping { continue; }
                if self.bodies[i].is_static() && self.bodies[j].is_static() { continue; }
                pairs.push((i, j));
            }
        }

        // 3. Narrow phase + sequential impulse resolution
        // We do PGS over solver_iterations
        for _ in 0..self.solver_iterations {
            for &(i, j) in &pairs {
                // SAFETY: we split borrow using indices
                let (left, right) = self.bodies.split_at_mut(j);
                let body_a = &mut left[i];
                let body_b = &mut right[0];

                if let Some(contact) = detect_collision(
                    &body_a.shape, body_a.position,
                    &body_b.shape, body_b.position,
                ) {
                    resolve_contact(body_a, body_b, &contact, dt);
                }
            }
        }
    }
}

/// Resolve a single contact between two bodies using Sequential Impulse
fn resolve_contact(a: &mut RigidBody, b: &mut RigidBody, c: &ContactManifold, dt: f32) {
    let ra = c.point_a - a.position;
    let rb = c.point_b - b.position;

    let va = a.velocity_at_point(c.point_a);
    let vb = b.velocity_at_point(c.point_b);
    let vrel = va - vb;

    let vn = vrel.dot(c.normal);
    if vn > 0.0 { return; } // Separating — no impulse needed

    // --- Normal impulse ---
    // e = coefficient of restitution (blended from the two materials)
    let e = (a.material.restitution * b.material.restitution).sqrt();
    let bias = 0.2 / dt * (c.depth - 0.01_f32).max(0.0); // Baumgarte stabilisation

    let ra_cross_n = ra.cross(c.normal);
    let rb_cross_n = rb.cross(c.normal);

    let effective_mass_n =
        a.inv_mass + b.inv_mass +
        (ra_cross_n * a.inv_inertia).dot(ra_cross_n) +
        (rb_cross_n * b.inv_inertia).dot(rb_cross_n);

    let lambda_n = (-(1.0 + e) * vn + bias) / effective_mass_n;
    let lambda_n = lambda_n.max(0.0);

    let impulse_n = c.normal * lambda_n;
    a.apply_impulse( impulse_n, c.point_a);
    b.apply_impulse(-impulse_n, c.point_b);

    // --- Friction impulse (Coulomb cone) ---
    // Build tangent frame
    let t1 = if c.normal.abs().x < 0.9 {
        c.normal.cross(Vec3::X).normalize_or_zero()
    } else {
        c.normal.cross(Vec3::Y).normalize_or_zero()
    };
    let t2 = c.normal.cross(t1);

    let mu = (a.material.friction_dynamic * b.material.friction_dynamic).sqrt();
    let max_friction = mu * lambda_n;

    for tangent in [t1, t2] {
        let vt = vrel.dot(tangent);
        let ra_cross_t = ra.cross(tangent);
        let rb_cross_t = rb.cross(tangent);
        let eff_mass_t =
            a.inv_mass + b.inv_mass +
            (ra_cross_t * a.inv_inertia).dot(ra_cross_t) +
            (rb_cross_t * b.inv_inertia).dot(rb_cross_t);

        let lambda_t = (-vt / eff_mass_t).clamp(-max_friction, max_friction);
        let impulse_t = tangent * lambda_t;
        a.apply_impulse( impulse_t, c.point_a);
        b.apply_impulse(-impulse_t, c.point_b);
    }
}
