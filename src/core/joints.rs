/// Phase 15: Joints & Constraints
///
/// Each joint generates one or more velocity-level constraint rows that feed
/// into the Sequential Impulse solver. This keeps the implementation modular
/// and consistent with the rest of the pipeline.

use macroquad::prelude::Vec3;
use crate::core::solver::RigidBody;

/// A spring joint: applies Hooke's law force between two anchor points
pub struct SpringJoint {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_local_a: Vec3, // Attachment point in A's local space
    pub anchor_local_b: Vec3, // Attachment point in B's local space
    pub rest_length:    f32,
    pub stiffness:      f32,  // k [N/m]
    pub damping:        f32,  // b [Ns/m]
}

impl SpringJoint {
    pub fn apply(&self, bodies: &mut Vec<RigidBody>, dt: f32) {
        let pos_a = bodies[self.body_a].position + self.anchor_local_a;
        let pos_b = bodies[self.body_b].position + self.anchor_local_b;

        let delta = pos_b - pos_a;
        let length = delta.length();
        if length < 1e-6 { return; }
        let dir = delta / length;

        let extension = length - self.rest_length;
        let vel_a = bodies[self.body_a].velocity;
        let vel_b = bodies[self.body_b].velocity;
        let rel_vel = (vel_b - vel_a).dot(dir);

        let force_magnitude = self.stiffness * extension + self.damping * rel_vel;
        let force = dir * force_magnitude;

        bodies[self.body_a].apply_force_at_point( force, pos_a);
        bodies[self.body_b].apply_force_at_point(-force, pos_b);
    }
}

/// A ball-socket joint: locks two anchor points together (3 DOF position constraint)
pub struct BallSocketJoint {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_local_a: Vec3,
    pub anchor_local_b: Vec3,
    pub bias_factor: f32, // Baumgarte [0..1]
}

impl BallSocketJoint {
    pub fn apply(&self, bodies: &mut Vec<RigidBody>, dt: f32) {
        let world_a = bodies[self.body_a].position + self.anchor_local_a;
        let world_b = bodies[self.body_b].position + self.anchor_local_b;

        let positional_error = world_b - world_a;
        let bias = positional_error * self.bias_factor / dt;

        // Apply corrective impulse along error direction
        let force = bias;
        let inv_m_a = bodies[self.body_a].inv_mass;
        let inv_m_b = bodies[self.body_b].inv_mass;
        let total_inv_m = inv_m_a + inv_m_b;

        if total_inv_m < 1e-10 { return; }
        let impulse = force / total_inv_m;

        bodies[self.body_a].velocity += impulse * inv_m_a;
        bodies[self.body_b].velocity -= impulse * inv_m_b;
    }
}

/// A hinge joint: constrains two bodies to rotate around a shared axis only
pub struct HingeJoint {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_local_a: Vec3,
    pub anchor_local_b: Vec3,
    pub axis: Vec3,           // Rotation axis in world space
    pub min_angle: f32,       // Limit (radians)
    pub max_angle: f32,
}

impl HingeJoint {
    /// Apply angular limit as a corrective impulse
    pub fn apply_limits(&self, bodies: &mut Vec<RigidBody>, dt: f32) {
        let rel_ang_vel = bodies[self.body_a].ang_velocity - bodies[self.body_b].ang_velocity;
        let ang_vel_along_axis = rel_ang_vel.dot(self.axis);

        let inv_ia = bodies[self.body_a].inv_inertia;
        let inv_ib = bodies[self.body_b].inv_inertia;

        let eff_inertia = (self.axis * inv_ia).dot(self.axis) + (self.axis * inv_ib).dot(self.axis);
        if eff_inertia < 1e-10 { return; }

        let lambda = -ang_vel_along_axis / eff_inertia;
        let ang_impulse = self.axis * lambda;

        bodies[self.body_a].ang_velocity += ang_impulse * inv_ia;
        bodies[self.body_b].ang_velocity -= ang_impulse * inv_ib;
    }
}

/// Container for all joints in the physics world
pub struct JointSystem {
    pub springs:      Vec<SpringJoint>,
    pub ball_sockets: Vec<BallSocketJoint>,
    pub hinges:       Vec<HingeJoint>,
}

impl JointSystem {
    pub fn new() -> Self {
        Self { springs: Vec::new(), ball_sockets: Vec::new(), hinges: Vec::new() }
    }

    pub fn apply_all(&self, bodies: &mut Vec<RigidBody>, dt: f32) {
        for j in &self.springs      { j.apply(bodies, dt); }
        for j in &self.ball_sockets { j.apply(bodies, dt); }
        for j in &self.hinges       { j.apply_limits(bodies, dt); }
    }
}
