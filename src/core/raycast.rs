/// Phase 18: Raycast API
///
/// Fires a ray through the physics world and returns the closest hit.
/// Used for editor mouse picking, gameplay queries, and light occlusion.

use macroquad::prelude::Vec3;
use crate::core::shapes::Shape;
use crate::core::solver::RigidBody;

pub struct Ray {
    pub origin:    Vec3,
    pub direction: Vec3, // Should be normalized
}

pub struct HitResult {
    pub body_id:  usize,
    pub point:    Vec3,  // World-space hit point
    pub normal:   Vec3,  // Surface normal at hit
    pub distance: f32,   // Distance along the ray
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize_or_zero() }
    }

    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Ray vs. Sphere
fn ray_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - center;
    let a  = ray.direction.dot(ray.direction);
    let b  = 2.0 * oc.dot(ray.direction);
    let c  = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 { return None; }
    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t > 0.001 { Some(t) } else { None }
}

/// Ray vs. AABB (slab method)
fn ray_aabb(ray: &Ray, center: Vec3, half_extents: Vec3) -> Option<f32> {
    let min = center - half_extents;
    let max = center + half_extents;

    let inv_dir = Vec3::new(
        if ray.direction.x.abs() > 1e-8 { 1.0 / ray.direction.x } else { f32::MAX },
        if ray.direction.y.abs() > 1e-8 { 1.0 / ray.direction.y } else { f32::MAX },
        if ray.direction.z.abs() > 1e-8 { 1.0 / ray.direction.z } else { f32::MAX },
    );

    let t1 = (min - ray.origin) * inv_dir;
    let t2 = (max - ray.origin) * inv_dir;

    let t_min = t1.min(t2);
    let t_max = t1.max(t2);

    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit  = t_max.x.min(t_max.y).min(t_max.z);

    if t_enter < t_exit && t_exit > 0.0 {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

/// Ray vs. Capsule (approximate: ray vs. two spheres + cylinder)
fn ray_capsule(ray: &Ray, center: Vec3, radius: f32, half_height: f32) -> Option<f32> {
    // Approximation: treat as a sphere at center for picking purposes
    ray_sphere(ray, center, radius + half_height * 0.5)
}

/// Compute surface normal at a hit point on a shape
fn surface_normal(shape: &Shape, center: Vec3, hit_point: Vec3) -> Vec3 {
    match shape {
        Shape::Sphere { .. }                 => (hit_point - center).normalize_or_zero(),
        Shape::Box { half_extents }          => {
            let local = hit_point - center;
            // Find the closest face
            let dx = (local.x.abs() - half_extents.x).abs();
            let dy = (local.y.abs() - half_extents.y).abs();
            let dz = (local.z.abs() - half_extents.z).abs();
            if dx < dy && dx < dz { Vec3::new(if local.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0) }
            else if dy < dz       { Vec3::new(0.0, if local.y >= 0.0 { 1.0 } else { -1.0 }, 0.0) }
            else                  { Vec3::new(0.0, 0.0, if local.z >= 0.0 { 1.0 } else { -1.0 }) }
        }
        Shape::Capsule { .. }                => (hit_point - center).normalize_or_zero(),
    }
}

/// Cast a ray against all rigid bodies. Returns the closest hit.
pub fn ray_cast(ray: &Ray, bodies: &[RigidBody]) -> Option<HitResult> {
    let mut closest: Option<HitResult> = None;

    for body in bodies {
        let t = match &body.shape {
            Shape::Sphere { radius }               => ray_sphere(ray, body.position, *radius),
            Shape::Box    { half_extents }         => ray_aabb(ray, body.position, *half_extents),
            Shape::Capsule { radius, half_height } => ray_capsule(ray, body.position, *radius, *half_height),
        };

        if let Some(t) = t {
            if closest.as_ref().map(|h| t < h.distance).unwrap_or(true) {
                let hit_point = ray.point_at(t);
                let normal    = surface_normal(&body.shape, body.position, hit_point);
                closest = Some(HitResult { body_id: body.id, point: hit_point, normal, distance: t });
            }
        }
    }

    closest
}

/// Unproject a 2D screen point to a world-space Ray (for editor mouse picking)
pub fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    screen_w: f32,
    screen_h: f32,
    cam_pos: Vec3,
    cam_target: Vec3,
    cam_up: Vec3,
    fov_y_rad: f32,
) -> Ray {
    let aspect = screen_w / screen_h;
    let half_h = (fov_y_rad * 0.5).tan();

    let forward = (cam_target - cam_pos).normalize_or_zero();
    let right   = forward.cross(cam_up).normalize_or_zero();
    let up      = right.cross(forward);

    let ndc_x = (screen_x / screen_w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_y / screen_h) * 2.0;

    let dir = (forward + right * ndc_x * half_h * aspect + up * ndc_y * half_h).normalize_or_zero();
    Ray::new(cam_pos, dir)
}
