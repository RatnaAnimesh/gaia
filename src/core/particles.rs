/// Phase 17: Particle System with SPH Density Interaction
///
/// Each emitter spawns particles with configurable velocity, lifetime, colour, and size.
/// Particles interact via SPH (Smoothed Particle Hydrodynamics) pressure forces.

use macroquad::prelude::{Vec3, Color, Vec4};

pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub color:    Vec4,   // RGBA with A fading over lifetime
    pub size:     f32,
    pub lifetime:     f32,   // Remaining seconds
    pub max_lifetime: f32,
    pub density:  f32,   // Computed by SPH kernel
}

#[derive(Clone, Debug)]
pub enum EmitterShape {
    Point,
    Sphere { radius: f32 },
    Cone   { radius: f32, height: f32 },
}

pub struct ParticleEmitter {
    pub position:       Vec3,
    pub shape:          EmitterShape,
    pub spawn_rate:     f32,   // Particles per second
    pub initial_vel:    Vec3,
    pub vel_spread:     f32,
    pub lifetime:       f32,
    pub size_start:     f32,
    pub size_end:       f32,
    pub color_start:    Vec4,
    pub color_end:      Vec4,
    pub sph_enabled:    bool,
    pub sph_kernel_h:   f32,   // Smoothing length
    pub sph_pressure_k: f32,   // Pressure stiffness

    accumulator: f32,
}

impl ParticleEmitter {
    pub fn new(position: Vec3, spawn_rate: f32) -> Self {
        Self {
            position,
            shape: EmitterShape::Point,
            spawn_rate,
            initial_vel: Vec3::new(0.0, 3.0, 0.0),
            vel_spread: 0.5,
            lifetime: 2.0,
            size_start: 0.15,
            size_end: 0.0,
            color_start: Vec4::new(1.0, 0.5, 0.1, 1.0),
            color_end: Vec4::new(0.5, 0.0, 0.0, 0.0),
            sph_enabled: false,
            sph_kernel_h: 1.0,
            sph_pressure_k: 1.0,
            accumulator: 0.0,
        }
    }

    fn random_offset(&self) -> Vec3 {
        let t = (macroquad::time::get_time() as f32 * 1000.0).fract();
        let t2 = (macroquad::time::get_time() as f32 * 2000.0 + 1.0).fract();
        let t3 = (macroquad::time::get_time() as f32 * 3000.0 + 2.0).fract();
        match &self.shape {
            EmitterShape::Point => Vec3::ZERO,
            EmitterShape::Sphere { radius } => {
                let v = Vec3::new(t * 2.0 - 1.0, t2 * 2.0 - 1.0, t3 * 2.0 - 1.0).normalize_or_zero();
                v * (*radius) * t
            }
            EmitterShape::Cone { radius, height } => {
                let h = t * height;
                let r = (h / height) * radius;
                Vec3::new((t2 * 2.0 - 1.0) * r, h, (t3 * 2.0 - 1.0) * r)
            }
        }
    }

    fn random_vel_spread(&self) -> Vec3 {
        let t  = (macroquad::time::get_time() as f32 * 7000.0).fract();
        let t2 = (macroquad::time::get_time() as f32 * 11000.0).fract();
        let t3 = (macroquad::time::get_time() as f32 * 13000.0).fract();
        Vec3::new(t * 2.0 - 1.0, t2 * 2.0 - 1.0, t3 * 2.0 - 1.0) * self.vel_spread
    }

    pub fn emit(&mut self, particles: &mut Vec<Particle>, dt: f32) {
        self.accumulator += self.spawn_rate * dt;
        while self.accumulator >= 1.0 {
            particles.push(Particle {
                position:     self.position + self.random_offset(),
                velocity:     self.initial_vel + self.random_vel_spread(),
                color:        self.color_start,
                size:         self.size_start,
                lifetime:     self.lifetime,
                max_lifetime: self.lifetime,
                density:      0.0,
            });
            self.accumulator -= 1.0;
        }
    }
}

pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub emitters:  Vec<ParticleEmitter>,
    pub gravity:   Vec3,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self { particles: Vec::new(), emitters: Vec::new(), gravity: Vec3::new(0.0, -4.0, 0.0) }
    }

    pub fn step(&mut self, dt: f32) {
        // Emit new particles
        for emitter in &mut self.emitters {
            emitter.emit(&mut self.particles, dt);
        }

        // SPH density (simple N for small counts)
        let enabled = self.emitters.iter().any(|e| e.sph_enabled);
        let h = self.emitters.first().map(|e| e.sph_kernel_h).unwrap_or(1.0);
        let h2 = h * h;

        if enabled {
            for i in 0..self.particles.len() {
                let mut density = 0.0_f32;
                for j in 0..self.particles.len() {
                    let r2 = (self.particles[i].position - self.particles[j].position).length_squared();
                    if r2 < h2 {
                        // Poly6 kernel: W = (h-r)
                        let w = (h2 - r2).powi(3);
                        density += w;
                    }
                }
                self.particles[i].density = density;
            }

            // SPH pressure force
            let k = self.emitters.first().map(|e| e.sph_pressure_k).unwrap_or(1.0);
            let rest_density = 1.0_f32;
            for i in 0..self.particles.len() {
                let mut force = Vec3::ZERO;
                let pi = self.particles[i].density - rest_density;
                let pos_i = self.particles[i].position;
                for j in 0..self.particles.len() {
                    if i == j { continue; }
                    let r = pos_i - self.particles[j].position;
                    let len = r.length();
                    if len < 1e-6 || len > h { continue; }
                    let pj = self.particles[j].density - rest_density;
                    let grad = r / len * (-3.0 * (h - len).powi(2)); // Spiky kernel gradient
                    force += grad * k * (pi + pj) * 0.5;
                }
                self.particles[i].velocity += force * dt;
            }
        }

        // Integrate & update
        let grav = self.gravity;
        self.particles.retain_mut(|p| {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 { return false; }

            p.velocity += grav * dt;
            p.position += p.velocity * dt;

            // Floor bounce
            if p.position.y < 0.0 { p.position.y = 0.0; p.velocity.y *= -0.4; }

            // Interpolate colour and size over lifetime
            let t = 1.0 - (p.lifetime / p.max_lifetime);
            // We'll do a simple lerp on the colour vec4 components
            p.color = Vec4::new(
                lerp(p.color.x, 0.0, t), // Fade towards end color
                lerp(p.color.y, 0.0, t),
                lerp(p.color.z, 0.0, t),
                lerp(p.color.w, 0.0, t),
            );
            // Shrink size
            p.size = p.size * (1.0 - t * 0.5).max(0.01);

            true
        });
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
