use glam::Vec3A;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContactManifold {
    pub entity_a: u32,
    pub entity_b: u32,
    pub normal: Vec3A,
    pub depth: f32,
    pub point: Vec3A,
    pub active: bool,
    // Add 3 bytes of padding to align the 1-byte bool 
    // to the Metal struct's 4-byte boundaries if needed, 
    // although glam's Vec3A is usually 16-byte aligned.
    _padding: [u8; 3], 
}
