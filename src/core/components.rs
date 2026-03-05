use glam::Vec3A;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Position {
    pub current: Vec3A,
    pub predicted: Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Velocity {
    pub current: Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Mass {
    pub inv_mass: f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BoundingBox {
    pub min: Vec3A,
    pub max: Vec3A,
}
