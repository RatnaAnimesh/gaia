use hecs::World;
use crate::core::components::{Position, Velocity, Mass, BoundingBox};
use glam::Vec3A;

pub struct PhysicsWorld {
    pub ecs: World,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut ecs = World::new();
        
        // Let's spawn a couple of dummy entities to test collisions
        ecs.spawn((
            Position {
                current: Vec3A::new(0.0, 10.0, 0.0),
                predicted: Vec3A::ZERO,
            },
            Velocity {
                current: Vec3A::new(0.0, -9.8, 0.0),
            },
            Mass {
                inv_mass: 1.0,
            },
            BoundingBox {
                min: Vec3A::new(-1.0, -1.0, -1.0),
                max: Vec3A::new(1.0, 1.0, 1.0),
            }
        ));
        
        ecs.spawn((
            Position {
                current: Vec3A::new(0.0, 0.0, 0.0),
                predicted: Vec3A::ZERO,
            },
            Velocity {
                current: Vec3A::ZERO,
            },
            Mass {
                inv_mass: 0.0, // Static object
            },
            BoundingBox {
                min: Vec3A::new(-10.0, -1.0, -10.0),
                max: Vec3A::new(10.0, 1.0, 10.0),
            }
        ));
        
        println!("Initialized Archetype ECS with dummy entities.");
        
        Self {
            ecs
        }
    }
}
