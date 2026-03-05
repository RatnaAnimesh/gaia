use hecs::World;
use crate::core::components::{Position, Velocity, Mass};
use glam::Vec3A;

pub struct PhysicsWorld {
    pub ecs: World,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut ecs = World::new();
        
        // Let's spawn a dummy entity to prove ECS allocation works
        ecs.spawn((
            Position {
                current: Vec3A::new(0.0, 10.0, 0.0),
                predicted: Vec3A::ZERO,
            },
            Velocity {
                current: Vec3A::new(1.0, 0.0, 0.0),
            },
            Mass {
                inv_mass: 1.0,
            }
        ));
        
        println!("Initialized Archetype ECS with dummy entities.");
        
        Self {
            ecs
        }
    }
}
