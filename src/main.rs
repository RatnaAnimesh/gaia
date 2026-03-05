use crate::metal::buffers::SharedBuffer;

pub mod core;
pub mod metal;

fn main() {
    println!("Gaia Physics Engine: Initialization sequence started.");
    
    // Boot up the ECS
    let mut world = core::world::PhysicsWorld::new();
    
    // Boot up the Metal Context
    match metal::device::MetalContext::new() {
        Ok(mut context) => {
            println!("✅ Successfully bound to Apple Silicon GPU: {}", context.device.name());
            
            // Dummy frame loop validation
            let delta_time: f32 = 0.016;
            println!("Simulating 1 frame with dt: {}", delta_time);
            
            // In a real loop, we would pull Archetypes into SharedBuffers.
            // For now we'll just demonstrate the buffer binding.
            let entity_count = 2; // Hardcoded dummy count
            let max_pairs = 1000;
            let mut pos_buffer = SharedBuffer::<core::components::Position>::new(&context.device, entity_count);
            let mut aabb_buffer = SharedBuffer::<core::components::BoundingBox>::new(&context.device, entity_count);
            
            // We need a flat array for the hash table (e.g. 10000 cells * 32 entities max)
            let hash_entries = SharedBuffer::<u32>::new(&context.device, 10000 * 32);
            let hash_counters = SharedBuffer::<u32>::new(&context.device, 10000);
            let mut pair_buffer = SharedBuffer::<crate::metal::device::CollisionPair>::new(&context.device, max_pairs);
            let mut pair_count = SharedBuffer::<u32>::new(&context.device, 1);
            
            // Initialize count to 0
            pair_count.as_slice_mut()[0] = 0;
            
            // TODO: CPU Integration -> Encode -> Dispatch compute shader logic goes here.
            
            println!("Simulation frame complete.");
        },
        Err(e) => {
            eprintln!("❌ Failed to initialize Metal framework: {:?}", e);
        }
    }
}
