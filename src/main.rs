pub mod core;
pub mod metal;

fn main() {
    println!("Gaia Physics Engine: Initialization sequence started.");
    
    // Boot up the ECS
    let _world = core::world::PhysicsWorld::new();
    
    // Boot up the Metal Context
    match metal::device::MetalContext::new() {
        Ok(context) => {
            println!("✅ Successfully bound to Apple Silicon GPU: {}", context.device.name());
            
            // Dummy frame loop validation
            let delta_time: f32 = 0.016;
            println!("Simulating 1 frame with dt: {}", delta_time);
            // TODO: dispatch actually pipeline here
            println!("Simulation frame complete.");
        },
        Err(e) => {
            eprintln!("❌ Failed to initialize Metal framework: {:?}", e);
        }
    }
}
