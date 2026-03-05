use crate::metal::buffers::SharedBuffer;

pub mod core;
pub mod metal;

fn dispatch_compute<'a>(
    encoder: &'a ::metal::ComputeCommandEncoderRef,
    pipeline: &::metal::ComputePipelineState,
    thread_count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    
    // For scaffolding, we use simple threadgroup math
    let w = pipeline.thread_execution_width();
    let threads_per_threadgroup = ::metal::MTLSize::new(w, 1, 1);
    
    // Calculate how many threadgroups we need to cover all items
    let groups = (thread_count + w - 1) / w;
    let threadgroups_per_grid = ::metal::MTLSize::new(groups, 1, 1);
    
    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
}

fn main() {
    println!("Gaia Physics Engine: Initialization sequence started.");
    
    // Boot up the ECS
    let _world = core::world::PhysicsWorld::new();
    
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
            let mut vel_buffer = SharedBuffer::<core::components::Velocity>::new(&context.device, entity_count);
            let mut mass_buffer = SharedBuffer::<core::components::Mass>::new(&context.device, entity_count);
            let mut aabb_buffer = SharedBuffer::<core::components::BoundingBox>::new(&context.device, entity_count);
            
            // We need a flat array for the hash table (e.g. 10000 cells * 32 entities max)
            let hash_entries = SharedBuffer::<u32>::new(&context.device, 10000 * 32);
            let hash_counters = SharedBuffer::<u32>::new(&context.device, 10000);
            let mut pair_buffer = SharedBuffer::<crate::metal::device::CollisionPair>::new(&context.device, max_pairs);
            let mut pair_count = SharedBuffer::<u32>::new(&context.device, 1);
            let mut manifold_buffer = SharedBuffer::<crate::metal::manifolds::ContactManifold>::new(&context.device, max_pairs);
            let mut dt_buffer = SharedBuffer::<f32>::new(&context.device, 1);
            let mut max_pairs_buffer = SharedBuffer::<u32>::new(&context.device, 1);
            let mut entity_count_buffer = SharedBuffer::<u32>::new(&context.device, 1);
            
            // Initialize flat memory
            pair_count.as_slice_mut()[0] = 0;
            dt_buffer.as_slice_mut()[0] = delta_time;
            max_pairs_buffer.as_slice_mut()[0] = max_pairs as u32;
            entity_count_buffer.as_slice_mut()[0] = entity_count as u32;
            
            // ----------------------------------------------------
            // METAL COMMAND ENCODING
            // ----------------------------------------------------
            let command_buffer = context.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            // 1. Clear Spatial Hash
            encoder.set_buffer(0, Some(&hash_counters.buffer), 0);
            dispatch_compute(encoder, &context.clear_hash_pipeline, 10000);
            
            // 2. Build Spatial Hash
            encoder.set_buffer(0, Some(&pos_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&hash_entries.buffer), 0);
            encoder.set_buffer(2, Some(&hash_counters.buffer), 0);
            encoder.set_buffer(3, Some(&entity_count_buffer.buffer), 0);
            dispatch_compute(encoder, &context.build_hash_pipeline, entity_count as u64);
            
            // 3. Find Broad-phase Pairs
            encoder.set_buffer(0, Some(&pos_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&aabb_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&hash_entries.buffer), 0);
            encoder.set_buffer(3, Some(&hash_counters.buffer), 0);
            encoder.set_buffer(4, Some(&pair_buffer.buffer), 0);
            encoder.set_buffer(5, Some(&pair_count.buffer), 0);
            encoder.set_buffer(6, Some(&entity_count_buffer.buffer), 0);
            encoder.set_buffer(7, Some(&max_pairs_buffer.buffer), 0);
            dispatch_compute(encoder, &context.find_pairs_pipeline, entity_count as u64);
            
            // Note: In a real simulation, we must wait for GPU to tell us `pair_count` here or just 
            // dispatch blindly over `max_pairs` and let the kernel early-exit if ID >= pair_count.
            // For scaffolding, we dispatch blindly over `max_pairs`.
            
            // 4. Narrow-Phase GJK / EPA
            encoder.set_buffer(0, Some(&pos_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&aabb_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&pair_buffer.buffer), 0);
            encoder.set_buffer(3, Some(&pair_count.buffer), 0);
            encoder.set_buffer(4, Some(&manifold_buffer.buffer), 0);
            dispatch_compute(encoder, &context.narrowphase_pipeline, max_pairs as u64);
            
            // 5. XPBD Constraint Solver
            encoder.set_buffer(0, Some(&pos_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&mass_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&manifold_buffer.buffer), 0);
            encoder.set_buffer(3, Some(&dt_buffer.buffer), 0);
            encoder.set_buffer(4, Some(&pair_count.buffer), 0);
            // Typically executed in sub-steps (e.g. 10 iterations)
            for _ in 0..10 {
                dispatch_compute(encoder, &context.xpbd_solve_pipeline, max_pairs as u64);
            }
            
            // 6. Velocity derivation
            encoder.set_buffer(0, Some(&pos_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&vel_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&dt_buffer.buffer), 0);
            encoder.set_buffer(3, Some(&entity_count_buffer.buffer), 0);
            dispatch_compute(encoder, &context.xpbd_velocity_pipeline, entity_count as u64);
            
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed(); // Blocking wait to synchronize with CPU
            
            println!("Simulation frame complete. Executed all 6 Compute passes on Apple Silicon.");
        },
        Err(e) => {
            eprintln!("❌ Failed to initialize Metal framework: {:?}", e);
        }
    }
}
