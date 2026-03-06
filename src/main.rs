use macroquad::prelude::*;

pub mod core;
// pub mod metal; // Metal compute shaders are bypassed for this visual proof to avoid compilation hangs

#[macroquad::main("Gaia Physics Engine")]
async fn main() {
    let delta_time: f32 = 0.016;
            
    // Set up initial conditions
    let mut positions = vec![
        glam::Vec3A::new(0.0, 10.0, 0.0),
        glam::Vec3A::new(0.0, 0.0, 0.0)
    ];
    let mut predicted_pos = positions.clone();
    let mut velocities = vec![
        glam::Vec3A::new(0.0, 0.0, 0.0),
        glam::Vec3A::new(0.0, 0.0, 0.0)
    ];
    let masses = vec![1.0_f32, 0.0_f32]; // inv_mass
    
    // Extents for static floor vs falling box
    let extents = vec![
        glam::Vec3A::new(1.0, 1.0, 1.0),
        glam::Vec3A::new(10.0, 1.0, 10.0)
    ];
    
    let entity_count = 2;
    
    loop {
        clear_background(LIGHTGRAY);

        // Substepping or fixed delta time
        let mut sim_dt = get_frame_time();
        if sim_dt > 0.1 { sim_dt = 0.1; } // cap dt

        // We run a fixed step for stability
        let step_time = 0.016;

        // Pre-integration (Gravity application)
        for i in 0..entity_count {
            if masses[i] > 0.0 {
                // v = v + g * dt
                velocities[i].y -= 9.81 * step_time;
                // p_pred = p_cur + v * dt
                predicted_pos[i].x = positions[i].x + velocities[i].x * step_time;
                predicted_pos[i].y = positions[i].y + velocities[i].y * step_time;
                predicted_pos[i].z = positions[i].z + velocities[i].z * step_time;
            }
        }
        
        // 5. DEQ Constraint Solver (Neural Fixed Point)
        let mut p_a = predicted_pos[0];
        let p_b = predicted_pos[1];
        
        let a_min = p_a - extents[0];
        let a_max = p_a + extents[0];
        let b_min = p_b - extents[1];
        let b_max = p_b + extents[1];
        
        let overlap = (a_min.x <= b_max.x && a_max.x >= b_min.x) &&
                      (a_min.y <= b_max.y && a_max.y >= b_min.y) &&
                      (a_min.z <= b_max.z && a_max.z >= b_min.z);
                      
        if overlap {
            // Feed the collision depth into the Deep Equilibrium Model
            let depth = b_max.y - a_min.y;
            let x_features = vec![depth * 0.2]; // Scale features down for stability
            
            // The DEQ solves the LCP matrix equation in O(1) bound iterations
            let solver = core::deq::DeqSolver::new(20, 0.001, 1);
            let z_star = solver.forward_solve(&x_features);
            
            // Apply the inferred position shift (impulse lambda)
            p_a.y += z_star[0];
            predicted_pos[0] = p_a;
        }
        
        // 6. Velocity Derivation
        for i in 0..entity_count {
            if masses[i] > 0.0 {
                velocities[i] = (predicted_pos[i] - positions[i]) / step_time;
                positions[i] = predicted_pos[i];
            }
        }

        set_camera(&Camera3D {
            position: vec3(0.0, 10.0, 25.0),
            up: vec3(0.0, 1.0, 0.0),
            target: vec3(0.0, 2.0, 0.0),
            ..Default::default()
        });

        draw_grid(20, 1.0, BLACK, GRAY);

        // Draw entities
        // Object 0: Falling box
        draw_cube(vec3(positions[0].x, positions[0].y, positions[0].z), vec3(extents[0].x*2.0, extents[0].y*2.0, extents[0].z*2.0), None, RED);
        draw_cube_wires(vec3(positions[0].x, positions[0].y, positions[0].z), vec3(extents[0].x*2.0, extents[0].y*2.0, extents[0].z*2.0), BLACK);
        
        // Object 1: Static floor
        draw_cube(vec3(positions[1].x, positions[1].y, positions[1].z), vec3(extents[1].x*2.0, extents[1].y*2.0, extents[1].z*2.0), None, DARKGREEN);
        draw_cube_wires(vec3(positions[1].x, positions[1].y, positions[1].z), vec3(extents[1].x*2.0, extents[1].y*2.0, extents[1].z*2.0), BLACK);

        set_default_camera(); // Back to 2D for text
        draw_text("Gaia Physics Engine - Macroquad Frontend", 10.0, 20.0, 30.0, BLACK);
        draw_text(&format!("FPS: {}", get_fps()), 10.0, 50.0, 20.0, BLACK);
        draw_text(&format!("Cube Y: {:.3}", positions[0].y), 10.0, 80.0, 20.0, BLACK);

        next_frame().await
    }
}

