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
    
    // Soft Body Scaffolding
    // ----------------------------------------------------
    let mut soft_body = core::soft_body::MatrixFreeSoftBody::new(100.0, 500.0);
    soft_body.particles.push(vec3(-2.0, 10.0, 0.0));
    soft_body.particles.push(vec3(2.0, 10.0, 0.0));
    soft_body.particles.push(vec3(0.0, 10.0, -2.0));
    soft_body.particles.push(vec3(0.0, 14.0, 0.0));
    
    for _ in 0..4 {
        soft_body.velocities.push(Vec3::ZERO);
        soft_body.masses.push(0.5);
    }
    
    let d_m = Mat3::from_cols(
        soft_body.particles[1] - soft_body.particles[0],
        soft_body.particles[2] - soft_body.particles[0],
        soft_body.particles[3] - soft_body.particles[0],
    );
    let inv_rest = d_m.inverse();
    let vol = d_m.determinant().abs() / 6.0;
    
    soft_body.elements.push(core::soft_body::Tetrahedron {
        v0: 0, v1: 1, v2: 2, v3: 3,
        inv_rest_shape: inv_rest,
        volume: vol,
    });
    // ----------------------------------------------------
    
    // ----------------------------------------------------
    // Fluid Simulation (Chebyshev-Preconditioned)
    // ----------------------------------------------------
    let fluid_res = 16usize;
    let mut fluid = core::fluid::FluidGrid::new(fluid_res, fluid_res, fluid_res, 0.5);
    // Seed an upward splash at the center
    fluid.add_impulse(fluid_res / 2, 2, fluid_res / 2, 5.0);
    // ----------------------------------------------------

    // ----------------------------------------------------
    // Hamiltonian Wavefront Light Transport (Phase 10)
    // ----------------------------------------------------
    let mut light_propagator = core::light::HamiltonianPropagator::new(64, 64);
    let light_pos = [0.0f32, 18.0, 0.0];
    light_propagator.emit_from_point(light_pos, 64, [1.0, 0.9, 0.6]);
    // ----------------------------------------------------

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

        // --- Step the Soft Body ---
        soft_body.step(step_time);
        
        // --- Step the Fluid ---
        fluid.step(step_time);

        // --- Propagate Light Wavefronts ---
        light_propagator.propagate(2); // 2 steps per frame = real-time
        // Re-seed light if all absorbed
        if light_propagator.wavefronts.is_empty() {
            light_propagator.emit_from_point(light_pos, 64, [1.0, 0.9, 0.6]);
        }

        set_camera(&Camera3D {
            position: vec3(0.0, 10.0, 25.0),
            up: vec3(0.0, 1.0, 0.0),
            target: vec3(0.0, 2.0, 0.0),
            ..Default::default()
        });

        draw_grid(20, 1.0, BLACK, GRAY);

        // Draw entities
        // Object 0: Falling box (Rigid DEQ)
        draw_cube(vec3(positions[0].x, positions[0].y, positions[0].z), vec3(extents[0].x*2.0, extents[0].y*2.0, extents[0].z*2.0), None, RED);
        draw_cube_wires(vec3(positions[0].x, positions[0].y, positions[0].z), vec3(extents[0].x*2.0, extents[0].y*2.0, extents[0].z*2.0), BLACK);
        
        // Object 1: Static floor
        draw_cube(vec3(positions[1].x, positions[1].y, positions[1].z), vec3(extents[1].x*2.0, extents[1].y*2.0, extents[1].z*2.0), None, DARKGREEN);
        draw_cube_wires(vec3(positions[1].x, positions[1].y, positions[1].z), vec3(extents[1].x*2.0, extents[1].y*2.0, extents[1].z*2.0), BLACK);

        // Soft Body: Draw Jello Tetrahedron
        for tet in &soft_body.elements {
            let p0 = soft_body.particles[tet.v0];
            let p1 = soft_body.particles[tet.v1];
            let p2 = soft_body.particles[tet.v2];
            let p3 = soft_body.particles[tet.v3];
            
            draw_line_3d(vec3(p0.x, p0.y, p0.z), vec3(p1.x, p1.y, p1.z), BLUE);
            draw_line_3d(vec3(p0.x, p0.y, p0.z), vec3(p2.x, p2.y, p2.z), BLUE);
            draw_line_3d(vec3(p1.x, p1.y, p1.z), vec3(p2.x, p2.y, p2.z), BLUE);
            draw_line_3d(vec3(p3.x, p3.y, p3.z), vec3(p0.x, p0.y, p0.z), BLUE);
            draw_line_3d(vec3(p3.x, p3.y, p3.z), vec3(p1.x, p1.y, p1.z), BLUE);
            draw_line_3d(vec3(p3.x, p3.y, p3.z), vec3(p2.x, p2.y, p2.z), BLUE);
        }

        // Render active wavefronts as tiny yellow points
        for wf in &light_propagator.wavefronts {
            draw_sphere(vec3(wf.x[0], wf.x[1], wf.x[2]), 0.1, None, YELLOW);
        }

        set_default_camera();
        draw_text("Gaia Physics Engine - Macroquad Frontend", 10.0, 20.0, 30.0, BLACK);
        draw_text(&format!("FPS: {}", get_fps()), 10.0, 50.0, 20.0, BLACK);
        draw_text(&format!("Cube Y: {:.3}", positions[0].y), 10.0, 80.0, 20.0, BLACK);
        draw_text(&format!("Soft Body Particles: {}", soft_body.particles.len()), 10.0, 110.0, 20.0, BLUE);
        draw_text("Fluid: Chebyshev PCG (8 iters, no global sync)", 10.0, 140.0, 18.0, DARKBLUE);

        next_frame().await
    }
}

