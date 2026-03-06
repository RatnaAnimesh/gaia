use macroquad::prelude::*;
use egui_macroquad::egui;

pub mod core;
pub mod editor;
pub mod ui;

use editor::EditorState;
use ui::{apply_blender_theme, draw_menu_bar, draw_toolbar, draw_right_panels, draw_status_bar};
use core::soft_body::{MatrixFreeSoftBody, Tetrahedron};
use core::fluid::FluidGrid;
use core::light::HamiltonianPropagator;
use core::deq::DeqSolver;

fn window_conf() -> Conf {
    Conf {
        window_title: "Gaia — Physics Engine & Editor".to_string(),
        window_width: 1400,
        window_height: 900,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut state = EditorState::new();

    // ── Physics backends ───────────────────────────────────────
    // Rigid body simulation (positions driven by DEQ)
    let mut rb_pos   = vec![Vec3::new(0.0, 8.0, 0.0)];
    let mut rb_vel   = vec![Vec3::ZERO];
    let rb_extents   = vec![Vec3::new(1.0, 1.0, 1.0)];
    let floor_pos    = Vec3::new(0.0, -1.0, 0.0);
    let floor_ext    = Vec3::new(10.0, 1.0, 10.0);

    // Soft body
    let mut soft = MatrixFreeSoftBody::new(80.0, 400.0);
    soft.particles.push(vec3(-5.0, 10.0, 0.0));
    soft.particles.push(vec3(-3.0, 10.0, 0.0));
    soft.particles.push(vec3(-4.0, 10.0, -2.0));
    soft.particles.push(vec3(-4.0, 13.0, 0.0));
    for _ in 0..4 { soft.velocities.push(Vec3::ZERO); soft.masses.push(0.5); }
    let dm = Mat3::from_cols(
        soft.particles[1] - soft.particles[0],
        soft.particles[2] - soft.particles[0],
        soft.particles[3] - soft.particles[0],
    );
    let inv_dm = dm.inverse();
    let vol    = dm.determinant().abs() / 6.0;
    soft.elements.push(Tetrahedron { v0: 0, v1: 1, v2: 2, v3: 3, inv_rest_shape: inv_dm, volume: vol });

    // Fluid
    let fluid_res = 16usize;
    let mut fluid = FluidGrid::new(fluid_res, fluid_res, fluid_res, 0.5);
    fluid.add_impulse(fluid_res / 2, 2, fluid_res / 2, 5.0);

    // Light
    let mut light = HamiltonianPropagator::new(64, 64);
    let lp = [0.0f32, 18.0, 0.0];
    light.emit_from_point(lp, 64, [1.0, 0.9, 0.6]);
    // ──────────────────────────────────────────────────────────

    let dt = 0.016_f32; // fixed physics step

    loop {
        // ── Input & Camera ────────────────────────────────────
        state.camera.update();

        // Keyboard tool shortcuts (only when egui doesn't want keyboard)
        if is_key_pressed(KeyCode::Q) { state.active_tool = editor::ActiveTool::Select; }
        if is_key_pressed(KeyCode::G) { state.active_tool = editor::ActiveTool::Move;   }
        if is_key_pressed(KeyCode::R) { state.active_tool = editor::ActiveTool::Rotate; }
        if is_key_pressed(KeyCode::S) { state.active_tool = editor::ActiveTool::Scale;  }
        if is_key_pressed(KeyCode::Space) { state.simulation_playing = !state.simulation_playing; }

        // ── Physics step ──────────────────────────────────────
        if state.simulation_playing {
            state.frame += 1;

            // Rigid body: gravity + DEQ floor constraint
            for (i, pos) in rb_pos.iter_mut().enumerate() {
                rb_vel[i].y -= 9.81 * dt;
                *pos += rb_vel[i] * dt;

                // DEQ constraint with floor
                let a_min = *pos - rb_extents[i];
                let a_max = *pos + rb_extents[i];
                let b_min = floor_pos - floor_ext;
                let b_max = floor_pos + floor_ext;

                if a_min.x <= b_max.x && a_max.x >= b_min.x
                    && a_min.y <= b_max.y && a_max.y >= b_min.y
                    && a_min.z <= b_max.z && a_max.z >= b_min.z
                {
                    let depth = b_max.y - a_min.y;
                    let solver = DeqSolver::new(20, 0.001, 1);
                    let z_star = solver.forward_solve(&[depth * 0.2]);
                    pos.y    += z_star[0];
                    rb_vel[i].y = rb_vel[i].y.max(0.0);
                }
            }

            // Sync rigid body positions to EditorState
            if let Some(obj) = state.objects.get_mut(0) {
                obj.position = rb_pos[0];
            }

            // Soft body
            soft.step(dt);
            // Sync soft body centroid
            if soft.particles.len() >= 4 {
                let ctr = (soft.particles[0] + soft.particles[1] + soft.particles[2] + soft.particles[3]) / 4.0;
                if let Some(obj) = state.objects.get_mut(2) { obj.position = ctr; }
            }

            // Fluid + light
            fluid.step(dt);
            light.propagate(2);
            if light.wavefronts.is_empty() {
                light.emit_from_point(lp, 64, [1.0, 0.9, 0.6]);
            }
        }

        // ── Rendering ─────────────────────────────────────────
        clear_background(Color::from_rgba(30, 30, 30, 255));

        // 3D Viewport
        set_camera(&state.camera.to_camera3d());

        if state.show_grid {
            draw_grid(30, 1.0, Color::from_rgba(80, 80, 80, 255), Color::from_rgba(55, 55, 55, 255));
        }

        // Floor
        draw_cube(floor_pos, floor_ext * 2.0, None, Color::from_rgba(60, 90, 60, 255));
        if state.show_wireframe {
            draw_cube_wires(floor_pos, floor_ext * 2.0, Color::from_rgba(80, 80, 80, 255));
        }

        // Scene objects
        for (i, obj) in state.objects.iter().enumerate() {
            if !obj.visible { continue; }
            let is_selected = state.selected == Some(i);
            let col = if is_selected {
                Color::from_rgba(255, 165, 0, 255) // orange highlight like Blender
            } else {
                obj.color
            };

            match obj.physics_type {
                editor::PhysicsType::Rigid => {
                    if i < rb_pos.len() {
                        let ext = rb_extents[0] * obj.scale;
                        draw_cube(rb_pos[i], ext * 2.0, None, col);
                        draw_cube_wires(rb_pos[i], ext * 2.0, WHITE);
                    }
                }
                editor::PhysicsType::SoftBody => {
                    // Draw FEM tetrahedron wireframe
                    for tet in &soft.elements {
                        let p = [
                            soft.particles[tet.v0],
                            soft.particles[tet.v1],
                            soft.particles[tet.v2],
                            soft.particles[tet.v3],
                        ];
                        let c = if is_selected { Color::from_rgba(255, 165, 0, 255) } else { BLUE };
                        draw_line_3d(p[0], p[1], c);
                        draw_line_3d(p[0], p[2], c);
                        draw_line_3d(p[1], p[2], c);
                        draw_line_3d(p[3], p[0], c);
                        draw_line_3d(p[3], p[1], c);
                        draw_line_3d(p[3], p[2], c);
                    }
                    // Render each vertex as a small sphere
                    for p in &soft.particles {
                        draw_sphere(*p, 0.18, None, col);
                    }
                }
                editor::PhysicsType::Static => {
                    // Point Light representation
                    draw_sphere(obj.position, 0.3, None, col);
                    // Light rays
                    for wf in &light.wavefronts {
                        draw_sphere(vec3(wf.x[0], wf.x[1], wf.x[2]), 0.08, None, Color::new(1.0, 0.95, 0.5, 0.6));
                    }
                }
                editor::PhysicsType::Fluid => {
                    draw_cube(obj.position, Vec3::new(4.0, 4.0, 4.0), None, Color::new(0.3, 0.7, 1.0, 0.3));
                    draw_cube_wires(obj.position, Vec3::new(4.0, 4.0, 4.0), SKYBLUE);
                }
            }
        }

        // Selection bounding box highlight (orange outline)
        if let Some(idx) = state.selected {
            if let Some(obj) = state.objects.get(idx) {
                draw_cube_wires(obj.position, obj.scale * 2.1, Color::from_rgba(255, 140, 0, 255));
            }
        }

        set_default_camera();

        // ── egui UI Panels ────────────────────────────────────
        egui_macroquad::ui(|ctx| {
            apply_blender_theme(ctx);
            draw_menu_bar(ctx, &mut state);
            draw_status_bar(ctx, &state, get_fps());
            draw_toolbar(ctx, &mut state);
            draw_right_panels(ctx, &mut state);
        });
        egui_macroquad::draw();

        next_frame().await;
    }
}
