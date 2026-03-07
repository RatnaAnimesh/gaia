use macroquad::prelude::*;
use egui_macroquad::egui;

pub mod core;
pub mod editor;
pub mod ui;

use editor::{EditorState, PhysicsType};
use ui::{apply_blender_theme, draw_menu_bar, draw_toolbar, draw_right_panels, draw_status_bar};
use core::solver::{PhysicsWorld, RigidBody};
use core::shapes::{Shape, PhysicsMaterial};
use core::soft_body::{MatrixFreeSoftBody, Tetrahedron};
use core::fluid::FluidGrid;
use core::light::HamiltonianPropagator;
use core::cloth::Cloth;
use core::particles::ParticleSystem;
use core::raycast::{ray_cast, screen_to_ray};

fn window_conf() -> Conf {
    Conf {
        window_title: "gaia  Physics Engine & Editor".to_string(),
        window_width: 1400,
        window_height: 900,
        high_dpi: true,
        ..Default::default()
    }
}

//  Colour helpers 
fn rb_color(id: usize) -> Color {
    let palette = [
        Color::from_rgba(220, 60, 60, 255),
        Color::from_rgba(60, 160, 220, 255),
        Color::from_rgba(80, 200, 100, 255),
        Color::from_rgba(200, 130, 50, 255),
        Color::from_rgba(180, 80, 220, 255),
        Color::from_rgba(50, 200, 200, 255),
    ];
    palette[id % palette.len()]
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut state = EditorState::new();

    // 
    //   PHYSICS WORLD (GJK + Sequential Impulse)         
    // 
    let mut phys = PhysicsWorld::new();

    // Floor (static, infinite mass)
    let floor_mat = PhysicsMaterial { restitution: 0.3, friction_static: 0.7, friction_dynamic: 0.5, density: 0.0 };
    let mut floor_body = RigidBody::new(0, Shape::Box { half_extents: Vec3::new(10.0, 1.0, 10.0) }, Vec3::new(0.0, -1.0, 0.0), floor_mat);
    floor_body.inv_mass = 0.0; // Static
    floor_body.inv_inertia = Vec3::ZERO;
    phys.add_body(floor_body);

    // Starting falling cube
    let box_mat = PhysicsMaterial::default();
    phys.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 8.0, 0.0), box_mat));

    // Starting sphere
    let sph_mat = PhysicsMaterial { restitution: 0.6, ..Default::default() };
    phys.add_body(RigidBody::new(2, Shape::Sphere { radius: 1.0 }, Vec3::new(1.5, 12.0, 0.5), sph_mat));

    // 
    //   Soft Body (FEM Neo-Hookean)                      
    // 
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
    soft.elements.push(Tetrahedron {
        v0: 0, v1: 1, v2: 2, v3: 3,
        inv_rest_shape: dm.inverse(),
        volume: dm.determinant().abs() / 6.0,
    });

    // 
    //   Cloth (PBD)                                      
    // 
    let mut cloth = Cloth::grid(8, 8, 0.6, Vec3::new(4.0, 8.0, -2.0));

    // 
    //   Particle System                                  
    // 
    let mut particles = ParticleSystem::new();
    let mut emitter = core::particles::ParticleEmitter::new(Vec3::new(-8.0, 1.0, 0.0), 20.0);
    emitter.initial_vel = Vec3::new(0.5, 4.0, 0.0);
    emitter.vel_spread  = 0.8;
    particles.emitters.push(emitter);

    // 
    //   Fluid (Chebyshev)                               
    // 
    let fluid_res = 16usize;
    let mut fluid = FluidGrid::new(fluid_res, fluid_res, fluid_res, 0.5);
    fluid.add_impulse(fluid_res / 2, 2, fluid_res / 2, 5.0);

    // 
    //   Light (Hamiltonian Wavefront)                    
    // 
    let mut light = HamiltonianPropagator::new(64, 64);
    let lp = [0.0f32, 18.0, 0.0];
    light.emit_from_point(lp, 64, [1.0, 0.9, 0.6]);

    let dt = 0.016_f32;

    loop {
        //  Input 
        state.camera.update();

        if is_key_pressed(KeyCode::Q) { state.active_tool = editor::ActiveTool::Select; }
        if is_key_pressed(KeyCode::G) { state.active_tool = editor::ActiveTool::Move;   }
        if is_key_pressed(KeyCode::R) { state.active_tool = editor::ActiveTool::Rotate; }
        if is_key_pressed(KeyCode::S) { state.active_tool = editor::ActiveTool::Scale;  }
        if is_key_pressed(KeyCode::Space) { state.simulation_playing = !state.simulation_playing; }

        // Delete selected object (X key)
        if is_key_pressed(KeyCode::X) {
            if let Some(idx) = state.selected {
                if idx > 0 && idx < state.objects.len() {
                    state.objects.remove(idx);
                    // Remove corresponding rigid body if it's a rigid type
                    if idx + 1 < phys.bodies.len() {
                        phys.bodies.remove(idx + 1);
                    }
                    state.selected = None;
                }
            }
        }

        // Mouse click  raycast pick
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let cam = state.camera.to_camera3d();
            let ray = screen_to_ray(
                mx, my,
                screen_width(), screen_height(),
                cam.position, cam.target, cam.up,
                std::f32::consts::FRAC_PI_4,
            );
            if let Some(hit) = ray_cast(&ray, &phys.bodies) {
                // Find editor object whose position matches body
                for (i, obj) in state.objects.iter().enumerate() {
                    if (obj.position - phys.bodies[hit.body_id].position).length() < 2.0 {
                        state.selected = Some(i);
                        break;
                    }
                }
            }
        }

        // "Add Cube" from editor state  spawn a rigid body from the Add menu
        // The UI sets a pending_add flag we check here
        if state.pending_add_cube {
            state.pending_add_cube = false;
            let pos = Vec3::new(0.0, 10.0 + (phys.bodies.len() as f32) * 2.5, 0.0);
            let id = phys.bodies.len();
            phys.add_body(RigidBody::new(id, Shape::Box { half_extents: Vec3::ONE }, pos, PhysicsMaterial::default()));
        }
        if state.pending_add_sphere {
            state.pending_add_sphere = false;
            let pos = Vec3::new(1.0, 12.0 + (phys.bodies.len() as f32) * 2.0, 0.0);
            let id = phys.bodies.len();
            phys.add_body(RigidBody::new(id, Shape::Sphere { radius: 1.0 }, pos, PhysicsMaterial { restitution: 0.6, ..Default::default() }));
        }

        //  Physics Step 
        if state.simulation_playing {
            state.frame += 1;
            phys.step(dt);
            soft.step(dt);
            cloth.step(dt);
            particles.step(dt);
            fluid.step(dt);
            light.propagate(2);
            if light.wavefronts.is_empty() {
                light.emit_from_point(lp, 64, [1.0, 0.9, 0.6]);
            }

            // Sync rigid body positions back to EditorState objects
            for (i, obj) in state.objects.iter_mut().enumerate() {
                if obj.physics_type == PhysicsType::Rigid && i + 1 < phys.bodies.len() {
                    obj.position = phys.bodies[i + 1].position;
                }
            }
        }

        //  3D Rendering 
        clear_background(Color::from_rgba(28, 28, 28, 255));
        set_camera(&state.camera.to_camera3d());

        if state.show_grid {
            draw_grid(30, 1.0,
                Color::from_rgba(70, 70, 70, 255),
                Color::from_rgba(48, 48, 48, 255));
        }

        // Render all rigid bodies from PhysicsWorld
        for body in &phys.bodies {
            let is_static = body.inv_mass == 0.0;
            let is_selected = state.selected
                .and_then(|idx| state.objects.get(idx))
                .map(|obj| (obj.position - body.position).length() < 2.0)
                .unwrap_or(false);

            let col = if is_selected {
                Color::from_rgba(255, 165, 0, 255)
            } else if is_static {
                Color::from_rgba(55, 80, 55, 255)
            } else {
                rb_color(body.id)
            };

            match &body.shape {
                Shape::Box { half_extents } => {
                    draw_cube(body.position, *half_extents * 2.0, None, col);
                    if state.show_wireframe || is_selected {
                        draw_cube_wires(body.position, *half_extents * 2.0, if is_selected { ORANGE } else { Color::from_rgba(80, 80, 80, 255) });
                    }
                }
                Shape::Sphere { radius } => {
                    draw_sphere(body.position, *radius, None, col);
                    if is_selected { draw_sphere(body.position, radius * 1.02, None, Color::from_rgba(255, 140, 0, 80)); }
                }
                Shape::Capsule { radius, half_height } => {
                    let top = body.position + Vec3::Y * *half_height;
                    let bot = body.position - Vec3::Y * *half_height;
                    draw_sphere(top, *radius, None, col);
                    draw_sphere(bot, *radius, None, col);
                    draw_line_3d(top, bot, col);
                }
            }

            // Velocity vector (debug)
            if state.simulation_playing && !is_static && body.velocity.length_squared() > 0.1 {
                draw_line_3d(body.position, body.position + body.velocity * 0.2, LIME);
            }
        }

        // Soft body FEM tetrahedron
        for tet in &soft.elements {
            let p = [soft.particles[tet.v0], soft.particles[tet.v1],
                     soft.particles[tet.v2], soft.particles[tet.v3]];
            let c = Color::from_rgba(60, 120, 240, 255);
            draw_line_3d(p[0], p[1], c); draw_line_3d(p[0], p[2], c);
            draw_line_3d(p[1], p[2], c); draw_line_3d(p[3], p[0], c);
            draw_line_3d(p[3], p[1], c); draw_line_3d(p[3], p[2], c);
            for &vp in &p { draw_sphere(vp, 0.18, None, c); }
        }

        // PBD Cloth mesh
        let cw = 8usize;
        let ch = 8usize;
        for r in 0..ch {
            for c in 0..cw {
                let idx = |rr, cc| rr * cw + cc;
                let pa = cloth.particles[idx(r, c)].position;
                if c + 1 < cw {
                    let pb = cloth.particles[idx(r, c + 1)].position;
                    draw_line_3d(pa, pb, Color::from_rgba(180, 140, 60, 200));
                }
                if r + 1 < ch {
                    let pb = cloth.particles[idx(r + 1, c)].position;
                    draw_line_3d(pa, pb, Color::from_rgba(180, 140, 60, 200));
                }
            }
        }

        // Particle system
        for p in &particles.particles {
            let col = Color::new(p.color.x, p.color.y, p.color.z, p.color.w);
            draw_sphere(p.position, p.size, None, col);
        }

        // Hamiltonian light wavefronts
        for wf in &light.wavefronts {
            draw_sphere(vec3(wf.x[0], wf.x[1], wf.x[2]), 0.07, None, Color::new(1.0, 0.95, 0.5, 0.55));
        }

        set_default_camera();

        //  egui UI Panels 
        egui_macroquad::ui(|ctx| {
            apply_blender_theme(ctx);
            draw_menu_bar(ctx, &mut state);
            draw_status_bar(ctx, &state, get_fps(), phys.bodies.len(), particles.particles.len());
            draw_toolbar(ctx, &mut state);
            draw_right_panels(ctx, &mut state);
        });
        egui_macroquad::draw();

        next_frame().await;
    }
}
