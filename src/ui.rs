use egui_macroquad::egui::{self, Context, RichText, Color32, Frame, Margin, Vec2};
use crate::editor::{EditorState, ActiveTool, PhysicsType};

/// Dark theme colours mimicking Blender's dark UI
const PANEL_BG:     Color32 = Color32::from_rgb(45, 45, 45);
const HEADER_BG:    Color32 = Color32::from_rgb(35, 35, 35);
const ACCENT:       Color32 = Color32::from_rgb(70, 120, 255);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(220, 220, 220);
const TEXT_DIM:     Color32 = Color32::from_rgb(140, 140, 140);
const SEL_BG:       Color32 = Color32::from_rgb(55, 90, 175);

/// Apply a Blender-dark theme to egui
pub fn apply_blender_theme(ctx: &Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill            = PANEL_BG;
    visuals.window_fill           = PANEL_BG;
    visuals.override_text_color   = Some(TEXT_PRIMARY);
    visuals.selection.bg_fill     = ACCENT;
    visuals.widgets.active.bg_fill   = ACCENT;
    visuals.widgets.hovered.bg_fill  = Color32::from_rgb(60, 100, 200);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(58, 58, 58);
    ctx.set_visuals(visuals);
    
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing  = Vec2::new(4.0, 4.0);
    style.spacing.button_padding = Vec2::new(8.0, 4.0);
    ctx.set_style(style);
}

/// Menu bar across the top
pub fn draw_menu_bar(ctx: &Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("menu_bar")
        .frame(Frame::none().fill(HEADER_BG).inner_margin(Margin { left: 8, right: 8, top: 4, bottom: 4 }))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Logo/Title
                ui.label(RichText::new(" GAIA").size(16.0).color(ACCENT).strong());
                ui.separator();

                // Menus
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene").clicked() { state.objects.clear(); }
                    ui.separator();
                    ui.button("Save");
                    ui.button("Open");
                    ui.separator();
                    ui.button("Export");
                });
                ui.menu_button("Edit", |ui| {
                    ui.button("Undo");
                    ui.button("Redo");
                    ui.separator();
                    ui.button("Preferences");
                });
                ui.menu_button("Add", |ui| {
                    ui.label(RichText::new("Mesh").color(TEXT_DIM));
                    if ui.button("   Cube (Rigid)").clicked() {
                        state.pending_add_cube = true;
                        ui.close_menu();
                    }
                    if ui.button("   Sphere (Rigid)").clicked() {
                        state.pending_add_sphere = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label(RichText::new("Soft / Fluid").color(TEXT_DIM));
                    if ui.button("   Soft Body (FEM)").clicked() {
                        state.objects.push(crate::editor::SceneObject::new(
                            "SoftBody",
                            macroquad::prelude::Vec3::new(2.0, 8.0, 0.0),
                            macroquad::prelude::BLUE,
                            PhysicsType::SoftBody,
                        ));
                        ui.close_menu();
                    }
                    if ui.button("   Fluid Volume").clicked() {
                        state.objects.push(crate::editor::SceneObject::new(
                            "Fluid",
                            macroquad::prelude::Vec3::new(-3.0, 4.0, 0.0),
                            macroquad::prelude::SKYBLUE,
                            PhysicsType::Fluid,
                        ));
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label(RichText::new("Light").color(TEXT_DIM));
                    if ui.button("   Point Light").clicked() {
                        state.objects.push(crate::editor::SceneObject::new(
                            "Light",
                            macroquad::prelude::Vec3::new(0.0, 15.0, 0.0),
                            macroquad::prelude::YELLOW,
                            PhysicsType::Static,
                        ));
                        ui.close_menu();
                    }
                });
                ui.menu_button("Render", |ui| {
                    ui.button("Render Image");
                    ui.button("Render Animation");
                    ui.separator();
                    if ui.checkbox(&mut state.show_wireframe, "Wireframe").clicked() {}
                    if ui.checkbox(&mut state.show_grid, "Grid").clicked() {}
                });
                ui.menu_button("Help", |ui| {
                    ui.label("gaia Physics Engine");
                    ui.label(RichText::new("github.com/RatnaAnimesh/gaia").color(TEXT_DIM).small());
                });

                // Right-aligned simulation controls
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let play_label = if state.simulation_playing { " Pause" } else { " Play" };
                    if ui.button(RichText::new(play_label).color(if state.simulation_playing { Color32::LIGHT_GREEN } else { TEXT_PRIMARY })).clicked() {
                        state.simulation_playing = !state.simulation_playing;
                    }
                    ui.label(RichText::new(format!("Frame {}", state.frame)).color(TEXT_DIM));
                });
            });
        });
}

/// Left vertical toolbar (Select / Move / Rotate / Scale)
pub fn draw_toolbar(ctx: &Context, state: &mut EditorState) {
    egui::SidePanel::left("toolbar")
        .resizable(false)
        .exact_width(44.0)
        .frame(Frame::none().fill(HEADER_BG))
        .show(ctx, |ui| {
            ui.add_space(4.0);
            let tools = [
                ("",  ActiveTool::Select,  "Select (Q)"),
                ("",  ActiveTool::Move,    "Grab/Move (G)"),
                ("",  ActiveTool::Rotate,  "Rotate (R)"),
                ("",  ActiveTool::Scale,   "Scale (S)"),
            ];
            for (icon, tool, tooltip) in tools {
                let active = state.active_tool == tool;
                let btn = egui::Button::new(RichText::new(icon).size(18.0).color(if active { Color32::WHITE } else { TEXT_DIM }))
                    .fill(if active { SEL_BG } else { Color32::TRANSPARENT })
                    .min_size(Vec2::new(36.0, 36.0));
                if ui.add(btn).on_hover_text(tooltip).clicked() {
                    state.active_tool = tool;
                }
                ui.add_space(2.0);
            }
            ui.separator();
            // Extra tools
            for (icon, tip) in [("", "Box Select"), ("", "Circle Select")] {
                let btn = egui::Button::new(RichText::new(icon).size(16.0).color(TEXT_DIM))
                    .fill(Color32::TRANSPARENT)
                    .min_size(Vec2::new(36.0, 36.0));
                ui.add(btn).on_hover_text(tip);
                ui.add_space(2.0);
            }
        });
}

/// Right panels: top = Outliner, bottom = Properties
pub fn draw_right_panels(ctx: &Context, state: &mut EditorState) {
    egui::SidePanel::right("right_panels")
        .resizable(true)
        .default_width(240.0)
        .min_width(180.0)
        .frame(Frame::none().fill(PANEL_BG))
        .show(ctx, |ui| {
            //  Outliner 
            ui.add_space(2.0);
            ui.label(RichText::new("    Scene Collection").color(TEXT_DIM).small().strong());
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(220.0)
                .show(ui, |ui| {
                    for (i, obj) in state.objects.iter().enumerate() {
                        let is_selected = state.selected == Some(i);
                        let icon = match obj.physics_type {
                            PhysicsType::Rigid    => "",
                            PhysicsType::SoftBody => "",
                            PhysicsType::Fluid    => "",
                            PhysicsType::Static   => "",
                        };
                        let row = egui::Frame::none()
                            .fill(if is_selected { SEL_BG } else { Color32::TRANSPARENT })
                            .inner_margin(Margin { left: 4, right: 4, top: 2, bottom: 2 });
                        row.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let label = format!("{} {}", icon, obj.name);
                                let resp = ui.selectable_label(is_selected, RichText::new(label).color(if is_selected { Color32::WHITE } else { TEXT_PRIMARY }));
                                if resp.clicked() {
                                    // mutable borrow workaround  we'll update after
                                }
                            });
                        });
                        if ui.interact(ui.min_rect(), ui.id().with(i), egui::Sense::click()).is_pointer_button_down_on() {
                            state.selected = Some(i);
                        }
                    }
                });

            ui.separator();

            //  Properties Panel 
            ui.add_space(4.0);
            ui.label(RichText::new("    Properties").color(TEXT_DIM).small().strong());
            ui.separator();

            if let Some(idx) = state.selected {
                if let Some(obj) = state.objects.get_mut(idx) {
                    egui::CollapsingHeader::new(RichText::new("Transform").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(RichText::new("Location").color(TEXT_DIM).small());
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("X").color(Color32::from_rgb(255, 80, 80)).small());
                                ui.add(egui::DragValue::new(&mut obj.position.x).speed(0.1).suffix(" m"));
                            });
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Y").color(Color32::from_rgb(80, 200, 80)).small());
                                ui.add(egui::DragValue::new(&mut obj.position.y).speed(0.1).suffix(" m"));
                            });
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Z").color(Color32::from_rgb(80, 80, 255)).small());
                                ui.add(egui::DragValue::new(&mut obj.position.z).speed(0.1).suffix(" m"));
                            });

                            ui.add_space(4.0);
                            ui.label(RichText::new("Scale").color(TEXT_DIM).small());
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut obj.scale.x).speed(0.01).prefix("X "));
                                ui.add(egui::DragValue::new(&mut obj.scale.y).speed(0.01).prefix("Y "));
                                ui.add(egui::DragValue::new(&mut obj.scale.z).speed(0.01).prefix("Z "));
                            });
                        });

                    egui::CollapsingHeader::new(RichText::new("Physics").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            let label = obj.physics_type.label();
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Type:").color(TEXT_DIM).small());
                                ui.label(RichText::new(label).color(ACCENT).small().strong());
                            });
                            ui.checkbox(&mut obj.visible, "Visible in Viewport");
                        });
                }
            } else {
                ui.label(RichText::new("Nothing selected.").color(TEXT_DIM).italics());
            }
        });
}

/// Bottom status bar
pub fn draw_status_bar(ctx: &Context, state: &EditorState, fps: i32, body_count: usize, particle_count: usize) {
    egui::TopBottomPanel::bottom("status_bar")
        .frame(Frame::none().fill(HEADER_BG).inner_margin(Margin { left: 8, right: 8, top: 3, bottom: 3 }))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("FPS: {fps}")).color(
                    if fps >= 60 { Color32::LIGHT_GREEN } else if fps >= 30 { Color32::YELLOW } else { Color32::RED }
                ).small());
                ui.separator();
                ui.label(RichText::new(format!("Bodies: {body_count}")).color(TEXT_DIM).small());
                ui.separator();
                ui.label(RichText::new(format!("Particles: {particle_count}")).color(TEXT_DIM).small());
                ui.separator();
                ui.label(RichText::new(format!("Tool: {:?}", state.active_tool)).color(TEXT_DIM).small());
                ui.separator();
                ui.label(RichText::new("GJK  FEM  Chebyshev  PBD  SPH  Wavefront").color(ACCENT).small());
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mode = if state.simulation_playing { " SIMULATING" } else { " STOPPED" };
                    let col = if state.simulation_playing { Color32::LIGHT_GREEN } else { TEXT_DIM };
                    ui.label(RichText::new(mode).color(col).small().strong());
                });
            });
        });
}
