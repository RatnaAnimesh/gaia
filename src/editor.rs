use macroquad::prelude::*;

/// What tool is currently active in the toolbar
#[derive(Debug, Clone, PartialEq)]
pub enum ActiveTool {
    Select,
    Move,
    Rotate,
    Scale,
}

/// The physics type of a scene object
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsType {
    Rigid,
    SoftBody,
    Fluid,
    Static,
}

impl PhysicsType {
    pub fn label(&self) -> &str {
        match self {
            PhysicsType::Rigid => "Rigid Body (DEQ)",
            PhysicsType::SoftBody => "Soft Body (FEM)",
            PhysicsType::Fluid => "Fluid (Chebyshev)",
            PhysicsType::Static => "Static",
        }
    }
}

/// A single object in the scene, equivalent to Blender's object data
#[derive(Debug, Clone)]
pub struct SceneObject {
    pub name: String,
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Vec3, // Euler angles (degrees)
    pub color: Color,
    pub physics_type: PhysicsType,
    pub visible: bool,
}

impl SceneObject {
    pub fn new(name: &str, position: Vec3, color: Color, physics_type: PhysicsType) -> Self {
        Self {
            name: name.to_string(),
            position,
            scale: Vec3::ONE,
            rotation: Vec3::ZERO,
            color,
            physics_type,
            visible: true,
        }
    }
}

/// Orbit camera state
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub pivot: Vec3,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            yaw: 45.0f32.to_radians(),
            pitch: 25.0f32.to_radians(),
            radius: 25.0,
            pivot: Vec3::new(0.0, 2.0, 0.0),
        }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::new(
            self.pivot.x + self.radius * self.pitch.cos() * self.yaw.sin(),
            self.pivot.y + self.radius * self.pitch.sin(),
            self.pivot.z + self.radius * self.pitch.cos() * self.yaw.cos(),
        )
    }

    pub fn update(&mut self) {
        // Left mouse drag  orbit
        if is_mouse_button_down(MouseButton::Left) {
            let delta = mouse_delta_position();
            self.yaw   -= delta.x * 2.5;
            self.pitch  = (self.pitch + delta.y * 2.5).clamp(-1.5, 1.5);
        }
        // Middle mouse drag  pan
        if is_mouse_button_down(MouseButton::Middle) {
            let delta = mouse_delta_position();
            // Build right/up vectors from yaw
            let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
            let up    = Vec3::Y;
            self.pivot -= right * delta.x * self.radius * 0.5;
            self.pivot += up    * delta.y * self.radius * 0.5;
        }
        // Scroll  zoom
        let (_sx, sy) = mouse_wheel();
        self.radius = (self.radius - sy * 1.5).clamp(2.0, 200.0);
    }

    pub fn to_camera3d(&self) -> Camera3D {
        Camera3D {
            position: self.position(),
            target:   self.pivot,
            up:       Vec3::Y,
            ..Default::default()
        }
    }
}

/// The full editor state  equivalent to Blender's bpy.context
pub struct EditorState {
    pub objects: Vec<SceneObject>,
    pub selected: Option<usize>,
    pub active_tool: ActiveTool,
    pub camera: OrbitCamera,
    pub show_wireframe: bool,
    pub show_grid: bool,
    pub simulation_playing: bool,
    pub frame: u64,
    // Pending spawn requests (set by UI, consumed by main loop)
    pub pending_add_cube:   bool,
    pub pending_add_sphere: bool,
}

impl EditorState {
    pub fn new() -> Self {
        // Populate a default scene (like Blender's startup cube)
        let objects = vec![
            SceneObject::new("Falling Cube",    Vec3::new(0.0, 8.0, 0.0),   RED,       PhysicsType::Rigid),
            SceneObject::new("Floor",            Vec3::new(0.0, -1.0, 0.0),  DARKGREEN, PhysicsType::Static),
            SceneObject::new("Jello (FEM)",      Vec3::new(-5.0, 10.0, 0.0), BLUE,      PhysicsType::SoftBody),
            SceneObject::new("Point Light",      Vec3::new(0.0, 18.0, 0.0),  YELLOW,    PhysicsType::Static),
            SceneObject::new("Fluid Volume",     Vec3::new(5.0, 5.0, 0.0),   SKYBLUE,   PhysicsType::Fluid),
        ];

        Self {
            objects,
            selected: Some(0),
            active_tool: ActiveTool::Select,
            camera: OrbitCamera::new(),
            show_wireframe: true,
            show_grid: true,
            simulation_playing: false,
            frame: 0,
            pending_add_cube:   false,
            pending_add_sphere: false,
        }
    }
}
