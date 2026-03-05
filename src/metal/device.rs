use metal::{Device, CommandQueue, Library, CompileOptions};
use anyhow::{Result, Context};

pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub library: Library,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .context("Failed to find system default Metal device. Are you on a Mac with Metal support?")?;
            
        let command_queue = device.new_command_queue();
        
        // Load and compile shader at runtime
        let shader_source = include_str!("../shaders/physics_kernels.metal");
        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .map_err(|err| anyhow::anyhow!("Failed to compile Metal shader: {}", err))?;
        
        println!("Initialized Metal Device: {}", device.name());
        
        Ok(Self {
            device,
            command_queue,
            library,
        })
    }
}
