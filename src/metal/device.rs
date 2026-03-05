use metal::{ComputePipelineState, Device, Library, CompileOptions, CommandQueue};
use anyhow::{Result, Context};
use std::mem;

// Matches the scalar struct generated in Metal
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CollisionPair {
    pub entity_a: u32,
    pub entity_b: u32,
}

pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub library: Library,
    
    // Compute Pipelines
    pub clear_hash_pipeline: ComputePipelineState,
    pub build_hash_pipeline: ComputePipelineState,
    pub find_pairs_pipeline: ComputePipelineState,
    pub narrowphase_pipeline: ComputePipelineState,
    pub xpbd_solve_pipeline: ComputePipelineState,
    pub xpbd_velocity_pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .context("Failed to find system default Metal device.")?;
            
        let command_queue = device.new_command_queue();
        
        let shader_source = include_str!("../shaders/physics_kernels.metal");
        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .map_err(|err| anyhow::anyhow!("Failed to compile Metal shader: {}", err))?;
            
        let clear_hash_pipeline = Self::build_pipeline(&device, &library, "clear_spatial_hash")?;
        let build_hash_pipeline = Self::build_pipeline(&device, &library, "build_spatial_hash")?;
        let find_pairs_pipeline = Self::build_pipeline(&device, &library, "find_broadphase_pairs")?;
        let narrowphase_pipeline = Self::build_pipeline(&device, &library, "narrowphase_gjk_epa")?;
        
        let xpbd_solve_pipeline = Self::build_pipeline(&device, &library, "xpbd_solve_contacts")?;
        let xpbd_velocity_pipeline = Self::build_pipeline(&device, &library, "xpbd_velocity_update")?;
        
        println!("Initialized Metal Device: {}", device.name());
        
        Ok(Self {
            device,
            command_queue,
            library,
            clear_hash_pipeline,
            build_hash_pipeline,
            find_pairs_pipeline,
            narrowphase_pipeline,
            xpbd_solve_pipeline,
            xpbd_velocity_pipeline,
        })
    }
    
    fn build_pipeline(device: &Device, library: &Library, name: &str) -> std::result::Result<ComputePipelineState, anyhow::Error> {
        let function = library.get_function(name, None)
            .map_err(|e| anyhow::anyhow!("Failed to find function {}: {}", name, e))?;
        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline for {}: {}", name, e))
    }
}
