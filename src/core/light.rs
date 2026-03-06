/// Hamiltonian Wavefront Light Transport
///
/// Instead of tracing individual rays (which causes GPU SIMD divergence when secondary
/// rays scatter in random directions), we propagate a WAVEFRONT of light.
///
/// Each cell of the wavefront grid stores a lumiance value L(x, omega) at discrete
/// directions. The wavefront evolves according to Hamiltonian ray mechanics:
///
///   dx/dt = grad_p H
///   dp/dt = -grad_x H
///
/// Where H(x, p) = c(x) * |p| is the optical Hamiltonian.
/// In uniform media, this simplifies to straight rays, but the formulation allows
/// heterogeneous media (glass, water, fog) at zero extra algorithmic cost.
///
/// The key SIMD advantage: all wavefront cells update with the SAME instruction sequence
/// (evaluate H, propagate L), eliminating warp divergence entirely.

/// A single photon wavefront sample
pub struct WaveFront {
    /// Position in world space
    pub x: [f32; 3],
    /// Momentum direction (unit vector * refractive index)
    pub p: [f32; 3],
    /// Luminance carried by this wavelet
    pub luminance: [f32; 3], // RGB
}

/// A grid of accumulated photon energy (the "photon map" for SPPM)
pub struct PhotonMap {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<[f32; 3]>, // RGB luminance accumulation
    pub sample_count: Vec<u32>,
}

impl PhotonMap {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self { width, height, cells: vec![[0.0; 3]; n], sample_count: vec![0; n] }
    }

    /// Project a world position to image plane and accumulate photon contribution
    pub fn accumulate(&mut self, world_x: f32, world_y: f32, luminance: [f32; 3]) {
        // Simple orthographic projection
        let px = ((world_x / 20.0 + 0.5) * self.width as f32) as usize;
        let py = ((world_y / 20.0 + 0.5) * self.height as f32) as usize;
        
        if px < self.width && py < self.height {
            let i = px + py * self.width;
            self.cells[i][0] += luminance[0];
            self.cells[i][1] += luminance[1];
            self.cells[i][2] += luminance[2];
            self.sample_count[i] += 1;
        }
    }
    
    /// Returns the normalized luminance at a pixel
    pub fn get_normalized(&self, px: usize, py: usize) -> [f32; 3] {
        let i = px + py * self.width;
        let n = self.sample_count[i].max(1) as f32;
        [self.cells[i][0] / n, self.cells[i][1] / n, self.cells[i][2] / n]
    }
}

/// Hamiltonian wavefront propagator
pub struct HamiltonianPropagator {
    pub wavefronts: Vec<WaveFront>,
    pub photon_map: PhotonMap,
    pub step_size: f32,
}

impl HamiltonianPropagator {
    pub fn new(map_width: usize, map_height: usize) -> Self {
        Self {
            wavefronts: Vec::new(),
            photon_map: PhotonMap::new(map_width, map_height),
            step_size: 0.1,
        }
    }

    /// Emit wavefronts from a point light source.
    /// All wavefronts share the same update logic => zero SIMD divergence.
    pub fn emit_from_point(&mut self, pos: [f32; 3], num_samples: usize, color: [f32; 3]) {
        self.wavefronts.clear();
        for i in 0..num_samples {
            // Stratified hemisphere sampling
            let phi = 2.0 * std::f32::consts::PI * (i as f32) / (num_samples as f32);
            let theta = std::f32::consts::FRAC_PI_4; // 45 degree cone
            
            let px = theta.sin() * phi.cos();
            let py = -theta.cos().abs(); // Downward (toward scene)
            let pz = theta.sin() * phi.sin();
            
            self.wavefronts.push(WaveFront {
                x: pos,
                p: [px, py, pz],
                luminance: color,
            });
        }
    }

    /// Propagate all wavefronts by one step using the Hamiltonian equations.
    /// dx/dt = p/|p| (normalized wavefront direction)
    /// dp/dt = -grad(c(x)) where c(x) is the refractive index field (uniform = no bending)
    pub fn step_uniform_medium(&mut self) {
        for wf in &mut self.wavefronts {
            let len = (wf.p[0] * wf.p[0] + wf.p[1] * wf.p[1] + wf.p[2] * wf.p[2]).sqrt().max(1e-6);
            wf.x[0] += (wf.p[0] / len) * self.step_size;
            wf.x[1] += (wf.p[1] / len) * self.step_size;
            wf.x[2] += (wf.p[2] / len) * self.step_size;
            
            // Floor absorption: when wavefront hits y <= 0, it stops and deposits energy
            if wf.x[1] <= 0.0 {
                self.photon_map.accumulate(wf.x[0], wf.x[2], wf.luminance);
                wf.luminance = [0.0; 3]; // Absorbed
            }
        }
        // Remove absorbed wavefronts
        self.wavefronts.retain(|wf| wf.luminance[0] > 0.001 || wf.luminance[1] > 0.001 || wf.luminance[2] > 0.001);
    }

    /// Step multiple times to propagate across the scene
    pub fn propagate(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_uniform_medium();
        }
    }
}
