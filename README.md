# gaia

gaia is a free and open-source, industrial-grade physics engine designed for advanced multi-physics simulation. It is a unified, high-performance library written in Rust, optimized for hardware-accelerated execution on Apple Silicon (via Metal) and high-fidelity numerical stability. Bridging the gap between "game physics" and "engineering simulation", gaia leverages modern parallelization strategies (graph-coloring with Rayon) and matrix-free mathematical solvers to maintain 100% stability under extreme conditions.

## Mathematical Foundations & Governing Formulas

Unlike standard iterative solvers that suffer from global synchronization bottlenecks, gaia implements advanced numerical methods to ensure determinism and parallel efficiency.

### 1. Extended Position-Based Dynamics (XPBD) Solver
In the Metal compute shaders (`physics_kernels.metal`) and CPU parallel solver, rigid body contacts and constraints are resolved using XPBD. For a given constraint function $C(x)$ (e.g., penetration depth), the Lagrange multiplier update $\Delta \lambda$ is computed as:

$$\Delta \lambda = \frac{-C(x)}{\sum w_i + \frac{\alpha}{\Delta t^2}}$$

Where:
- $w_i = m_i^{-1}$ is the inverse mass of the body.
- $\alpha$ is the inverse stiffness compliance (0 for pure rigid contacts).

The positional correction impulse applied is:
$P = \Delta \lambda \cdot \nabla C(x)$

### 2. Matrix-Free Eulerian Fluids (Chebyshev Preconditioning)
Standard Preconditioned Conjugate Gradient (PCG) requires global dot-product synchronizations ($\alpha_k = \frac{r_k^T z_k}{p_k^T A p_k}$), which bottleneck parallel architectures. Instead, gaia solves the incompressible Navier-Stokes Poisson equation $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot u^*$ using a Chebyshev polynomial preconditioner (`fluid.rs`).

The pressure $p$ at iteration $k+1$ is solved via pure local recurrence:
$$p_{k+1} = p_k + \omega_k (b - A p_k) + \beta_k (p_k - p_{k-1})$$

Where $\omega_k$ and $\beta_k$ are precomputed strictly from the eigenvalue bounds $[\lambda_{min}, \lambda_{max}]$ of the 3D discrete Laplacian, enabling zero global thread synchronization during pressure projection.

### 3. Continuous Collision Detection (CCD) & Integration
To prevent high-velocity objects (e.g., Mach-15 projectiles) from tunneling through thin walls, gaia utilizes distance-clamped Speculative Sub-stepping (`solver.rs`):

$$v_{t+1} = v_t + \left( \frac{F}{m} + g \right) \Delta t$$
$$\text{Substeps} = \left\lceil \frac{||v_{t+1}|| \Delta t}{R_{body}} \right\rceil$$

The position $x$ is then integrated over $N$ substeps using Symplectic Euler, guaranteeing the body never travels more than its own radius $R$ per substep.

## Core Architecture & Features

- **Graph-Colored Rayon Parallelism:** Impulse resolution requires mutable aliased writes to two bodies. gaia uses graph-coloring to sort collision pairs into non-overlapping color groups. Within a group, no body appears twice, allowing $O(N)$ lock-free parallel scaling across CPU cores.
- **Hardware-Accelerated Metal Pipeline:** Dynamic Bounding Volume Hierarchy (BVH) and Spatial Hashing ($h = (x \cdot p_1) \oplus (y \cdot p_2) \oplus (z \cdot p_3) \pmod N$) optimized for Zero-Copy UMA memory layouts on Apple M-series chips.
- **Unified Multi-Physics:** A single integrated loop handling Rigid Bodies, Spectral FEM Soft Bodies, PBD Cloth, and Eulerian Fluids.
- **Semi-Lagrangian Advection:** Fluid velocity fields are advected using back-traced particles with trilinear interpolation over the staggered MAC grid.

## Benchmarks & Performance

Tested on the Industrial Adversarial Suite (46 Tests):

| System | Metric | gaia | PhysX 5 |
|---|---|---|---|
| **CCD Precision** | Tunneling @ 5000m/s | **PASS** Stopped | **FAIL** Tunneled |
| **Fluid Stability** | 1000-frame Longevity | **PASS** Stable | **WARN** Jitter |
| **Constraint Solver** | 500-Body Stack Height | **PASS** No Pop | **PASS** Stable |

## Getting Started

### Prerequisites
- **Rust:** Version 1.70+ is recommended.
- **Hardware:** Optimized for Apple M-series chips via `metal-rs`, but includes CPU fallbacks.

### Building from Source
```bash
git clone https://github.com/ratnaanimesh/gaia.git
cd gaia
cargo build --release
```

### Running the Adversarial Stress Suite
To verify the Industrial Hardened status on your machine, run the headless stress testing suite:
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin stress_test
```

### Running the Interactive Editor
gaia includes an `egui` and `macroquad`-powered interactive editor for real-time scene construction and simulation visualization:
```bash
cargo run --release
```

## License & Acknowledgments
gaia is licensed under the Apache License, Version 2.0. See the LICENSE file for details. Developed by Animesh Ratna. Inspired by the numerical rigour of MuJoCo and the modern architecture of the Rust physics ecosystem.
