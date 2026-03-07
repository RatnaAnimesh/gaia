# <p align="center"> gaia Physics Engine </p>

<p align="center">
  <a href="https://github.com/ratnaanimesh/gaia/actions"><img src="https://github.com/ratnaanimesh/gaia/workflows/build/badge.svg" alt="Build Status"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

gaia is a free and open-source, industrial-grade physics engine designed for advanced multi-physics simulation. It is a unified, high-performance library written in Rust, optimized for hardware-accelerated execution on Apple Silicon (via Metal) and high-fidelity numerical stability.

### Key Features

gaia provides a unique combination of speed and precision, ranking in the **Elite Tier** of modern physics engines:

*   **Continuous Collision Detection (CCD):** Global sub-stepping pipeline capable of capturing hyper-sonic interactions (e.g., a Mach-15 projectile hitting a thin wall without tunneling).
*   **Unified Multi-Physics Solver:** A single integrated loop handling Rigid Bodies, Spectral FEM Soft Bodies, PBD Cloth, and Eulerian Fluids.
*   **DEQ-Stabilized Solver:** Utilizes Deep Equilibrium concepts for fixed-point root-finding in LCP constraints, ensuring jitter-free stacks even at high mass ratios.
*   **Spectral Matrix-Free Fluids:** Chebyshev-preconditioned pressure projection for incompressible flow without global synchronization bottlenecks.
*   **Spectral FEM:** Matrix-free hyperelastic soft bodies (Neo-Hookean) that bypass $O(N^2)$ Jacobian assembly, enabling high-resolution volumetric deformation.
*   **Hardware-Accelerated Broadphase:** Dynamic Bounding Volume Hierarchy (BVH) and Spatial Hashing optimized for Zero-Copy UMA memory layouts.

---

## Benchmarks & Performance

gaia is built to bridge the gap between "game physics" and "engineering simulation." In the **Industrial Adversarial Suite (46 Tests)**, gaia maintains 100% stability under extreme conditions:

| System | Metric | gaia | PhysX 5 |
|---|---|---|---|
| **CCD Precision** | Tunneling @ 5000m/s | **PASS Stopped** | FAIL Tunneled |
| **Fluid Stability** | 1000-frame Longevity | **PASS Stable** | WARN Jitter |
| **Constraint Solver** | 500-Body Stack Height | **PASS No Pop** | PASS Stable |

For a comprehensive comparison, see the [Universal Ranking Report](docs/ranking_report.md).

---

## Getting Started

### Prerequisites

-   **Rust:** 1.70+ recommended.
-   **Hardware:** Optimized for Apple M-series chips (Metal), but includes a cross-platform fallback.

### Building from Source

```bash
git clone https://github.com/ratnaanimesh/gaia.git
cd gaia
cargo build --release
```

### Running the Adversarial Stress Suite

To verify the "Industrial Hardened" status on your machine, run the 46-test headless suite:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin stress_test
```

### Running the Interactive Editor

gaia includes a Blender-inspired interactive editor for real-time scene construction:

```bash
cargo run --release
```

---

## Documentation

Comprehensive technical documentation is available in the [`docs/`](docs/) directory:

-   [Architecture Blueprint](docs/gaia_architecture_blueprint.md): Design patterns and ECS layout.
-   [Mathematical Bottlenecks](docs/omniphysics_mathematical_bottlenecks.md): Detailed analysis of the MLCP and Poisson solve breakthroughs.
-   [Ranking Report](docs/ranking_report.md): Industrial benchmarking against PhysX, Jolt, and Havok.

---

## License

gaia is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

Developed by **Animesh Ratna** and the **SAGE/Antigravity** team. Inspired by the numerical rigour of MuJoCo and the modern architecture of the Rust physics ecosystem.
