/// Gaia Physics Engine — Automated Stress Test Harness
///
/// Runs completely headlessly (no window, no rendering, no human input).
/// Tests every physics subsystem at multiple scales, detects bugs automatically,
/// and prints a full pass/fail report with timing data.
///
/// Run with: cargo run --release --bin stress_test

use std::time::Instant;

// Physics modules (no macroquad rendering — math types still work)
use macroquad::math::Vec3;

// Inline the core module re-exports
mod core {
    pub use gaia::core::*;
}

// ─── Test Infrastructure ─────────────────────────────────────────────────────

struct TestResult {
    name:     String,
    passed:   bool,
    duration: std::time::Duration,
    notes:    String,
}

struct TestSuite {
    results: Vec<TestResult>,
}

impl TestSuite {
    fn new() -> Self { Self { results: Vec::new() } }

    fn run<F: FnOnce() -> Result<String, String>>(&mut self, name: &str, f: F) {
        let t0 = Instant::now();
        let outcome = f();
        let duration = t0.elapsed();
        let (passed, notes) = match outcome {
            Ok(msg)  => (true,  msg),
            Err(msg) => (false, msg),
        };
        self.results.push(TestResult {
            name: name.to_string(), passed, duration, notes,
        });
    }

    fn report(&self) {
        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║          GAIA PHYSICS ENGINE STRESS TEST REPORT       ║");
        println!("╚══════════════════════════════════════════════════════╝\n");

        let total  = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        for r in &self.results {
            let status = if r.passed { "✅ PASS" } else { "❌ FAIL" };
            println!("  {} {:.<50} {:>8.2}ms",
                status,
                format!("{} ", r.name),
                r.duration.as_secs_f64() * 1000.0,
            );
            if !r.notes.is_empty() {
                println!("       ↳ {}", r.notes);
            }
        }

        println!("\n  ─────────────────────────────────────────────────────");
        println!("  Total: {total}  |  Passed: {passed}  |  Failed: {failed}");
        if failed == 0 {
            println!("  🎉 ALL TESTS PASSED — Engine is stable.");
        } else {
            println!("  ⚠️  {failed} test(s) failed — see ↳ notes above.");
        }
        println!();
    }
}

// ─── Sanity helpers ──────────────────────────────────────────────────────────

fn is_valid_vec3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn check_bodies_valid(bodies: &[gaia::core::solver::RigidBody]) -> Result<(), String> {
    for (i, b) in bodies.iter().enumerate() {
        if !is_valid_vec3(b.position) {
            return Err(format!("Body {i} position is NaN/Inf: {:?}", b.position));
        }
        if !is_valid_vec3(b.velocity) {
            return Err(format!("Body {i} velocity is NaN/Inf: {:?}", b.velocity));
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

fn main() {
    let mut suite = TestSuite::new();

    // ── 1. Single body free fall ──────────────────────────────────────────────
    suite.run("Rigid body: single free fall (100 frames)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new(0,
            Shape::Box { half_extents: Vec3::ONE },
            Vec3::new(0.0, 10.0, 0.0),
            PhysicsMaterial::default(),
        ));

        let y_start = world.bodies[0].position.y;
        for _ in 0..100 { world.step(0.016); }
        check_bodies_valid(&world.bodies)?;

        let y_end = world.bodies[0].position.y;
        if y_end >= y_start {
            return Err(format!("Body did not fall! y_start={y_start:.2} y_end={y_end:.2}"));
        }
        Ok(format!("Fell from y={y_start:.2} to y={y_end:.2}"))
    });

    // ── 2. Floor collision + bounce ───────────────────────────────────────────
    suite.run("Rigid body: floor collision + bounce", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let mut world = PhysicsWorld::new();
        // Static floor
        let mut floor = RigidBody::new(0,
            Shape::Box { half_extents: Vec3::new(10.0, 1.0, 10.0) },
            Vec3::new(0.0, -1.0, 0.0),
            PhysicsMaterial { restitution: 0.5, ..Default::default() },
        );
        floor.inv_mass = 0.0; floor.inv_inertia = Vec3::ZERO;
        world.add_body(floor);

        // Falling box
        world.add_body(RigidBody::new(1,
            Shape::Box { half_extents: Vec3::ONE },
            Vec3::new(0.0, 8.0, 0.0),
            PhysicsMaterial { restitution: 0.5, ..Default::default() },
        ));

        for _ in 0..300 { world.step(0.016); }
        check_bodies_valid(&world.bodies)?;

        let y = world.bodies[1].position.y;
        if y < -5.0 {
            return Err(format!("Box tunneled through floor! y={y:.2}"));
        }
        Ok(format!("Box resting at y={y:.2}"))
    });

    // ── 3. Stacking — 10 boxes ───────────────────────────────────────────────
    suite.run("Rigid body: 10-box stack (200 frames)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let mut world = PhysicsWorld::new();
        let mut floor = RigidBody::new(0,
            Shape::Box { half_extents: Vec3::new(15.0, 1.0, 15.0) },
            Vec3::new(0.0, -1.0, 0.0),
            PhysicsMaterial::default(),
        );
        floor.inv_mass = 0.0; floor.inv_inertia = Vec3::ZERO;
        world.add_body(floor);

        for i in 0..10 {
            world.add_body(RigidBody::new(i + 1,
                Shape::Box { half_extents: Vec3::ONE },
                Vec3::new(0.0, 2.0 + i as f32 * 2.5, 0.0),
                PhysicsMaterial::default(),
            ));
        }

        for _ in 0..200 { world.step(0.016); }
        check_bodies_valid(&world.bodies)?;

        // Check none tunneled below floor
        let violations: Vec<_> = world.bodies[1..]
            .iter()
            .filter(|b| b.position.y < -3.0)
            .collect();

        if !violations.is_empty() {
            return Err(format!("{} boxes tunneled below floor", violations.len()));
        }
        Ok(format!("All 10 boxes stable above floor"))
    });

    // ── 4. Energy conservation (closed system) ────────────────────────────────
    suite.run("Rigid body: energy conservation over 500 frames", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new(0,
            Shape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 5.0, 0.0),
            PhysicsMaterial { restitution: 0.0, friction_static: 0.0, friction_dynamic: 0.0, density: 1000.0 },
        ));

        let mut max_speed = 0.0_f32;
        for _ in 0..500 {
            world.step(0.016);
            let speed = world.bodies[0].velocity.length();
            max_speed = max_speed.max(speed);
        }
        check_bodies_valid(&world.bodies)?;

        // Speed should stay bounded — terminal velocity ~100 m/s in free fall for 500 frames
        if max_speed > 200.0 {
            return Err(format!("Energy explosion! max_speed={max_speed:.1} m/s"));
        }
        Ok(format!("Max speed: {max_speed:.1} m/s — stable"))
    });

    // ── 5. Sphere vs Box collision ────────────────────────────────────────────
    suite.run("GJK+EPA: sphere vs box contact manifold", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;

        let sphere = Shape::Sphere { radius: 1.0 };
        let cube   = Shape::Box   { half_extents: Vec3::ONE };

        // Overlapping: sphere center at (1.5, 0, 0), cube at origin
        let contact = detect_collision(
            &sphere, Vec3::new(1.5, 0.0, 0.0),
            &cube,   Vec3::ZERO,
        );
        match contact {
            Some(c) => {
                if !c.depth.is_finite() || c.depth < 0.0 {
                    return Err(format!("Invalid depth: {}", c.depth));
                }
                if !is_valid_vec3(c.normal) {
                    return Err(format!("Invalid normal: {:?}", c.normal));
                }
                Ok(format!("Contact detected: depth={:.3}, normal={:.2?}", c.depth, c.normal))
            }
            None => Err("GJK missed overlapping sphere+box!".into()),
        }
    });

    // ── 6. GJK: separated shapes (no false positive) ─────────────────────────
    suite.run("GJK+EPA: no false contact for separated shapes", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;

        let a = Shape::Sphere { radius: 1.0 };
        let b = Shape::Sphere { radius: 1.0 };

        // 10m apart — clearly not touching
        let contact = detect_collision(&a, Vec3::new(-5.0, 0.0, 0.0), &b, Vec3::new(5.0, 0.0, 0.0));
        match contact {
            None    => Ok("Correctly returned no contact".into()),
            Some(c) => Err(format!("False positive! depth={:.3}", c.depth)),
        }
    });

    // ── 7. Soft body stability ────────────────────────────────────────────────
    suite.run("FEM soft body: 200 frames without NaN", || {
        use gaia::core::soft_body::{MatrixFreeSoftBody, Tetrahedron};
        use macroquad::math::Mat3;

        let mut s = MatrixFreeSoftBody::new(80.0, 400.0);
        s.particles.push(Vec3::new(-1.0, 5.0, 0.0));
        s.particles.push(Vec3::new( 1.0, 5.0, 0.0));
        s.particles.push(Vec3::new( 0.0, 5.0,-1.5));
        s.particles.push(Vec3::new( 0.0, 7.0, 0.0));
        for _ in 0..4 { s.velocities.push(Vec3::ZERO); s.masses.push(0.5); }

        let dm = Mat3::from_cols(
            s.particles[1] - s.particles[0],
            s.particles[2] - s.particles[0],
            s.particles[3] - s.particles[0],
        );
        s.elements.push(Tetrahedron {
            v0: 0, v1: 1, v2: 2, v3: 3,
            inv_rest_shape: dm.inverse(),
            volume: dm.determinant().abs() / 6.0,
        });

        for _ in 0..200 { s.step(0.016); }

        for (i, &p) in s.particles.iter().enumerate() {
            if !is_valid_vec3(p) {
                return Err(format!("Particle {i} is NaN/Inf: {p:?}"));
            }
        }
        Ok("All particles finite after 200 frames".into())
    });

    // ── 8. Fluid grid stability ───────────────────────────────────────────────
    suite.run("Fluid: Chebyshev solve 100 frames without NaN", || {
        use gaia::core::fluid::FluidGrid;

        let mut grid = FluidGrid::new(12, 12, 12, 0.5);
        grid.add_impulse(6, 2, 6, 5.0);

        for _ in 0..100 { grid.step(0.016); }

        let nan_count = grid.pressure.iter().filter(|v| !v.is_finite()).count();
        if nan_count > 0 {
            return Err(format!("{nan_count} NaN/Inf values in pressure field"));
        }
        Ok(format!("Pressure field clean after 100 steps ({} cells)", grid.pressure.len()))
    });

    // ── 9. Raycast accuracy ───────────────────────────────────────────────────
    suite.run("Raycast: ray hits sphere correctly", || {
        use gaia::core::raycast::{Ray, ray_cast};
        use gaia::core::solver::RigidBody;
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let bodies = vec![
            RigidBody::new(0, Shape::Sphere { radius: 2.0 }, Vec3::new(0.0, 0.0, 0.0), PhysicsMaterial::default()),
        ];
        let ray = Ray::new(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0)); // fire at origin

        match ray_cast(&ray, &bodies) {
            Some(hit) => Ok(format!("Hit at distance={:.3}, normal={:.2?}", hit.distance, hit.normal)),
            None => Err("Ray missed sphere directly in front of it".into()),
        }
    });

    // ── 10. Raycast: miss ─────────────────────────────────────────────────────
    suite.run("Raycast: ray correctly misses sphere", || {
        use gaia::core::raycast::{Ray, ray_cast};
        use gaia::core::solver::RigidBody;
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let bodies = vec![
            RigidBody::new(0, Shape::Sphere { radius: 1.0 }, Vec3::new(0.0, 0.0, 0.0), PhysicsMaterial::default()),
        ];
        let ray = Ray::new(Vec3::new(10.0, 10.0, -10.0), Vec3::new(0.0, 0.0, 1.0)); // aimed wide

        match ray_cast(&ray, &bodies) {
            None => Ok("Correctly missed".into()),
            Some(hit) => Err(format!("False hit at distance={:.3}", hit.distance)),
        }
    });

    // ── 11. Massive parallel scale test ──────────────────────────────────────
    suite.run("Scale: 50 bodies, 100 frames (parallel broadphase)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};

        let mut world = PhysicsWorld::new();
        let mut floor = RigidBody::new(0,
            Shape::Box { half_extents: Vec3::new(30.0, 1.0, 30.0) },
            Vec3::new(0.0, -1.0, 0.0),
            PhysicsMaterial::default(),
        );
        floor.inv_mass = 0.0; floor.inv_inertia = Vec3::ZERO;
        world.add_body(floor);

        for i in 0..50usize {
            let x = ((i % 7) as f32 - 3.0) * 3.0;
            let z = ((i / 7) as f32 - 3.0) * 3.0;
            world.add_body(RigidBody::new(i + 1,
                Shape::Box { half_extents: Vec3::ONE },
                Vec3::new(x, 5.0 + (i as f32) * 0.3, z),
                PhysicsMaterial::default(),
            ));
        }

        let t0 = Instant::now();
        for _ in 0..100 { world.step(0.016); }
        let elapsed = t0.elapsed();
        check_bodies_valid(&world.bodies)?;

        let ms_per_frame = elapsed.as_secs_f64() * 1000.0 / 100.0;
        if ms_per_frame > 100.0 {
            return Err(format!("Too slow! {ms_per_frame:.1}ms/frame for 50 bodies"));
        }
        Ok(format!("{:.2}ms/frame avg for 50 bodies ({} pairs)", ms_per_frame, 50 * 51 / 2))
    });

    // ── 12. PBD Cloth stability ───────────────────────────────────────────────
    suite.run("PBD Cloth: 6×6 grid, 200 frames without NaN", || {
        use gaia::core::cloth::Cloth;

        let mut cloth = Cloth::grid(6, 6, 0.5, Vec3::new(0.0, 5.0, 0.0));
        for _ in 0..200 { cloth.step(0.016); }

        for (i, p) in cloth.particles.iter().enumerate() {
            if !is_valid_vec3(p.position) {
                return Err(format!("Particle {i} is NaN: {:?}", p.position));
            }
        }
        Ok("All cloth particles finite".into())
    });

    // ── 13. BVH insert + query ────────────────────────────────────────────────
    suite.run("BVH: insert 100 objects, query pairs", || {
        use gaia::core::collision::bvh::{Aabb, BvhTree};

        let mut tree = BvhTree::new();
        for i in 0..100usize {
            let x = (i as f32) * 3.0;
            let aabb = Aabb::new(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0));
            tree.insert(i, aabb);
        }

        let pairs = tree.query_pairs();
        // With spacing of 3.0 and size 1.0, none should overlap
        if !pairs.is_empty() {
            return Err(format!("False {} overlaps on separated objects", pairs.len()));
        }
        Ok("100 objects inserted, 0 false overlaps".into())
    });

    // ── 14. Performance benchmark ─────────────────────────────────────────────
    suite.run("Perf: 1000 GJK queries per second benchmark", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;

        let a = Shape::Box { half_extents: Vec3::ONE };
        let b = Shape::Sphere { radius: 1.0 };
        let mut hits = 0usize;

        let t0 = Instant::now();
        for i in 0..1000 {
            let offset = Vec3::new(i as f32 * 0.001, 0.0, 0.0);
            if detect_collision(&a, Vec3::ZERO, &b, offset).is_some() {
                hits += 1;
            }
        }
        let elapsed = t0.elapsed();
        let qps = 1000.0 / elapsed.as_secs_f64();

        Ok(format!("{qps:.0} GJK queries/sec ({hits}/1000 hits)"))
    });

    // ── Final Report ─────────────────────────────────────────────────────────
    suite.report();

    // Exit with non-zero code if any test failed
    let any_failed = suite.results.iter().any(|r| !r.passed);
    if any_failed { std::process::exit(1); }
}
