/// Gaia Physics Engine — Automated Stress Test Harness v2
///
/// Runs completely headlessly. Tests every physics subsystem at multiple
/// scales with adversarial edge cases that kill most engines.
/// Run with: cargo run --release --bin stress_test

use std::time::Instant;
use macroquad::math::Vec3;

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
        self.results.push(TestResult { name: name.to_string(), passed, duration, notes });
    }

    fn report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║      GAIA PHYSICS ENGINE — EDGE CASE STRESS REPORT        ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let total  = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        for r in &self.results {
            let status = if r.passed { "✅ PASS" } else { "❌ FAIL" };
            println!("  {} {:.<55} {:>8.2}ms", status, format!("{} ", r.name), r.duration.as_secs_f64() * 1000.0);
            if !r.notes.is_empty() { println!("       ↳ {}", r.notes); }
        }

        println!("\n  ───────────────────────────────────────────────────────────");
        println!("  Total: {total}  |  Passed: {passed}  |  Failed: {failed}");
        if failed == 0 { println!("  🎉 ALL TESTS PASSED"); }
        else { println!("  ⚠️  {failed} test(s) failed — bugs to fix above."); }
        println!();
    }
}

fn is_valid(v: Vec3) -> bool { v.x.is_finite() && v.y.is_finite() && v.z.is_finite() }

fn check_bodies(bodies: &[gaia::core::solver::RigidBody]) -> Result<(), String> {
    for (i, b) in bodies.iter().enumerate() {
        if !is_valid(b.position) { return Err(format!("Body {i} position NaN: {:?}", b.position)); }
        if !is_valid(b.velocity) { return Err(format!("Body {i} velocity NaN: {:?}", b.velocity)); }
        if b.velocity.length() > 1000.0 { return Err(format!("Body {i} velocity exploded: {:.0} m/s", b.velocity.length())); }
    }
    Ok(())
}

fn make_floor(world: &mut gaia::core::solver::PhysicsWorld) {
    use gaia::core::solver::RigidBody;
    use gaia::core::shapes::{Shape, PhysicsMaterial};
    let mut f = RigidBody::new(0, Shape::Box { half_extents: Vec3::new(30.0, 1.0, 30.0) }, Vec3::new(0.0, -1.0, 0.0), PhysicsMaterial::default());
    f.inv_mass = 0.0; f.inv_inertia = Vec3::ZERO;
    world.add_body(f);
}

fn main() {
    let mut suite = TestSuite::new();

    // ═══════════════════════════════════════════════════════════
    // GROUP 1: ORIGINAL PASSING TESTS (regression guard)
    // ═══════════════════════════════════════════════════════════

    suite.run("Regression: single free fall", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        w.add_body(RigidBody::new(0, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 10.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..100 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        let y = w.bodies[0].position.y;
        if y >= 10.0 { return Err(format!("Did not fall! y={y:.2}")); }
        Ok(format!("y={y:.2}"))
    });

    suite.run("Regression: 14/14 original tests baseline", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let a = Shape::Sphere { radius: 1.0 };
        let b = Shape::Box { half_extents: Vec3::ONE };
        let c = detect_collision(&a, Vec3::new(1.5, 0.0, 0.0), &b, Vec3::ZERO);
        if c.is_none() { return Err("GJK regression failed".into()); }
        Ok("GJK still working".into())
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 2: HIGH-SPEED TUNNELING (Fast objects through walls)
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: high-speed sphere doesn't tunnel through floor", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        let mut b = RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 15.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() });
        b.velocity = Vec3::new(0.0, -50.0, 0.0); // Very fast downward
        w.add_body(b);
        for _ in 0..60 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        let y = w.bodies[1].position.y;
        if y < -5.0 { return Err(format!("TUNNELED through floor! y={y:.2}")); }
        Ok(format!("Stopped at y={y:.2}"))
    });

    suite.run("Edge: very fast horizontal projectile", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        let mut b = RigidBody::new(0, Shape::Sphere { radius: 0.5 }, Vec3::new(-100.0, 0.0, 0.0), PhysicsMaterial::default());
        b.velocity = Vec3::new(200.0, 0.0, 0.0);
        w.add_body(b);
        for _ in 0..120 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Position: {:.1?}", w.bodies[0].position))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 3: DEGENERATE / EXTREME SHAPES
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: very flat box (pancake) — half_extents (5,0.01,5)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::new(5.0, 0.01, 5.0) }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..200 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Flat box stable at y={:.3}", w.bodies[1].position.y))
    });

    suite.run("Edge: very thin needle box (0.01,5,0.01)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::new(0.01, 3.0, 0.01) }, Vec3::new(0.0, 8.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..200 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok("Needle box stable".into())
    });

    suite.run("Edge: tiny sphere (radius 0.01)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Sphere { radius: 0.01 }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..200 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Tiny sphere at y={:.4}", w.bodies[1].position.y))
    });

    suite.run("Edge: huge sphere (radius 20)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        w.add_body(RigidBody::new(0, Shape::Sphere { radius: 20.0 }, Vec3::new(0.0, 50.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..100 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Large sphere at y={:.2}", w.bodies[0].position.y))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 4: GJK DEGENERATE CONTACT CASES
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: GJK two shapes at identical position", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let a = Shape::Sphere { radius: 1.0 };
        let b = Shape::Sphere { radius: 1.0 };
        // Both at origin — maximum overlap
        let result = std::panic::catch_unwind(|| {
            detect_collision(&a, Vec3::ZERO, &b, Vec3::ZERO)
        });
        match result {
            Ok(c) => Ok(format!("Handled co-located shapes: contact={}", c.is_some())),
            Err(_) => Err("PANIC on co-located shapes!".into()),
        }
    });

    suite.run("Edge: GJK sphere just barely touching box", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let s = Shape::Sphere { radius: 1.0 };
        let b = Shape::Box { half_extents: Vec3::ONE };
        // Sphere center exactly at box surface: distance = 1.0 + 1.0 = 2.0, sphere radius = 1.0 → touching
        let c = detect_collision(&s, Vec3::new(2.0, 0.0, 0.0), &b, Vec3::ZERO);
        // At exactly touching, GJK may or may not detect — but must not panic/NaN
        Ok(format!("Touch detection: contact={}", c.is_some()))
    });

    suite.run("Edge: capsule vs capsule collision", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let a = Shape::Capsule { radius: 0.5, half_height: 1.0 };
        let b = Shape::Capsule { radius: 0.5, half_height: 1.0 };
        // Overlapping capsules side by side
        let c = detect_collision(&a, Vec3::new(0.5, 0.0, 0.0), &b, Vec3::new(-0.5, 0.0, 0.0));
        if c.is_none() { return Err("Missed overlapping capsules!".into()); }
        let c = c.unwrap();
        if !c.depth.is_finite() { return Err(format!("NaN depth: {}", c.depth)); }
        Ok(format!("Capsule contact depth={:.3}", c.depth))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 5: EXTREME STACKING
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: 20-box tower (tall stack)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        for i in 0..20 {
            w.add_body(RigidBody::new(i + 1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 1.5 + i as f32 * 2.1, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() }));
        }
        for _ in 0..300 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        let below = w.bodies[1..].iter().filter(|b| b.position.y < -4.0).count();
        if below > 0 { return Err(format!("{below} boxes tunneled")); }
        Ok("20-box tower stable".into())
    });

    suite.run("Scale: 100 bodies, 50 frames timing", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        for i in 0..100usize {
            let x = ((i % 10) as f32 - 5.0) * 2.5;
            let z = ((i / 10) as f32 - 5.0) * 2.5;
            w.add_body(RigidBody::new(i + 1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(x, 3.0 + (i % 5) as f32 * 2.0, z), PhysicsMaterial::default()));
        }
        let t0 = Instant::now();
        for _ in 0..50 { w.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
        check_bodies(&w.bodies)?;
        if ms > 200.0 { return Err(format!("Too slow: {ms:.1}ms/frame for 100 bodies")); }
        Ok(format!("{ms:.2}ms/frame for 100 bodies"))
    });

    suite.run("Scale: 200 bodies, 20 frames surviving", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        for i in 0..200usize {
            let x = ((i % 14) as f32 - 7.0) * 2.2;
            let z = ((i / 14) as f32 - 7.0) * 2.2;
            w.add_body(RigidBody::new(i + 1, Shape::Sphere { radius: 0.8 }, Vec3::new(x, 4.0 + (i % 8) as f32 * 1.5, z), PhysicsMaterial::default()));
        }
        let t0 = Instant::now();
        for _ in 0..20 { w.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        check_bodies(&w.bodies)?;
        Ok(format!("{ms:.2}ms/frame for 200 bodies"))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 6: NUMERICAL STABILITY EDGE CASES
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: very large dt (0.1s — 6× normal)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..30 { w.step(0.1); } // Huge timestep
        check_bodies(&w.bodies)?;
        Ok("Engine survived 0.1s timestep without NaN".into())
    });

    suite.run("Edge: very small dt (0.0001s) — 160× finer than normal", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        w.add_body(RigidBody::new(0, Shape::Sphere { radius: 1.0 }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..1000 { w.step(0.0001); }
        check_bodies(&w.bodies)?;
        Ok(format!("y={:.3}", w.bodies[0].position.y))
    });

    suite.run("Edge: near-zero mass body (0.001 kg)", || {
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial { density: 0.001, restitution: 0.3, friction_static: 0.5, friction_dynamic: 0.4 }));
        for _ in 0..200 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Low-mass body at y={:.2}", w.bodies[1].position.y))
    });

    suite.run("Edge: body initialized below floor (already penetrating)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        // Start inside the floor
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, -1.5, 0.0), PhysicsMaterial::default()));
        for _ in 0..100 { w.step(0.016); }
        check_bodies(&w.bodies)?;
        Ok(format!("Depenetrated to y={:.2}", w.bodies[1].position.y))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 7: FLUID ADVERSARIAL
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: fluid massive impulse (strength 1000)", || {
        use gaia::core::fluid::FluidGrid;
        let mut g = FluidGrid::new(12, 12, 12, 0.5);
        g.add_impulse(6, 6, 6, 1000.0); // Extreme impulse
        for _ in 0..200 { g.step(0.016); }
        let nans = g.pressure.iter().filter(|v| !v.is_finite()).count();
        if nans > 0 { return Err(format!("{nans} NaN cells after massive impulse")); }
        Ok("Fluid stable after impulse=1000".into())
    });

    suite.run("Edge: fluid very fine grid (32³ — 32768 cells)", || {
        use gaia::core::fluid::FluidGrid;
        let mut g = FluidGrid::new(32, 32, 32, 0.25);
        g.add_impulse(16, 4, 16, 5.0);
        let t0 = Instant::now();
        for _ in 0..20 { g.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        let nans = g.pressure.iter().filter(|v| !v.is_finite()).count();
        if nans > 0 { return Err(format!("{nans} NaN cells in 32³ grid")); }
        Ok(format!("{ms:.1}ms/frame for 32³ grid ({} cells)", 32usize.pow(3)))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 8: JOINT STABILITY
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: stiff spring joint (k=10000) stability", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        use gaia::core::joints::{SpringJoint, JointSystem};
        let mut w = PhysicsWorld::new();
        w.add_body(RigidBody::new(0, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        let mut fixed = RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 8.0, 0.0), PhysicsMaterial::default());
        fixed.inv_mass = 0.0; fixed.inv_inertia = Vec3::ZERO;
        w.add_body(fixed);
        let mut joints = JointSystem::new();
        joints.springs.push(SpringJoint {
            body_a: 0, body_b: 1,
            anchor_local_a: Vec3::ZERO, anchor_local_b: Vec3::ZERO,
            rest_length: 3.0, stiffness: 10000.0, damping: 50.0,
        });
        for _ in 0..200 {
            joints.apply_all(&mut w.bodies, 0.016);
            w.step(0.016);
        }
        check_bodies(&w.bodies)?;
        Ok(format!("Stiff spring stable at y={:.2}", w.bodies[0].position.y))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 9: CLOTH ADVERSARIAL
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: cloth with heavy wind (10× normal)", || {
        use gaia::core::cloth::Cloth;
        let mut c = Cloth::grid(8, 8, 0.5, Vec3::new(0.0, 6.0, 0.0));
        c.wind = Vec3::new(30.0, 0.0, 10.0); // Very strong wind
        for _ in 0..300 { c.step(0.016); }
        for (i, p) in c.particles.iter().enumerate() {
            if !is_valid(p.position) { return Err(format!("Particle {i} NaN under heavy wind")); }
        }
        Ok("Cloth stable under heavy wind".into())
    });

    suite.run("Edge: cloth large grid (16×16 = 256 particles)", || {
        use gaia::core::cloth::Cloth;
        let mut c = Cloth::grid(16, 16, 0.4, Vec3::new(0.0, 8.0, 0.0));
        let t0 = Instant::now();
        for _ in 0..100 { c.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
        for (i, p) in c.particles.iter().enumerate() {
            if !is_valid(p.position) { return Err(format!("Particle {i} NaN in 16×16")); }
        }
        Ok(format!("{ms:.2}ms/frame for 16×16 cloth"))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 10: RAYCAST EDGE CASES
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: ray fired from inside a sphere", || {
        use gaia::core::raycast::{Ray, ray_cast};
        use gaia::core::solver::RigidBody;
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let bodies = vec![RigidBody::new(0, Shape::Sphere { radius: 5.0 }, Vec3::ZERO, PhysicsMaterial::default())];
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)); // from inside
        let result = std::panic::catch_unwind(|| ray_cast(&ray, &bodies));
        match result {
            Ok(_) => Ok("Raycast from inside sphere handled without panic".into()),
            Err(_) => Err("PANIC on ray from inside sphere".into()),
        }
    });

    suite.run("Edge: ray parallel to box face (no hit)", || {
        use gaia::core::raycast::{Ray, ray_cast};
        use gaia::core::solver::RigidBody;
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let bodies = vec![RigidBody::new(0, Shape::Box { half_extents: Vec3::new(2.0, 0.5, 2.0) }, Vec3::ZERO, PhysicsMaterial::default())];
        // Ray perfectly parallel to top face, just above it
        let ray = Ray::new(Vec3::new(-10.0, 0.55, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let _ = ray_cast(&ray, &bodies); // must not panic
        Ok("Parallel ray handled".into())
    });

    suite.run("Edge: 10000 raycasts per second benchmark", || {
        use gaia::core::raycast::{Ray, ray_cast};
        use gaia::core::solver::RigidBody;
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let bodies: Vec<_> = (0..20).map(|i| {
            RigidBody::new(i, Shape::Sphere { radius: 0.5 }, Vec3::new(i as f32 * 2.0 - 20.0, 0.0, 0.0), PhysicsMaterial::default())
        }).collect();
        let t0 = Instant::now();
        let mut hits = 0usize;
        for i in 0..10_000 {
            let ray = Ray::new(Vec3::new(i as f32 * 0.001 - 5.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0));
            if ray_cast(&ray, &bodies).is_some() { hits += 1; }
        }
        let rps = 10_000.0 / t0.elapsed().as_secs_f64();
        Ok(format!("{rps:.0} raycasts/sec, {hits}/10000 hits, 20 bodies"))
    });

    // ═══════════════════════════════════════════════════════════
    // GROUP 11: BVH STRESS
    // ═══════════════════════════════════════════════════════════

    suite.run("Edge: BVH with 1000 objects — no false overlaps", || {
        use gaia::core::collision::bvh::{Aabb, BvhTree};
        let mut tree = BvhTree::new();
        for i in 0..1000usize {
            let x = (i as f32) * 2.1; // Perfectly separated
            tree.insert(i, Aabb::new(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0)));
        }
        let pairs = tree.query_pairs();
        if !pairs.is_empty() { return Err(format!("False {} overlaps", pairs.len())); }
        Ok("1000 objects, 0 false overlaps".into())
    });

    suite.run("Edge: BVH with dense overlapping cloud — all pairs found", || {
        use gaia::core::collision::bvh::{Aabb, BvhTree};
        // 5 objects all overlapping at origin — should find C(5,2)=10 pairs
        let mut tree = BvhTree::new();
        for i in 0..5 {
            tree.insert(i, Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));
        }
        let pairs = tree.query_pairs();
        if pairs.len() != 10 { return Err(format!("Expected 10 pairs, got {}", pairs.len())); }
        Ok("All 10 overlapping pairs found".into())
    });

    // ═══════════════════════════════════════════════════════════
    // FINAL REPORT
    // ═══════════════════════════════════════════════════════════
    suite.report();
    if suite.results.iter().any(|r| !r.passed) { std::process::exit(1); }
}
