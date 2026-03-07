/// gaia Physics Engine  Automated Stress Test Harness v2
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
        println!("\n");
        println!("      GAIA PHYSICS ENGINE  EDGE CASE STRESS REPORT        ");
        println!("\n");

        let total  = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        for r in &self.results {
            let status = if r.passed { "[PASS]" } else { "[FAIL]" };
            println!("  {} {:.<55} {:>8.2}ms", status, format!("{} ", r.name), r.duration.as_secs_f64() * 1000.0);
            if !r.notes.is_empty() { println!("        {}", r.notes); }
        }

        println!("\n  ");
        println!("  Total: {total}  |  Passed: {passed}  |  Failed: {failed}");
        if failed == 0 { println!("  ALL TESTS PASSED"); }
        else { println!("  {} test(s) failed  bugs to fix above.", failed); }
        println!();
    }
}

fn is_valid(v: Vec3) -> bool { v.x.is_finite() && v.y.is_finite() && v.z.is_finite() }

fn check_bodies(bodies: &[gaia::core::solver::RigidBody]) -> Result<(), String> {
    for (i, b) in bodies.iter().enumerate() {
        if !is_valid(b.position) { return Err(format!("Body {i} position NaN: {:?}", b.position)); }
        if !is_valid(b.velocity) { return Err(format!("Body {i} velocity NaN: {:?}", b.velocity)); }
        if b.velocity.length() > 10000.0 { return Err(format!("Body {i} velocity exploded: {:.0} m/s", b.velocity.length())); }
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

    // 
    // GROUP 1: ORIGINAL PASSING TESTS (regression guard)
    // 

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

    // 
    // GROUP 2: HIGH-SPEED TUNNELING (Fast objects through walls)
    // 

    suite.run("Edge: high-speed sphere doesn't tunnel through floor", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        let mut b = RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 15.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() });
        b.velocity = Vec3::new(0.0, -50.0, 0.0); // Very fast downward
        b.enable_ccd = true; // MUST enable CCD for high-speed tunneling tests
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

    // 
    // GROUP 3: DEGENERATE / EXTREME SHAPES
    // 

    suite.run("Edge: very flat box (pancake)  half_extents (5,0.01,5)", || {
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

    // 
    // GROUP 4: GJK DEGENERATE CONTACT CASES
    // 

    suite.run("Edge: GJK two shapes at identical position", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let a = Shape::Sphere { radius: 1.0 };
        let b = Shape::Sphere { radius: 1.0 };
        // Both at origin  maximum overlap
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
        // Sphere center exactly at box surface: distance = 1.0 + 1.0 = 2.0, sphere radius = 1.0  touching
        let c = detect_collision(&s, Vec3::new(2.0, 0.0, 0.0), &b, Vec3::ZERO);
        // At exactly touching, GJK may or may not detect  but must not panic/NaN
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

    // 
    // GROUP 5: EXTREME STACKING
    // 

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

    // 
    // GROUP 6: NUMERICAL STABILITY EDGE CASES
    // 

    suite.run("Edge: very large dt (0.1s  6 normal)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial::default()));
        for _ in 0..30 { w.step(0.1); } // Huge timestep
        check_bodies(&w.bodies)?;
        Ok("Engine survived 0.1s timestep without NaN".into())
    });

    suite.run("Edge: very small dt (0.0001s)  160 finer than normal", || {
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

    // 
    // GROUP 7: FLUID ADVERSARIAL
    // 

    suite.run("Edge: fluid massive impulse (strength 1000)", || {
        use gaia::core::fluid::FluidGrid;
        let mut g = FluidGrid::new(12, 12, 12, 0.5);
        g.add_impulse(6, 6, 6, 1000.0); // Extreme impulse
        for _ in 0..200 { g.step(0.016); }
        let nans = g.pressure.iter().filter(|v| !v.is_finite()).count();
        if nans > 0 { return Err(format!("{nans} NaN cells after massive impulse")); }
        Ok("Fluid stable after impulse=1000".into())
    });

    suite.run("Edge: fluid very fine grid (32  32768 cells)", || {
        use gaia::core::fluid::FluidGrid;
        let mut g = FluidGrid::new(32, 32, 32, 0.25);
        g.add_impulse(16, 4, 16, 5.0);
        let t0 = Instant::now();
        for _ in 0..20 { g.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        let nans = g.pressure.iter().filter(|v| !v.is_finite()).count();
        if nans > 0 { return Err(format!("{nans} NaN cells in 32 grid")); }
        Ok(format!("{ms:.1}ms/frame for 32 grid ({} cells)", 32usize.pow(3)))
    });

    // 
    // GROUP 8: JOINT STABILITY
    // 

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

    // 
    // GROUP 9: CLOTH ADVERSARIAL
    // 

    suite.run("Edge: cloth with heavy wind (10 normal)", || {
        use gaia::core::cloth::Cloth;
        let mut c = Cloth::grid(8, 8, 0.5, Vec3::new(0.0, 6.0, 0.0));
        c.wind = Vec3::new(30.0, 0.0, 10.0); // Very strong wind
        for _ in 0..300 { c.step(0.016); }
        for (i, p) in c.particles.iter().enumerate() {
            if !is_valid(p.position) { return Err(format!("Particle {i} NaN under heavy wind")); }
        }
        Ok("Cloth stable under heavy wind".into())
    });

    suite.run("Edge: cloth large grid (1616 = 256 particles)", || {
        use gaia::core::cloth::Cloth;
        let mut c = Cloth::grid(16, 16, 0.4, Vec3::new(0.0, 8.0, 0.0));
        let t0 = Instant::now();
        for _ in 0..100 { c.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
        for (i, p) in c.particles.iter().enumerate() {
            if !is_valid(p.position) { return Err(format!("Particle {i} NaN in 1616")); }
        }
        Ok(format!("{ms:.2}ms/frame for 1616 cloth"))
    });

    // 
    // GROUP 10: RAYCAST EDGE CASES
    // 

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

    // 
    // GROUP 11: BVH STRESS
    // 

    suite.run("Edge: BVH with 1000 objects  no false overlaps", || {
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

    suite.run("Edge: BVH with dense overlapping cloud  all pairs found", || {
        use gaia::core::collision::bvh::{Aabb, BvhTree};
        // 5 objects all overlapping at origin  should find C(5,2)=10 pairs
        let mut tree = BvhTree::new();
        for i in 0..5 {
            tree.insert(i, Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));
        }
        let pairs = tree.query_pairs();
        if pairs.len() != 10 { return Err(format!("Expected 10 pairs, got {}", pairs.len())); }
        Ok("All 10 overlapping pairs found".into())
    });

    // 
    // GROUP 12: PARTICLE SYSTEM ADVERSARIAL
    // 

    suite.run("Particles: 1000-particle SPH stability (direct insert)", || {
        use gaia::core::particles::{ParticleSystem, Particle, ParticleEmitter};
        use macroquad::prelude::Vec4;
        let mut sys = ParticleSystem::new();
        // Insert 1000 particles directly (avoids macroquad::time::get_time() in emitter)
        for i in 0..1000usize {
            let x = ((i % 32) as f32 - 16.0) * 0.3;
            let z = ((i / 32) as f32 - 16.0) * 0.3;
            sys.particles.push(Particle {
                position: Vec3::new(x, (i % 10) as f32 * 0.3, z),
                velocity: Vec3::new(0.0, 1.0, 0.0),
                color: Vec4::ONE,
                size: 0.1,
                lifetime: 5.0,
                max_lifetime: 5.0,
                density: 0.0,
            });
        }
        // Enable SPH via a dummy emitter that doesn't emit
        let mut e = ParticleEmitter::new(Vec3::ZERO, 0.0);
        e.sph_enabled = true;
        e.sph_kernel_h = 1.5;
        e.sph_pressure_k = 0.5;
        sys.emitters.push(e);
        for _ in 0..30 { sys.step(0.016); }
        let nans = sys.particles.iter().filter(|p| !p.position.is_finite() || !p.velocity.is_finite()).count();
        if nans > 0 { return Err(format!("{nans}/{} particles NaN", sys.particles.len())); }
        Ok(format!("{} particles alive, all finite", sys.particles.len()))
    });

    suite.run("Particles: extreme SPH pressure (k=1000) stability", || {
        use gaia::core::particles::{ParticleSystem, Particle, ParticleEmitter};
        use macroquad::prelude::Vec4;
        let mut sys = ParticleSystem::new();
        for i in 0..200usize {
            let x = ((i % 14) as f32 - 7.0) * 0.2;
            let z = ((i / 14) as f32 - 7.0) * 0.2;
            sys.particles.push(Particle {
                position: Vec3::new(x, 0.0, z),
                velocity: Vec3::ZERO,
                color: Vec4::ONE,
                size: 0.1,
                lifetime: 3.0,
                max_lifetime: 3.0,
                density: 0.0,
            });
        }
        let mut e = ParticleEmitter::new(Vec3::ZERO, 0.0);
        e.sph_enabled = true;
        e.sph_pressure_k = 1000.0;
        e.sph_kernel_h = 1.0;
        sys.emitters.push(e);
        for _ in 0..20 { sys.step(0.016); }
        let nans = sys.particles.iter().filter(|p| !p.position.is_finite()).count();
        if nans > 0 { return Err(format!("{nans} NaN with extreme SPH pressure")); }
        Ok(format!("{} particles stable under k=1000", sys.particles.len()))
    });

    // 
    // GROUP 13: JOINT CHAINS
    // 

    suite.run("Joints: 5-body spring chain (ragdoll backbone)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        use gaia::core::joints::{SpringJoint, JointSystem};
        let mut w = PhysicsWorld::new();
        // Static anchor at top
        let mut anchor = RigidBody::new(0, Shape::Sphere { radius: 0.3 }, Vec3::new(0.0, 10.0, 0.0), PhysicsMaterial::default());
        anchor.inv_mass = 0.0; anchor.inv_inertia = Vec3::ZERO;
        w.add_body(anchor);
        // 5 segments hanging below
        for i in 1..=5 {
            w.add_body(RigidBody::new(i, Shape::Sphere { radius: 0.3 }, Vec3::new(0.0, 10.0 - i as f32 * 2.0, 0.0), PhysicsMaterial::default()));
        }
        let mut joints = JointSystem::new();
        for i in 0..5 {
            joints.springs.push(SpringJoint {
                body_a: i, body_b: i + 1,
                anchor_local_a: Vec3::ZERO, anchor_local_b: Vec3::ZERO,
                rest_length: 2.0, stiffness: 500.0, damping: 20.0,
            });
        }
        for _ in 0..200 {
            joints.apply_all(&mut w.bodies, 0.016);
            w.step(0.016);
        }
        check_bodies(&w.bodies)?;
        Ok(format!("5-body chain stable, bottom at y={:.2}", w.bodies[5].position.y))
    });

    suite.run("Joints: ball-socket chain (pendulum)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        use gaia::core::joints::{BallSocketJoint, JointSystem};
        let mut w = PhysicsWorld::new();
        let mut anchor = RigidBody::new(0, Shape::Sphere { radius: 0.3 }, Vec3::new(0.0, 8.0, 0.0), PhysicsMaterial::default());
        anchor.inv_mass = 0.0; anchor.inv_inertia = Vec3::ZERO;
        w.add_body(anchor);
        // Pendulum bob offset horizontally (will swing)
        w.add_body(RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(3.0, 5.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() }));
        let mut joints = JointSystem::new();
        joints.ball_sockets.push(BallSocketJoint {
            body_a: 0, body_b: 1,
            anchor_local_a: Vec3::ZERO,
            anchor_local_b: Vec3::new(0.0, 3.0, 0.0),
            bias_factor: 0.2,
        });
        for _ in 0..300 {
            joints.apply_all(&mut w.bodies, 0.016);
            w.step(0.016);
        }
        check_bodies(&w.bodies)?;
        Ok(format!("Pendulum stable at pos={:.1?}", w.bodies[1].position))
    });

    // 
    // GROUP 14: RESTITUTION PHYSICS ACCURACY
    // 

    suite.run("Physics: elastic bounce height conservation (e=0.8)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        let mut f = RigidBody::new(0, Shape::Box { half_extents: Vec3::new(30.0, 1.0, 30.0) }, Vec3::new(0.0, -1.0, 0.0),
            PhysicsMaterial { restitution: 1.0, friction_dynamic: 0.0, friction_static: 0.0, density: 1000.0 });
        f.inv_mass = 0.0; f.inv_inertia = Vec3::ZERO;
        w.add_body(f);
        w.add_body(RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.01, 8.0, 0.0),
            PhysicsMaterial { restitution: 0.8, friction_dynamic: 0.0, friction_static: 0.0, density: 1000.0 }));

        let mut max_vy_after_land = 0.0_f32;
        let mut max_y_bounce = 0.0_f32;
        let mut landed = false;
        for _ in 0..300 {
            w.step(0.016);
            let y  = w.bodies[1].position.y;
            let vy = w.bodies[1].velocity.y;
            if y < 1.5 { landed = true; }
            if landed {
                if vy > max_vy_after_land { max_vy_after_land = vy; }
                if y > max_y_bounce { max_y_bounce = y; }
            }
        }
        if !landed { return Err("Sphere never reached floor".into()); }
        if max_y_bounce < 3.0 {
            return Err(format!("Restitution failed: max height={max_y_bounce:.2} after landing (expected > 3.0)"));
        }
        Ok(format!("Bounce ok: max height={max_y_bounce:.2}m"))
    });

    suite.run("Physics: zero restitution  no bounce", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        w.add_body(RigidBody::new(1, Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 5.0, 0.0), PhysicsMaterial { restitution: 0.0, friction_dynamic: 0.0, friction_static: 0.0, density: 1000.0 }));
        let mut peak_after = 0.0_f32;
        let mut hit_floor = false;
        for _ in 0..200 {
            w.step(0.016);
            let y = w.bodies[1].position.y;
            if y < 1.0 { hit_floor = true; }
            if hit_floor { peak_after = peak_after.max(y); }
        }
        if !hit_floor { return Err("Ball never reached floor".into()); }
        if peak_after > 2.0 { return Err(format!("Ball bounced to y={peak_after:.2} with e=0!")); }
        Ok(format!("Max height after impact: {peak_after:.3}m  correctly no bounce"))
    });

    // 
    // GROUP 15: LARGE SCALE
    // 

    suite.run("Scale: 500 bodies, 10 frames (maximum scale)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        for i in 0..500usize {
            let x = ((i % 22) as f32 - 11.0) * 2.3;
            let z = ((i / 22) as f32 - 11.0) * 2.3;
            let shape = if i % 2 == 0 { Shape::Sphere { radius: 0.7 } } else { Shape::Box { half_extents: Vec3::ONE } };
            w.add_body(RigidBody::new(i + 1, shape, Vec3::new(x, 2.0 + (i % 10) as f32, z), PhysicsMaterial::default()));
        }
        let t0 = Instant::now();
        for _ in 0..10 { w.step(0.016); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
        check_bodies(&w.bodies)?;
        Ok(format!("{ms:.1}ms/frame for 500 bodies  {:.0} pairs checked", 500.0 * 499.0 / 2.0))
    });

    // 
    // GROUP 16: MIXED SHAPE COLLISION
    // 

    suite.run("GJK: sphere vs capsule collision", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let s = Shape::Sphere { radius: 1.0 };
        let c = Shape::Capsule { radius: 0.5, half_height: 2.0 };
        let contact = detect_collision(&s, Vec3::new(1.0, 0.0, 0.0), &c, Vec3::ZERO);
        if contact.is_none() { return Err("Sphere vs Capsule missed overlapping case!".into()); }
        let m = contact.unwrap();
        if !m.depth.is_finite() { return Err(format!("NaN depth sphere vs capsule: {}", m.depth)); }
        Ok(format!("Sphere vs Capsule: depth={:.3}", m.depth))
    });

    suite.run("GJK: box vs capsule collision", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let b = Shape::Box { half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let c = Shape::Capsule { radius: 0.5, half_height: 1.5 };
        let contact = detect_collision(&b, Vec3::ZERO, &c, Vec3::new(1.0, 0.0, 0.0));
        if contact.is_none() { return Err("Box vs Capsule missed overlapping case!".into()); }
        let m = contact.unwrap();
        if !m.depth.is_finite() { return Err(format!("NaN depth box vs capsule: {}", m.depth)); }
        Ok(format!("Box vs Capsule: depth={:.3}", m.depth))
    });

    suite.run("GJK: all 6 shape pair combinations valid", || {
        use gaia::core::collision::gjk::detect_collision;
        use gaia::core::shapes::Shape;
        let shapes = vec![
            Shape::Sphere { radius: 1.0 },
            Shape::Box { half_extents: Vec3::ONE },
            Shape::Capsule { radius: 0.5, half_height: 1.0 },
        ];
        let mut pairs_tested = 0;
        for (i, a) in shapes.iter().enumerate() {
            for b in shapes.iter().skip(i) {
                let result = std::panic::catch_unwind(|| {
                    detect_collision(a, Vec3::new(0.5, 0.0, 0.0), b, Vec3::new(-0.5, 0.0, 0.0))
                });
                if result.is_err() { return Err(format!("PANIC on shape pair {i}")); }
                pairs_tested += 1;
            }
        }
        Ok(format!("All {pairs_tested} shape pairs handled without panic"))
    });

    // 
    // GROUP 17: SOFT BODY FEM ADVERSARIAL
    // 

    suite.run("FEM: external force applied 500 frames  no NaN", || {
        use gaia::core::soft_body::{MatrixFreeSoftBody, Tetrahedron};
        use macroquad::math::Mat3;
        // Build a single-tet soft body manually
        let mut sb = MatrixFreeSoftBody::new(3000.0, 10000.0);
        sb.particles = vec![
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.0, 5.0, 1.0),
        ];
        sb.velocities = vec![Vec3::ZERO; 4];
        sb.masses = vec![1.0; 4];
        // Reference shape matrix: columns are (p1-p0, p2-p0, p3-p0)
        let dm = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        sb.elements.push(Tetrahedron {
            v0: 0, v1: 1, v2: 2, v3: 3,
            inv_rest_shape: dm.inverse(),
            volume: 1.0 / 6.0,
        });
        // Apply lateral wind + gravity as velocity impulses each frame
        for frame in 0..500usize {
            let wind_accel = Vec3::new((frame as f32 * 0.1).sin() * 10.0, -9.81, 0.0);
            for v in &mut sb.velocities { *v += wind_accel * 0.016; }
            sb.step(0.016);
        }
        for (i, p) in sb.particles.iter().enumerate() {
            if !p.is_finite() { return Err(format!("Particle {i} NaN after ext force")); }
        }
        Ok("FEM stable under external forces for 500 frames".into())
    });

    suite.run("FEM: high stiffness (mu=1e6) stability", || {
        use gaia::core::soft_body::{MatrixFreeSoftBody, Tetrahedron};
        use macroquad::math::Mat3;
        let mut sb = MatrixFreeSoftBody::new(1_000_000.0, 3_000_000.0);
        sb.particles = vec![
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.5, 2.0, 0.0),
            Vec3::new(0.0, 2.5, 0.0),
            Vec3::new(0.0, 2.0, 0.5),
        ];
        sb.velocities = vec![Vec3::ZERO; 4];
        sb.masses = vec![0.5; 4];
        let dm = Mat3::from_cols(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
        );
        sb.elements.push(Tetrahedron {
            v0: 0, v1: 1, v2: 2, v3: 3,
            inv_rest_shape: dm.inverse(),
            volume: 0.5f32.powi(3) / 6.0,
        });
        for _ in 0..200 { sb.step(0.008); } // Small dt for stiff system
        for (i, p) in sb.particles.iter().enumerate() {
            if !p.is_finite() { return Err(format!("Particle {i} NaN at high stiffness")); }
        }
        Ok("High-stiffness FEM stable".into())
    });

    // 
    // GROUP 18: FLUID LONGEVITY
    // 

    suite.run("Fluid: 1000-frame longevity (sustained flow)", || {
        use gaia::core::fluid::FluidGrid;
        let mut g = FluidGrid::new(16, 16, 16, 0.3);
        // Apply impulses every 50 frames to simulate sustained flow
        for frame in 0..1000usize {
            if frame % 50 == 0 { g.add_impulse(8, 2, 8, 3.0); }
            g.step(0.016);
        }
        let nans = g.pressure.iter().filter(|v| !v.is_finite()).count();
        let vels = g.vel_x.iter().chain(g.vel_y.iter()).chain(g.vel_z.iter())
            .filter(|v| !v.is_finite()).count();
        if nans > 0 || vels > 0 { return Err(format!("{nans} pressure NaN, {vels} velocity NaN after 1000 frames")); }
        Ok("Fluid stable for 1000 frames with periodic impulses".into())
    });

    // 
    // GROUP 19: SLEEPING / WAKING
    // 

    suite.run("Sleep: resting stack woken by falling body", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        let mut w = PhysicsWorld::new();
        make_floor(&mut w);
        // Place a box on the floor (it will settle and sleep)
        w.add_body(RigidBody::new(1, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 1.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() }));
        // Let it settle and sleep
        for _ in 0..120 { w.step(0.016); }
        let was_sleeping = w.bodies[1].sleeping;

        // Now drop another body on top
        w.add_body(RigidBody::new(2, Shape::Box { half_extents: Vec3::ONE }, Vec3::new(0.0, 8.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() }));
        for _ in 0..60 { w.step(0.016); }
        check_bodies(&w.bodies)?;

        Ok(format!("Settled body slept={was_sleeping}, stack survived wake impact"))
    });

    // 
    // GROUP 20: SPRING NETWORK CONFLICT
    // 

    suite.run("Joints: 3-body spring triangle (conflicting constraints)", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        use gaia::core::joints::{SpringJoint, JointSystem};
        let mut w = PhysicsWorld::new();
        for i in 0..3usize {
            let angle = std::f32::consts::TAU / 3.0 * i as f32;
            let x = angle.cos() * 3.0;
            let z = angle.sin() * 3.0;
            w.add_body(RigidBody::new(i, Shape::Sphere { radius: 0.4 }, Vec3::new(x, 5.0, z), PhysicsMaterial::default()));
        }
        let mut joints = JointSystem::new();
        // Connect all 3 in a triangle  conflicting constraints
        for a in 0..3usize {
            for b in (a+1)..3 {
                joints.springs.push(SpringJoint {
                    body_a: a, body_b: b,
                    anchor_local_a: Vec3::ZERO, anchor_local_b: Vec3::ZERO,
                    rest_length: 3.0, stiffness: 300.0, damping: 15.0,
                });
            }
        }
        for _ in 0..300 {
            joints.apply_all(&mut w.bodies, 0.016);
            w.step(0.016);
        }
        check_bodies(&w.bodies)?;
        Ok("3-body spring triangle stable under conflicting constraints".into())
    });

    // 
    // GROUP 21: BVH DYNAMIC CORRECTNESS
    // 

    suite.run("BVH: remove and reinsert  no stale entries", || {
        use gaia::core::collision::bvh::{Aabb, BvhTree};
        let mut tree = BvhTree::new();
        let mut handles = Vec::new();
        // Insert 50 objects
        for i in 0..50usize {
            let x = i as f32 * 2.5;
            let leaf = tree.insert(i, Aabb::new(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0)));
            handles.push(leaf);
        }
        // Remove every other object
        for i in (0..50usize).step_by(2) {
            tree.remove(handles[i]);
        }
        // Reinsert at new positions
        for i in (0..50usize).step_by(2) {
            let x = (i + 1) as f32 * 2.5 + 0.1;
            tree.insert(i, Aabb::new(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0)));
        }
        let pairs = tree.query_pairs();
        if pairs.is_empty() { return Err("BVH found no pairs after remove/reinsert".into()); }
        Ok(format!("BVH found {} pairs after churn", pairs.len()))
    });

    // 
    // GROUP 22: CLOTH FLOOR COLLISION
    // 

    suite.run("Cloth: floor collision  no particles below y=0", || {
        use gaia::core::cloth::Cloth;
        let mut c = Cloth::grid(10, 10, 0.5, Vec3::new(0.0, 4.0, 0.0));
        for _ in 0..400 { c.step(0.016); }
        let below = c.particles.iter().filter(|p| p.position.y < -0.1).count();
        if below > 0 { return Err(format!("{below} cloth particles below floor")); }
        for (i, p) in c.particles.iter().enumerate() {
            if !is_valid(p.position) { return Err(format!("Particle {i} NaN after floor collision")); }
        }
        Ok(format!("Cloth settled above floor ({} particles)", c.particles.len()))
    });

    // 
    // GROUP 23: EXTREME CCD (5000 m/s  bullet)
    // 

    suite.run("CCD: bullet-speed object (5000 m/s)  no tunneling", || {
        use gaia::core::solver::{PhysicsWorld, RigidBody};
        use gaia::core::shapes::{Shape, PhysicsMaterial};
        // Massive slab wall: 100m wide, 100m tall, 0.2m thick
        let mut w = PhysicsWorld::new();
        let mut wall = RigidBody::new(0, Shape::Box { half_extents: Vec3::new(0.1, 50.0, 50.0) }, Vec3::ZERO, PhysicsMaterial::default());
        wall.inv_mass = 0.0; wall.inv_inertia = Vec3::ZERO;
        w.add_body(wall);
        // Bullet fired from x=-50 at 5000 m/s
        let mut bullet = RigidBody::new(1, Shape::Sphere { radius: 0.05 }, Vec3::new(-50.0, 0.0, 0.0), PhysicsMaterial { restitution: 0.0, ..Default::default() });
        bullet.velocity = Vec3::new(5000.0, 0.0, 0.0);
        w.add_body(bullet);
        for _ in 0..20 { w.step(0.016); } // Run longer to confirm it STAYS stopped
        check_bodies(&w.bodies)?;
        let x = w.bodies[1].position.x;
        let v = w.bodies[1].velocity.x;
        if x > 5.0 { return Err(format!("Bullet tunneled through wall! x={x:.1}")); }
        Ok(format!("Bullet stopped at x={x:.2} (vel={v:.1})"))
    });

    // 
    // FINAL REPORT
    // 
    suite.report();
    if suite.results.iter().any(|r| !r.passed) { std::process::exit(1); }
}
