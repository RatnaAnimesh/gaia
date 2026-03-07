/// Phase 12: GJK (Gilbert-Johnson-Keerthi) + EPA Narrow Phase
///
/// GJK finds the minimum distance between two convex shapes (or determines they overlap).
/// EPA expands the simplex from GJK into the global contact normal and penetration depth.
///
/// The key insight: we never need the shapes' geometry directly 
/// only their `support()` function (farthest point along a direction).

use macroquad::prelude::Vec3;
use crate::core::shapes::Shape;

/// The output of a successful collision detection
pub struct ContactManifold {
    pub normal:    Vec3,  // Collision normal pointing from B to A
    pub depth:     f32,   // Penetration depth > 0
    pub point_a:   Vec3,  // Contact point on shape A
    pub point_b:   Vec3,  // Contact point on shape B
}

/// Minkowski difference support function
fn support(shape_a: &Shape, pos_a: Vec3, shape_b: &Shape, pos_b: Vec3, dir: Vec3) -> Vec3 {
    let sup_a = pos_a + shape_a.support(dir);
    let sup_b = pos_b + shape_b.support(-dir);
    sup_a - sup_b
}

/// GJK simplex  up to 4 points in the Minkowski difference
#[derive(Default)]
struct Simplex {
    pts: [Vec3; 4],
    size: usize,
}

impl Simplex {
    fn push(&mut self, p: Vec3) {
        // Shift existing and add new point at front
        for i in (0..self.size).rev() {
            self.pts[i + 1] = self.pts[i];
        }
        self.pts[0] = p;
        self.size += 1;
    }
    fn set(&mut self, pts: &[Vec3]) {
        self.size = pts.len();
        for (i, &p) in pts.iter().enumerate() { self.pts[i] = p; }
    }
    fn a(&self) -> Vec3 { self.pts[0] }
    fn b(&self) -> Vec3 { self.pts[1] }
    fn c(&self) -> Vec3 { self.pts[2] }
    fn d(&self) -> Vec3 { self.pts[3] }
}

/// Returns the next search direction and whether the origin is enclosed
fn next_simplex(simplex: &mut Simplex, dir: &mut Vec3) -> bool {
    match simplex.size {
        2 => line_case(simplex, dir),
        3 => triangle_case(simplex, dir),
        4 => tetrahedron_case(simplex, dir),
        _ => false,
    }
}

fn same_dir(a: Vec3, b: Vec3) -> bool { a.dot(b) > 0.0 }

fn line_case(s: &mut Simplex, dir: &mut Vec3) -> bool {
    let ab = s.b() - s.a();
    let ao = -s.a();
    if same_dir(ab, ao) { *dir = ab.cross(ao).cross(ab); }
    else { s.set(&[s.a()]); *dir = ao; }
    false
}

fn triangle_case(s: &mut Simplex, dir: &mut Vec3) -> bool {
    let (a, b, c) = (s.a(), s.b(), s.c());
    let ab = b - a; let ac = c - a; let ao = -a;
    let abc = ab.cross(ac);

    if same_dir(abc.cross(ac), ao) {
        if same_dir(ac, ao) { s.set(&[a, c]); *dir = ac.cross(ao).cross(ac); }
        else { return line_case(&mut { let mut tmp = Simplex::default(); tmp.set(&[a, b]); tmp }, dir);  }
    } else if same_dir(ab.cross(abc), ao) {
        return line_case(&mut { let mut tmp = Simplex::default(); tmp.set(&[a, b]); tmp }, dir);
    } else if same_dir(abc, ao) {
        *dir = abc;
    } else {
        s.set(&[a, c, b]); *dir = -abc;
    }
    false
}

fn tetrahedron_case(s: &mut Simplex, dir: &mut Vec3) -> bool {
    let (a, b, c, d) = (s.a(), s.b(), s.c(), s.d());
    let ab = b - a; let ac = c - a; let ad = d - a; let ao = -a;
    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if same_dir(abc, ao) { s.set(&[a, b, c]); return triangle_case(s, dir); }
    if same_dir(acd, ao) { s.set(&[a, c, d]); return triangle_case(s, dir); }
    if same_dir(adb, ao) { s.set(&[a, d, b]); return triangle_case(s, dir); }
    true // Origin enclosed
}

/// Run GJK. Returns `true` if shapes intersect, and a Simplex enclosing the origin.
pub fn gjk(shape_a: &Shape, pos_a: Vec3, shape_b: &Shape, pos_b: Vec3) -> Option<Simplex> {
    let mut dir  = pos_a - pos_b;
    if dir.length_squared() < 1e-10 { dir = Vec3::X; }

    let mut simplex = Simplex::default();
    simplex.push(support(shape_a, pos_a, shape_b, pos_b, dir));

    dir = -simplex.a();

    for _ in 0..256 {
        let a = support(shape_a, pos_a, shape_b, pos_b, dir);
        if a.dot(dir) < 0.0 { return None; } // No intersection
        simplex.push(a);
        if next_simplex(&mut simplex, &mut dir) {
            return Some(simplex);
        }
    }
    None
}

/// EPA  Expanding Polytope Algorithm
/// Extracts penetration depth and contact normal from GJK simplex.
pub fn epa(
    simplex: Simplex,
    shape_a: &Shape, pos_a: Vec3,
    shape_b: &Shape, pos_b: Vec3,
) -> ContactManifold {
    // Start with tetrahedron faces
    let pts = [simplex.pts[0], simplex.pts[1], simplex.pts[2], simplex.pts[3]];

    // Represent polytope as list of triangles (tuples of indices)
    let mut polytope: Vec<Vec3> = pts.to_vec();
    let mut faces: Vec<[usize; 3]> = vec![[0,1,2],[0,3,1],[0,2,3],[1,3,2]];

    for _ in 0..128 {
        // Find closest face to origin
        let mut min_dist = f32::MAX;
        let mut min_normal = Vec3::Y;
        let mut min_face = 0;

        for (fi, face) in faces.iter().enumerate() {
            let a = polytope[face[0]];
            let b = polytope[face[1]];
            let c = polytope[face[2]];
            let mut n = (b - a).cross(c - a).normalize_or_zero();
            let mut dist = n.dot(a);
            if dist < 0.0 { n = -n; dist = -dist; } // Ensure normal points outward
            if dist < min_dist { min_dist = dist; min_normal = n; min_face = fi; }
        }

        // Expand toward new support
        let new_pt = support(shape_a, pos_a, shape_b, pos_b, min_normal);
        let new_dist = min_normal.dot(new_pt);

        if (new_dist - min_dist).abs() < 1e-4 {
            // Converged
            let contact_normal = min_normal;
            let depth = min_dist;
            
            // Stable contact points: use boundary points shifted by half-depth
            // This is more stable than raw support points which can "jitter" at extreme speeds.
            return ContactManifold {
                normal: contact_normal,
                depth,
                point_a: pos_a - contact_normal * depth * 0.5,
                point_b: pos_b + contact_normal * depth * 0.5,
            };
        }

        // Remove faces facing the new point, add new ones
        let remove_face = faces[min_face];
        faces.remove(min_face);

        let ni = polytope.len();
        polytope.push(new_pt);
        faces.push([remove_face[0], remove_face[1], ni]);
        faces.push([remove_face[1], remove_face[2], ni]);
        faces.push([remove_face[2], remove_face[0], ni]);
    }

    ContactManifold {
        normal: Vec3::Y,
        depth: 0.0,
        point_a: pos_a,
        point_b: pos_b,
    }
}

/// Convenience: full narrow phase for two shapes at given positions.
pub fn detect_collision(
    shape_a: &Shape, pos_a: Vec3,
    shape_b: &Shape, pos_b: Vec3,
) -> Option<ContactManifold> {
    let simplex = gjk(shape_a, pos_a, shape_b, pos_b)?;
    Some(epa(simplex, shape_a, pos_a, shape_b, pos_b))
}
