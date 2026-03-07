/// Phase 12: GJK + EPA Narrow Phase (Industrial Hardened)

use macroquad::prelude::Vec3;
use crate::core::shapes::Shape;

pub struct ContactManifold {
    pub normal:    Vec3,
    pub depth:     f32,
    pub point_a:   Vec3,
    pub point_b:   Vec3,
}

#[derive(Default, Clone, Copy)]
pub struct Simplex {
    pub pts: [Vec3; 4],
    pub size: usize,
}

impl Simplex {
    fn push(&mut self, p: Vec3) {
        for i in (0..self.size).rev() { self.pts[i+1] = self.pts[i]; }
        self.pts[0] = p;
        self.size += 1;
    }
    fn a(&self) -> Vec3 { self.pts[0] }
    fn b(&self) -> Vec3 { self.pts[1] }
    fn c(&self) -> Vec3 { self.pts[2] }
    fn d(&self) -> Vec3 { self.pts[3] }
}

fn support(shape_a: &Shape, pos_a: Vec3, shape_b: &Shape, pos_b: Vec3, dir: Vec3) -> Vec3 {
    (pos_a + shape_a.support(dir)) - (pos_b + shape_b.support(-dir))
}

pub fn gjk(shape_a: &Shape, pos_a: Vec3, shape_b: &Shape, pos_b: Vec3) -> Option<Simplex> {
    let mut dir = pos_b - pos_a; // Direction from A to B (towards B)
    if dir.length_squared() < 1e-6 { dir = Vec3::Y; }
    
    let mut s = Simplex::default();
    let p0 = support(shape_a, pos_a, shape_b, pos_b, dir);
    s.push(p0);
    dir = -p0;
    
    for _ in 0..128 { // Increased iterations
        let p = support(shape_a, pos_a, shape_b, pos_b, dir);
        if p.dot(dir) < 0.0 { return None; } // No intersection
        s.push(p);
        if next_simplex(&mut s, &mut dir) { return Some(s); }
        if dir.length_squared() < 1e-9 { return Some(s); } // Origin is on boundary
    }
    None
}

fn next_simplex(s: &mut Simplex, dir: &mut Vec3) -> bool {
    match s.size {
        2 => {
            let ab = s.b() - s.a();
            let ao = -s.a();
            if ab.dot(ao) > 0.0 {
                *dir = ab.cross(ao).cross(ab);
                if dir.length_squared() < 1e-6 { *dir = if ab.x.abs() < 0.9 { ab.cross(Vec3::X) } else { ab.cross(Vec3::Y) }; }
            } else {
                s.size = 1;
                *dir = ao;
            }
        }
        3 => {
            let (a, b, c) = (s.a(), s.b(), s.c());
            let ab = b - a; let ac = c - a; let ao = -a;
            let abc = ab.cross(ac);
            if abc.cross(ac).dot(ao) > 0.0 {
                if ac.dot(ao) > 0.0 {
                    s.size = 2; s.pts[1] = c;
                    *dir = ac.cross(ao).cross(ac);
                } else {
                    s.size = 2; s.pts[1] = b;
                    return next_simplex(s, dir);
                }
            } else if ab.cross(abc).dot(ao) > 0.0 {
                s.size = 2; s.pts[1] = b;
                return next_simplex(s, dir);
            } else {
                if abc.dot(ao) > 0.0 { *dir = abc; }
                else { s.pts[1] = c; s.pts[2] = b; *dir = -abc; }
            }
        }
        4 => {
            let (a, b, c, d) = (s.a(), s.b(), s.c(), s.d());
            let ab = b - a; let ac = c - a; let ad = d - a; let ao = -a;
            let abc = ab.cross(ac); let acd = ac.cross(ad); let adb = ad.cross(ab);
            if abc.dot(ao) > 0.0 { s.size = 3; return next_simplex(s, dir); }
            if acd.dot(ao) > 0.0 { s.size = 3; s.pts[1] = c; s.pts[2] = d; return next_simplex(s, dir); }
            if adb.dot(ao) > 0.0 { s.size = 3; s.pts[1] = d; s.pts[2] = b; return next_simplex(s, dir); }
            return true;
        }
        _ => {}
    }
    false
}

pub fn epa(s: Simplex, shape_a: &Shape, pos_a: Vec3, shape_b: &Shape, pos_b: Vec3) -> ContactManifold {
    let mut polytope = s.pts.to_vec();
    let mut faces = vec![[0,1,2], [0,2,3], [0,3,1], [1,3,2]];
    
    // Ensure outward normals
    let center = (polytope[0] + polytope[1] + polytope[2] + polytope[3]) * 0.25;
    for face in faces.iter_mut() {
        let n = (polytope[face[1]] - polytope[face[0]]).cross(polytope[face[2]] - polytope[face[0]]);
        if n.dot(polytope[face[0]] - center) < 0.0 { face.swap(1, 2); }
    }

    let mut min_dist = f32::MAX;
    let mut min_norm = Vec3::Y;

    for _ in 0..64 {
        min_dist = f32::MAX;
        min_norm = Vec3::Y;
        let mut min_face = 0;

        for (i, f) in faces.iter().enumerate() {
            let n = (polytope[f[1]] - polytope[f[0]]).cross(polytope[f[2]] - polytope[f[0]]).normalize_or_zero();
            let d = n.dot(polytope[f[0]]);
            if d < min_dist { min_dist = d; min_norm = n; min_face = i; }
        }

        let p = (pos_a + shape_a.support(min_norm)) - (pos_b + shape_b.support(-min_norm));
        if (min_norm.dot(p) - min_dist).abs() < 1e-4 {
            let norm = min_norm;
            let depth = min_dist;
            return ContactManifold {
                normal: norm, depth,
                point_a: pos_a + shape_a.support(norm),
                point_b: pos_b + shape_b.support(-norm),
            };
        }

        let mut edges = Vec::new();
        let mut i = 0;
        while i < faces.len() {
            let f = faces[i];
            let n = (polytope[f[1]] - polytope[f[0]]).cross(polytope[f[2]] - polytope[f[0]]).normalize_or_zero();
            if n.dot(p - polytope[f[0]]) > 0.0 {
                let f = faces.remove(i);
                for e in [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]] {
                    if let Some(pos) = edges.iter().position(|&(e0, e1): &(usize, usize)| e0 == e[1] && e1 == e[0]) {
                        edges.remove(pos);
                    } else { edges.push((e[0], e[1])); }
                }
            } else { i += 1; }
        }

        let ni = polytope.len();
        polytope.push(p);
        for (e0, e1) in edges { faces.push([e0, e1, ni]); }
    }

    ContactManifold {
        normal: min_norm, depth: min_dist,
        point_a: pos_a + shape_a.support(min_norm),
        point_b: pos_b + shape_b.support(-min_norm),
    }
}

pub fn detect_collision(sa: &Shape, pa: Vec3, sb: &Shape, pb: Vec3) -> Option<ContactManifold> {
    let s = gjk(sa, pa, sb, pb)?;
    let mut m = epa(s, sa, pa, sb, pb);
    m.normal = -m.normal;
    Some(m)
}
