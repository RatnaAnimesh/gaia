/// Phase 11: AABB Bounding Volume Hierarchy (BVH)  Broad Phase
///
/// Provides O(N log N) collision pair detection.
/// Each node in the tree stores an AABB that tightly encloses all its children.
/// `query_pairs()` returns all leaf-leaf pairs whose AABBs overlap.

use macroquad::prelude::Vec3;

#[derive(Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn from_point(p: Vec3, margin: f32) -> Self {
        Self { min: p - Vec3::splat(margin), max: p + Vec3::splat(margin) }
    }

    pub fn merge(a: &Aabb, b: &Aabb) -> Self {
        Aabb {
            min: a.min.min(b.min),
            max: a.max.max(b.max),
        }
    }

    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn contains(&self, other: &Aabb) -> bool {
        self.min.x <= other.min.x && self.max.x >= other.max.x &&
        self.min.y <= other.min.y && self.max.y >= other.max.y &&
        self.min.z <= other.min.z && self.max.z >= other.max.z
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
}

/// A node in the BVH tree
#[derive(Clone, Debug)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub object_id: Option<usize>, // leaf only
}

impl BvhNode {
    fn is_leaf(&self) -> bool { self.left.is_none() && self.right.is_none() }
}

/// Dynamic AABB BVH tree
pub struct BvhTree {
    pub nodes: Vec<BvhNode>,
    root: Option<usize>,
    free_list: Vec<usize>,
}

impl BvhTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), root: None, free_list: Vec::new() }
    }

    fn alloc_node(&mut self) -> usize {
        if let Some(i) = self.free_list.pop() { return i; }
        let i = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
            parent: None, left: None, right: None, object_id: None,
        });
        i
    }

    fn free_node(&mut self, idx: usize) {
        self.free_list.push(idx);
    }

    /// Insert an object with an AABB leaf
    pub fn insert(&mut self, object_id: usize, aabb: Aabb) -> usize {
        let leaf = self.alloc_node();
        self.nodes[leaf].aabb = aabb;
        self.nodes[leaf].object_id = Some(object_id);
        self.nodes[leaf].left = None;
        self.nodes[leaf].right = None;

        if self.root.is_none() {
            self.nodes[leaf].parent = None;
            self.root = Some(leaf);
            return leaf;
        }

        // Find best sibling using surface area heuristic
        let mut best = self.root.unwrap();
        loop {
            let node = &self.nodes[best];
            if node.is_leaf() { break; }

            let left  = node.left.unwrap();
            let right = node.right.unwrap();

            let merged_area = Aabb::merge(&self.nodes[best].aabb, &self.nodes[leaf].aabb).surface_area();
            let left_area   = Aabb::merge(&self.nodes[left].aabb,  &self.nodes[leaf].aabb).surface_area();
            let right_area  = Aabb::merge(&self.nodes[right].aabb, &self.nodes[leaf].aabb).surface_area();

            if merged_area < left_area && merged_area < right_area { break; }
            best = if left_area < right_area { left } else { right };
        }

        // Create new parent for leaf + best
        let new_parent = self.alloc_node();
        let old_parent = self.nodes[best].parent;
        self.nodes[new_parent].aabb   = Aabb::merge(&self.nodes[leaf].aabb, &self.nodes[best].aabb);
        self.nodes[new_parent].parent  = old_parent;
        self.nodes[new_parent].left    = Some(best);
        self.nodes[new_parent].right   = Some(leaf);
        self.nodes[new_parent].object_id = None;

        self.nodes[best].parent = Some(new_parent);
        self.nodes[leaf].parent = Some(new_parent);

        match old_parent {
            None => self.root = Some(new_parent),
            Some(p) => {
                if self.nodes[p].left == Some(best) { self.nodes[p].left = Some(new_parent); }
                else { self.nodes[p].right = Some(new_parent); }
            }
        }

        // Refit ancestors
        self.refit(new_parent);
        leaf
    }

    fn refit(&mut self, mut idx: usize) {
        loop {
            let left  = self.nodes[idx].left;
            let right = self.nodes[idx].right;
            if let (Some(l), Some(r)) = (left, right) {
                self.nodes[idx].aabb = Aabb::merge(&self.nodes[l].aabb, &self.nodes[r].aabb);
            }
            match self.nodes[idx].parent {
                None => break,
                Some(p) => idx = p,
            }
        }
    }

    /// Returns all overlapping (object_id_a, object_id_b) pairs
    pub fn query_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        if let Some(root) = self.root {
            self.query_node(root, root, &mut pairs);
        }
        pairs
    }

    fn query_node(&self, a: usize, b: usize, pairs: &mut Vec<(usize, usize)>) {
        let na = &self.nodes[a];
        let nb = &self.nodes[b];
        if !na.aabb.overlaps(&nb.aabb) { return; }

        if na.is_leaf() && nb.is_leaf() {
            if a != b {
                let id_a = na.object_id.unwrap();
                let id_b = nb.object_id.unwrap();
                if id_a < id_b { pairs.push((id_a, id_b)); }
            }
            return;
        }

        if nb.is_leaf() || (!na.is_leaf() && na.aabb.surface_area() >= nb.aabb.surface_area()) {
            if let (Some(l), Some(r)) = (na.left, na.right) {
                self.query_node(l, b, pairs);
                self.query_node(r, b, pairs);
            }
        } else {
            if let (Some(l), Some(r)) = (nb.left, nb.right) {
                self.query_node(a, l, pairs);
                self.query_node(a, r, pairs);
            }
        }
    }

    /// Update a leaf's AABB (after the object moves)
    pub fn update(&mut self, leaf: usize, new_aabb: Aabb) {
        if self.nodes[leaf].aabb.contains(&new_aabb) { return; } // no refit needed
        // Remove and reinsert
        let obj_id = self.nodes[leaf].object_id.unwrap();
        self.remove(leaf);
        self.insert(obj_id, new_aabb);
    }

    pub fn remove(&mut self, leaf: usize) {
        if self.root == Some(leaf) { self.root = None; self.free_node(leaf); return; }
        let parent = self.nodes[leaf].parent.unwrap();
        let grandparent = self.nodes[parent].parent;
        let sibling = if self.nodes[parent].left == Some(leaf) {
            self.nodes[parent].right.unwrap()
        } else {
            self.nodes[parent].left.unwrap()
        };
        match grandparent {
            None => { self.nodes[sibling].parent = None; self.root = Some(sibling); }
            Some(gp) => {
                self.nodes[sibling].parent = Some(gp);
                if self.nodes[gp].left == Some(parent) { self.nodes[gp].left = Some(sibling); }
                else { self.nodes[gp].right = Some(sibling); }
                self.refit(gp);
            }
        }
        self.free_node(parent);
        self.free_node(leaf);
    }
}
