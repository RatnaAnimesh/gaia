#include <metal_stdlib>
using namespace metal;

struct Position {
    float3 current;
    float3 predicted;
};

struct Velocity {
    float3 current;
};

struct Mass {
    float inv_mass;
};

struct BoundingBox {
    float3 min;
    float3 max;
};

struct CollisionPair {
    uint entity_a;
    uint entity_b;
};

// Simple spatial hash parameters (should match CPU)
constant const float CELL_SIZE = 10.0;
constant const uint HASH_TABLE_SIZE = 10000;

// Hash function
uint spatial_hash(float3 position) {
    int x = int(floor(position.x / CELL_SIZE));
    int y = int(floor(position.y / CELL_SIZE));
    int z = int(floor(position.z / CELL_SIZE));
    
    // Large primes for hashing
    uint h = (uint(x) * 73856093) ^ (uint(y) * 19349663) ^ (uint(z) * 83492791);
    return h % HASH_TABLE_SIZE;
}

// 1. Clear the spatial hash grid counters
kernel void clear_spatial_hash(
    device atomic_uint* hash_counters [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < HASH_TABLE_SIZE) {
        atomic_store_explicit(&hash_counters[id], 0, memory_order_relaxed);
    }
}

// 2. Insert entities into grid
kernel void build_spatial_hash(
    device Position* positions [[buffer(0)]],
    device uint* hash_entries [[buffer(1)]],  // Flattened array
    device atomic_uint* hash_counters [[buffer(2)]],
    constant uint& entity_count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= entity_count) return;

    uint hash = spatial_hash(positions[id].current);
    
    // Increment counter and get index within cell
    uint index_in_cell = atomic_fetch_add_explicit(&hash_counters[hash], 1, memory_order_relaxed);
    
    // Store entity ID in flattened hash grid (assuming max 32 entities per cell)
    // Warning: Hardcoded max entities per cell for scaffolding!
    uint MAX_PER_CELL = 32;
    if (index_in_cell < MAX_PER_CELL) {
        hash_entries[hash * MAX_PER_CELL + index_in_cell] = id;
    }
}

// 3. Find Broadphase Pairs
kernel void find_broadphase_pairs(
    device Position* positions [[buffer(0)]],
    device BoundingBox* aabbs [[buffer(1)]],
    device uint* hash_entries [[buffer(2)]],
    device atomic_uint* hash_counters [[buffer(3)]],
    device CollisionPair* pair_buffer [[buffer(4)]],
    device atomic_uint* pair_count [[buffer(5)]],
    constant uint& entity_count [[buffer(6)]],
    constant uint& max_pairs [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= entity_count) return;

    // VERY naive pair generation: look at our own cell and test AABBs.
    // In a full AAA setup, we must look at 27 neighboring cells.
    uint hash = spatial_hash(positions[id].current);
    uint entities_in_cell = atomic_load_explicit(&hash_counters[hash], memory_order_relaxed);
    
    uint MAX_PER_CELL = 32;
    entities_in_cell = min(entities_in_cell, MAX_PER_CELL);

    // Bounding box of current entity
    float3 a_min = aabbs[id].min + positions[id].current;
    float3 a_max = aabbs[id].max + positions[id].current;

    for (uint i = 0; i < entities_in_cell; i++) {
        uint other_id = hash_entries[hash * MAX_PER_CELL + i];
        
        // Only test one way to avoid duplicate pairs (id < other_id)
        if (id < other_id) {
            float3 b_min = aabbs[other_id].min + positions[other_id].current;
            float3 b_max = aabbs[other_id].max + positions[other_id].current;

            // Simple AABB intersection test
            bool overlap = (a_min.x <= b_max.x && a_max.x >= b_min.x) &&
                           (a_min.y <= b_max.y && a_max.y >= b_min.y) &&
                           (a_min.z <= b_max.z && a_max.z >= b_min.z);
                           
            if (overlap) {
                // We have a temporal narrowphase candidate
                uint pair_idx = atomic_fetch_add_explicit(pair_count, 1, memory_order_relaxed);
                if (pair_idx < max_pairs) {
                    pair_buffer[pair_idx].entity_a = id;
                    pair_buffer[pair_idx].entity_b = other_id;
                }
            }
        }
    }
}

// ----------------------------------------------------
// NARROW PHASE: SUPPORT MATHEMATICS
// ----------------------------------------------------

// Support function for a simple box
float3 support_box(float3 extents, float3 dir) {
    return float3(
        sign(dir.x) * extents.x,
        sign(dir.y) * extents.y,
        sign(dir.z) * extents.z
    );
}

// Support function for a sphere
float3 support_sphere(float radius, float3 dir) {
    return normalize(dir) * radius;
}

// Minkowski difference support mapping: S_A(dir) - S_B(-dir)
float3 minkowski_support(
    float3 pos_a, float3 extents_a, /* ... other shape params */
    float3 pos_b, float3 extents_b, /* ... other shape params */
    float3 dir
) {
    // For scaffolding, we assume boxes
    float3 p_a = pos_a + support_box(extents_a, dir);
    float3 p_b = pos_b + support_box(extents_b, -dir);
    return p_a - p_b;
}

// ----------------------------------------------------
// NARROW PHASE: GJK / EPA LOOP
// ----------------------------------------------------

struct ContactManifold {
    uint entity_a;
    uint entity_b;
    float3 normal;
    float depth;
    float3 point;
    bool active;
};

// Main GPU execution for resolving actual volumes.
// Note: We use fixed-sized local arrays to prevent dynamic allocation.
kernel void narrowphase_gjk_epa(
    device Position* positions [[buffer(0)]],
    device BoundingBox* aabbs [[buffer(1)]],
    device CollisionPair* pairs [[buffer(2)]],
    constant uint& pair_count [[buffer(3)]],
    device ContactManifold* manifolds [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= pair_count) return;
    
    uint id_a = pairs[id].entity_a;
    uint id_b = pairs[id].entity_b;
    
    // We derive extents from AABBs for the dummy representation
    float3 extents_a = (aabbs[id_a].max - aabbs[id_a].min) * 0.5;
    float3 extents_b = (aabbs[id_b].max - aabbs[id_b].min) * 0.5;
    
    // Initial search direction
    float3 search_dir = normalize(positions[id_a].current - positions[id_b].current);
    if (length_squared(search_dir) < 0.001) { search_dir = float3(1, 0, 0); }
    
    // Simplex storage (fixed size 4 for tetrahedron)
    float3 simplex[4];
    uint simplex_count = 0;
    
    // 1. Initial point
    float3 c = minkowski_support(positions[id_a].current, extents_a, positions[id_b].current, extents_b, search_dir);
    simplex[0] = c;
    simplex_count = 1;
    
    search_dir = -c;
    
    bool intersecting = false;
    
    // 2. Main GJK loop
    for (int iter = 0; iter < 32; ++iter) {
        float3 p = minkowski_support(positions[id_a].current, extents_a, positions[id_b].current, extents_b, search_dir);
        
        if (dot(p, search_dir) < 0.0) {
            // Origin is outside the Minkowski difference
            intersecting = false;
            break;
        }
        
        simplex[simplex_count++] = p;
        
        if (simplex_count == 2) {
            // Line segment
            float3 ab = simplex[0] - simplex[1]; // B is [1], A is [0] if we view the newest as A
            float3 ao = -simplex[1];
            
            if (dot(ab, ao) > 0) {
                search_dir = cross(cross(ab, ao), ab);
            } else {
                simplex[0] = simplex[1];
                simplex_count = 1;
                search_dir = ao;
            }
        } else if (simplex_count == 3) {
            // Triangle
            float3 a = simplex[2];
            float3 b = simplex[1];
            float3 c = simplex[0];
            
            float3 ab = b - a;
            float3 ac = c - a;
            float3 ao = -a;
            
            float3 abc = cross(ab, ac);
            
            if (dot(cross(abc, ac), ao) > 0) {
                if (dot(ac, ao) > 0) {
                    simplex[0] = c;
                    simplex[1] = a;
                    simplex_count = 2;
                    search_dir = cross(cross(ac, ao), ac);
                } else {
                    if (dot(ab, ao) > 0) {
                        simplex[0] = b;
                        simplex[1] = a;
                        simplex_count = 2;
                        search_dir = cross(cross(ab, ao), ab);
                    } else {
                        simplex[0] = a;
                        simplex_count = 1;
                        search_dir = ao;
                    }
                }
            } else {
                if (dot(cross(ab, abc), ao) > 0) {
                    if (dot(ab, ao) > 0) {
                        simplex[0] = b;
                        simplex[1] = a;
                        simplex_count = 2;
                        search_dir = cross(cross(ab, ao), ab);
                    } else {
                        simplex[0] = a;
                        simplex_count = 1;
                        search_dir = ao;
                    }
                } else {
                    if (dot(abc, ao) > 0) {
                        search_dir = abc;
                    } else {
                        // Reverse vertex order to point normal toward origin
                        simplex[0] = b;
                        simplex[1] = c;
                        simplex[2] = a;
                        search_dir = -abc;
                    }
                }
            }
        } else if (simplex_count == 4) {
            // Tetrahedron - if we are here, and the origin is inside, we are intersecting.
            // (A true implementation evaluates 4 triangle faces; for scaffolding, 
            // if we assembled 4 points bounding the origin without rejection, we call it a hit).
            intersecting = true;
            break;
        }
    }
    
    // 3. EPA / Output Mapping
    if (intersecting) {
        // EPA Algorithm: expand polytope using fixed-size buffers inside thread memory.
        // MAX_VERTS and MAX_FACES act as our hard memory limit to avoid dynamic allocations.
        const int MAX_VERTS = 64;
        const int MAX_FACES = 128;
        
        float3 polytope_verts[MAX_VERTS];
        uint3 polytope_faces[MAX_FACES];
        float3 polytope_normals[MAX_FACES]; // Pre-computed normals
        float face_distances[MAX_FACES];   // Pre-computed distances to origin
        
        uint num_verts = 0;
        uint num_faces = 0;
        
        // Initialize Polytope from GJK Tetrahedron (Simplex 4)
        for(int i=0; i<4; i++) {
            polytope_verts[num_verts++] = simplex[i];
        }
        
        // Define initial 4 faces of the tetrahedron (0,1,2), (0,3,1), (0,2,3), (1,3,2)
        // Note: For a real physics engine, we must ensure winding order points OUTWARD rigidly here.
        polytope_faces[0] = uint3(0, 1, 2);
        polytope_faces[1] = uint3(0, 3, 1);
        polytope_faces[2] = uint3(0, 2, 3);
        polytope_faces[3] = uint3(1, 3, 2);
        num_faces = 4;
        
        // Calculate initial normals and distances
        for(uint i=0; i<num_faces; i++) {
            uint3 face = polytope_faces[i];
            float3 a = polytope_verts[face.x];
            float3 b = polytope_verts[face.y];
            float3 c = polytope_verts[face.z];
            
            float3 n = normalize(cross(b-a, c-a));
            polytope_normals[i] = n;
            face_distances[i] = dot(n, a);
        }
        
        float min_dist = 999999.0;
        float3 min_normal = float3(0, 1, 0);
        
        // EPA Iteration Loop
        for(int epa_iter = 0; epa_iter < 16; ++epa_iter) {
            // Find closest face to origin
            int closest_face = -1;
            min_dist = 999999.0;
            
            for(uint i=0; i<num_faces; i++) {
                if (face_distances[i] < min_dist) {
                    min_dist = face_distances[i];
                    closest_face = i;
                }
            }
            
            if (closest_face == -1) break;
            
            min_normal = polytope_normals[closest_face];
            
            // Get support point in direction of the closest face normal
            float3 support_pt = minkowski_support(positions[id_a].current, extents_a, positions[id_b].current, extents_b, min_normal);
            float support_dist = dot(min_normal, support_pt);
            
            // If the support point doesn't expand the polytope significantly, we've found the boundary.
            if (abs(support_dist - min_dist) < 0.001) {
                break;
            }
            
            // If we run out of geometry buffer space, bail out and use best guess.
            if (num_verts >= MAX_VERTS || num_faces >= MAX_FACES) {
                break;
            }
            
            // [Implementation placeholder:
            // 1. Identify all faces that "see" the new support_pt (dot > 0).
            // 2. Remove those faces.
            // 3. Extract the boundary edges of the "hole" left behind.
            // 4. Create new faces bridging the boundary edges to the new support_pt.
            // 5. Update normals and face_distances.]
            break; // Scaffolding: break immediately after setup to not compile-timeout or hang GPU.
        }
        
        // Write out the contact manifold
        manifolds[id].entity_a = id_a;
        manifolds[id].entity_b = id_b;
        manifolds[id].normal = min_normal; // The vector strictly pointing out of shape A
        manifolds[id].depth = min_dist;    // Penetration distance
        manifolds[id].active = true;
    } else {
        manifolds[id].active = false;
    }
}

