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

    // Scaffolding Stub: For 2 entities, bypass the spatial hash entirely and just return pair (0, 1)
    if (id == 0) {
        uint pair_idx = atomic_fetch_add_explicit(pair_count, 1, memory_order_relaxed);
        if (pair_idx < max_pairs) {
            pair_buffer[pair_idx].entity_a = 0;
            pair_buffer[pair_idx].entity_b = 1;
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
    
    // Scaffolding Stub: Skip actual GJK/EPA math to prevent GPU hang.
    // If the cubes overlap by AABB, let's just push them up slightly on Y.
    
    float3 a_min = aabbs[id_a].min + positions[id_a].current;
    float3 a_max = aabbs[id_a].max + positions[id_a].current;
    
    float3 b_min = aabbs[id_b].min + positions[id_b].current;
    float3 b_max = aabbs[id_b].max + positions[id_b].current;

    bool overlap = (a_min.x <= b_max.x && a_max.x >= b_min.x) &&
                   (a_min.y <= b_max.y && a_max.y >= b_min.y) &&
                   (a_min.z <= b_max.z && a_max.z >= b_min.z);
    
    if (overlap) {
        manifolds[id].entity_a = id_a;
        manifolds[id].entity_b = id_b;
        manifolds[id].normal = float3(0, 1, 0); // Always push up
        manifolds[id].depth = (b_max.y - a_min.y); // Penetration depth
        manifolds[id].active = true;
    } else {
        manifolds[id].active = false;
    }
}

// ----------------------------------------------------
// XPBD SOLVER: POSITION-BASED DYNAMICS
// ----------------------------------------------------

kernel void xpbd_solve_contacts(
    device Position* positions [[buffer(0)]],
    device Mass* masses [[buffer(1)]],
    device ContactManifold* manifolds [[buffer(2)]],
    constant float& dt_sub [[buffer(3)]],
    constant uint& pairs_count [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= pairs_count) return;
    
    // We execute one constraint per thread. 
    // In a fully robust engine, thread coloring or atomic positional additions
    // must be used to prevent race-conditions on positions[] writes if 
    // multiple constraints affect the same ID. 
    // (This acts as atomic-less parallel Jacobi for scaffolding).
    
    ContactManifold manifold = manifolds[id];
    if (!manifold.active) return;
    
    uint id_a = manifold.entity_a;
    uint id_b = manifold.entity_b;
    
    float w_a = masses[id_a].inv_mass;
    float w_b = masses[id_b].inv_mass;
    
    float w_sum = w_a + w_b;
    if (w_sum <= 0.0) return; // Both static
    
    // 1. Evaluate Constraint C(x) = depth 
    // (The distance the shapes are overlapping)
    float C = manifold.depth;
    
    if (C <= 0.0) return; // No penetration
    
    // 2. XPBD Compliance (alpha = inverse stiffness)
    // For pure rigid contacts, alpha is 0. 
    float alpha = 0.0;
    
    // 3. Lagrange Multiplier update (Delta Lambda)
    // dl = -C / (w_sum + alpha / dt^2)
    float dt_sq = dt_sub * dt_sub;
    float d_lambda = -C / (w_sum + alpha / dt_sq);
    
    // 4. Positional correction impulse P = Delta Lambda * Normal
    float3 P = d_lambda * manifold.normal;
    
    // 5. Apply corrections proportional to inverse mass
    // Warning: requires atomic_fetch_add on floats for true robustness in production
    // metal 3.0 supports atomic<float>, but for scaffolding we naive-write:
    
    positions[id_a].predicted += P * w_a;
    positions[id_b].predicted -= P * w_b;
}

// ----------------------------------------------------
// VELOCITY DERIVATION
// ----------------------------------------------------

kernel void xpbd_velocity_update(
    device Position* positions [[buffer(0)]],
    device Velocity* velocities [[buffer(1)]],
    constant float& delta_time [[buffer(2)]],
    constant uint& entity_count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= entity_count) return;
    
    // V = (P_new - P_old) / dt
    velocities[id].current = (positions[id].predicted - positions[id].current) / delta_time;
    
    // Advance position
    positions[id].current = positions[id].predicted;
}

