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
