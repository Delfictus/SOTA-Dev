// Floyd-Warshall All-Pairs Shortest Paths - Blocked Algorithm
// WebGPU Compute Shader for PRISM-LBS
//
// Implements the 3-phase blocked Floyd-Warshall algorithm for GPU:
// - Phase 1: Process diagonal block (k,k) - serial within block
// - Phase 2: Process row k and column k blocks - parallel
// - Phase 3: Process all remaining blocks - fully parallel
//
// Reference: Venkataraman et al. (2003) "A blocked all-pairs shortest-paths algorithm"

// Workgroup size - must match Rust code
const BLOCK_SIZE: u32 = 32u;

// Distance matrix (read-write)
@group(0) @binding(0)
var<storage, read_write> dist: array<f32>;

// Matrix dimension
@group(0) @binding(1)
var<uniform> params: Params;

struct Params {
    n: u32,           // Matrix dimension (padded to BLOCK_SIZE multiple)
    k_block: u32,     // Current k block index
    phase: u32,       // 1, 2, or 3
    n_blocks: u32,    // Total number of blocks
}

// Shared memory for block tiles
var<workgroup> tile_ik: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;
var<workgroup> tile_kj: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;
var<workgroup> tile_ij: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;

// Helper: Get index in flat distance array
fn idx(i: u32, j: u32) -> u32 {
    return i * params.n + j;
}

// Helper: Load a block tile into shared memory
fn load_tile(
    tile: ptr<workgroup, array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>>,
    block_i: u32,
    block_j: u32,
    local_i: u32,
    local_j: u32
) {
    let global_i = block_i * BLOCK_SIZE + local_i;
    let global_j = block_j * BLOCK_SIZE + local_j;

    if (global_i < params.n && global_j < params.n) {
        (*tile)[local_i][local_j] = dist[idx(global_i, global_j)];
    } else {
        (*tile)[local_i][local_j] = 1e30; // Infinity for out-of-bounds
    }
}

// Helper: Store a block tile back to global memory
fn store_tile(
    tile: ptr<workgroup, array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>>,
    block_i: u32,
    block_j: u32,
    local_i: u32,
    local_j: u32
) {
    let global_i = block_i * BLOCK_SIZE + local_i;
    let global_j = block_j * BLOCK_SIZE + local_j;

    if (global_i < params.n && global_j < params.n) {
        dist[idx(global_i, global_j)] = (*tile)[local_i][local_j];
    }
}

// Phase 1: Process diagonal block (k,k)
// All k values within the block processed sequentially
@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn phase1(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_i = local_id.x;
    let local_j = local_id.y;
    let k_block = params.k_block;

    // Load diagonal block into shared memory
    load_tile(&tile_ij, k_block, k_block, local_i, local_j);
    workgroupBarrier();

    // Process all k values within block
    for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
        // Floyd-Warshall relaxation: d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        let d_ik = tile_ij[local_i][k];
        let d_kj = tile_ij[k][local_j];
        let new_dist = d_ik + d_kj;

        if (new_dist < tile_ij[local_i][local_j]) {
            tile_ij[local_i][local_j] = new_dist;
        }

        workgroupBarrier();
    }

    // Store result back
    store_tile(&tile_ij, k_block, k_block, local_i, local_j);
}

// Phase 2: Process row k and column k blocks (excluding diagonal)
@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn phase2(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let local_i = local_id.x;
    let local_j = local_id.y;
    let k_block = params.k_block;
    let block_idx = group_id.x;

    // Skip diagonal block (handled in phase 1)
    var target_block = block_idx;
    if (target_block >= k_block) {
        target_block = target_block + 1u;
    }

    // Determine if we're processing row or column
    let is_row = group_id.y == 0u;

    // Load diagonal block (k,k) - needed for all phase 2 computations
    load_tile(&tile_kj, k_block, k_block, local_i, local_j);
    workgroupBarrier();

    if (is_row) {
        // Process block (k, target_block) - row k
        load_tile(&tile_ij, k_block, target_block, local_i, local_j);
        workgroupBarrier();

        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            let d_ik = tile_kj[local_i][k];  // From diagonal block
            let d_kj = tile_ij[k][local_j];  // From current block
            let new_dist = d_ik + d_kj;

            if (new_dist < tile_ij[local_i][local_j]) {
                tile_ij[local_i][local_j] = new_dist;
            }
            workgroupBarrier();
        }

        store_tile(&tile_ij, k_block, target_block, local_i, local_j);
    } else {
        // Process block (target_block, k) - column k
        load_tile(&tile_ij, target_block, k_block, local_i, local_j);
        workgroupBarrier();

        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            let d_ik = tile_ij[local_i][k];  // From current block
            let d_kj = tile_kj[k][local_j];  // From diagonal block
            let new_dist = d_ik + d_kj;

            if (new_dist < tile_ij[local_i][local_j]) {
                tile_ij[local_i][local_j] = new_dist;
            }
            workgroupBarrier();
        }

        store_tile(&tile_ij, target_block, k_block, local_i, local_j);
    }
}

// Phase 3: Process all remaining blocks (fully parallel)
@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn phase3(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let local_i = local_id.x;
    let local_j = local_id.y;
    let k_block = params.k_block;

    // Compute block indices, skipping row k and column k
    var block_i = group_id.x;
    var block_j = group_id.y;

    if (block_i >= k_block) {
        block_i = block_i + 1u;
    }
    if (block_j >= k_block) {
        block_j = block_j + 1u;
    }

    // Load the three tiles we need
    // tile_ik: block (block_i, k_block) - from row k of block_i
    // tile_kj: block (k_block, block_j) - from column k of block_j
    // tile_ij: block (block_i, block_j) - target block

    load_tile(&tile_ik, block_i, k_block, local_i, local_j);
    load_tile(&tile_kj, k_block, block_j, local_i, local_j);
    load_tile(&tile_ij, block_i, block_j, local_i, local_j);
    workgroupBarrier();

    // Process all k values within block
    for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
        let d_ik = tile_ik[local_i][k];
        let d_kj = tile_kj[k][local_j];
        let new_dist = d_ik + d_kj;

        if (new_dist < tile_ij[local_i][local_j]) {
            tile_ij[local_i][local_j] = new_dist;
        }
        // No barrier needed here - each thread writes to its own location
    }

    // Store result
    store_tile(&tile_ij, block_i, block_j, local_i, local_j);
}

// Simple single-threaded Floyd-Warshall for small matrices
// Used when n < BLOCK_SIZE
@compute @workgroup_size(1, 1, 1)
fn simple_floyd_warshall() {
    let n = params.n;

    for (var k: u32 = 0u; k < n; k = k + 1u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let d_ik = dist[idx(i, k)];
            if (d_ik >= 1e30) {
                continue;
            }

            for (var j: u32 = 0u; j < n; j = j + 1u) {
                let d_kj = dist[idx(k, j)];
                if (d_kj >= 1e30) {
                    continue;
                }

                let new_dist = d_ik + d_kj;
                if (new_dist < dist[idx(i, j)]) {
                    dist[idx(i, j)] = new_dist;
                }
            }
        }
    }
}
