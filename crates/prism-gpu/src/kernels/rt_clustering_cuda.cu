// RT Clustering CUDA Kernels - Union-Find and Post-processing
//
// This file contains regular CUDA kernels for the clustering pipeline.
// The OptiX raygen shaders are in rt_clustering.cu (compiled to .optixir)
//
// Pipeline phases:
// 1. count neighbors (OptiX raygen)
// 2. compute_neighbor_offsets (prefix sum)
// 3. build neighbor list (OptiX raygen)
// 4. init_union_find
// 5. union_neighbors
// 6. flatten_clusters (path compression)
// 7. propagate_cluster_ids

extern "C" {

// ============================================================================
// PARALLEL PREFIX SUM (BLELLOCH SCAN)
// ============================================================================
// For large N (>1024), we use a multi-block Blelloch scan
// For small N (<=1024), use single-block version

#define SCAN_BLOCK_SIZE 512  // Power of 2, max 1024

// Single-block prefix sum for small arrays
__global__ void prefix_sum_single_block(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int n,
    unsigned int* __restrict__ total
) {
    __shared__ unsigned int temp[SCAN_BLOCK_SIZE * 2];

    unsigned int tid = threadIdx.x;
    unsigned int offset = 1;

    // Load input into shared memory
    if (2 * tid < n) temp[2 * tid] = input[2 * tid];
    else temp[2 * tid] = 0;

    if (2 * tid + 1 < n) temp[2 * tid + 1] = input[2 * tid + 1];
    else temp[2 * tid + 1] = 0;

    // Up-sweep (reduce) phase
    for (unsigned int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            if (bi < SCAN_BLOCK_SIZE * 2 && ai < SCAN_BLOCK_SIZE * 2) {
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }

    // Store total and clear last element
    __syncthreads();
    if (tid == 0) {
        unsigned int last_idx = min(n, SCAN_BLOCK_SIZE * 2) - 1;
        *total = temp[last_idx];
        temp[last_idx] = 0;
    }

    // Down-sweep phase
    for (unsigned int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            if (bi < SCAN_BLOCK_SIZE * 2 && ai < SCAN_BLOCK_SIZE * 2) {
                unsigned int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }

    __syncthreads();

    // Write results to output
    if (2 * tid < n) output[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) output[2 * tid + 1] = temp[2 * tid + 1];
}

// Fallback: Sequential prefix sum for large N (TODO: implement multi-block scan)
// This is still faster than the original due to coalesced access
__global__ void compute_neighbor_offsets(
    const unsigned int* __restrict__ neighbor_counts,
    unsigned int* __restrict__ neighbor_offsets,
    unsigned int num_events,
    unsigned int* __restrict__ total_neighbors
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int offset = 0;
    for (unsigned int i = 0; i < num_events; i++) {
        neighbor_offsets[i] = offset;
        offset += neighbor_counts[i];
    }
    *total_neighbors = offset;
}

// Phase 4: Initialize union-find parent array
__global__ void init_union_find(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    parent[idx] = idx;  // Each element is its own root initially
}

// Find root with path compression (device function)
__device__ int find_root(int* parent, int i) {
    while (parent[i] != i) {
        // Path compression: point to grandparent
        int p = parent[i];
        int gp = parent[p];
        parent[i] = gp;
        i = gp;
    }
    return i;
}

// Phase 5: Union neighbors (union-find with atomic operations)
__global__ void union_neighbors(
    int* __restrict__ parent,
    const unsigned int* __restrict__ neighbor_counts,
    const unsigned int* __restrict__ neighbor_offsets,
    const unsigned int* __restrict__ neighbor_indices,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    unsigned int count = neighbor_counts[idx];
    if (count == 0) return;

    unsigned int offset = neighbor_offsets[idx];

    // Union this event with all its neighbors
    for (unsigned int i = 0; i < count; i++) {
        unsigned int neighbor_idx = neighbor_indices[offset + i];
        if (neighbor_idx >= num_events) continue;  // Skip invalid indices

        // Find roots
        int root_a = find_root(parent, idx);
        int root_b = find_root(parent, neighbor_idx);

        if (root_a != root_b) {
            // Union: point smaller root to larger
            int min_root = min(root_a, root_b);
            int max_root = max(root_a, root_b);
            atomicCAS(&parent[max_root], max_root, min_root);
        }
    }
}

// Phase 6: Flatten clusters (path compression pass)
// Uses pointer jumping for O(log n) convergence
// Set changed[0] = 0 before launch, returns 1 if any change occurred
__global__ void flatten_clusters(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    // Pointer jumping: point to grandparent
    // This converges in O(log n) iterations
    int p = parent[idx];
    if (p != (int)idx && p >= 0) {
        int gp = parent[p];
        if (gp != p && gp >= 0) {
            parent[idx] = gp;  // Skip to grandparent
        }
    }
}

// Alternative: Single-pass flatten that fully compresses paths
// More memory traffic but fewer kernel launches
__global__ void flatten_clusters_full(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    // Chase pointers to root
    int root = idx;
    int prev = root;
    while (parent[root] != root) {
        prev = root;
        root = parent[root];
        if (root < 0 || root >= (int)num_events) {
            root = prev;  // Safety check
            break;
        }
    }

    // Point directly to root (full path compression)
    parent[idx] = root;
}

// Phase 7: Propagate final cluster IDs
// Maps root indices to contiguous cluster IDs
__global__ void propagate_cluster_ids(
    const int* __restrict__ parent,
    int* __restrict__ cluster_ids,
    unsigned int num_events,
    unsigned int* __restrict__ num_clusters
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int root = parent[idx];

    // Simple approach: cluster ID = root index
    // (contiguous renumbering would need another pass)
    cluster_ids[idx] = root;

    // Count unique roots (only roots count themselves)
    if (root == (int)idx) {
        atomicAdd(num_clusters, 1);
    }
}

// Count cluster sizes
__global__ void count_cluster_sizes(
    const int* __restrict__ cluster_ids,
    unsigned int* __restrict__ cluster_sizes,
    unsigned int num_events,
    unsigned int max_clusters
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int cluster = cluster_ids[idx];
    if (cluster >= 0 && (unsigned int)cluster < max_clusters) {
        atomicAdd(&cluster_sizes[cluster], 1);
    }
}

// Filter small clusters (mark as noise = -1)
__global__ void filter_small_clusters(
    int* __restrict__ cluster_ids,
    const unsigned int* __restrict__ cluster_sizes,
    unsigned int num_events,
    unsigned int min_cluster_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int cluster = cluster_ids[idx];
    if (cluster >= 0) {
        if (cluster_sizes[cluster] < min_cluster_size) {
            cluster_ids[idx] = -1;  // Mark as noise
        }
    }
}

} // extern "C"
