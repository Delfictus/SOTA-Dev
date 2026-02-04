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

// Phase 2: Compute prefix sum of neighbor counts
// Single-threaded version for simplicity (works for small N)
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
// Run multiple iterations until convergence
__global__ void flatten_clusters(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    // Chase pointers to root
    int root = idx;
    while (parent[root] != root) {
        root = parent[root];
    }

    // Point directly to root (path compression)
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
