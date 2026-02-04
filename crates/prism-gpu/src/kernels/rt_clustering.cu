// [RT-CLUSTERING] OptiX RT-Core Accelerated Spatial Clustering
//
// Hardware-accelerated spatial clustering using RTX RT cores.
// Replaces O(N²) CPU DBSCAN with O(N) GPU spatial queries.
//
// Architecture:
// 1. Build BVH from event positions (spheres with radius = eps/2)
// 2. For each event, query neighbors within epsilon using RT cores
// 3. Build connectivity graph in CSR format
// 4. Run parallel Union-Find for connected components
//
// Target: NVIDIA RTX 5080 (84 RT cores, sm_120)

#include <optix.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Launch Parameters
// ═══════════════════════════════════════════════════════════════════════════════

struct RtClusteringParams {
    // BVH for event positions
    OptixTraversableHandle traversable;

    // Input: Event positions
    float3* event_positions;        // [num_events] positions
    unsigned int num_events;

    // Clustering parameters
    float epsilon;                  // Neighborhood radius (Å)
    unsigned int min_points;        // Minimum points for core point
    unsigned int rays_per_event;    // Rays to cast per event (for neighbor finding)

    // Output: Neighbor counts and indices
    unsigned int* neighbor_counts;  // [num_events] count of neighbors
    unsigned int* neighbor_offsets; // [num_events+1] CSR offsets
    unsigned int* neighbor_indices; // [total_neighbors] neighbor event IDs

    // Output: Cluster assignments
    int* cluster_ids;               // [num_events] cluster ID (-1 = noise)
    int* parent;                    // [num_events] Union-Find parent array

    // Statistics
    unsigned int* total_neighbors;  // Single value: total neighbor pairs
    unsigned int* num_clusters;     // Single value: number of clusters found
};

extern "C" {
    __constant__ RtClusteringParams params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility: Sphere Direction Generation (Fibonacci lattice)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float3 fibonacci_sphere_direction(unsigned int idx, unsigned int total) {
    const float golden_ratio = 1.618033988749895f;
    const float pi = 3.14159265358979f;

    float theta = 2.0f * pi * idx / golden_ratio;
    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / total);

    float sin_phi = sinf(phi);
    return make_float3(
        sin_phi * cosf(theta),
        sin_phi * sinf(theta),
        cosf(phi)
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 1: Count Neighbors (Phase 1)
// Cast rays to find all neighbors within epsilon
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __raygen__count_neighbors() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int event_id = idx.x;
    unsigned int ray_id = idx.y;

    if (event_id >= params.num_events) return;
    if (ray_id >= params.rays_per_event) return;

    // Get event position
    float3 origin = params.event_positions[event_id];

    // Generate ray direction
    float3 direction = fibonacci_sphere_direction(ray_id, params.rays_per_event);

    // Trace ray with epsilon as max distance
    unsigned int p0 = 0;  // hit_event_id
    unsigned int p1 = 0;  // hit_count

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,                         // tmin (avoid self-intersection)
        params.epsilon,                 // tmax = epsilon
        0.0f,                           // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,            // Want all hits, not just closest
        0,                              // SBT offset
        1,                              // SBT stride
        0,                              // missSBTIndex
        p0, p1
    );

    // p1 contains the number of hits for this ray
    // Atomically add to neighbor count
    if (p1 > 0) {
        atomicAdd(&params.neighbor_counts[event_id], p1);
    }
}

extern "C" __global__ void __closesthit__count_neighbors() {
    unsigned int hit_event_id = optixGetPrimitiveIndex();

    // Increment hit count in payload
    unsigned int p1 = optixGetPayload_1();
    optixSetPayload_1(p1 + 1);

    // Store hit event ID (for later neighbor list building)
    optixSetPayload_0(hit_event_id);
}

extern "C" __global__ void __miss__count_neighbors() {
    // No action needed for miss
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 2: Compute CSR Offsets (Prefix Sum)
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void compute_neighbor_offsets(
    unsigned int* neighbor_counts,
    unsigned int* neighbor_offsets,
    unsigned int num_events,
    unsigned int* total_neighbors
) {
    // Simple sequential prefix sum (could be parallelized with scan)
    // For production, use CUB::DeviceScan::ExclusiveSum
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < num_events; i++) {
            neighbor_offsets[i] = sum;
            sum += neighbor_counts[i];
        }
        neighbor_offsets[num_events] = sum;
        *total_neighbors = sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 3: Build Neighbor List (Phase 2)
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __raygen__build_neighbors() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int event_id = idx.x;
    unsigned int ray_id = idx.y;

    if (event_id >= params.num_events) return;
    if (ray_id >= params.rays_per_event) return;

    float3 origin = params.event_positions[event_id];
    float3 direction = fibonacci_sphere_direction(ray_id, params.rays_per_event);

    // Get write offset for this event
    unsigned int base_offset = params.neighbor_offsets[event_id];

    // Trace and store neighbor IDs
    unsigned int p0 = event_id;     // Current event ID
    unsigned int p1 = base_offset;  // Write offset

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,
        params.epsilon,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1
    );
}

extern "C" __global__ void __closesthit__build_neighbors() {
    unsigned int hit_event_id = optixGetPrimitiveIndex();
    unsigned int source_event = optixGetPayload_0();
    unsigned int write_offset = optixGetPayload_1();

    // Avoid self-references and duplicates (only store if hit_id > source_id)
    if (hit_event_id != source_event && hit_event_id > source_event) {
        // Atomically claim a slot
        unsigned int slot = atomicAdd(&params.neighbor_counts[source_event], 1);
        unsigned int actual_offset = params.neighbor_offsets[source_event] + slot;

        // SAFETY: Bounds check using prefix sum offsets
        // max_valid_offset is neighbor_offsets[source_event + 1] (exclusive upper bound)
        unsigned int max_valid_offset = params.neighbor_offsets[source_event + 1];
        if (actual_offset < max_valid_offset) {
            params.neighbor_indices[actual_offset] = hit_event_id;
        }
        // If we exceed bounds, the slot is wasted but we don't corrupt memory
    }

    optixSetPayload_1(write_offset + 1);
}

extern "C" __global__ void __miss__build_neighbors() {
    // No action needed
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 4: Parallel Union-Find Initialization
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void init_union_find(
    int* parent,
    unsigned int num_events
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    // Each element is its own parent initially
    parent[i] = i;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 5: Union-Find with Path Compression
// ═══════════════════════════════════════════════════════════════════════════════

__device__ int find_root(int* parent, int x) {
    int root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    // Path compression
    while (parent[x] != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}

__device__ void union_sets(int* parent, int a, int b) {
    int root_a = find_root(parent, a);
    int root_b = find_root(parent, b);

    if (root_a != root_b) {
        // Always point larger to smaller for consistency
        if (root_a < root_b) {
            atomicMin(&parent[root_b], root_a);
        } else {
            atomicMin(&parent[root_a], root_b);
        }
    }
}

extern "C" __global__ void union_neighbors(
    const unsigned int* neighbor_offsets,
    const unsigned int* neighbor_indices,
    int* parent,
    unsigned int num_events,
    int* changed
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    unsigned int start = neighbor_offsets[i];
    unsigned int end = neighbor_offsets[i + 1];

    int my_root = find_root(parent, i);

    for (unsigned int k = start; k < end; k++) {
        unsigned int neighbor = neighbor_indices[k];
        int nbr_root = find_root(parent, neighbor);

        if (my_root != nbr_root) {
            union_sets(parent, i, neighbor);
            atomicOr(changed, 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 6: Flatten and Assign Cluster IDs
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void flatten_clusters(
    int* parent,
    int* cluster_ids,
    const unsigned int* neighbor_counts,
    unsigned int num_events,
    unsigned int min_points,
    int* cluster_map,      // [num_events] maps root -> cluster_id
    unsigned int* num_clusters
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    // Find root with path compression
    int root = find_root(parent, i);
    parent[i] = root;  // Full compression

    // Mark noise points (not enough neighbors)
    if (neighbor_counts[i] < min_points) {
        cluster_ids[i] = -1;  // Noise
        return;
    }

    // Assign cluster ID based on root
    // Root elements get assigned new cluster IDs
    if (i == root) {
        int cluster_id = atomicAdd(num_clusters, 1);
        cluster_map[i] = cluster_id;
        cluster_ids[i] = cluster_id;
    }
}

extern "C" __global__ void propagate_cluster_ids(
    const int* parent,
    int* cluster_ids,
    const int* cluster_map,
    unsigned int num_events
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    if (cluster_ids[i] == -1) return;  // Skip noise

    int root = parent[i];
    cluster_ids[i] = cluster_map[root];
}

// ═══════════════════════════════════════════════════════════════════════════════
// POSTPROCESSING: Filter Small Clusters
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void count_cluster_sizes(
    const int* cluster_ids,
    unsigned int* cluster_sizes,
    unsigned int num_events,
    unsigned int max_clusters
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    int cid = cluster_ids[i];
    if (cid >= 0 && cid < max_clusters) {
        atomicAdd(&cluster_sizes[cid], 1);
    }
}

extern "C" __global__ void filter_small_clusters(
    int* cluster_ids,
    const unsigned int* cluster_sizes,
    unsigned int num_events,
    unsigned int min_cluster_size
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_events) return;

    int cid = cluster_ids[i];
    if (cid >= 0 && cluster_sizes[cid] < min_cluster_size) {
        cluster_ids[i] = -1;  // Mark as noise
    }
}
