// CUDA Kernel for Geometry Stress Analysis
//
// ASSUMPTIONS:
// - Graph positions stored as (x, y) coordinates in f32 pairs (row-major: [x0, y0, x1, y1, ...])
// - MAX_VERTICES = 100,000 (validated by caller before launch)
// - Precision: f32 for coordinates and distances
// - Block size: 256 threads for coalesced memory access
// - Grid size: ceil(num_vertices / 256) for overlap density and hotspot kernels
// - Requires: sm_75+ for warp-level primitives
//
// REFERENCE: Metaphysical Telemetry Coupling - Geometry Sensor Layer

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

// Maximum supported vertices (guardrail)
#define MAX_VERTICES 100000

// Constants for stress analysis
#define OVERLAP_THRESHOLD 0.1f  // Distance threshold for overlap detection
#define HOTSPOT_RADIUS 0.5f     // Radius for anchor hotspot detection

//-----------------------------------------------------------------------------
// Kernel 1: Compute Pairwise Overlap Density
//
// Computes the number of vertex pairs within OVERLAP_THRESHOLD distance.
// Each thread processes one vertex and counts overlaps with all other vertices.
//
// Block/Grid: 256 threads/block, ceil(num_vertices/256) blocks
// Shared Memory: None (global memory only)
// Output: overlap_counts[v] = number of vertices within OVERLAP_THRESHOLD of v
//-----------------------------------------------------------------------------
extern "C" __global__ void compute_overlap_density(
    const float* positions,     // [num_vertices * 2] (x, y pairs)
    float* overlap_counts,      // [num_vertices] output: overlap count per vertex
    unsigned int num_vertices
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices) return;

    // Load this vertex's position
    float vx = positions[v * 2];
    float vy = positions[v * 2 + 1];

    float overlap_count = 0.0f;

    // Count overlaps with all other vertices
    for (unsigned int u = 0; u < num_vertices; u++) {
        if (u == v) continue;

        float ux = positions[u * 2];
        float uy = positions[u * 2 + 1];

        float dx = vx - ux;
        float dy = vy - uy;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < OVERLAP_THRESHOLD) {
            overlap_count += 1.0f;
        }
    }

    overlap_counts[v] = overlap_count;
}

//-----------------------------------------------------------------------------
// Kernel 2: Compute Bounding Box Stress
//
// Computes min/max coordinates and bounding box area.
// Uses parallel reduction to find global min/max across all vertices.
//
// Block/Grid: Single block with 256 threads (sufficient for reduction)
// Shared Memory: 512 floats (4 arrays of 256 for min_x, max_x, min_y, max_y)
// Output: bbox[0]=min_x, bbox[1]=max_x, bbox[2]=min_y, bbox[3]=max_y
//-----------------------------------------------------------------------------
extern "C" __global__ void compute_bounding_box(
    const float* positions,     // [num_vertices * 2]
    float* bbox,                // [4] output: [min_x, max_x, min_y, max_y]
    unsigned int num_vertices
) {
    __shared__ float s_min_x[256];
    __shared__ float s_max_x[256];
    __shared__ float s_min_y[256];
    __shared__ float s_max_y[256];

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    // Initialize with first vertex (or infinity if no vertices)
    float local_min_x = (num_vertices > 0) ? positions[0] : INFINITY;
    float local_max_x = (num_vertices > 0) ? positions[0] : -INFINITY;
    float local_min_y = (num_vertices > 0) ? positions[1] : INFINITY;
    float local_max_y = (num_vertices > 0) ? positions[1] : -INFINITY;

    // Each thread processes multiple vertices (grid-stride loop)
    for (unsigned int v = tid; v < num_vertices; v += stride) {
        float x = positions[v * 2];
        float y = positions[v * 2 + 1];

        local_min_x = fminf(local_min_x, x);
        local_max_x = fmaxf(local_max_x, x);
        local_min_y = fminf(local_min_y, y);
        local_max_y = fmaxf(local_max_y, y);
    }

    // Store thread-local results to shared memory
    s_min_x[tid] = local_min_x;
    s_max_x[tid] = local_max_x;
    s_min_y[tid] = local_min_y;
    s_max_y[tid] = local_max_y;

    __syncthreads();

    // Parallel reduction (tree-based)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        bbox[0] = s_min_x[0]; // min_x
        bbox[1] = s_max_x[0]; // max_x
        bbox[2] = s_min_y[0]; // min_y
        bbox[3] = s_max_y[0]; // max_y
    }
}

//-----------------------------------------------------------------------------
// Kernel 3: Detect Anchor Hotspots
//
// Identifies spatial regions with high anchor density.
// For each vertex, counts nearby anchors within HOTSPOT_RADIUS.
//
// Block/Grid: 256 threads/block, ceil(num_vertices/256) blocks
// Shared Memory: None
// Output: hotspot_scores[v] = number of anchors within HOTSPOT_RADIUS of v
//-----------------------------------------------------------------------------
extern "C" __global__ void detect_anchor_hotspots(
    const float* positions,         // [num_vertices * 2]
    const unsigned int* anchors,    // [num_anchors] vertex indices
    float* hotspot_scores,          // [num_vertices] output: anchor density
    unsigned int num_vertices,
    unsigned int num_anchors
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices) return;

    float vx = positions[v * 2];
    float vy = positions[v * 2 + 1];

    float score = 0.0f;

    // Count anchors within hotspot radius
    for (unsigned int i = 0; i < num_anchors; i++) {
        unsigned int anchor_idx = anchors[i];

        if (anchor_idx >= num_vertices) continue; // Guard against invalid indices

        float ax = positions[anchor_idx * 2];
        float ay = positions[anchor_idx * 2 + 1];

        float dx = vx - ax;
        float dy = vy - ay;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < HOTSPOT_RADIUS) {
            score += 1.0f;
        }
    }

    hotspot_scores[v] = score;
}

//-----------------------------------------------------------------------------
// Kernel 4: Compute Curvature Stress
//
// Approximates local curvature by analyzing vertex neighborhoods.
// Curvature stress = variance of edge lengths in 1-hop neighborhood.
//
// Block/Grid: 256 threads/block, ceil(num_vertices/256) blocks
// Shared Memory: None
// Output: curvature_stress[v] = variance of edge lengths at v
//-----------------------------------------------------------------------------
extern "C" __global__ void compute_curvature_stress(
    const float* positions,         // [num_vertices * 2]
    const unsigned int* row_ptr,    // [num_vertices + 1] CSR row pointers
    const unsigned int* col_idx,    // [num_edges] CSR column indices
    float* curvature_stress,        // [num_vertices] output: curvature stress
    unsigned int num_vertices
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices) return;

    float vx = positions[v * 2];
    float vy = positions[v * 2 + 1];

    // Get neighbors from CSR adjacency
    unsigned int start = row_ptr[v];
    unsigned int end = row_ptr[v + 1];
    unsigned int degree = end - start;

    if (degree == 0) {
        curvature_stress[v] = 0.0f;
        return;
    }

    // Compute mean edge length
    float sum_lengths = 0.0f;
    for (unsigned int i = start; i < end; i++) {
        unsigned int u = col_idx[i];
        if (u >= num_vertices) continue;

        float ux = positions[u * 2];
        float uy = positions[u * 2 + 1];

        float dx = vx - ux;
        float dy = vy - uy;
        float length = sqrtf(dx * dx + dy * dy);

        sum_lengths += length;
    }

    float mean_length = sum_lengths / degree;

    // Compute variance
    float variance = 0.0f;
    for (unsigned int i = start; i < end; i++) {
        unsigned int u = col_idx[i];
        if (u >= num_vertices) continue;

        float ux = positions[u * 2];
        float uy = positions[u * 2 + 1];

        float dx = vx - ux;
        float dy = vy - uy;
        float length = sqrtf(dx * dx + dy * dy);

        float diff = length - mean_length;
        variance += diff * diff;
    }

    variance /= degree;

    curvature_stress[v] = variance;
}
