// SNDC Stage 2: Spike Density Grid CUDA Kernels
//
// Gaussian splatting of spike events into a continuous 3D density field
// with intensity² weighting, plus 3D non-maximum suppression for peak finding.
//
// Kernels:
//   scatter_spike_density — splat each spike into nearby voxels
//   find_density_peaks    — 3×3×3 NMS to extract local maxima

#include <stdint.h>

extern "C" {

// Kernel 1: Scatter spikes into density grid with Gaussian splatting
// Each spike contributes intensity² × exp(-r²/2σ²) to nearby voxels
// σ = 2.0Å → 99% energy within 6Å radius → ~6³ = 216 voxels per spike
// With 40K spikes: 40K × 216 = 8.6M atomicAdd operations
// On RTX 5080: ~0.1ms (trivially fast)
__global__ void scatter_spike_density(
    const float* __restrict__ spike_positions,  // [N, 3]
    const float* __restrict__ spike_intensities, // [N]
    float* __restrict__ density_grid,            // [Dx, Dy, Dz]
    int N, int Dx, int Dy, int Dz,
    float origin_x, float origin_y, float origin_z,
    float spacing, float sigma
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = spike_positions[tid * 3 + 0];
    float py = spike_positions[tid * 3 + 1];
    float pz = spike_positions[tid * 3 + 2];
    float w = spike_intensities[tid];
    w = w * w;  // intensity²

    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    int radius = (int)ceilf(3.0f * sigma / spacing);  // 3σ cutoff

    int cx = (int)((px - origin_x) / spacing);
    int cy = (int)((py - origin_y) / spacing);
    int cz = (int)((pz - origin_z) / spacing);

    for (int dx = -radius; dx <= radius; dx++) {
        int ix = cx + dx;
        if (ix < 0 || ix >= Dx) continue;
        float fx = (ix * spacing + origin_x) - px;
        for (int dy = -radius; dy <= radius; dy++) {
            int iy = cy + dy;
            if (iy < 0 || iy >= Dy) continue;
            float fy = (iy * spacing + origin_y) - py;
            for (int dz = -radius; dz <= radius; dz++) {
                int iz = cz + dz;
                if (iz < 0 || iz >= Dz) continue;
                float fz = (iz * spacing + origin_z) - pz;
                float r2 = fx*fx + fy*fy + fz*fz;
                float val = w * expf(-r2 * inv_2sigma2);
                atomicAdd(&density_grid[ix * Dy * Dz + iy * Dz + iz], val);
            }
        }
    }
}

// Kernel 2: 3D non-maximum suppression to find density peaks
// A voxel is a peak if it's the maximum in its 3×3×3 neighborhood
// These peaks seed the hierarchical clustering in Stage 3
__global__ void find_density_peaks(
    const float* __restrict__ density_grid,
    uint32_t* __restrict__ peak_mask,
    int Dx, int Dy, int Dz,
    float min_density
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Dx || iy >= Dy || iz >= Dz) return;

    float val = density_grid[ix * Dy * Dz + iy * Dz + iz];
    if (val < min_density) { peak_mask[ix * Dy * Dz + iy * Dz + iz] = 0; return; }

    bool is_max = true;
    for (int dx = -1; dx <= 1 && is_max; dx++)
        for (int dy = -1; dy <= 1 && is_max; dy++)
            for (int dz = -1; dz <= 1 && is_max; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix+dx, ny = iy+dy, nz = iz+dz;
                if (nx >= 0 && nx < Dx && ny >= 0 && ny < Dy && nz >= 0 && nz < Dz)
                    if (density_grid[nx * Dy * Dz + ny * Dz + nz] > val)
                        is_max = false;
            }
    peak_mask[ix * Dy * Dz + iy * Dz + iz] = is_max ? 1 : 0;
}

} // extern "C"
