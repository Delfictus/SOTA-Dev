// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Device-Side Spike Compaction + Ring Buffer Push
//
// Eliminates the CPU round-trip that was the #1 coupling latency source:
//   OLD: GPU spike_events → CPU download → CPU repack → CPU upload → GPU ring
//   NEW: GPU spike_events → GPU compact kernel → GPU ring (zero PCIe traffic)
//
// Reads d_spike_count and d_spike_events directly from the physics engine's
// GPU buffers. Compacts GpuSpikeEvent (92B) → RingSpikeEvent (48B) and
// pushes directly into the ring buffer.
//
// Grid sizing: launched with gridDim read from d_spike_count via
// device-updated graph parameters (the GPU sizes its own kernel).
//
// Compiled with: nvcc -arch=sm_120 -O3 --use_fast_math
// ═══════════════════════════════════════════════════════════════════════

#include <cstdint>

// Match the ring buffer struct layout from ring_buffer.cu
struct RingSpikeEvent {
    int timestep;
    int voxel_idx;
    float x, y, z;
    float intensity;
    float vibrational_energy;
    float water_density;
    int n_nearby_excited;
    int spike_source;
    float wavelength_nm;
    int pad;
};

// GpuSpikeEvent byte offsets (verified against fused_engine.rs:225-241)
// These are the byte positions within each 92-byte GpuSpikeEvent record.
// NOTE: Rust struct layout is NOT repr(C), but empirically these offsets
// are stable because all fields are i32/f32 (4-byte aligned, no padding).
#define GPU_SPIKE_SIZE 92
#define OFF_TIMESTEP           0
#define OFF_VOXEL_IDX          4
#define OFF_POSITION_X         8
#define OFF_POSITION_Y        12
#define OFF_POSITION_Z        16
#define OFF_INTENSITY         20
#define OFF_SPIKE_SOURCE      60
#define OFF_WAVELENGTH_NM     64
#define OFF_WATER_DENSITY     76
#define OFF_VIB_ENERGY        80
#define OFF_N_NEARBY_EXCITED  84

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Compact + Push
//
// Each thread handles one spike. Reads from the physics engine's raw
// spike buffer, extracts the 10 fields needed for the ring buffer,
// and atomically pushes into the ring.
//
// This kernel is designed to be captured in a CUDA graph with
// device-updated grid dimensions read from d_new_spike_count.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void device_compact_and_push(
    // Source: physics engine spike buffer (read-only)
    const unsigned char* __restrict__ gpu_spike_buf,  // d_spike_events, 92B stride
    const int* __restrict__ d_prev_count,              // [1] — start index (exclusive)
    const int* __restrict__ d_curr_count,              // [1] — end index (from d_spike_count)

    // Destination: ring buffer (write)
    RingSpikeEvent* __restrict__ ring_buffer,          // [ring_capacity]
    unsigned int* __restrict__ ring_head,               // [1] — atomic write position
    unsigned int* __restrict__ ring_overflow,            // [1] — overflow counter
    unsigned int ring_capacity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int prev = *d_prev_count;
    int curr = *d_curr_count;
    int n_new = curr - prev;

    // Clamp to ring capacity (take most recent spikes if too many)
    int start_idx = (n_new > (int)ring_capacity) ? (curr - (int)ring_capacity) : prev;
    n_new = curr - start_idx;

    if (tid >= n_new) return;

    int spike_idx = start_idx + tid;
    int offset = spike_idx * GPU_SPIKE_SIZE;
    const unsigned char* src = &gpu_spike_buf[offset];

    // Extract fields by byte offset (type-punned via memcpy)
    RingSpikeEvent ev;
    memcpy(&ev.timestep,           src + OFF_TIMESTEP,          4);
    memcpy(&ev.voxel_idx,          src + OFF_VOXEL_IDX,         4);
    memcpy(&ev.x,                  src + OFF_POSITION_X,        4);
    memcpy(&ev.y,                  src + OFF_POSITION_Y,        4);
    memcpy(&ev.z,                  src + OFF_POSITION_Z,        4);
    memcpy(&ev.intensity,          src + OFF_INTENSITY,          4);
    memcpy(&ev.vibrational_energy, src + OFF_VIB_ENERGY,        4);
    memcpy(&ev.water_density,      src + OFF_WATER_DENSITY,     4);
    memcpy(&ev.n_nearby_excited,   src + OFF_N_NEARBY_EXCITED,  4);
    memcpy(&ev.spike_source,       src + OFF_SPIKE_SOURCE,      4);
    memcpy(&ev.wavelength_nm,      src + OFF_WAVELENGTH_NM,     4);
    ev.pad = 0;

    // Atomic push into ring buffer
    unsigned int write_pos = atomicAdd(ring_head, 1);
    if (write_pos >= ring_capacity) {
        // Overflow: ring is full, count it but still write (circular)
        atomicAdd(ring_overflow, 1);
    }
    ring_buffer[write_pos % ring_capacity] = ev;
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Update prev_count after compaction
//
// After device_compact_and_push completes, update d_prev_count to
// d_curr_count so the next step knows where to start.
// Single-thread kernel, launched as a graph node after compact.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void update_prev_spike_count(
    int* __restrict__ d_prev_count,       // [1] — updated to curr
    const int* __restrict__ d_curr_count  // [1] — current spike count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_prev_count = *d_curr_count;
    }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Compute grid size for compact kernel
//
// Reads d_spike_count and d_prev_count, computes ceil(delta/256),
// writes to d_grid_dim. This grid_dim is used by the CUDA graph's
// device-updated kernel parameters to dynamically size the compact
// kernel's grid.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void compute_compact_grid_size(
    const int* __restrict__ d_curr_count,   // [1]
    const int* __restrict__ d_prev_count,   // [1]
    unsigned int* __restrict__ d_grid_dim_x, // [1] — output grid dim
    unsigned int ring_capacity
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int n_new = *d_curr_count - *d_prev_count;
        if (n_new <= 0) {
            *d_grid_dim_x = 1;  // minimum 1 block (kernel will early-exit)
            return;
        }
        // Clamp to ring capacity
        if (n_new > (int)ring_capacity) n_new = (int)ring_capacity;
        // ceil(n_new / 256)
        *d_grid_dim_x = ((unsigned int)n_new + 255) / 256;
    }
}
