// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Spike Ring Buffer Module
//
// Standalone circular buffer for continuous spike exchange between
// twin MD streams. Operates BETWEEN kernel launches — does NOT
// modify nhs_amber_fused.cu.
//
// Architecture:
//   AFTER fused kernel: ring_buffer_push_batch (copy new spikes to ring)
//   BEFORE fused kernel: ring_buffer_read_and_adapt (apply to thresholds)
//   PERIODICALLY: ring_buffer_threshold_recovery (prevent permanent reduction)
// ═══════════════════════════════════════════════════════════════════════

#ifndef RING_BUFFER_CU
#define RING_BUFFER_CU

#define RING_DEFAULT_CAPACITY 8192

// Must match GpuSpikeEvent in fused_engine.rs
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
    int pad;  // alignment to 48 bytes
};

struct RingBufferState {
    unsigned int head;           // total writes (monotonic, never reset)
    unsigned int tail;           // total reads (monotonic, never reset)
    unsigned int capacity;       // ring size
    unsigned int overflow_count; // times head overtook tail+capacity
};


// ─────────────────────────────────────────────────────────────────────
// KERNEL 1: Push batch of new spikes into ring buffer
// Called AFTER each fused kernel launch completes.
// Only copies spikes [prev_count .. curr_count) from the fused spike buffer.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_push_batch(
    RingSpikeEvent* __restrict__ ring_buffer,    // [capacity]
    unsigned int* __restrict__ head_ptr,          // single uint: write pointer
    unsigned int* __restrict__ tail_ptr,          // single uint: read pointer
    unsigned int* __restrict__ overflow_ptr,      // single uint: overflow counter
    unsigned int capacity,
    const RingSpikeEvent* __restrict__ spike_buffer, // fused kernel output
    unsigned int prev_count,
    unsigned int curr_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_new = (int)(curr_count - prev_count);
    if (idx >= n_new) return;

    // Reserve a write position
    unsigned int write_pos = atomicAdd(head_ptr, 1);

    // Check overflow
    unsigned int current_tail = *tail_ptr;
    if (write_pos - current_tail >= capacity) {
        atomicAdd(overflow_ptr, 1);
    }

    // Write spike to ring (circular)
    ring_buffer[write_pos % capacity] = spike_buffer[prev_count + idx];
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 2: Read spikes from ring buffer and adapt thresholds
// Called BEFORE each fused kernel launch.
// Single-threaded (ring buffer typically has <1000 entries per step).
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_read_and_adapt(
    const RingSpikeEvent* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int capacity,
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float voxel_size,
    float sensitivity_boost,
    float max_reduction_fraction,
    unsigned int current_step,
    float decay_constant
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int head = *head_ptr;
    unsigned int tail = *tail_ptr;
    unsigned int n_to_process = head - tail;

    if (n_to_process > capacity) {
        tail = head - capacity;
        n_to_process = capacity;
    }

    for (unsigned int i = 0; i < n_to_process; i++) {
        unsigned int ring_idx = (tail + i) % capacity;
        RingSpikeEvent spike = ring_buffer[ring_idx];

        // Map spike position to voxel
        int vx = (int)((spike.x - grid_origin_x) / voxel_size);
        int vy = (int)((spike.y - grid_origin_y) / voxel_size);
        int vz = (int)((spike.z - grid_origin_z) / voxel_size);

        if (vx < 0 || vx >= n_voxels_x) continue;
        if (vy < 0 || vy >= n_voxels_y) continue;
        if (vz < 0 || vz >= n_voxels_z) continue;

        int voxel_idx = vx + vy * n_voxels_x + vz * n_voxels_x * n_voxels_y;

        float age = (float)((int)current_step - spike.timestep);
        if (age < 0.0f) age = 0.0f;
        float time_weight = expf(-age / decay_constant);
        float boost = sensitivity_boost * spike.intensity * time_weight;

        float base = base_thresholds[voxel_idx];
        float floor_val = base * (1.0f - max_reduction_fraction);
        float current = osc_thresholds[voxel_idx];
        float new_val = fmaxf(floor_val, current - boost);
        osc_thresholds[voxel_idx] = new_val;
    }

    *tail_ptr = head;
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 3: Threshold recovery (gradual restore toward baseline)
// Called periodically to prevent permanent threshold suppression.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_threshold_recovery(
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    unsigned int n_voxels_total,
    float recovery_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)n_voxels_total) return;

    float current = osc_thresholds[idx];
    float base = base_thresholds[idx];
    osc_thresholds[idx] = current + recovery_rate * (base - current);
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 4: Reset ring buffer state (for initialization)
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_reset(
    unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int* __restrict__ overflow_ptr
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *head_ptr = 0;
        *tail_ptr = 0;
        *overflow_ptr = 0;
    }
}

#endif // RING_BUFFER_CU
