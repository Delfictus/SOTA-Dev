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

#include "protocol_state.cuh"

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
    int primary_residue_id;  // lossless residue attribution (was: pad)
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
    float decay_constant,
    // ASC steering: coupling kernel writes when it detects hotspots
    ProtocolState* __restrict__ d_protocol  // nullable: pass nullptr to skip ASC
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int head = *head_ptr;
    unsigned int tail = *tail_ptr;
    unsigned int n_to_process = head - tail;

    if (n_to_process > capacity) {
        tail = head - capacity;
        n_to_process = capacity;
    }

    // ASC: track max threshold reduction for hotspot detection
    float max_reduction = 0.0f;
    int hotspot_residue = -1;

    // ── Stage 2: Closed-loop ASC steering ──
    //
    // Snapshot the steering focus list once per kernel launch (it's small —
    // up to 64 entries — and the rest of the loop only reads it). Reading
    // through the device pointer once amortizes the global-memory load.
    //
    // The autonomous CUDA Graph captures `d_protocol` as a device pointer;
    // the contents change between graph replays via the host's
    // cuMemcpyHtoDAsync writes. This kernel re-loads the focus list every
    // launch (every chunk), so it always sees the latest decisions from
    // the ASC controller without any graph re-capture.
    int n_focus = 0;
    if (d_protocol != nullptr) {
        n_focus = d_protocol->steering_focus_count;
        if (n_focus < 0) n_focus = 0;
        if (n_focus > 64) n_focus = 64;
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

        // ── Stage 2 steering boost ──
        //
        // Extract the spike's primary residue id from the bit-packed
        // primary_residue_id field (low 16 bits hold the residue id; bits
        // 16-25 hold the 10-bit phase angle from compact_and_push). Then
        // walk the focus list and check if this spike's residue is in it.
        // If it is, multiply the threshold-reduction boost by
        // `(1 + steering_gain × weight)`.
        //
        // STEERING_GAIN = 2.0 means a synergy_fraction of 1.0 produces a
        // 3× boost on the spike's threshold reduction. This is moderate
        // enough not to overwhelm the natural physics signal but strong
        // enough to actually shift the cross-group divergence. Documented
        // as a tunable in the kernel comment so future calibration work
        // can swap it without searching.
        float steer_weight = 0.0f;
        if (n_focus > 0) {
            int spike_residue = spike.primary_residue_id & 0xFFFF;
            #pragma unroll 8
            for (int k = 0; k < 64; k++) {
                if (k >= n_focus) break;
                if (d_protocol->steering_focus_residues[k].residue_id == spike_residue) {
                    steer_weight = d_protocol->steering_focus_residues[k].weight;
                    break;
                }
            }
        }
        const float STEERING_GAIN = 2.0f;
        boost *= (1.0f + STEERING_GAIN * steer_weight);

        float base = base_thresholds[voxel_idx];
        float floor_val = base * (1.0f - max_reduction_fraction);
        float current = osc_thresholds[voxel_idx];
        float new_val = fmaxf(floor_val, current - boost);
        osc_thresholds[voxel_idx] = new_val;

        // ASC: track largest threshold reduction (hotspot candidate)
        float reduction = (base - new_val) / fmaxf(base, 1e-6f);
        if (reduction > max_reduction) {
            max_reduction = reduction;
            hotspot_residue = spike.primary_residue_id;
        }
    }

    *tail_ptr = head;

    // ── ASC Steering Hook ──
    // When coupling detects a hotspot (>20% threshold reduction at a voxel),
    // write steering fields. Director reads these next step and adjusts UV/temp.
    if (d_protocol != nullptr && max_reduction > 0.20f) {
        d_protocol->steering_uv_boost = 1.5f;          // 50% UV boost at hotspot
        d_protocol->steering_focus_residue = hotspot_residue;
        d_protocol->steering_flags |= 0x1;              // phase_lock flag
    }
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

// ═══════════════════════════════════════════════════════════════════════
// GATE 3: Device-side spike compaction + push
// ═══════════════════════════════════════════════════════════════════════
//
// Reads GpuSpikeEvent (92 bytes) directly from d_spike_events on GPU,
// extracts the 12 fields that RingSpikeEvent needs (48 bytes), and pushes
// directly into the ring buffer — ALL on GPU, zero CPU memcpy.
//
// This kernel replaces the CPU-side push_compacted() path:
//   Old: GPU spikes → memcpy_dtoh → CPU field extraction → memcpy_htod → push kernel
//   New: GPU spikes → this kernel → ring buffer (never leaves VRAM)

// GpuSpikeEvent layout — must match SpikeEvent in nhs_amber_fused.cu (92 bytes)
struct GpuSpikeEvent {
    int timestep;               // 0
    int voxel_idx;              // 4
    float pos_x, pos_y, pos_z;  // 8, 12, 16 (float3 = 3 contiguous floats)
    float intensity;            // 20
    int nearby_residues[8];     // 24-55
    int n_residues;             // 56
    int spike_source;           // 60
    float wavelength_nm;        // 64
    int aromatic_type;          // 68
    int aromatic_residue_id;    // 72
    float water_density;        // 76
    float vibrational_energy;   // 80
    int n_nearby_excited;       // 84
    float wd_change;            // 88
};  // total = 92 bytes

extern "C" __global__ void compact_and_push(
    const GpuSpikeEvent* __restrict__ spike_events, // [max_spikes] source (92 bytes each)
    const int* __restrict__ spike_count,              // [1] number of valid spikes
    unsigned char* __restrict__ ring_buffer,           // ring buffer storage
    unsigned int* __restrict__ head_ptr,               // monotonic write counter
    unsigned int* __restrict__ overflow_ptr,            // overflow counter
    unsigned int capacity,                             // ring buffer capacity
    const ProtocolState* __restrict__ d_protocol       // for phase angle computation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_spikes = *spike_count;
    if (tid >= n_spikes) return;

    // Read from the fat GpuSpikeEvent
    const GpuSpikeEvent* src = &spike_events[tid];

    // Compact into RingSpikeEvent (48 bytes)
    RingSpikeEvent compact;
    compact.timestep          = src->timestep;
    compact.voxel_idx         = src->voxel_idx;
    compact.x                 = src->pos_x;
    compact.y                 = src->pos_y;
    compact.z                 = src->pos_z;
    compact.intensity         = src->intensity;
    compact.vibrational_energy = src->vibrational_energy;
    compact.water_density     = src->water_density;
    compact.n_nearby_excited  = src->n_nearby_excited;
    compact.spike_source      = src->spike_source;
    compact.wavelength_nm     = src->wavelength_nm;

    // ── BIT-PACKED PHASE ANGLE + RESIDUE ID ──
    // Bits 0-15:  primary residue ID (0-65535)
    // Bits 16-25: quantized CCNS phase angle (0-1023 → 0-2π)
    // Bits 26-31: reserved
    //
    // Phase angle is read from ProtocolState at the INSTANT of this spike,
    // not derived from timestep. This survives ASC timeline steering.
    int residue_id = (src->n_residues > 0) ? src->nearby_residues[0] : -1;
    int residue_bits = residue_id & 0xFFFF; // 16-bit residue ID

    // Compute instantaneous phase from ProtocolState
    int phase_bits = 0;
    if (d_protocol != nullptr) {
        unsigned int step = d_protocol->current_step;
        unsigned int total = d_protocol->total_steps;
        if (total > 0) {
            // Quantize to 10 bits (0-1023)
            phase_bits = (int)((unsigned long long)step * 1023ULL / (unsigned long long)total);
            phase_bits = min(max(phase_bits, 0), 1023);
        }
    }

    // Pack: residue_id[15:0] | phase_angle[25:16]
    compact.primary_residue_id = residue_bits | (phase_bits << 16);

    // Atomic increment head to claim a slot
    unsigned int slot = atomicAdd(head_ptr, 1);
    unsigned int ring_idx = slot % capacity;

    // Write compact event to ring buffer
    RingSpikeEvent* dst = (RingSpikeEvent*)ring_buffer + ring_idx;
    *dst = compact;

    // Track overflow (informational — ring overwrites old data, which is correct)
    if (slot >= capacity) {
        atomicAdd(overflow_ptr, 1);
    }
}

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
