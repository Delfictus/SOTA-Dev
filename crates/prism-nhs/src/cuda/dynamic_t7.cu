// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / Dynamic T7 — substrate-aware noise-floor calibration (impl)
// ═══════════════════════════════════════════════════════════════════════════

#include "dynamic_t7.cuh"
#include "adjudicator.cuh"  // pulls in InterferometricAdjudicatorFfi
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage

// ─── Capture: read adj->current_divergence, accumulate into acc[idx] ──────
//
// Single thread.  Atomic increment of idx with saturation at N_SAMPLES so
// the buffer never overflows once calibration is complete.  Subsequent
// launches no-op (idx stays at N_SAMPLES; acc[N_SAMPLES] write is gated).
extern "C"
__global__ void prism_dynamic_t7_capture_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    float*    __restrict__ acc,
    uint32_t* __restrict__ idx
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Atomic claim of next slot.  atomicAdd returns the OLD value before
    // increment, so i is the slot we own.
    uint32_t i = atomicAdd(idx, 1u);
    if (i < PRISM_DYNT7_N_SAMPLES) {
        // adj->current_divergence is written by the Adjudicator step
        // kernel earlier in the captured-graph epoch.  __threadfence()
        // inside that kernel + captured-graph dependency edge guarantee
        // L2 visibility here.
        acc[i] = adj->current_divergence;
    } else {
        // Saturate idx at N_SAMPLES.  atomicMin clamps any over-shoot
        // (multiple captures in same launch would never happen with
        // the single-thread guard, but defensively cap anyway).
        atomicMin(idx, PRISM_DYNT7_N_SAMPLES);
    }
}

// ─── Reduce: mean + stddev across acc[0..min(idx, N)] ─────────────────────
//
// Single block, 256 threads.  Two-pass shared-memory reduction:
//   Pass 1: sum → mean
//   Pass 2: variance sum → stddev (population std with Bessel correction)
extern "C"
__global__ void prism_dynamic_t7_reduce_kernel(
    const float*    __restrict__ acc,
    const uint32_t* __restrict__ idx,
    float*          __restrict__ stats
) {
    __shared__ float sdata[256];
    const uint32_t tid = threadIdx.x;
    const uint32_t n   = min(*idx, PRISM_DYNT7_N_SAMPLES);

    if (n == 0u) {
        if (tid == 0u) { stats[0] = 0.0f; stats[1] = 0.0f; }
        return;
    }

    // ── Pass 1: sum ────────────────────────────────────────────────
    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += 256u) {
        sum += acc[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction (sum, descending stride).
    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);
    __syncthreads();

    // ── Pass 2: variance ──────────────────────────────────────────
    float var_sum = 0.0f;
    for (uint32_t i = tid; i < n; i += 256u) {
        const float d = acc[i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0u) {
        // Bessel-corrected sample stddev (n-1 denominator); fall back to
        // n=1 case to avoid div-by-zero on tiny windows.
        const float denom = (n > 1u) ? static_cast<float>(n - 1u) : 1.0f;
        const float variance = sdata[0] / denom;
        stats[0] = mean;
        stats[1] = sqrtf(fmaxf(variance, 0.0f));
    }
}

// ─── Apply: write substrate-derived priors to adj noise floor ────────────
//
// Single thread.  Gated on n >= N_MIN; below that, the kernel is a no-op
// and the FFI struct retains the constants written by apply_t7_calibration
// or set_noise_floor_constants at build time.
extern "C"
__global__ void prism_dynamic_t7_apply_kernel(
    const float*    __restrict__ stats,
    const uint32_t* __restrict__ idx,
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t n = *idx;
    if (n < PRISM_DYNT7_N_MIN) return;

    // The Adjudicator's threshold formula reads ONLY band 0:
    //   threshold = noise_floor_mu[0] + 3 * noise_floor_sigma[0]
    // Other 5 bands stay at whatever apply_t7_calibration set them to;
    // future enhancement: per-band Δ_AB samples + per-band priors.
    adj->noise_floor_mu[0]    = stats[0];
    adj->noise_floor_sigma[0] = stats[1];
}

// ─── C-ABI host launcher: enqueues all three kernels in order ─────────────

extern "C"
int prism_dynamic_t7_launch(
    const void* adj,
    void*       acc_dev,
    void*       idx_dev,
    void*       stats_dev,
    void*       stream
) {
    if (adj == nullptr || acc_dev == nullptr ||
        idx_dev == nullptr || stats_dev == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const InterferometricAdjudicatorFfi* adj_ptr =
        static_cast<const InterferometricAdjudicatorFfi*>(adj);

    // Capture.
    prism_dynamic_t7_capture_kernel<<<1, 1, 0, s>>>(
        adj_ptr,
        static_cast<float*>(acc_dev),
        static_cast<uint32_t*>(idx_dev)
    );

    // Reduce (block-wide).
    prism_dynamic_t7_reduce_kernel<<<1, 256, 0, s>>>(
        static_cast<const float*>(acc_dev),
        static_cast<const uint32_t*>(idx_dev),
        static_cast<float*>(stats_dev)
    );

    // Apply.  Cast away const for the apply kernel (writes to adj's μ/σ).
    prism_dynamic_t7_apply_kernel<<<1, 1, 0, s>>>(
        static_cast<const float*>(stats_dev),
        static_cast<const uint32_t*>(idx_dev),
        const_cast<InterferometricAdjudicatorFfi*>(adj_ptr)
    );

    return static_cast<int>(cudaGetLastError());
}
