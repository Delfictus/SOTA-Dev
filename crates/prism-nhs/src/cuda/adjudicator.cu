// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Adjudicator — T1 Quantum-Photonic Bridge + T2 KL-Divergence
//                          (Blackwell sm_120, Anti-Greenfield-compliant,
//                           Total Function over all 32-bit f32 inputs)
// ═══════════════════════════════════════════════════════════════════════
//
// Two device-side primitives + one host-callable adjudicator step:
//
//   T1 — `prism_compute_quantum_weight(intensity_packed)` returns
//        Q_s = μ_01²(λ) · I, branchless via __constant__ LUT.
//
//   T2 — `prism_adjudicator_step_kernel` is a single-thread <<<1, 1, 0,
//        stream>>> launch that consumes one (relaxed, perturbed)
//        ContactShellTile pair and writes:
//        * current_divergence (Σ KL contributions)
//        * adjudication_code (0/1/2 per pre_rank::AdjudicationCode)
//        * start_clock / stop_clock (clock64 cycles for T4 telemetry)
//        * legacy_centroid_fallback (Path A: AABB midpoint of relaxed)
//
// Numerical defense (T2):
//   - C++ pre-check: `isfinite(p_raw) && isfinite(q_raw)` increments a
//     violation counter on dirty input.
//   - C++ defensive padding: `fmaxf(., epsilon)` clamps zeros / -Inf.
//     Inf is overwritten with epsilon before this if violation_count was
//     bumped, so the ratio stays finite for log evaluation.
//   - Inline PTX guard: `setp.nan.f32` + `setp.le.f32` zero out log_ratio
//     on any residual badness. Defense in depth.
//   - Final code: violation_count > 0 ⇒ Violation (2); else 3-σ threshold
//     decides Prune (0) / Construct (1).
//
// LUT derivation (T1, Anti-Greenfield-compliant):
//   μ_01²(λ) ∝ ε(λ_probe), with ε approximated via gaussian attenuation
//   around each chromophore's λ_max (per uv_bias.rs § Chromophore
//   Absorption Profiles). Normalized by ABSORPTION_NORMALIZATION
//   (= TRP_EXTINCTION_280 = 5600 from crate::config).
//
//     260 nm: PHE-dominant.
//             ε_260 ≈ PHE_EXTINCTION_280 · exp(−((260−258) / 10)²)
//                   ≈ 200 · exp(−0.04) ≈ 192.16
//             μ²    ≈ 192.16 / 5600 ≈ 0.0343
//
//     280 nm: TRP primary trigger.
//             ε_280 = TRP_EXTINCTION_280 = 5600
//             μ²    = 5600 / 5600 = 1.0000
//
//     305 nm: TRP gaussian tail.
//             ε_305 ≈ 5600 · exp(−((305−280) / 15)²) ≈ 5600 · 0.0621
//             μ²    ≈ 0.0621
//
//     320 nm: TRP control / baseline.
//             ε_320 ≈ 5600 · exp(−((320−280) / 15)²) ≈ 5600 · 0.000810
//             μ²    ≈ 0.000810
//
// ═══════════════════════════════════════════════════════════════════════

#include "adjudicator.cuh"  // transitively #includes so3_project.cuh (header fusion)
#include <cuda_runtime.h>
#include <math_constants.h>

// ════════════════════════════════════════════════════════════════════
// __constant__ memory LUT — populated by init_constants() at startup.
// Static initializer is the canonical CUDA pattern; no init kernel
// needed. nvcc emits a module-load-time copy from .nv_const to
// constant memory when the symbol is first referenced.
// ════════════════════════════════════════════════════════════════════

__constant__ float c_mu_01_sq[PRISM_MU_01_SQ_N] = {
    0.0343f,    // 260 nm — PHE-dominant
    1.0000f,    // 280 nm — TRP primary
    0.0621f,    // 305 nm — TRP gaussian tail
    0.000810f,  // 320 nm — TRP control / baseline
};

// ════════════════════════════════════════════════════════════════════
// T1 — prism_compute_quantum_weight (out-of-line definition)
// ════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float
prism_compute_quantum_weight(uint32_t intensity_packed) {
    uint32_t uv_code = prism_extract_uv_code(intensity_packed);
    float    intensity = prism_extract_intensity(intensity_packed);
    // Branchless LDC: the wavelength code indexes into c_mu_01_sq.
    return c_mu_01_sq[uv_code & 0x3u] * intensity;
}

// ════════════════════════════════════════════════════════════════════
// T2 — KL-divergence Adjudicator kernel
// ════════════════════════════════════════════════════════════════════

namespace {

// PTX-guarded log2 — defense-in-depth: even if fmaxf-clamped inputs
// produce a finite ratio, this guard catches any residual NaN / le0.
//
// On clean input: emits `lg2.approx.f32` (≈ 4-cycle hardware op).
// On NaN: log_ratio = 0.0
// On ratio ≤ 0: log_ratio = 0.0
__device__ __forceinline__ float prism_log2_with_guard(float ratio) {
    // PTX-correct guard: PTX `setp.nan.f32` is BINARY (returns true
    // when either operand is NaN); we pass `%1, %1` so the predicate
    // fires exactly when `ratio` is NaN. PTX also allows only ONE
    // guard predicate per instruction, so the "either-or" path is
    // expressed via `or.pred` into a combined predicate.
    float log_ratio;
    asm volatile (
        "{\n\t"
        " .reg .pred p_nan, p_le0, p_bad;\n\t"
        " setp.nan.f32 p_nan, %1, %1;\n\t"
        " setp.le.f32  p_le0, %1, 0f00000000;\n\t"
        " or.pred      p_bad, p_nan, p_le0;\n\t"
        " @p_bad  mov.f32        %0, 0f00000000;\n\t"
        " @!p_bad lg2.approx.f32 %0, %1;\n\t"
        "}"
        : "=f"(log_ratio)
        : "f"(ratio)
    );
    return log_ratio;
}

}  // anonymous namespace

/// Single-thread Adjudicator kernel.
/// Launches as <<<1, 1, 0, stream>>>. Captured into the F1
/// cudaGraphConditionalNode predicate evaluator. Race-free: the
/// __threadfence() between the divergence write and the
/// adjudication_code write enforces L2 visibility before the SWITCH
/// node read; the captured-graph topology orders this kernel before
/// the SWITCH read.
// Amendment 3.9 — SIMT Vector Adjudicator.
//
// Launch contract: `<<<1, n_clusters>>>` where n_clusters ≤ N_MAX_CLUSTERS
// (operator-mandated 64).  Each thread `cid = threadIdx.x` independently:
//   1. Loads its own (relaxed[cid], perturbed[cid]) ContactShellTile pair
//   2. Computes its own KL-divergence across the 6 geo SH bands
//   3. Applies the 3-σ noise-floor threshold to produce its local code
//   4. Honors the SISR force_prune_mask bit-`cid` if set
//   5. Writes its local code into `per_cluster_codes[cid]`
//   6. Participates in a __reduce_or_sync warp aggregation
// Block-level reduction across warps yields global_adjudication_summary,
// written by thread 0.  F1 SWITCH continues to read offset 52 — the byte
// is now a u32-OR over all per-thread codes (0=all-prune, 1=any-burst,
// 2=any-violation).
__global__ void prism_interferometric_adjudicator_step_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adjudicator
) {
    if (blockIdx.x != 0) return;
    const uint32_t cid = threadIdx.x;

    const ContactShellTile* relaxed_base   = adjudicator->relaxed_manifold_ptr;
    const ContactShellTile* perturbed_base = adjudicator->perturbed_manifold_ptr;

    // Defensive null-pointer guard. Thread 0 stamps Violation; all threads
    // skip per-cluster processing.
    if (relaxed_base == nullptr || perturbed_base == nullptr) {
        if (cid == 0) {
            adjudicator->current_divergence = 0.0f;
            __threadfence();
            adjudicator->global_adjudication_summary = PRISM_ADJ_VIOLATION;
        }
        return;
    }

    // Each thread indexes its OWN cluster pair via the contiguous tile
    // arrays (cfg.n_clusters slots, allocated by SiteManifestFfi::alloc_bytes).
    const ContactShellTile& relaxed   = relaxed_base[cid];
    const ContactShellTile& perturbed = perturbed_base[cid];

    // ─── Per-thread KL-divergence over the geometry plane SH bands l=0..5
    float    total_kl_div    = 0.0f;
    uint32_t violation_count = 0u;
    constexpr float epsilon  = 1.0e-7f;

    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        float p_raw = relaxed.geo_power_spectrum[l];
        float q_raw = perturbed.geo_power_spectrum[l];
        if (!isfinite(p_raw) || !isfinite(q_raw)) {
            violation_count++;
            if (!isfinite(p_raw)) p_raw = epsilon;
            if (!isfinite(q_raw)) q_raw = epsilon;
        }
        const float p = fmaxf(p_raw, epsilon);
        const float q = fmaxf(q_raw, epsilon);
        const float log_ratio = prism_log2_with_guard(p / q);
        total_kl_div += p * log_ratio;
    }

    uint32_t local_code;
    if (violation_count > 0u) {
        local_code = PRISM_ADJ_VIOLATION;
    } else {
        const float threshold =
            adjudicator->noise_floor_mu[0]
            + 3.0f * adjudicator->noise_floor_sigma[0];
        local_code = (total_kl_div > threshold) ? PRISM_ADJ_CONSTRUCT
                                                 : PRISM_ADJ_PRUNE;
    }

    // ─── G28 SISR per-cluster symmetry veto (Amendment 3.4 / 3.9) ──
    // bit-cid set ⇒ this cluster failed bilateral-truth check.
    if (adjudicator->force_prune_mask != nullptr) {
        const uint64_t mask = *adjudicator->force_prune_mask;
        if ((mask & (1ull << cid)) != 0ull) {
            local_code = PRISM_ADJ_PRUNE;
        }
    }

    // ─── Vector output: write per-cluster code ──────────────────────
    if (adjudicator->per_cluster_codes != nullptr) {
        adjudicator->per_cluster_codes[cid] = local_code;
    }

    // ─── Warp-level OR reduction → block-level reduction → global summary
    // OR-reducing {0,1,2} preserves max (since 0|x = x, 1|2 = 3 — not in
    // valid set; we clamp to PRISM_ADJ_VIOLATION below).
    const uint32_t warp_summary = __reduce_or_sync(0xFFFFFFFFu, local_code);
    __shared__ uint32_t warp_buf[2];   // up to 64 threads = 2 warps
    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t warp_id = threadIdx.x >> 5;
    if (lane == 0u && warp_id < 2u) {
        warp_buf[warp_id] = warp_summary;
    }
    __syncthreads();

    if (cid == 0) {
        uint32_t global = warp_buf[0] | warp_buf[1];
        // Clamp to the canonical 3-value enum: any violation (2 set) →
        // PRISM_ADJ_VIOLATION; otherwise any burst (1) → CONSTRUCT;
        // else PRUNE.  Bitwise OR of {0,1,2} can yield 3 (=1|2), which
        // we resolve to VIOLATION (the higher-severity outcome).
        if ((global & 0x2u) != 0u) global = PRISM_ADJ_VIOLATION;
        else if ((global & 0x1u) != 0u) global = PRISM_ADJ_CONSTRUCT;
        else global = PRISM_ADJ_PRUNE;

        // Stamp current_divergence with thread 0's value (cluster 0 KL).
        // Vector codes carry per-cluster KL implicitly via per_cluster_codes;
        // current_divergence remains a single-cluster diagnostic for legacy
        // T4 telemetry consumers.
        adjudicator->current_divergence = total_kl_div;
        __threadfence();
        adjudicator->global_adjudication_summary = global;

        // Anti-Greenfield § 6.2 legacy_centroid_fallback — cluster 0 only.
        adjudicator->legacy_centroid_fallback[0] =
            0.5f * (relaxed.aabb_min[0] + relaxed.aabb_max[0]);
        adjudicator->legacy_centroid_fallback[1] =
            0.5f * (relaxed.aabb_min[1] + relaxed.aabb_max[1]);
        adjudicator->legacy_centroid_fallback[2] =
            0.5f * (relaxed.aabb_min[2] + relaxed.aabb_max[2]);
    }
}

/// Zero-fill kernel — initialises a freshly F2-pool-allocated
/// InterferometricAdjudicatorFfi to all-zero state. Called from
/// `prism_interferometric_adjudicator_create` after the pool alloc
/// returns. <<<1, 1, 0, stream>>> single-thread launch.
__global__ void prism_interferometric_adjudicator_zero_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        adj->noise_floor_mu[i] = 0.0f;
        adj->noise_floor_sigma[i] = 0.0f;
    }
    adj->current_divergence = 0.0f;
    adj->global_adjudication_summary = PRISM_ADJ_PRUNE;
    adj->relaxed_manifold_ptr = nullptr;
    adj->perturbed_manifold_ptr = nullptr;
    adj->start_clock = 0u;
    adj->stop_clock = 0u;
    adj->legacy_centroid_fallback[0] = 0.0f;
    adj->legacy_centroid_fallback[1] = 0.0f;
    adj->legacy_centroid_fallback[2] = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        adj->_reserved[i] = 0u;
    }
}

/// Noise-floor update kernel — Welford-style running mean+stddev over
/// the latest "Cool" frame's 6-band power spectrum. Single-thread.
__global__ void prism_interferometric_adjudicator_noise_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj,
    const float* __restrict__ cool_power_spectrum
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // First-pass running update: μ_new = 0.5·μ_old + 0.5·sample.
    // This is an EMA, not strict Welford, to keep the noise-floor
    // adaptive on a per-frame timescale. σ tracks abs-deviation as a
    // rough proxy for stddev — a future enhancement is the proper
    // Welford recurrence M2 += (sample−μ_old)·(sample−μ_new).
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float sample = cool_power_spectrum[l];
        if (!isfinite(sample)) continue;  // skip dirty samples
        const float mu_old = adj->noise_floor_mu[l];
        const float mu_new = 0.5f * mu_old + 0.5f * sample;
        const float dev    = fabsf(sample - mu_new);
        const float sig_old = adj->noise_floor_sigma[l];
        const float sig_new = 0.5f * sig_old + 0.5f * dev;
        adj->noise_floor_mu[l] = mu_new;
        adj->noise_floor_sigma[l] = sig_new;
    }
}

/// Link-probe kernel — returns a sentinel via a stream-ordered single
/// write. Used by the Rust-side test to detect FFI / build-system
/// drift.
__global__ void prism_interferometric_adjudicator_probe_kernel(uint32_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out = 0xAD31u;
}

// ════════════════════════════════════════════════════════════════════
// T3 — ASC Boundary Repulsion Tensor kernel
// ════════════════════════════════════════════════════════════════════

/// Per-atom outward-repulsion kernel. Reads the relaxed manifold's
/// AABB, computes the centroid X_c, and atomicAdds an outward force
/// F_i = α · Δ_AB · (x_i − X_c) into the existing fused_engine
/// d_forces buffer. Anti-Greenfield § 2.1 surgical extension.
///
/// Pointer-stability invariant: every device pointer read here MUST
/// be allocated outside the captured-graph WHILE region. The F2
/// pool's UINT64_MAX release threshold guarantees that.
///
/// Branchless cluster-membership: `d_atom_in_cluster[i]` is `0` or `1`;
/// the atomicAdds are gated by an integer multiply against that mask
/// rather than an early-return branch, eliminating warp divergence
/// when a warp straddles the cluster boundary.
__global__ void prism_asc_apply_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    float*          __restrict__ d_forces,
    const float*    __restrict__ d_atom_positions,
    const uint32_t* __restrict__ d_atom_in_cluster,
    int32_t         n_atoms,
    float           steering_gain_alpha
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    // Defensive: only fire on Construct (code 1). The F1 SWITCH
    // already routed us here only when global_adjudication_summary == 1
    // (any cluster went burst), but read-back guards against
    // pointer-stability bugs upstream.  Amendment 3.9: vector path
    // promotes single-cluster scalar to OR-reduced summary.
    const uint32_t code = adj->global_adjudication_summary;
    if (code != PRISM_ADJ_CONSTRUCT) return;

    const ContactShellTile* tile = adj->relaxed_manifold_ptr;
    if (tile == nullptr) return;

    const float xc0 = 0.5f * (tile->aabb_min[0] + tile->aabb_max[0]);
    const float xc1 = 0.5f * (tile->aabb_min[1] + tile->aabb_max[1]);
    const float xc2 = 0.5f * (tile->aabb_min[2] + tile->aabb_max[2]);

    const float xi0 = d_atom_positions[i * 3 + 0];
    const float xi1 = d_atom_positions[i * 3 + 1];
    const float xi2 = d_atom_positions[i * 3 + 2];

    const float dx = xi0 - xc0;
    const float dy = xi1 - xc1;
    const float dz = xi2 - xc2;

    // Branchless mask multiply — keeps every lane in lockstep.
    const float mask  = static_cast<float>(d_atom_in_cluster[i]);
    const float scale = steering_gain_alpha * adj->current_divergence * mask;

    atomicAdd(&d_forces[i * 3 + 0], scale * dx);
    atomicAdd(&d_forces[i * 3 + 1], scale * dy);
    atomicAdd(&d_forces[i * 3 + 2], scale * dz);
}

// ════════════════════════════════════════════════════════════════════
// T4 — clock64 pipeline timing bookends (PTX-direct)
// ════════════════════════════════════════════════════════════════════

/// Returns the SM cycle counter via the canonical `mov.u64 %0, %clock64`
/// PTX instruction. One PTX instruction; <1ns overhead. Equivalent to
/// the CUDA `clock64()` intrinsic but emitted as inline PTX per the
/// operator's directive § 3.1.
__device__ __forceinline__ uint64_t prism_clock64_ptx() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
}

__global__ void prism_pipeline_clock_start_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    adj->start_clock = prism_clock64_ptx();
}

__global__ void prism_pipeline_clock_stop_kernel(
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    adj->stop_clock = prism_clock64_ptx();
}

// ════════════════════════════════════════════════════════════════════
// Host-callable launchers (extern "C") — match Rust FFI declarations
// in interferometric_adjudicator.rs::ffi.
// ════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_interferometric_adjudicator_link_probe(void) {
    // Allocate a 4-byte pinned host buffer, run the probe, sync, return.
    // This pattern mirrors vram_pool's link probe (which uses a static
    // device buffer); we use a heap one-shot here since the probe is
    // one-time-only at startup.
    uint32_t* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, sizeof(uint32_t));
    if (err != cudaSuccess) return 0xDEADu;
    prism_interferometric_adjudicator_probe_kernel<<<1, 1, 0, 0>>>(d_out);
    cudaDeviceSynchronize();
    uint32_t value = 0;
    err = cudaMemcpy(&value, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return (err == cudaSuccess) ? value : 0xDEADu;
}

int prism_interferometric_adjudicator_init_constants(void* stream_v) {
    // The __constant__ symbol c_mu_01_sq is statically initialised at
    // module load (see file-scope init above). No host-side memcpy is
    // required; this function exists for explicit ABI symmetry with
    // future LUTs that need runtime calibration.
    //
    // Returns cudaSuccess unconditionally — callers can rely on this
    // being a stable no-op while the LUT is compile-time-constant.
    (void)stream_v;
    return static_cast<int>(cudaSuccess);
}

int prism_interferometric_adjudicator_create(
    void* pool_v, void* stream_v,
    InterferometricAdjudicatorFfi** out_ptr
) {
    // F2 pool allocation via the existing vram_pool wrapper. We do
    // NOT cudaMalloc — that's host-syncing and breaks captured-graph
    // replay (Anti-Greenfield § 3.2 forbids it for live state).
    extern int prism_vram_pool_alloc_async(void*, uint64_t, void*, void**);
    void* raw = nullptr;
    int rc = prism_vram_pool_alloc_async(
        pool_v,
        static_cast<uint64_t>(sizeof(InterferometricAdjudicatorFfi)),
        stream_v, &raw);
    if (rc != static_cast<int>(cudaSuccess)) return rc;
    if (raw == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);

    // Zero-init on the same stream — stream-ordered ensures the alloc
    // is visible before the kernel's writes.
    auto* adj = static_cast<InterferometricAdjudicatorFfi*>(raw);
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_zero_kernel<<<1, 1, 0, s>>>(adj);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);

    *out_ptr = adj;
    return static_cast<int>(cudaSuccess);
}

int prism_interferometric_adjudicator_destroy(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    extern int prism_vram_pool_free_async(void*, void*);
    return prism_vram_pool_free_async(static_cast<void*>(adj), stream_v);
}

int prism_interferometric_adjudicator_step(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    // Amendment 3.9 — SIMT vector launch. blockDim.x = N_MAX_CLUSTERS = 64
    // (operator mandate; matches PipelineConfig.n_clusters bake-in).
    // Each thread cid computes cluster cid's KL-divergence; thread 0
    // writes the OR-reduced global_adjudication_summary.  Single warp
    // pair (2 warps × 32 threads = 64 threads), warp_buf shared mem
    // sized to 2 entries.
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_step_kernel<<<1, 64, 0, s>>>(adj);
    return static_cast<int>(cudaGetLastError());
}

int prism_interferometric_adjudicator_update_noise_floor(
    InterferometricAdjudicatorFfi* adj,
    const float* cool_power_spectrum,
    void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_interferometric_adjudicator_noise_kernel<<<1, 1, 0, s>>>(
        adj, cool_power_spectrum);
    return static_cast<int>(cudaGetLastError());
}

// ────────────────────────────────────────────────────────────────────
// T3 / T4 host launchers
// ────────────────────────────────────────────────────────────────────

int prism_asc_apply(
    const InterferometricAdjudicatorFfi* adj,
    float* d_forces,
    const float* d_atom_positions,
    const uint32_t* d_atom_in_cluster,
    int32_t n_atoms,
    float steering_gain_alpha,
    void* stream_v
) {
    if (n_atoms <= 0) return static_cast<int>(cudaSuccess);
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    constexpr int BLOCK = 256;
    const int grid = (n_atoms + BLOCK - 1) / BLOCK;
    prism_asc_apply_kernel<<<grid, BLOCK, 0, s>>>(
        adj, d_forces, d_atom_positions, d_atom_in_cluster,
        n_atoms, steering_gain_alpha);
    return static_cast<int>(cudaGetLastError());
}

int prism_pipeline_clock_start(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_pipeline_clock_start_kernel<<<1, 1, 0, s>>>(adj);
    return static_cast<int>(cudaGetLastError());
}

int prism_pipeline_clock_stop(
    InterferometricAdjudicatorFfi* adj, void* stream_v
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_v);
    prism_pipeline_clock_stop_kernel<<<1, 1, 0, s>>>(adj);
    return static_cast<int>(cudaGetLastError());
}

// ────────────────────────────────────────────────────────────────────
// DAG-COND-WIRE host launchers
// ────────────────────────────────────────────────────────────────────

const uint32_t* prism_get_adjudication_code_devptr(
    const InterferometricAdjudicatorFfi* adj
) {
    // Field is at byte offset 52 within the 128-aligned struct.
    // Pointer is stable for the campaign per F2 pool's
    // ReleaseThreshold = UINT64_MAX guarantee. Always 4-byte aligned;
    // never crosses a 128-byte L1 sector boundary (the entire
    // 128-byte struct fits in one sector).
    return &adj->global_adjudication_summary;
}

}  // extern "C"

// ────────────────────────────────────────────────────────────────────
// F1 SWITCH translation kernel (CUDA 12.4+: cudaGraphSetConditional)
// ────────────────────────────────────────────────────────────────────
//
// Conditional-node support requires CUDA 12.4+. Older toolkits define
// neither `cudaGraphConditionalHandle` nor `cudaGraphSetConditional`,
// so we feature-gate via the toolkit version macro to keep the
// archive linkable on transitional dev environments.
//
// `cudaGraphSetConditional` is documented in:
//   CUDA Programming Guide § 3.2.8.7.4 "Conditional Node Setting"
// Available since CUDA 12.4 (toolkit version 12040). Project ships on
// CUDA 13.x (per build.rs and the e9628229 F2-pool commit), so this
// path is always live in production builds.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

__global__ void prism_adj_set_conditional_kernel(
    cudaGraphConditionalHandle handle,
    const InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Read the memory-resident code written by T2's adjudicator
    // kernel. The captured-graph dependency edge from T2 → here
    // guarantees stream-order visibility (operator mandate § 2.3).
    const uint32_t code = adj->global_adjudication_summary;
    // Forward to the F1 SWITCH handle. The 3 valid values
    // (PRUNE=0, CONSTRUCT=1, VIOLATION=2) map to the SWITCH's
    // 3 sub-graph branches.
    cudaGraphSetConditional(handle, code);
}

extern "C" int prism_adj_set_conditional(
    uint64_t handle_v,
    const InterferometricAdjudicatorFfi* adj,
    void* stream_v
) {
    // cudaGraphConditionalHandle is `typedef unsigned long long`
    // (driver_types.h:3229), so the value is a direct copy.
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    prism_adj_set_conditional_kernel<<<1, 1, 0, stream>>>(handle, adj);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

extern "C" int prism_adj_set_conditional(
    uint64_t /*handle_v*/,
    const InterferometricAdjudicatorFfi* /*adj*/,
    void* /*stream_v*/
) {
    // Pre-CUDA-12.4 toolchain: conditional nodes are unavailable.
    // Return cudaErrorNotSupported so callers can fail loudly.
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040

// ════════════════════════════════════════════════════════════════════
// V2 C-ABI BYPASS — Native F1 SWITCH forge
// ════════════════════════════════════════════════════════════════════
//
// Single-call host-side helper that:
//   1. Creates a cudaGraphConditionalHandle bound to `graph`.
//   2. Populates cudaGraphNodeParams for a 3-way SWITCH conditional node.
//   3. Adds the conditional node to the graph.
//   4. Locks the happens-before edge: adjudicator_node → conditional_node.
//
// Requires CUDA 12.6+ for cudaGraphCondTypeSwitch. Production toolchain
// is CUDA 13.x (CUDART_VERSION = 13020 per system headers), so the
// path below is always live; the gate exists for transitional
// dev-environment safety.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

extern "C" int prism_wire_f1_switch_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  adjudicator_node,
    const uint32_t*  predicate_dev_ptr,
    cudaGraphNode_t* out_conditional_node
) {
    if (graph == nullptr || out_conditional_node == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Step 1: Create the F1 SWITCH handle bound to this graph.
    // defaultLaunchValue = 0 ⇒ Prune by default if the bridge kernel
    // hasn't updated the handle (e.g., first frame before any
    // adjudicator step has run).
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle, graph,
        /*defaultLaunchValue=*/ 0u,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // `predicate_dev_ptr` is wiring metadata — it is the device
    // address of `adj->adjudication_code` (offset 52 inside the
    // 128-aligned InterferometricAdjudicatorFfi). The runtime
    // conditional-handle API is handle-based, not pointer-based, so
    // the pointer is not consumed directly here; the downstream
    // `prism_adj_set_conditional_kernel` reads from it to forward
    // the value into the handle via `cudaGraphSetConditional`.
    // Retained in the function signature for wiring-contract clarity
    // and forward-compatibility with a possible future device-
    // pointer-conditional API.
    (void)predicate_dev_ptr;

    // Step 2: Populate cudaGraphNodeParams for the conditional node.
    // The 3-way SWITCH maps directly to the F1 routing table:
    //   Case 0 → Prune sub-graph
    //   Case 1 → Construct (bisimulation + ASC) sub-graph
    //   Case 2 → Violation (PTX trap) sub-graph
    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeSwitch;
    nodeParams.conditional.size     = 3u;
    // ctx = NULL ⇒ runtime fills in from the current device context
    // at node-creation time. Avoids dragging the driver-API CUcontext
    // into this TU.
    nodeParams.conditional.ctx      = nullptr;

    // Step 3: Add the conditional node. Pass NULL for both
    // pDependencies and dependencyData (CUDA 13.x signature) — we
    // install the dependency edge in step 4 to make the wiring
    // contract explicit + match the operator's numbered ordering.
    //
    // CUDA 13.x cudaGraphAddNode signature:
    //   cudaError_t cudaGraphAddNode(
    //       cudaGraphNode_t* pGraphNode,
    //       cudaGraph_t graph,
    //       const cudaGraphNode_t* pDependencies,
    //       const cudaGraphEdgeData* dependencyData,   ← NEW in 13.x
    //       size_t numDependencies,
    //       cudaGraphNodeParams* nodeParams
    //   );
    err = cudaGraphAddNode(
        out_conditional_node, graph,
        /*pDependencies=*/   nullptr,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ 0u,
        &nodeParams);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Step 4: Gate G19 — happens-before edge.
    // CUDA 13.x cudaGraphAddDependencies signature:
    //   cudaError_t cudaGraphAddDependencies(
    //       cudaGraph_t graph,
    //       const cudaGraphNode_t* from,
    //       const cudaGraphNode_t* to,
    //       const cudaGraphEdgeData* edgeData,         ← NEW in 13.x
    //       size_t numDependencies
    //   );
    // Both from/to are 1-element arrays via &adjudicator_node and
    // out_conditional_node respectively. edgeData = nullptr selects
    // the default cudaGraphEdgeTypeProgram + cudaGraphFullPort
    // edge attributes (the ordinary "happens-before" edge).
    err = cudaGraphAddDependencies(
        graph,
        /*from=*/             &adjudicator_node,
        /*to=*/               out_conditional_node,
        /*edgeData=*/         nullptr,
        /*numDependencies=*/  1u);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaSuccess);
}

#else  // CUDART_VERSION < 12060

extern "C" int prism_wire_f1_switch_ffi(
    cudaGraph_t      /*graph*/,
    cudaGraphNode_t  /*adjudicator_node*/,
    const uint32_t*  /*predicate_dev_ptr*/,
    cudaGraphNode_t* /*out_conditional_node*/
) {
    // Pre-CUDA-12.6: cudaGraphCondTypeSwitch (3-way SWITCH) is not
    // available. Caller should fall back to a chain of If-conditional
    // nodes if 2-way branching is acceptable, or upgrade the toolkit.
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

// ════════════════════════════════════════════════════════════════════
// T7 — Noise-floor calibration writeback
// ════════════════════════════════════════════════════════════════════
//
// Two stream-ordered cudaMemcpyAsync(H2D) into the adjudicator
// struct's noise-floor fields. No kernel launch — the runtime API
// itself is captured-graph-compatible (memcpy-async nodes).
//
// Field layout (per CSR-C invariants in interferometric_adjudicator.rs):
//   noise_floor_mu     [f32; 6]  offset  0..24    24 B
//   noise_floor_sigma  [f32; 6]  offset 24..48    24 B

extern "C" int prism_adj_set_noise_floor_constants(
    InterferometricAdjudicatorFfi* adj,
    const float*                   mu_host,
    const float*                   sigma_host,
    void*                          stream_v
) {
    if (adj == nullptr || mu_host == nullptr || sigma_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);

    // Step 1: μ[6] → adj->noise_floor_mu (struct offset 0).
    cudaError_t err = cudaMemcpyAsync(
        &adj->noise_floor_mu[0],
        mu_host,
        sizeof(float) * 6,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Step 2: σ[6] → adj->noise_floor_sigma (struct offset 24).
    // Same stream ⇒ ordered after the μ copy.
    err = cudaMemcpyAsync(
        &adj->noise_floor_sigma[0],
        sigma_host,
        sizeof(float) * 6,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaSuccess);
}
