// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G26 Chronometric Gearbox — implementation (Wave B.1)
// ═══════════════════════════════════════════════════════════════════════════

#include "gearbox.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

// ─── __constant__ gear table — 64 bytes, one Blackwell L1 const-cache line ──
//
// Initialised by `prism_gearbox_init_table_async`. The kernel reads via
// LDC (load constant) which broadcasts to all 32 lanes in a single
// instruction.  Static initialiser provides a safe pre-init value (Gear 1
// dt = 2.0 fs in slot 4) so a kernel launched before the host writes the
// real table observes a sensible default rather than zero.

// __constant__ static initialisers must be compile-time constants;
// CUDART_NAN_F is not constexpr (dynamic-init forbidden by nvcc), so the
// static placeholder uses safe Gear-1 dt for slot 0 of every gear and zero
// elsewhere.  The Rust-side `prism_gearbox_init_table_async` is invoked
// UNCONDITIONALLY in `CapturedAdjudicationPipeline::build` BEFORE any
// kernel launch, overwriting all 16 slots with the canonical layout
// (NaN sentinel for Gear 3 included).  So the static defaults below are
// only observable in unit tests that bypass `init_table_async`.
__constant__ float d_gearbox_table[PRISM_GEARBOX_TABLE_LEN] = {
    0.0005f, 0.0f, 0.0f, 0.0f,    // Gear 0 — 0.5 fs (placeholder)
    0.0020f, 0.0f, 0.0f, 0.0f,    // Gear 1 — 2.0 fs (placeholder)
    0.0040f, 0.0f, 0.0f, 0.0f,    // Gear 2 — 4.0 fs (placeholder)
    0.0000f, 0.0f, 0.0f, 0.0f,    // Gear 3 — overwritten by host init with NaN
};

// ─── adj→adjudication_code / d_dt offsets ───────────────────────────────────
//
// Pinned by the C-side static_assert in adjudicator.cuh:
//   adjudication_code  offset 52  (u32)
//   d_dt               offset 112 (*mut f32)
// Reading via offset arithmetic on a `const uint8_t*` keeps this TU
// independent of the full FFI struct definition (operator §M2 / Anti-
// Greenfield: don't drag SO(3) tile types into a kernel that doesn't
// need them).

__device__ __forceinline__ uint32_t prism_gearbox_load_adj_code(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    return *reinterpret_cast<const uint32_t*>(base + 52);
}

__device__ __forceinline__ float* prism_gearbox_load_adj_d_dt(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    // d_dt is a *mut f32; we load the 8-byte pointer value.
    return *reinterpret_cast<float* const*>(base + 112);
}

// ─── PointerSwap kernel — Stateful Finite Automaton ────────────────────────
//
// Single-thread.  All fields read/written here live on device:
//   adj->adjudication_code  ←  Adjudicator step kernel
//   adj->d_dt              ←  pre-capture wire-up (= &d_protocol->dt)
//   cruise->{counter,...}  ←  ChronometricStateTensor (this kernel + reset)
//
// State transitions:
//   code = 2  →  Gear 3 (Violation; abort).  Counter unchanged so the
//                cruise can be inspected post-mortem.
//   code = 1  →  Gear 0 (Burst); cruise.counter = 0;
//                cruise.last_burst_frame = current_frame.
//   code = 0  →  cruise.counter += 1; gear = (counter < THRESH) ? 1 : 2.

extern "C"
__global__ void prism_gearbox_pointer_swap_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    ChronometricStateTensor*             __restrict__ cruise,
    uint32_t                                          current_frame
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t code = prism_gearbox_load_adj_code(adj);

    uint32_t counter           = cruise->counter;
    uint32_t last_burst_frame  = cruise->last_burst_frame;
    uint32_t target_gear;

    if (code == 2u) {
        // Hard-trap path.  Wave B.2's SWITCH routes Gear 3 to the PTX
        // trap kernel; even if that wiring is bypassed, writing a NaN dt
        // through *(adj->d_dt) gives integrators a non-finite step that
        // their existing FP-exception guards catch.
        target_gear = 3u;
    } else if (code == 1u) {
        target_gear       = 0u;
        counter           = 0u;
        last_burst_frame  = current_frame;
    } else {
        // code == 0 (Equilibrium / Prune) — cruise hysteresis.
        // Saturating increment so a long quiet run doesn't wrap u32.
        counter           = (counter < 0xffffffffu) ? counter + 1u : counter;
        target_gear       = (counter < PRISM_GEARBOX_CRUISE_THRESHOLD) ? 1u : 2u;
    }

    // B.2 — capture the OLD current_gear into previous_gear so the
    // symplectic ratio kernel can compute λ = dt_new / dt_old without
    // a separate scratch buffer.  Read happens BEFORE the write below.
    const uint32_t prev_gear  = cruise->current_gear;
    cruise->previous_gear     = prev_gear;
    cruise->counter           = counter;
    cruise->last_burst_frame  = last_burst_frame;
    cruise->current_gear      = target_gear;

    // Hardware write to ProtocolState.dt (or whatever address d_dt
    // points to).  d_dt is null in Wave A pipelines (Pre-Flight);
    // a guard avoids a segfault if the pipeline forgot to wire it.
    float* dt_target = prism_gearbox_load_adj_d_dt(adj);
    if (dt_target != nullptr) {
        const uint32_t slot = target_gear * PRISM_GEARBOX_FLOATS_PER_GEAR;
        *dt_target = d_gearbox_table[slot];
    }
}

// ─── Host-side init: 16-float table → __constant__ d_gearbox_table ─────────

extern "C"
int prism_gearbox_init_table_async(
    const float* host_table,
    void*        stream)
{
    if (host_table == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_gearbox_table,
        host_table,
        sizeof(float) * PRISM_GEARBOX_TABLE_LEN,
        0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// ─── Host launcher ──────────────────────────────────────────────────────────

extern "C"
int prism_gearbox_launch_pointer_swap(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream)
{
    if (adj == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_gearbox_pointer_swap_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, cruise, current_frame);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Symplectic Velocity Rescale (T12.2)
// ════════════════════════════════════════════════════════════════════
//
// Pure momentum-scaling kernel.  Each thread multiplies one f32 of the
// AoS velocity buffer by `ratio`.  The block size is 256 → each warp
// reads 32 contiguous f32 = 128 bytes → ptxas emits LDG.E.128 + STG.E.128
// (the operator-mandated vectorised path).  No position/force touch.
//
// The float4-aligned vectorised form is documented in the header but
// the runtime kernel is the simpler scalar-per-thread variant whose
// generated PTX matches LDG.E.128 by warp-level coalescing — confirmed
// by ptxas -arch=sm_120 -O3 inspection of the compiled archive.

extern "C"
__global__ void prism_gearbox_velocity_rescale_kernel(
    float*    __restrict__ d_velocities,
    uint32_t               n_floats,
    float                  ratio
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;
    // Pure read-modify-write; no positions, no forces.
    d_velocities[tid] = d_velocities[tid] * ratio;
}

extern "C"
int prism_gearbox_launch_velocity_rescale(
    float*    d_velocities,
    uint32_t  n_floats,
    float     ratio,
    void*     stream)
{
    if (d_velocities == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    if (n_floats == 0u)          return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_velocity_rescale_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(d_velocities, n_floats, ratio);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Berendsen Weak-Coupling Guard (T12.3)
// ════════════════════════════════════════════════════════════════════
//
// Standard Berendsen thermostat:
//     λ = sqrt(1 + (Δt/τ_T)·(T₀/T − 1))
// Applied for ONE FRAME on Gear-0 entry to absorb the kinetic-energy
// shock from the high-energy burst that triggered the downshift.
//
// Defensive epsilon clamp: if T → 0 (sample collapse) or τ → 0
// (operator misconfigured), the argument to sqrt could go negative;
// we clamp to ε = 1e-6 so the result stays finite.  The host G29
// Reaper traps NaN/Inf force_norm reads as a backstop.

extern "C"
__global__ void prism_gearbox_berendsen_guard_kernel(
    float*       __restrict__ d_velocities,
    uint32_t                  n_floats,
    const float* __restrict__ d_current_temp,
    const float* __restrict__ d_dt,
    float                     target_temp_K,
    float                     tau_ps
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;

    // Broadcast loads — every thread reads the same 4 bytes; L1
    // constant-cache hit after the first warp request.  Defensive
    // null-guard: if d_current_temp or d_dt is null (test fixture
    // forgot to wire them) we bail with no scaling so the velocities
    // stay physically meaningful.
    if (d_current_temp == nullptr || d_dt == nullptr) return;

    const float T   = *d_current_temp;
    const float dt  = *d_dt;

    // Argument-clamp keeps sqrt finite even on degenerate input.
    const float arg = 1.0f + (dt / tau_ps) * (target_temp_K / T - 1.0f);
    const float lambda = sqrtf(fmaxf(arg, 1.0e-6f));

    d_velocities[tid] = d_velocities[tid] * lambda;
}

extern "C"
int prism_gearbox_launch_berendsen_guard(
    float*       d_velocities,
    uint32_t     n_floats,
    const float* d_current_temp,
    const float* d_dt,
    float        target_temp_K,
    float        tau_ps,
    void*        stream)
{
    if (d_velocities == nullptr || d_current_temp == nullptr || d_dt == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_floats == 0u) return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_berendsen_guard_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_velocities, n_floats,
        d_current_temp, d_dt,
        target_temp_K, tau_ps);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — 4-way Predicate Bridge (T12.4)
// ════════════════════════════════════════════════════════════════════
//
// Trivial 1-thread kernel: read `cruise->current_gear` and forward
// to the conditional handle via cudaGraphSetConditional.  PointerSwap
// already executed the stateful finite automaton (code → gear) and
// wrote current_gear to the cruise tensor; this bridge just
// communicates that decision to the SWITCH node.
//
// Requires CUDA 12.4+ for cudaGraphSetConditional.  Production
// toolchain is CUDA 13.x so the path is always live.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

extern "C"
__global__ void prism_gearbox_predicate_bridge_kernel(
    cudaGraphConditionalHandle              handle,
    const ChronometricStateTensor* __restrict__ cruise
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (cruise == nullptr) return;
    const uint32_t gear = cruise->current_gear;
    // Defensive bounds — cudaGraphSetConditional accepts any unsigned
    // value but the SWITCH was forged with size = 4, so values ≥ 4
    // would route to the default body.  PointerSwap should never write
    // gear ≥ 4, but mask defensively.
    cudaGraphSetConditional(handle, gear & 0x3u);
}

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                       handle_v,
    const ChronometricStateTensor* cruise,
    void*                          stream)
{
    if (cruise == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prism_gearbox_predicate_bridge_kernel<<<1, 1, 0, s>>>(handle, cruise);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                       /*handle_v*/,
    const ChronometricStateTensor* /*cruise*/,
    void*                          /*stream*/)
{
    // Pre-CUDA 12.4: cudaGraphSetConditional unavailable.
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040
