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
