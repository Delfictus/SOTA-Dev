// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G26 Chronometric Gearbox — Wave B.1 (Foundation)
// ═══════════════════════════════════════════════════════════════════════════
//
// "The Gearbox is the sole master of the chronometric plane." — operator
// directive 2026-05-02 (RULING: Gearbox > Adaptive-DT).
//
// Wave B.1 deliverables (this header / TU):
//
//   1. `__constant__ float d_gearbox_table[16]`
//      Layout: 4 gears × 4 floats each; first slot per gear is dt (ps),
//      remaining three reserved for B.2 (VelocityRescale ratios + Berendsen
//      coupling tau).  Gear 3 is the abort sentinel — all four slots = NaN
//      so any kernel that accidentally reads from this gear's row produces
//      a non-finite result and trips the existing FP-exception traps.
//
//        Gear 0 (0.5 fs): {0.0005, 0, 0, 0}
//        Gear 1 (2.0 fs): {0.0020, 0, 0, 0}
//        Gear 2 (4.0 fs): {0.0040, 0, 0, 0}
//        Gear 3 (Abort) : {NaN,    NaN, NaN, NaN}
//
//   2. `prism_gearbox_init_table_async(host_table[16], stream)` host helper:
//      cudaMemcpyToSymbolAsync of the 64-byte table into __constant__
//      memory.  Called ONCE per pipeline at build time, BEFORE the capture
//      window opens (host-resident write — captured-graph-incompatible).
//
//   3. `prism_gearbox_pointer_swap_kernel`:
//      Single-thread <<<1,1>>> kernel.  State machine:
//        adj->adjudication_code == 2 (Violation) ⇒ Gear 3 (abort)
//        adj->adjudication_code == 1 (Burst)     ⇒ Gear 0; reset cruise.counter,
//                                                   stamp last_burst_frame
//        adj->adjudication_code == 0 (Prune)     ⇒ cruise.counter++, then:
//          counter <  500  ⇒ Gear 1 (2.0 fs safety)
//          counter >= 500  ⇒ Gear 2 (4.0 fs campaign)
//      Kernel writes d_gearbox_table[gear*4+0] into *(adj->d_dt) — the
//      ProtocolState.dt address that all three integrator kernels
//      (nhs_amber_fused_step, nhs_voxel_step, nhs_voxel_step_multi_lif)
//      read on their next launch.
//
//   4. `ChronometricStateTensor` — 16-byte F2-pool device buffer.
//      Persistent state across captured-graph launches; never touched
//      by host once the pipeline is built (Zero-CPU mandate).
//
// Wave B.2 will add VelocityRescale + Berendsen kernels (gear-transition
// momentum conservation) and the predicate bridge that wires this kernel
// into the G26 SWITCH body sub-graphs.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ─── ChronometricStateTensor — 16-byte persistent device state ──────────────
//
// Accessed via raw pointer (not __constant__) so the kernel can both READ
// and WRITE it — __constant__ is read-only from device code.  Layout
// pinned by the Rust-side `#[repr(C, align(16))]` mirror in gearbox.rs.

struct __align__(16) ChronometricStateTensor {
    uint32_t counter;            // frames since last burst (0..∞)
    uint32_t last_burst_frame;   // global frame index of most-recent Burst
    uint32_t current_gear;       // 0/1/2/3 — whatever PointerSwap most recently chose
    uint32_t _pad;               // padding to 16 bytes
};

static_assert(sizeof(ChronometricStateTensor) == 16,
              "ChronometricStateTensor MUST be 16 bytes (operator-mandated layout).");
static_assert(alignof(ChronometricStateTensor) == 16,
              "ChronometricStateTensor MUST be 16-byte aligned.");

// ─── Gear-table cardinality ─────────────────────────────────────────────────
//
// 4 gears × 4 floats = 16 floats = 64 bytes.  Pinned to 64 because
// the Blackwell sm_120 L1 constant cache fetches in 64-byte / 128-byte
// lines — the entire table fits in one cache line, broadcasting in a
// single LDC across all 32 lanes of any consuming warp.

#define PRISM_GEARBOX_N_GEARS         4u
#define PRISM_GEARBOX_FLOATS_PER_GEAR 4u
#define PRISM_GEARBOX_TABLE_LEN       (PRISM_GEARBOX_N_GEARS * PRISM_GEARBOX_FLOATS_PER_GEAR)

// Cruise hysteresis threshold — frames of consecutive code=0 (Equilibrium)
// before the gearbox upshifts from Gear 1 (safety) to Gear 2 (campaign).
#define PRISM_GEARBOX_CRUISE_THRESHOLD 500u

// Forward-declaration of the FFI struct from adjudicator.cuh.  We do NOT
// #include "adjudicator.cuh" here because the gearbox kernel reads only
// two fields (adjudication_code and d_dt) and including the full header
// would drag the SO(3) tile definitions into this TU.  Layout drift is
// caught by the static_assert in adjudicator.cuh which pins both fields.
struct InterferometricAdjudicatorFfi;

#ifdef __cplusplus
extern "C" {
#endif

/// Host helper: writes the 16-float gear table into `__constant__` memory.
/// Stream-ordered (cudaMemcpyToSymbolAsync). Call ONCE per pipeline build,
/// BEFORE the captured-graph window opens.  `host_table` is a 16-element
/// f32 array following the layout described at the top of this header.
int prism_gearbox_init_table_async(
    const float* host_table,   /* 16 floats */
    void*        stream);

/// Records the PointerSwap kernel onto `stream` as a single graph node.
/// The kernel is single-thread; reads adj->adjudication_code +
/// adj->d_dt, mutates the cruise tensor, writes the active gear's dt
/// through *(adj->d_dt).  Returns cudaError_t cast to int.
///
/// `current_frame` is the global frame index used to stamp
/// `cruise->last_burst_frame` on a Burst transition.  In Wave A the
/// caller passes `0` outside the capture window; B.2's predicate bridge
/// passes a device-resident frame counter.
int prism_gearbox_launch_pointer_swap(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream);

#ifdef __cplusplus
}  // extern "C"
#endif
