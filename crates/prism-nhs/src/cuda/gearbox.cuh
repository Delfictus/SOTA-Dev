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
    uint32_t counter;            // frames since last burst (0..∞)        @ 0
    uint32_t last_burst_frame;   // global frame index of most-recent Burst @ 4
    uint32_t current_gear;       // 0/1/2/3 — current PointerSwap choice  @ 8
    uint32_t previous_gear;      // B.2 — gear active BEFORE the latest    @ 12
                                 //       PointerSwap launch (symplectic
                                 //       ratio kernel reads it for
                                 //       λ = dt_new / dt_old).
    // M1.2.17 — Hamiltonian Stability Fuse state.  V_{t-1} for the
    // rolling-window drift check.  Initialised to 0.0 (sentinel for
    // "first frame; skip drift check"); SFA writes V_t here at the
    // end of each launch.
    double   v_prev;             // V at frame t-1 (kcal/mol, f64)        @ 16
    // Pad to 32 bytes for L2-cache-line-aligned access (24 → 32 via
    // align(16)).  Operator-locked layout with explicit padding.
    uint64_t _pad_v_prev;        //                                       @ 24
};

static_assert(sizeof(ChronometricStateTensor) == 32,
              "ChronometricStateTensor MUST be 32 bytes (M1.2.17 v_prev extension).");
static_assert(alignof(ChronometricStateTensor) == 16,
              "ChronometricStateTensor MUST be 16-byte aligned.");
static_assert(offsetof(ChronometricStateTensor, v_prev) == 16,
              "v_prev offset drift: must be 16.");

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

/// B.3.2 — SFA-only kernel (Logic/Action Bifurcation).
///
/// Decoupled from the dt-write path: this kernel runs the Stateful
/// Finite Automaton that maps `adj->adjudication_code` → `target_gear`
/// and updates the cruise tensor (counter, current_gear, previous_gear,
/// last_burst_frame).  Does NOT write `*(adj->d_dt)` — that
/// side-effect is owned by the SWITCH body's apply-fixed-dt kernel
/// after the SWITCH has routed the bodies.
///
/// Wired into the captured pipeline AFTER the Adjudicator step and
/// BEFORE the predicate-bridge kernel.  The cruise.current_gear it
/// writes is the predicate-bridge's input.
int prism_gearbox_launch_sfa(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream);

// ─── B.2 — Velocity Rescale + Berendsen Guard + Predicate Bridge ───────────
//
// Three new kernels per operator directive 2026-05-02 §3.1–§3.3.

/// T12.2 — Symplectic Velocity Rescale.
///
/// Pure momentum-scale kernel: v_i ← v_i · ratio for every f32 in the
/// `n_floats = n_atoms · 3` velocity buffer.  Does NOT touch positions
/// or forces.  Vectorised float4 main loop with a 1-thread tail handler
/// for the `n_floats % 4` trailing scalars; the f32 entry point keeps
/// the FFI surface pointer-aligned-agnostic (the kernel re-interprets
/// to float4 internally and asserts 16-byte alignment).
///
/// `ratio` should be `dt_new / dt_old` (computed by the symplectic
/// ratio kernel — B.3) or a Berendsen λ (computed by the Berendsen
/// guard — same call signature).
int prism_gearbox_launch_velocity_rescale(
    float*    d_velocities,        /* n_atoms × 3 contiguous f32 (AoS) */
    uint32_t  n_floats,
    float     ratio,
    void*     stream);

/// T12.3 — Berendsen weak-coupling guard (Gear 0 entry only).
///
/// Reads `*d_current_temp` (= d_protocol->current_temperature) and
/// `*d_dt` (= d_protocol->dt, the gear-0 timestep just written by
/// PointerSwap), computes
///     λ = sqrt(max(1 + (dt/τ_T)·(T₀/T − 1), ε))
/// and applies v_i ← v_i · λ across the whole velocity buffer.  Single
/// captured node: each thread re-derives λ from the broadcast loads of
/// d_current_temp + d_dt (perfect L1 const-cache hit after the first
/// warp; ~10 extra ALU ops per thread is negligible vs. avoiding a
/// separate scratch-buffer + compute kernel pair).
int prism_gearbox_launch_berendsen_guard(
    float*       d_velocities,
    uint32_t     n_floats,
    const float* d_current_temp,    /* device ptr to d_protocol->current_temperature */
    const float* d_dt,              /* device ptr to d_protocol->dt                 */
    float        target_temp_K,
    float        tau_ps,            /* Berendsen τ_T (operator default 0.5 ps)      */
    void*        stream);

/// T12.4 — 4-way Predicate Bridge.
///
/// Captured AFTER PointerSwap.  Reads cruise->current_gear (already
/// computed by PointerSwap's stateful finite automaton) and forwards
/// the 0/1/2/3 value into the G26 SWITCH conditional handle via
/// `cudaGraphSetConditional(handle, gear)`.  Trivial 1-thread kernel —
/// the heavy lifting is in PointerSwap.  This kernel is the FFI
/// handshake that completes the autonomous gear-shift loop.
///
/// `handle_v` is `cudaGraphConditionalHandle` cast to u64 (driver
/// typedef).  Created via `cudaGraphConditionalHandleCreate` at
/// pipeline build time and bound to the captured graph.
///
/// B.3.2 — additionally consults `adj->gear_override` (u32 at offset
/// 100).  When the operator writes 0..3 to that VRAM word, the bridge
/// short-circuits the SFA's calculated gear with the override.
/// Sentinel 0xFF = Auto.  Pass null `adj` for tests that don't have
/// an InterferometricAdjudicatorFfi fixture (override is simply not
/// consulted).
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream);

// ─── B.3-narrow — SWITCH body kernels + ratio matrix ───────────────────────

/// B.3 — Constant-memory transition-ratio matrix init.
///
/// 3×3 matrix of dt-ratio factors `[prev_gear * 3 + target_gear]` for
/// the three active gears (0=0.5fs / 1=2.0fs / 2=4.0fs).  Gear 3 (abort)
/// is OUT-OF-BAND — the body 3 sub-graph fires the trap kernel directly
/// without consulting this matrix.  Pre-baked at host build time:
///
///     d_rescale_ratios[0..3]  = { 1.0,    4.0,   8.0  }    // prev=0 → 0/1/2
///     d_rescale_ratios[3..6]  = { 0.25,   1.0,   2.0  }    // prev=1
///     d_rescale_ratios[6..9]  = { 0.125,  0.5,   1.0  }    // prev=2
///
/// Operator constraint (2026-05-02 §2): zero divisions in the body
/// kernels; the rescale kernel performs a single LDC + multiply.
int prism_gearbox_init_rescale_ratios_async(void* stream);

/// B.3 — Symplectic velocity rescale, ratio-matrix variant.
///
/// Reads `cruise->previous_gear` at runtime, indexes the constant
/// ratio matrix, multiplies every f32 in `d_velocities` by that ratio.
/// Single LDC for the ratio (broadcast across the warp); single LDG +
/// MUL + STG per thread.  Block size 256 ⇒ LDG.E.128 / STG.E.128 at
/// the warp level (sm_120 ptxas behaviour).
///
/// `target_gear` is the body's hardcoded 0/1/2.  Body 3 (abort) does
/// NOT call this kernel — the trap kernel runs instead.
int prism_gearbox_launch_rescale(
    float*                              d_velocities,
    uint32_t                            n_floats,
    const ChronometricStateTensor*      cruise,
    uint32_t                            target_gear,
    void*                               stream);

/// B.3 — Apply-fixed-dt kernel.  Single-thread <<<1,1>>> writes
/// `d_gearbox_table[target_gear * 4]` into `*(adj->d_dt)`.  Used inside
/// each body sub-graph (Gears 0/1/2) to commit the gear's timestep
/// after the velocity rescale has completed.  Idempotent: if the
/// upstream PointerSwap already wrote the same value, the second write
/// is a no-op observable.
int prism_gearbox_launch_apply_fixed_dt(
    const InterferometricAdjudicatorFfi* adj,
    uint32_t                             target_gear,
    void*                                stream);

/// B.3 — Hardware trap kernel.  Single-thread <<<1,1>>> issues
/// `asm volatile("trap;");`.  Used inside body 3 (Gear 3 / Abort).
/// Halts the GPU stream and surfaces a CUDA error to the host.
int prism_gearbox_launch_trap(void* stream);

/// B.3 — Populate the four phGraph_out body sub-graphs returned by
/// `prism_wire_g26_gearbox_ffi`.  For each active gear N ∈ {0, 1, 2},
/// adds rescale → [Berendsen if N==0] → apply_fixed_dt as kernel
/// nodes with intra-body dependency edges.  For Gear 3, adds the trap
/// kernel as a single node.
///
/// Returns cudaSuccess on full population.  If any phGraph_out[i] is
/// null (CUDA driver bug — the Wave A B.1 risk register flag),
/// returns `cudaErrorInvalidValue` early as the operator-mandated
/// "Smoking Gun" signal for the kernel-conditional fallback path.
///
/// Berendsen knobs (target_temp_K, tau_ps) are passed through to the
/// Gear 0 body kernel.  Pass null d_current_temp / d_dt to skip the
/// Berendsen guard (for tests that don't have a ProtocolState).
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          body_subgraphs,        /* [4] */
    const InterferometricAdjudicatorFfi*  adj,
    float*                                d_velocities,
    uint32_t                              n_floats,
    const ChronometricStateTensor*        cruise,
    const float*                          d_current_temp,        /* nullable */
    const float*                          d_dt,                  /* nullable */
    float                                 target_temp_K,
    float                                 tau_ps);

#ifdef __cplusplus
}  // extern "C"
#endif
