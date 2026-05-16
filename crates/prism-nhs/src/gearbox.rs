//! G26 Chronometric Gearbox — Wave B.1 (Foundation)
//!
//! Operator-mandated three-commit sequence (B.1 → B.2 → B.3) ratified
//! 2026-05-02.  This module lands the **B.1 foundation only**:
//!
//!   * `__constant__ float d_gearbox_table[16]` — 4 gears × 4 floats,
//!     dt slot per gear; Gear 3 = NaN sentinel (double-redundant abort).
//!   * `ChronometricStateTensor` — 16-byte F2-pool persistent state,
//!     `{counter, last_burst_frame, current_gear, _pad}`.
//!   * `prism_gearbox_pointer_swap_kernel` — Stateful Finite Automaton
//!     that reads `adj->adjudication_code` + cruise counter and writes
//!     the active gear's dt into `*(adj->d_dt)` (the address the
//!     T12 Pre-Flight commit wired to `&d_protocol->dt`).
//!
//! B.2 will add VelocityRescale + Berendsen kernels and the predicate
//! bridge that wires this kernel into the G26 SWITCH body sub-graphs.
//! B.3 lands the PTX-trap kernel for Gear 3 + the 1k-step Gear 2→0→2
//! transition test.
//!
//! # Hierarchy ruling (operator 2026-05-02)
//!
//! When the V2 captured pipeline is built, the legacy `--adaptive-dt`
//! host-side write to `engine.dt` is bypassed; the gearbox is the sole
//! master of the chronometric plane.  See `fused_engine.rs` for the
//! `gearbox_active` gate.

#![cfg(feature = "gpu")]

use std::ffi::c_void;

// ============================================================================
// ChronometricStateTensor — 16-byte persistent device-resident state
// ============================================================================
//
// Layout pinned by both the Rust `#[repr(C, align(16))]` and the C-side
// `__align__(16)` + `static_assert(sizeof == 16)` in `gearbox.cuh`.

/// Persistent gearbox state — lives in the F2 pool, mutated only by the
/// PointerSwap kernel (Zero-CPU mandate).  Allocated once per pipeline,
/// pointer-stable for the campaign.
///
/// **M1.2.17**: extended to 32 bytes to hold `v_prev: f64` for the
/// Hamiltonian Stability Fuse.  Layout pinned by const_assert and
/// the C-side static_assert in `gearbox.cuh`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct ChronometricStateTensor {
    /// Frames since the most recent Burst (`adjudication_code == 1`).  @ 0
    /// Drives the cruise hysteresis: counter < 500 → Gear 1 (2.0 fs
    /// safety); counter ≥ 500 → Gear 2 (4.0 fs campaign).
    pub counter: u32,
    /// Global frame index of the most-recent Burst.                    @ 4
    pub last_burst_frame: u32,
    /// Most-recent gear chosen by the kernel.                           @ 8
    pub current_gear: u32,
    /// Gear active BEFORE the most-recent PointerSwap launch.           @ 12
    /// Used by the symplectic-ratio kernel for λ = dt_new / dt_old.
    pub previous_gear: u32,
    /// **M1.2.17 — V_{t-1}** for the rolling-window stability fuse.    @ 16
    /// Initialised to 0.0 (first-frame sentinel; the SFA skips the
    /// drift check when v_prev == 0).  SFA writes `adj.d_potential_energy`
    /// here at the end of each launch so the next launch sees the
    /// correct V_{t-1}.
    pub v_prev: f64,
    /// Padding to 32-byte boundary for L2 cache-line alignment.        @ 24
    pub _pad_v_prev: u64,
}

impl ChronometricStateTensor {
    /// Default cruise state: counter = 0, no burst yet, current_gear = 1
    /// (the safety gear — 2.0 fs).  v_prev = 0.0 sentinel triggers
    /// first-frame-skip in the SFA stability fuse.
    pub const fn initial() -> Self {
        Self {
            counter: 0,
            last_burst_frame: 0,
            current_gear: 1,
            previous_gear: 1,
            v_prev: 0.0,
            _pad_v_prev: 0,
        }
    }
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<ChronometricStateTensor>() == 32);
    assert!(align_of::<ChronometricStateTensor>() == 16);
};

// ============================================================================
// Gearbox dt table — 4 gears × 4 floats = 64 bytes (one L1 const-cache line)
// ============================================================================

/// Number of gears in the gearbox (3 active + 1 abort sentinel).
pub const PRISM_GEARBOX_N_GEARS: u32 = 4;
/// Floats per gear (slot 0 = dt; 1..3 reserved for B.2 VelocityRescale +
/// Berendsen tau).
pub const PRISM_GEARBOX_FLOATS_PER_GEAR: u32 = 4;
/// Total table length, in floats.  16 floats × 4 bytes = 64 bytes.
pub const PRISM_GEARBOX_TABLE_LEN: usize = 16;

/// Cruise hysteresis threshold.  Frames of consecutive
/// `adjudication_code == 0` (Equilibrium) before the gearbox upshifts
/// from Gear 1 (safety) to Gear 2 (campaign velocity).
pub const PRISM_GEARBOX_CRUISE_THRESHOLD: u32 = 500;

/// Default gear-table values in picoseconds.  Slot 0 of each row is dt;
/// the remaining three slots are zero in B.1 and will hold VelocityRescale
/// ratios + Berendsen tau in B.2.  Gear 3 is NaN-everywhere — the
/// double-redundant abort sentinel.
pub fn default_gearbox_table() -> [f32; PRISM_GEARBOX_TABLE_LEN] {
    let nan = f32::NAN;
    [
        // Gear 0 — 0.5 fs (high-resolution capture).
        0.0005, 0.0, 0.0, 0.0,
        // Gear 1 — 2.0 fs (default monitoring / cruise safety).
        0.0020, 0.0, 0.0, 0.0, // Gear 2 — 4.0 fs (HMR-stabilised campaign sprint).
        0.0040, 0.0, 0.0, 0.0, // Gear 3 — abort sentinel.
        nan, nan, nan, nan,
    ]
}

// ============================================================================
// FFI — extern "C" forward declarations from gearbox.cu
// ============================================================================

pub mod ffi {
    use super::ChronometricStateTensor;
    use crate::interferometric_adjudicator::InterferometricAdjudicatorFfi;
    use std::ffi::c_void;

    extern "C" {
        /// Host-side cudaMemcpyToSymbolAsync of the 16-float gear table
        /// into `__constant__ d_gearbox_table`.  Stream-ordered.  Call
        /// ONCE per pipeline build, BEFORE the captured-graph window
        /// opens.
        pub fn prism_gearbox_init_table_async(
            host_table: *const f32, // 16 floats
            stream: *mut c_void,
        ) -> i32;

        /// Records the PointerSwap kernel onto `stream` as one graph
        /// node.  Single-thread <<<1,1>>>; reads adj->adjudication_code
        /// + adj->d_dt, mutates the cruise tensor, writes the active
        /// gear's dt through *(adj->d_dt).
        pub fn prism_gearbox_launch_pointer_swap(
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *mut ChronometricStateTensor,
            current_frame: u32,
            stream: *mut c_void,
        ) -> i32;

        // ── Wave B.2 — Velocity Rescale + Berendsen + Predicate Bridge ──

        /// T12.2 — Symplectic Velocity Rescale.  Records one bulk kernel
        /// node onto `stream`: each thread multiplies one f32 of the
        /// AoS velocity buffer by `ratio`.  Block size 256 ⇒ each warp
        /// reads 32 contiguous f32 = 128 bytes ⇒ ptxas emits LDG.E.128 +
        /// STG.E.128 (operator-mandated vectorised path).
        pub fn prism_gearbox_launch_velocity_rescale(
            d_velocities: *mut f32,
            n_floats: u32,
            ratio: f32,
            stream: *mut c_void,
        ) -> i32;

        /// T12.3 — Berendsen weak-coupling guard.  Reads
        /// `*d_current_temp` (= d_protocol->current_temperature) and
        /// `*d_dt` (= d_protocol->dt for the current gear), computes
        /// λ = sqrt(max(1 + (dt/τ_T)·(T₀/T − 1), ε)), applies
        /// v_i ← v_i · λ across the whole velocity buffer.  Single
        /// captured node; no scratch buffer.
        pub fn prism_gearbox_launch_berendsen_guard(
            d_velocities: *mut f32,
            n_floats: u32,
            d_current_temp: *const f32,
            d_dt: *const f32,
            target_temp_K: f32,
            tau_ps: f32,
            stream: *mut c_void,
        ) -> i32;

        /// T12.4 — 4-way Predicate Bridge.  Single-thread kernel reads
        /// `cruise->current_gear` and forwards the value to the G26
        /// SWITCH conditional handle via cudaGraphSetConditional.
        ///
        /// B.3.2 — also consults `adj->gear_override` (u32 at offset 100;
        /// Emergency Rectification 2026-05-02 reverted from M1.2.18.5
        /// attempt to relocate to 136).  When non-0xFF, the override
        /// short-circuits the SFA's calculated gear.  Pass null `adj`
        /// to skip override consult.
        pub fn prism_gearbox_launch_predicate_bridge(
            handle_v: u64,
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *const ChronometricStateTensor,
            stream: *mut c_void,
        ) -> i32;

        /// CHUNK13_DIAG — TRACED variant. Equivalent to
        /// `prism_gearbox_launch_predicate_bridge` plus an additional
        /// `branch_trace_dev` device pointer (pinned-mapped
        /// `PrismBranchTrace`). Pass 0 to disable trace writes.
        pub fn prism_gearbox_launch_predicate_bridge_traced(
            handle_v: u64,
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *const ChronometricStateTensor,
            stream: *mut c_void,
            branch_trace_dev: u64,
        ) -> i32;

        /// B.3.2 — SFA-only kernel (Logic/Action Bifurcation).
        /// Runs the same KL-driven state machine as
        /// prism_gearbox_pointer_swap_kernel but does NOT write
        /// *(adj->d_dt) — the dt-write side-effect is owned by the
        /// SWITCH body's apply-fixed-dt kernel.  Decoupling lets the
        /// Blackwell scheduler hide the SFA's logic-only cycles
        /// behind the integrator's global-memory writes.
        pub fn prism_gearbox_launch_sfa(
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *mut ChronometricStateTensor,
            current_frame: u32,
            stream: *mut c_void,
        ) -> i32;

        // ── B.3-narrow — SWITCH body kernels + populator ──

        /// B.3 — No-op on the canonical 0.5/2.0/4.0 fs gearbox; the
        /// 3×3 ratio matrix is statically initialised in __constant__
        /// memory.  Exists for ABI symmetry with future runtime
        /// re-tuning.  Returns cudaSuccess.
        pub fn prism_gearbox_init_rescale_ratios_async(stream: *mut c_void) -> i32;

        /// B.3 — Symplectic velocity rescale (ratio-matrix variant).
        /// Reads cruise->previous_gear, indexes the constant ratio
        /// matrix at [prev*3 + target_gear], multiplies every f32 in
        /// d_velocities by that ratio.  Single LDC + LDG + MUL + STG
        /// per thread; block 256 ⇒ LDG.E.128 / STG.E.128 warp-coalesced.
        pub fn prism_gearbox_launch_rescale(
            d_velocities: *mut f32,
            n_floats: u32,
            cruise: *const ChronometricStateTensor,
            target_gear: u32,
            stream: *mut c_void,
        ) -> i32;

        /// B.3 — Apply-fixed-dt kernel.  Single-thread <<<1,1>>> writes
        /// d_gearbox_table[target_gear*4] into *(adj->d_dt).  Used
        /// inside SWITCH body sub-graphs (Gears 0/1/2).
        pub fn prism_gearbox_launch_apply_fixed_dt(
            adj: *const InterferometricAdjudicatorFfi,
            target_gear: u32,
            stream: *mut c_void,
        ) -> i32;

        /// B.3 — Hardware trap kernel.  Single-thread asm("trap;").
        /// Used inside SWITCH body 3 (Gear 3 / Abort).
        pub fn prism_gearbox_launch_trap(stream: *mut c_void) -> i32;

        /// B.3 — Populate the four phGraph_out body sub-graphs returned
        /// by prism_wire_g26_gearbox_ffi with the gear-specific kernel
        /// node sequences.  Returns cudaErrorInvalidValue if any
        /// body_subgraphs[i] is null — the operator-mandated "Smoking
        /// Gun" signal that the caller must pivot to the kernel-
        /// conditional fallback pattern.
        ///
        /// `d_current_temp` and `d_dt` are nullable — pass null to skip
        /// the Berendsen guard in body 0 (e.g., for tests without a
        /// ProtocolState fixture).
        // Reconciled with captured_pipeline.rs:865 to silence
        // `clashing_extern_declarations`. The CUgraph alias in cudarc resolves
        // to `*mut CUgraph_st`, so `*mut CUgraph` is `*mut *mut CUgraph_st`,
        // ABI-identical to `*mut *mut c_void` but type-distinguishable to the
        // Rust compiler. Using cudarc's CUgraph here matches the canonical
        // captured-pipeline declaration verbatim. Same applies to the cruise
        // pointer: kept `*const c_void` to match captured_pipeline.rs (the
        // C++ producer takes `void*` and the Rust caller casts on the way in).
        pub fn prism_gearbox_populate_switch_bodies_ffi(
            body_subgraphs: *mut cudarc::driver::sys::CUgraph, // [4]
            adj: *const InterferometricAdjudicatorFfi,
            d_velocities: *mut f32,
            n_floats: u32,
            cruise: *const std::ffi::c_void, // ChronometricStateTensor*
            d_current_temp: *const f32,
            d_dt: *const f32,
            target_temp_K: f32,
            tau_ps: f32,
        ) -> i32;
    }
}

// ============================================================================
// Tests
// ============================================================================
//
// The Wave B.1 acceptance gate ("dt 2.0 fs → 4.0 fs without host
// intervention") lives in `crates/prism-nhs/tests/gearbox_b1_dt_flip.rs`
// — moved out of the lib `#[cfg(test)]` module because pre-existing
// E0063 breakage in unrelated lib tests (solvate.rs / rt_targets.rs /
// input.rs constructing `PrismPrepTopology` without `dimer_dyad`)
// prevents the lib test crate from compiling. The integration target
// compiles independently and exercises the same kernel via the same
// FFI surface.
