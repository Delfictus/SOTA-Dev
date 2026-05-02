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
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct ChronometricStateTensor {
    /// Frames since the most recent Burst (`adjudication_code == 1`).
    /// Drives the cruise hysteresis: counter < 500 → Gear 1 (2.0 fs
    /// safety); counter ≥ 500 → Gear 2 (4.0 fs campaign).  Saturating
    /// increment in the kernel so a long quiet run cannot wrap u32.
    pub counter: u32,
    /// Global frame index of the most-recent Burst.  Stamped by the
    /// kernel on the code=1 transition; passed in by the caller as
    /// `current_frame`.  Useful for offline replay / forensic timeline.
    pub last_burst_frame: u32,
    /// Most-recent gear chosen by the kernel.  0 = 0.5 fs, 1 = 2.0 fs,
    /// 2 = 4.0 fs, 3 = abort.  Read by the predicate bridge in B.2 to
    /// route the SWITCH; written by every PointerSwap launch.
    pub current_gear: u32,
    /// Wave B.2 — gear active BEFORE the most-recent PointerSwap launch.
    /// PointerSwap copies `current_gear` into this slot before computing
    /// the new gear, so the symplectic-ratio kernel can compute
    /// λ = dt_new / dt_old without a separate scratch buffer.  Replaces
    /// the former `_pad` slot; layout still 16 bytes, 16-aligned.
    pub previous_gear: u32,
}

impl ChronometricStateTensor {
    /// Default cruise state: counter = 0, no burst yet, current_gear = 1
    /// (the safety gear — 2.0 fs).  This matches the operator-mandated
    /// "Safety Gear" behaviour for the first ≤500 frames after build.
    pub const fn initial() -> Self {
        Self {
            counter: 0,
            last_burst_frame: 0,
            current_gear: 1,
            previous_gear: 1,    // matches initial current_gear so the first
                                 // symplectic-ratio compute returns 1.0
        }
    }
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<ChronometricStateTensor>()  == 16);
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
        0.0020, 0.0, 0.0, 0.0,
        // Gear 2 — 4.0 fs (HMR-stabilised campaign sprint).
        0.0040, 0.0, 0.0, 0.0,
        // Gear 3 — abort sentinel.
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
            host_table: *const f32,   // 16 floats
            stream:     *mut c_void,
        ) -> i32;

        /// Records the PointerSwap kernel onto `stream` as one graph
        /// node.  Single-thread <<<1,1>>>; reads adj->adjudication_code
        /// + adj->d_dt, mutates the cruise tensor, writes the active
        /// gear's dt through *(adj->d_dt).
        pub fn prism_gearbox_launch_pointer_swap(
            adj:           *const InterferometricAdjudicatorFfi,
            cruise:        *mut   ChronometricStateTensor,
            current_frame: u32,
            stream:        *mut   c_void,
        ) -> i32;

        // ── Wave B.2 — Velocity Rescale + Berendsen + Predicate Bridge ──

        /// T12.2 — Symplectic Velocity Rescale.  Records one bulk kernel
        /// node onto `stream`: each thread multiplies one f32 of the
        /// AoS velocity buffer by `ratio`.  Block size 256 ⇒ each warp
        /// reads 32 contiguous f32 = 128 bytes ⇒ ptxas emits LDG.E.128 +
        /// STG.E.128 (operator-mandated vectorised path).
        pub fn prism_gearbox_launch_velocity_rescale(
            d_velocities: *mut f32,
            n_floats:     u32,
            ratio:        f32,
            stream:       *mut c_void,
        ) -> i32;

        /// T12.3 — Berendsen weak-coupling guard.  Reads
        /// `*d_current_temp` (= d_protocol->current_temperature) and
        /// `*d_dt` (= d_protocol->dt for the current gear), computes
        /// λ = sqrt(max(1 + (dt/τ_T)·(T₀/T − 1), ε)), applies
        /// v_i ← v_i · λ across the whole velocity buffer.  Single
        /// captured node; no scratch buffer.
        pub fn prism_gearbox_launch_berendsen_guard(
            d_velocities:   *mut   f32,
            n_floats:       u32,
            d_current_temp: *const f32,
            d_dt:           *const f32,
            target_temp_K:  f32,
            tau_ps:         f32,
            stream:         *mut   c_void,
        ) -> i32;

        /// T12.4 — 4-way Predicate Bridge.  Single-thread kernel reads
        /// `cruise->current_gear` and forwards the value to the G26
        /// SWITCH conditional handle via cudaGraphSetConditional.
        /// `handle_v` is the cudaGraphConditionalHandle (driver typedef
        /// to `unsigned long long`) bound to the captured graph at
        /// pipeline-build time via prism_wire_g26_gearbox_ffi.
        pub fn prism_gearbox_launch_predicate_bridge(
            handle_v: u64,
            cruise:   *const ChronometricStateTensor,
            stream:   *mut   c_void,
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
