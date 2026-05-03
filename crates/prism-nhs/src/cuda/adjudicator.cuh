// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Adjudicator — T0 FFI struct mirror + T1 quantum-weight LUT
//                          + T2 KL-divergence kernel forward declarations
//                          (Blackwell sm_120, align(128), Anti-Greenfield)
// ═══════════════════════════════════════════════════════════════════════
//
// Mirror of crates/prism-nhs/src/interferometric_adjudicator.rs at the
// CUDA boundary. Layout-pinned by static_assert below + Rust-side
// `tests::ffi_field_offsets_match_csr_c_table`.
//
// Anti-Greenfield Doctrine compliance:
//   - Adjudication codes use `crate::pre_rank::AdjudicationCode` numeric
//     values (0=Prune, 1=Construct, 2=Violation; mirror in `pre_rank.cuh`).
//   - μ_01_sq LUT values DERIVED from existing extinction coefficients in
//     `crate::config` (TRP_EXTINCTION_280=5600, TYR_EXTINCTION_280=1400,
//     PHE_EXTINCTION_280=200) via Strickler-Berg-style relation
//     |μ_01|² ∝ ε(λ_probe). NOT hardcoded.
//   - F2 allocation via the existing `vram_pool` wrappers (no new allocator).
//   - Single-thread launch geometry (<<<1, 1, 0, stream>>>) mirrors the
//     established idiom from gpu_invariant.cu / vram_pool.cu / sh_basis.cu.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Cross-lane header fusion (operator mandate, Part 1.2):
// Read-only ingest of Claude-1's ContactShellTile definition.
// Single source of truth — no copy-paste, no parallel decl, no
// forward-decl indirection. Brings the full 1280-byte align(128)
// definition into scope so the Adjudicator kernel can resolve
// `tile->aabb_min` / `tile->geo_power_spectrum[l]` to bit-accurate
// LDG.E.128 displacements at the sm_120 ptxas pass.
#include "so3_project.cuh"

// Hoist `prism_nhs::so3_project::ContactShellTile` into the global
// scope so the FFI struct's pointer fields read naturally as
// `const ContactShellTile*` and match the Rust-side
// `*const ContactShellTile` ABI.
using ContactShellTile = prism_nhs::so3_project::ContactShellTile;

// Layout-pin sanity: ContactShellTile must remain 1280 bytes,
// 128-byte aligned. If Claude-1 ever changes the tile, this fires
// at compile time inside the adjudicator TU rather than letting a
// silent ABI drift propagate downstream.
static_assert(sizeof(prism_nhs::so3_project::ContactShellTile) == 1280,
              "ContactShellTile MUST be 1280 bytes (RECT-3.1.b layout pin).");
static_assert(alignof(prism_nhs::so3_project::ContactShellTile) == 128,
              "ContactShellTile MUST be 128-byte aligned (Blackwell L1 sector).");

// ════════════════════════════════════════════════════════════════════
// Adjudication codes — match crate::pre_rank::AdjudicationCode
// ════════════════════════════════════════════════════════════════════

#define PRISM_ADJ_PRUNE      0u  /* Δ_AB ≤ μ_noise + 3σ — prune the site */
#define PRISM_ADJ_CONSTRUCT  1u  /* Δ_AB > μ_noise + 3σ — bisimulation path */
#define PRISM_ADJ_VIOLATION  2u  /* NaN/Inf in raw input — F1 SAD-PATH abort */

// ════════════════════════════════════════════════════════════════════
// μ_01² LUT — 4 wavelengths packed into RichSpike.intensity_packed
// bits 30-31 per the operator's QI mapping (260/280/305/320 nm).
// ════════════════════════════════════════════════════════════════════

#define PRISM_MU_01_SQ_N   4
#define PRISM_UV_CODE_260  0u
#define PRISM_UV_CODE_280  1u
#define PRISM_UV_CODE_305  2u
#define PRISM_UV_CODE_320  3u

// Bit mask + shift constants for the QI extraction (operator-spec'd).
#define PRISM_INTENSITY_PAYLOAD_MASK  0x3FFFFFFFu  /* lower 30 bits */
#define PRISM_QI_SHIFT                30u

// ════════════════════════════════════════════════════════════════════
// InterferometricAdjudicatorFfi — 128 B / align(128) struct
// (byte-for-byte mirror of the Rust struct in
//  crates/prism-nhs/src/interferometric_adjudicator.rs)
// ════════════════════════════════════════════════════════════════════

struct __align__(128) InterferometricAdjudicatorFfi {
    // Header — running noise-floor estimates (offset 0..48, 48 B).
    float    noise_floor_mu[6];     // offset   0
    float    noise_floor_sigma[6];  // offset  24

    // Adjudicator outputs (offset 48..56, 8 B).
    float    current_divergence;    // offset  48
    uint32_t adjudication_code;     // offset  52  — F1 SWITCH selector

    // Manifold pointers (offset 56..72, 16 B).
    const ContactShellTile* relaxed_manifold_ptr;    // offset 56
    const ContactShellTile* perturbed_manifold_ptr;  // offset 64

    // Hardware clocks (offset 72..88, 16 B).
    uint64_t start_clock;           // offset  72  — clock64() at entry
    uint64_t stop_clock;            // offset  80  — clock64() at exit

    // Anti-Greenfield § 6.2 backward-compatibility shim (offset 88..100, 12 B).
    float    legacy_centroid_fallback[3];  // offset 88

    // ── B.3.2 — Manual Gear Override (Hardware Interlock) ──
    // u32 at offset 100 the host writes via cuMemcpyHtoDAsync to force
    // a gear shift, bypassing the KL-driven SFA.  Sentinel 0x000000FF
    // = Auto; values 0..3 = manual force.  Read by the predicate-bridge
    // kernel on every captured-graph launch.  Restored to its B.3.2
    // home (Emergency Rectification 2026-05-02 — reverted from the
    // M1.2.18.5 attempt to relocate to 136).
    uint32_t gear_override;         // offset 100..104  (Hardware Interlock)

    // ── G28 SISR symmetry prune mask pointer (offset 104..112, 8 B) ──
    // Pointer to a single u64 holding bit-flags per cluster.  The G28
    // kernel sets `bit[i]` when cluster `i` fails the C2-reflected AABB
    // partner search; the step kernel forces adjudication_code=0 on hit.
    // Null disables the gate (non-dimer / legacy targets).
    //
    // Renamed from `force_prune_mask` to `force_prune_mask_ptr` per
    // the operator's Zero-Trust §1.1 — the `_ptr` suffix makes the
    // pointer-vs-value distinction explicit at the FFI boundary.
    uint64_t* force_prune_mask_ptr; // offset 104

    // ── M1.2.17 — Hamiltonian VALUE + Gearbox dt POINTER ──
    // potential_energy: system-wide total potential energy (f64 VALUE,
    //   not pointer) at offset 112.  CUB DeviceReduce::Sum captured node
    //   writes it every step from the per-atom AMBER PE components
    //   buffer.  SFA reads it for the 1% drift Hamiltonian Stability Fuse.
    //   Renamed from `d_potential_energy` per Zero-Trust §1.1 — `d_`
    //   prefix is reserved for pointer fields.
    // d_dt: *mut f32 → integrator's d_protocol->dt at offset 120.
    //   The G26 SWITCH body sub-graphs write here via apply_fixed_dt_kernel.
    double    potential_energy;     // offset 112  (f64 VALUE)
    float*    d_dt;                 // offset 120

    // ── Total External Work POINTER (Hamiltonian audit) ──
    // double* at offset 128.  VRAM-native F2-pool accumulator.
    // atomicAdd-f64 destination for non-conservative energy injections
    // (UV velocity kicks; ASC; force/velocity clamps).  Multiple
    // kernels contribute asynchronously across the 2+2+2+2 graph —
    // pointer fusion eliminates the read-modify-write hazard that an
    // inline f64 would impose.
    //
    // SFA drift formula (Zero-Trust §3.2):
    //   Drift = |V_t − (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|
    //
    // Captured-graph emits cuMemsetD8Async at the head of every replay
    // window so *d_external_work is fresh at chunk start.  The pre-
    // capture wire-in MUST receive a non-null pointer (Zero-Host Guard
    // §3) — null at capture time is a LANE-BLOCKED escalation.
    double*   d_external_work;      // offset 128  (POINTER)

    // M1.2.20.C-A — Gradient Gasp handles (operator §4 explicit fields).
    //   gasp_gain_eta   @ 136..140  f32   "Stiffness" Handle (η_base)
    //   force_burst_step @ 140..144 u32   "Trigger" Handle (10× amp gate)
    //   _reserved        @ 144..256 u8[112]  pads to 2 × L1 sector.
    // Both fields are written by the host orchestrator at engine init
    // and read by prism_apply_gradient_gasp_kernel each replay.
    float    gasp_gain_eta;             // offset 136
    uint32_t force_burst_step;          // offset 140
    // M1.2.20.C-B — PTX Momentum Guard flag.  Set to 1 by
    // prism_momentum_guard_check_kernel when the gasp-induced
    // center-of-mass shift exceeds 1e-4 Å; the Adjudicator step
    // kernel reads this and forces adjudication_code = VIOLATION
    // (overrides the KL-divergence trigger).
    uint32_t momentum_violation_flag;   // offset 144
    // M1.2.20.C-C / T19 — LQI Quarantine.  Bit-31 (0x80000000) set when
    // the Dynamic T7 reduce kernel found σ² <= 0 after the 100-frame
    // cold-hold burn-in (degenerate noise floor, all samples identical).
    // Adjudicator step kernel reads this and forces VIOLATION.
    uint32_t lqi_flags;                 // offset 148
    uint8_t  _reserved_m1_2_20[104];    // offset 152..256
};
static_assert(sizeof(InterferometricAdjudicatorFfi) == 256,
              "FFI Size Mismatch — must be 256 bytes (Zero-Trust §1.1).");
static_assert(alignof(InterferometricAdjudicatorFfi) == 128,
              "FFI Alignment Mismatch — must be 128-byte aligned (Blackwell L1 sector).");
// Operator Zero-Trust §1.1 — mandatory offset asserts.  These mirror
// the Rust-side const-context asserts in interferometric_adjudicator.rs.
// Drift here breaks every kernel that reads via byte-offset arithmetic
// (gearbox SFA, predicate bridge, energy monitor, supervisor shim).
static_assert(offsetof(InterferometricAdjudicatorFfi, gear_override) == 100,
              "gear_override offset drift: must be 100.");
static_assert(offsetof(InterferometricAdjudicatorFfi, force_prune_mask_ptr) == 104,
              "force_prune_mask_ptr offset drift: must be 104 (8-byte aligned).");
static_assert(offsetof(InterferometricAdjudicatorFfi, potential_energy) == 112,
              "potential_energy offset drift: must be 112 (M1.2.17 Hamiltonian).");
static_assert(offsetof(InterferometricAdjudicatorFfi, d_dt) == 120,
              "d_dt offset drift: must be 120.");
static_assert(offsetof(InterferometricAdjudicatorFfi, d_external_work) == 128,
              "External Work Offset Drift — must be 128.");
static_assert(offsetof(InterferometricAdjudicatorFfi, gasp_gain_eta) == 136,
              "M1.2.20.C-A: gasp_gain_eta offset drift — must be 136.");
static_assert(offsetof(InterferometricAdjudicatorFfi, force_burst_step) == 140,
              "M1.2.20.C-A: force_burst_step offset drift — must be 140.");
static_assert(offsetof(InterferometricAdjudicatorFfi, momentum_violation_flag) == 144,
              "M1.2.20.C-B: momentum_violation_flag offset drift — must be 144.");
static_assert(offsetof(InterferometricAdjudicatorFfi, lqi_flags) == 148,
              "M1.2.20.C-C / T19: lqi_flags offset drift — must be 148.");
static_assert(offsetof(InterferometricAdjudicatorFfi, _reserved_m1_2_20) == 152,
              "M1.2.20.C-C: _reserved tail offset drift — must be 152 (post LQI carve).");

// ════════════════════════════════════════════════════════════════════
// __device__ helpers (T1 — Quantum-Photonic Bridge)
//
// No `#ifdef __CUDA_ARCH__` guard: this header is only included by
// .cu translation units processed by nvcc, which understands the
// `__device__` qualifier in both host and device compilation passes.
// Guarding behind `__CUDA_ARCH__` would hide these declarations
// during the host pass (where the macro is undefined), making them
// invisible when the device pass needs them in some toolchain
// versions — empirically caught at first build.
// ════════════════════════════════════════════════════════════════════

/// Extracts the 2-bit Quantum Identifier (UV code) from RichSpike's
/// `intensity_packed` field. Bits 30-31 → {0,1,2,3} → {260,280,305,320} nm.
/// Branchless. Single PTX shr.b32 instruction.
__device__ __forceinline__ uint32_t prism_extract_uv_code(uint32_t intensity_packed) {
    return (intensity_packed >> PRISM_QI_SHIFT) & 0x3u;
}

/// Recovers the f32 intensity from the lower 30 bits of `intensity_packed`.
/// **Contract**: the producer must encode intensities in [0, 2.0) before
/// packing; values ≥ 2.0 lose bit 30 of the IEEE-754 exponent and round-trip
/// incorrectly. Documented in interferometric_adjudicator.rs § wavelength
/// encoding.
__device__ __forceinline__ float prism_extract_intensity(uint32_t intensity_packed) {
    uint32_t clean_bits = intensity_packed & PRISM_INTENSITY_PAYLOAD_MASK;
    return __uint_as_float(clean_bits);
}

/// Computes the Quantum Scaling Factor Q_s = μ_01²(λ) · I for a single
/// RichSpike. Reads the wavelength from intensity_packed bits 30-31 and
/// the intensity from bits 0-29. Returns the weight to multiply with
/// the spike's contribution to the SO(3) a_lm before WMMA accumulation.
///
/// Branchless: the LUT lookup is a single LDC (load constant) instruction.
/// No warp divergence regardless of the wavelength of any individual lane.
__device__ __forceinline__ float prism_compute_quantum_weight(uint32_t intensity_packed);

// ════════════════════════════════════════════════════════════════════
// Host-callable launchers (extern "C" — match Rust FFI in
// crates/prism-nhs/src/interferometric_adjudicator.rs::ffi)
// ════════════════════════════════════════════════════════════════════

extern "C" {

/// Sentinel `0xAD31`. Confirms FFI ABI round-trip across the static
/// archive boundary. See Rust-side `LINK_PROBE_SENTINEL`.
uint32_t prism_interferometric_adjudicator_link_probe(void);

/// Initialise the __constant__ memory LUT with μ_01² values derived
/// from existing extinction coefficients (T1). Call ONCE per device
/// at MD-campaign init, before any adjudicator-step launch. Idempotent.
int prism_interferometric_adjudicator_init_constants(void* stream);

/// Allocate one InterferometricAdjudicatorFfi from the F2 pool, zero
/// the device-side state, return the device pointer in *out_ptr.
/// Pointer-stable for the entire campaign.
int prism_interferometric_adjudicator_create(
    void* pool, void* stream,
    InterferometricAdjudicatorFfi** out_ptr);

/// Free the F2-pool-backed adjudicator allocation.
int prism_interferometric_adjudicator_destroy(
    InterferometricAdjudicatorFfi* adj, void* stream);

/// Run the T13 SIMT 4-plane weighted KL Adjudicator kernel against
/// the (relaxed[], perturbed[]) ContactShellTile arrays currently
/// pointed-to by `adj`. Updates current_divergence, adjudication_code,
/// legacy_centroid_fallback in place. <<<1, 64, 0, stream>>> launch
/// — each thread processes one cluster, capped at 64.
///
/// `n_clusters` is the number of valid tiles in the (relaxed,
/// perturbed) arrays. Threads with id >= n_clusters skip the KL math
/// and contribute 0 to the reduction. Capture-time constant.
int prism_interferometric_adjudicator_step(
    InterferometricAdjudicatorFfi* adj,
    uint32_t                       n_clusters,
    void*                          stream);

/// Update running noise-floor estimates from the latest "Cool" frame.
/// <<<1, 1, 0, stream>>> launch — single-thread, no atomic contention.
int prism_interferometric_adjudicator_update_noise_floor(
    InterferometricAdjudicatorFfi* adj,
    const float* cool_power_spectrum, /* 6 floats, l=0..5 */
    void* stream);

// ════════════════════════════════════════════════════════════════════
// T3 — ASC Boundary Repulsion Tensor
// ════════════════════════════════════════════════════════════════════

/// Run the T3 ASC kernel: for every atom flagged in `d_atom_in_cluster`,
/// compute F_i = α · Δ_AB · (x_i − X_c) and atomicAdd into the existing
/// `d_forces` buffer (Anti-Greenfield § 2.1: surgical extension to the
/// fused_engine.rs::d_forces force-summation pipeline; we do NOT replace
/// the integrator).
///
/// **Behaviour by adjudication_code (read from `adj`):**
///   - 1 (Construct): force is written.
///   - 0 (Prune)  : kernel returns without writes.
///   - 2 (Violation) : kernel returns without writes.
///
/// **Pointer-stability invariant**: `adj`, `d_forces`, `d_atom_positions`,
/// `d_atom_in_cluster` MUST be pointer-stable across captured-graph
/// replays (operator mandate § 4). All four are F2-pool or
/// statically allocated upstream; this kernel never reallocates.
///
/// Launch geometry: <<<(n_atoms + 255)/256, 256, 0, stream>>>.
int prism_asc_apply(
    const InterferometricAdjudicatorFfi* adj,
    float*          d_forces,           /* [n_atoms × 3] */
    const float*    d_atom_positions,   /* [n_atoms × 3] */
    const uint32_t* d_atom_in_cluster,  /* [n_atoms]    */
    int32_t         n_atoms,
    float           steering_gain_alpha,
    void*           stream,
    /* M1.2.18-P3.2 — per-atom PE accumulator (nullable).  When non-
     * null, V_ASC = -½·α·Δ·‖x_i − X_c‖²·mask is atomic-added to
     * pe_components[i] so the SFA stability fuse sees the steering
     * potential as part of V_t. */
    double*         pe_components);

// ════════════════════════════════════════════════════════════════════
// T4 — clock64 pipeline timing bookends
// ════════════════════════════════════════════════════════════════════

/// Stamp `adj->start_clock` with `clock64()`. Single-thread launch.
/// Must run as the FIRST kernel in the captured pipeline (before
/// SO(3) projection) for the < 10μs gate to be meaningful.
int prism_pipeline_clock_start(
    InterferometricAdjudicatorFfi* adj,
    void* stream);

/// Stamp `adj->stop_clock` with `clock64()`. Single-thread launch.
/// Must run as the LAST kernel in the captured pipeline (after ASC
/// apply). Pipeline elapsed = stop_clock − start_clock cycles. At
/// 2.977 GHz boost (sm_120), 1 ns ≈ 2.977 cycles; the < 10μs gate is
/// < 29,770 cycles.
int prism_pipeline_clock_stop(
    InterferometricAdjudicatorFfi* adj,
    void* stream);

// ════════════════════════════════════════════════════════════════════
// DAG-COND-WIRE — F1 SWITCH bridge helpers
// ════════════════════════════════════════════════════════════════════

/// Returns a stable `const uint32_t*` to the `adjudication_code`
/// field (offset 52) inside the F2-pool-allocated adjudicator struct.
/// Pointer-stable for the entire MD campaign (G19 invariant).
///
/// Caller use cases:
///   - `cudaGraphConditionalHandleCreate(...)` if the operator's
///     hardware-resident predicate-pointer model is the chosen path.
///   - Direct passthrough into a `cudaGraphSetConditional` translation
///     kernel (see `prism_adj_set_conditional` below) for CUDA 12.x's
///     handle-based conditional-node API.
const uint32_t* prism_get_adjudication_code_devptr(
    const InterferometricAdjudicatorFfi* adj);

/// Single-thread kernel launcher: reads the memory-resident
/// `adj->adjudication_code` and forwards it to the F1 SWITCH via
/// device-side `cudaGraphSetConditional(handle, code)`.
///
/// This is the bridge between the memory-write semantics of T2's
/// adjudicator kernel (which writes 0/1/2 into the FFI struct) and
/// the handle-based semantics of CUDA 12.4+ `cudaGraphConditionalNode`.
///
/// Captured-graph topology: place this node DOWNSTREAM of the T2
/// adjudicator-step node and UPSTREAM of the conditional node, with
/// explicit `cudaGraphAddDependencies` edges on both sides (operator
/// mandate § 2.3 happens-before guarantee).
///
/// The `handle` parameter is the value returned by
/// `cudaGraphConditionalHandleCreate`. CUDA defines
/// `cudaGraphConditionalHandle` as `typedef unsigned long long`
/// (driver_types.h:3229), so we pass it as `uint64_t` across FFI
/// for direct ABI parity — no casting on either side.
int prism_adj_set_conditional(
    uint64_t handle,                             /* cudaGraphConditionalHandle */
    const InterferometricAdjudicatorFfi* adj,
    void* stream);

// ════════════════════════════════════════════════════════════════════
// V2 C-ABI BYPASS — Native F1 SWITCH forge
// (cudarc 595.45.04 conditional-node binding bug bypass)
// ════════════════════════════════════════════════════════════════════

/// Native single-call C++ forge that creates the F1 SWITCH conditional
/// node and locks the happens-before edge from the adjudicator node.
///
/// Internally executes (in order):
///   1. `cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault)`
///      — bind the handle (default value 0 ⇒ Prune).
///   2. Populate `cudaGraphNodeParams` with type=Conditional,
///      conditional.type=Switch, conditional.size=3.
///   3. `cudaGraphAddNode(out_conditional_node, graph, NULL, 0, &nodeParams)`.
///   4. `cudaGraphAddDependencies(graph, &adjudicator_node, out_conditional_node, 1)`
///      — Gate G19 happens-before lock.
///
/// `predicate_dev_ptr` is the device address of `adj->adjudication_code`
/// (from `prism_get_adjudication_code_devptr`). It is wiring metadata;
/// the runtime conditional API is handle-based, so the bridge kernel
/// `prism_adj_set_conditional_kernel` is the path that forwards the
/// memory-resident value into the conditional handle.
///
/// Returns 0 on success; CUDA error code cast to int otherwise.
/// Returns 801 (`cudaErrorNotSupported`) if the toolkit is pre-CUDA-12.6
/// (no SWITCH conditional support — production CUDA 13.x is always
/// supported).
int prism_wire_f1_switch_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  adjudicator_node,
    const uint32_t*  predicate_dev_ptr,
    cudaGraphNode_t* out_conditional_node);

// ════════════════════════════════════════════════════════════════════
// T12 Pre-Flight — G26 Chronometric Gearbox 4-way SWITCH forge
// ════════════════════════════════════════════════════════════════════
//
// Sibling of `prism_wire_f1_switch_ffi`: creates a 4-way SWITCH
// conditional node and returns BOTH the conditional-node handle and
// all four body sub-graph handles via `out_body_subgraphs[4]`.  The
// caller (Rust orchestrator, Wave B) populates each sub-graph with:
//   Case 0 → Gear 0 PointerSwap  (0.5 fs)
//   Case 1 → Gear 1 PointerSwap  (2.0 fs)
//   Case 2 → Gear 2 PointerSwap  (4.0 fs)
//   Case 3 → Gear 3 PTX trap     (abort)
//
// Wave A leaves the body sub-graphs unpopulated; the SWITCH fires the
// default value (0) for every frame, which is a no-op until Wave B's
// PointerSwap kernels land.
//
// `predicate_dev_ptr` is wiring metadata; a separate bridge kernel
// (Wave B) will forward the gear_id source value into the conditional
// handle via `cudaGraphSetConditional`. The Pre-Flight forge does NOT
// capture that bridge — it only stamps the SWITCH skeleton.
int prism_wire_g26_gearbox_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  predicate_node,
    const uint32_t*  predicate_dev_ptr,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraphs    /* [4] */);

// ─── B.3.2-FULL — capture-time handle creation + post-capture wire ─────────

/// Wrapper around `cudaGraphConditionalHandleCreate`.  Created
/// DURING capture (via cuStreamGetCaptureInfo to fetch the in-progress
/// graph handle) so the bridge kernel can capture the handle as a
/// kernel-node arg.  Pair with `prism_gearbox_wire_with_handle_ffi`
/// post-capture to add the SWITCH node referencing the same handle.
int prism_gearbox_create_handle_ffi(
    cudaGraph_t graph,
    uint32_t    default_value,
    uint64_t*   out_handle);

/// Post-capture: add a SWITCH (size=4) conditional node downstream of
/// `predicate_node` referencing an EXISTING `handle_v` (created via
/// `prism_gearbox_create_handle_ffi` during capture).  Returns the
/// four phGraph_out body sub-graphs ready for population.
int prism_gearbox_wire_with_handle_ffi(
    cudaGraph_t      graph,
    cudaGraphNode_t  predicate_node,
    uint64_t         handle_v,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraphs    /* [4] */);

// ════════════════════════════════════════════════════════════════════
// T7 — Noise-floor calibration writeback
// ════════════════════════════════════════════════════════════════════

/// Writes the calibrated 3-σ noise-floor priors (`μ_l`, `σ_l` for
/// `l = 0..5`) into the device-side `InterferometricAdjudicatorFfi`
/// after the T7 cold_hold calibration run. Avoids the need to
/// recompile with hardcoded constants — Claude-1's calibration script
/// captures the measured priors and writes them back via this setter.
///
/// Internally performs TWO stream-ordered host-to-device copies:
///   1. `mu_host[6]`    → `adj->noise_floor_mu`    (struct offset 0,  24 B)
///   2. `sigma_host[6]` → `adj->noise_floor_sigma` (struct offset 24, 24 B)
///
/// **Lifecycle**: typically called ONCE per MD campaign, AFTER the
/// adjudicator struct is created via `prism_interferometric_adjudicator_create`
/// and BEFORE the first captured-graph launch. May also be called
/// between captured-graph launches to recalibrate without rebuilding
/// the graph (the struct is pointer-stable per F2 pool guarantees).
///
/// **Caller-pointer contract**:
///   - `adj` must be a valid F2-pool-allocated InterferometricAdjudicatorFfi
///     device pointer.
///   - `mu_host` and `sigma_host` must each point to 6 contiguous f32
///     values. The runtime MAY synchronously stage from pageable host
///     memory; for true async behavior, callers should pass pinned
///     host memory (`cudaMallocHost`).
///   - `stream` is the F2-pool stream the adjudicator was created on
///     (or any stream — the writes are stream-ordered against
///     subsequent kernel launches on the same stream).
///
/// Returns 0 on success; CUDA error code cast to int otherwise.
int prism_adj_set_noise_floor_constants(
    InterferometricAdjudicatorFfi* adj,
    const float*                   mu_host,    /* [6] */
    const float*                   sigma_host, /* [6] */
    void*                          stream);

}  // extern "C"
