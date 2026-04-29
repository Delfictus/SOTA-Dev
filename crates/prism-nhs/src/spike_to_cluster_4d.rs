//! M1.1 — `SpikeToCluster4D` transform: type definitions only.
//!
//! This module is the type-definitions layer of the M1 producer
//! transform. The transform is the on-device counterpart to the
//! existing post-MD `clustering_to_clustered_sites::ClusteringToClusteredSites`
//! audited transform: it consumes the engine's already-on-device
//! spike buffer, performs voxel-to-cluster attribution and per-cluster
//! AABB construction natively in VRAM, and emits a frame-level
//! [`crate::entangled_manifold::EntangledManifold`] plus the
//! [`ConservationScalars`] required for the M1 algebraic conservation
//! audit.
//!
//! # What is in this commit (M1.1) and what is not
//!
//! Present in M1.1:
//!
//! * [`SpikeToCluster4DInput`] — borrowed input shape for the
//!   transform's `apply` method (M1.2).
//! * [`SpikeToCluster4DOutput`] — owned output shape, packaging
//!   the produced frame-level manifold + conservation scalars.
//! * [`ConservationScalars`] — three-scalar Family-A
//!   conservation-of-mass audit payload.
//! * [`ManifoldViewAabbFfi`] — `#[repr(C)]` POD struct returned by
//!   the three per-view extern-C accessors (declarations land in
//!   M1.2 alongside the `.cu` that backs them).
//!
//! Deliberately absent — land in subsequent intra-lane commits:
//!
//! * `extern "C" fn prism_get_*_aabb` declarations — M1.2.
//! * `impl AuditedTransform for SpikeToCluster4D` (with `apply` that
//!   launches the kernels and `verify` that runs the algebraic
//!   conservation check) — M1.2.
//! * Constructors that take [`crate::entangled_manifold::CausalSortKey`]
//!   on the role-view types — M1.3.
//! * The CUDA `.cu` source — M1.2.
//!
//! # Conservation of Mass — the law this transform enforces
//!
//! Per the M1 readiness report §5 and the M1 contract directive (1):
//!
//! ```text
//!     Σ(spikes in clusters) + background_count == total_attributed_spikes
//! ```
//!
//! Equivalently, on the wire as M1's three [`ConservationScalars`]
//! fields:
//!
//! ```text
//!     total_attributed + background_count == total_input_spikes
//! ```
//!
//! On-device, the producer kernel maintains two `atomicAdd` counters
//! (per-cluster spike counts and `background_count` for spikes
//! assigned to [`crate::site_manifest::ClusterId::UNCLUSTERED`]
//! / `UINT32_MAX`). A device-side segmented reduction
//! (CUB `DeviceReduce::Sum` on the per-cluster counters) produces
//! `total_attributed`. The engine's pre-kernel spike count is
//! captured into `total_input_spikes`. All three exit through the
//! FFI boundary into [`ConservationScalars`].
//!
//! Rust-side, the algebraic equality check happens in the
//! `AuditedTransform::verify` impl (M1.2). On violation, the
//! `TransformViolation` is routed `AuditRouting::Abort`, the
//! adjudicate path returns `AuditOutcome::Aborted`, and the run dies
//! before the `EntangledManifold` is released to downstream consumers.
//! There is no parallel "GPU-side auditor" type — the on-device work
//! is the *reduction*, the audit is the *equality check*, and the
//! latter lives in the trait impl per the M1 contract's §3.3
//! audit-bypass HALT predicate.

use serde::{Deserialize, Serialize};

use crate::entangled_manifold::{Aabb, EntangledManifold};
use crate::transform::{LawFamily, LawId, TransformId};

// ============================================================================
// Input — borrowed shape for SpikeToCluster4D::apply (M1.2)
// ============================================================================

/// Borrowed input to the M1 producer transform.
///
/// At M1.1 this is a placeholder type-shape: the kernel that consumes
/// it lands in M1.2 along with the FFI extern declarations and the
/// borrowed-handle types it requires (`SpikeBufferHandle`,
/// `SpatialHashParams`). Until then the type carries only the scalar
/// parameters the kernel needs and a phantom lifetime parameter to
/// reserve the borrowing slot for the M1.2 fields.
///
/// # Why a placeholder rather than the full shape now
///
/// The M1.2 kernel will borrow the engine's existing GPU spike buffer
/// (currently exposed via
/// `crate::fused_engine::NhsAmberFusedEngine::spike_buffer_gpu()` and
/// `spike_count_gpu()`). Wiring those handles into this type requires
/// either re-exporting their wrapper types or introducing the FFI
/// translation layer that wraps them — both of which are M1.2's job
/// (FFI declarations). M1.1 keeps the lifetime parameter so the
/// signature of `AuditedTransform::Input<'a>` is compatible with the
/// future borrowing shape without forcing M1.2 to break the type
/// signature.
pub struct SpikeToCluster4DInput<'a> {
    /// Number of valid spikes the engine has emitted into its GPU
    /// spike buffer at the current frame. Read by the engine via
    /// `spike_count_gpu()` immediately before kernel launch and
    /// captured into [`ConservationScalars::total_input_spikes`].
    pub spike_count: u32,
    /// MD frame index this clustering invocation is being performed
    /// at. Stamped into every produced [`crate::site_manifest::Centroid3D::frame`]
    /// and into the [`EntangledManifold::frame`] field.
    pub frame: u64,
    /// Spatial-hash cell size in Å; the M1 producer's grid resolution.
    /// Conventionally equals the clustering ε (so the 27-cell
    /// neighborhood exactly covers all points within ε of a query
    /// point, matching the legacy `gpu_cluster_backend.rs` convention).
    pub epsilon_angstrom: f32,
    /// Phantom lifetime for the borrowed GPU-buffer handles that
    /// land in M1.2. Reserves the lifetime slot in the type so the
    /// M1.2 signature change is additive, not breaking.
    pub _borrow: std::marker::PhantomData<&'a ()>,
}

// ============================================================================
// Output — owned shape returned by SpikeToCluster4D::apply (M1.2)
// ============================================================================

/// Owned output of the M1 producer transform.
///
/// Packages the frame-level
/// [`crate::entangled_manifold::EntangledManifold`] (the LBVH-ready
/// typed role views with AABBs) alongside the three
/// [`ConservationScalars`] required for the Family-A algebraic audit.
/// The transform's `verify` impl (M1.2) reads only the conservation
/// scalars; downstream consumers read only the manifold.
///
/// On audit violation the `AuditedTransform::adjudicate` default impl
/// will route this `Output` into `AuditOutcome::Aborted` and the
/// engine dies before the `manifold` is released to downstream code.
#[derive(Debug, Clone)]
pub struct SpikeToCluster4DOutput {
    /// Frame-level typed manifold: 3 role views with LBVH-ready AABBs.
    pub manifold: EntangledManifold,
    /// Three-scalar conservation-of-mass audit payload.
    pub conservation: ConservationScalars,
}

// ============================================================================
// ConservationScalars — Family-A algebraic conservation audit payload
// ============================================================================

/// Three-scalar conservation-of-mass audit payload.
///
/// Family A (Algebraic local) per the post-05dbc3dc canonical
/// architecture blueprint's law-family taxonomy. The audit law is the
/// equality
///
/// ```text
///     total_attributed + background_count == total_input_spikes
/// ```
///
/// enforced by `AuditedTransform::verify` on every `apply` call.
/// Determinism class for the producing transform is
/// `DeterminismClass::AtomicsAffected` (per-cluster `atomicAdd`
/// reduction induces order-dependence under realistic runtime
/// conditions); tolerance for *this* equality check is
/// `TolerancePolicy::BitExact` because all three scalars are integer
/// counts produced by counting integers — no floating-point
/// arithmetic in the path that would warrant a tolerance knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConservationScalars {
    /// Total spikes the engine emitted into the GPU spike buffer
    /// at this frame (captured pre-kernel from `spike_count_gpu()`).
    pub total_input_spikes: u64,
    /// Σ over the per-cluster `atomicAdd` counters: the count of
    /// spikes that were successfully attributed to *some* cluster.
    /// Computed on-device by a segmented `DeviceReduce::Sum` over
    /// the per-cluster counter array.
    pub total_attributed: u64,
    /// Count of spikes assigned to the
    /// [`crate::site_manifest::ClusterId::UNCLUSTERED`] sentinel
    /// (`UINT32_MAX`). Maintained on-device as a single
    /// `atomicAdd` counter incremented in the assignment kernel.
    pub background_count: u64,
}

impl ConservationScalars {
    /// Sum of attributed and background counts. The law's left-hand
    /// side. Compared against [`Self::total_input_spikes`] in the
    /// M1.2 `verify` impl.
    ///
    /// `checked_add` is used rather than wrapping addition so that an
    /// overflow on a maliciously corrupted output still surfaces as
    /// a violation rather than wrapping silently to a value that
    /// happens to equal `total_input_spikes` modulo `2^64`.
    pub fn attributed_plus_background(&self) -> Option<u64> {
        self.total_attributed.checked_add(self.background_count)
    }

    /// True iff the algebraic conservation law holds:
    /// `total_attributed + background_count == total_input_spikes`,
    /// with a `checked_add` overflow counting as a violation
    /// (returns `false`).
    pub fn is_conserved(&self) -> bool {
        match self.attributed_plus_background() {
            Some(sum) => sum == self.total_input_spikes,
            None => false,
        }
    }
}

// ============================================================================
// FFI sibling AABB struct (locked at §M9 of the blueprint)
// ============================================================================

/// FFI-only `#[repr(C)]` sibling struct that bundles a geometric
/// [`Aabb`] with its provenance scalars (`support_count`, `frame`).
///
/// Per blueprint §M9 the steering subsystem (post-M1 lane) consumes
/// per-view AABBs through three named extern-C accessors
/// (`prism_get_causal_driver_aabb`, `prism_get_lining_contact_aabb`,
/// `prism_get_localized_subcluster_aabb`); the per-view discrimination
/// is at the FFI symbol-name level rather than the Rust type system,
/// so the same sibling struct is returned by all three accessors.
/// This keeps the underlying [`Aabb`] a pure POD geometric type for
/// direct LBVH consumption and gives the steering subsystem a
/// self-contained per-call payload that includes the support-set
/// size and the frame index.
///
/// # Frame width
///
/// The MD engine carries `FrameIndex = u64` internally
/// (see [`crate::entangled_manifold::FrameIndex`]). The FFI surface
/// narrows it to `u32` here. The narrowing is safe by construction:
/// at typical 4 fs MD step sizes, `u32::MAX` ≈ 4.29 × 10⁹ steps
/// corresponds to ≈ 17.16 μs of simulated time, several orders of
/// magnitude above any single canonical-verification or campaign-
/// scale run that PRISM-4D performs (typical horizon: tens to
/// hundreds of nanoseconds). The downcast site (M1.2's FFI shim)
/// MUST carry an inline code comment that states this headroom
/// calculation explicitly, so a future reader sees the intentional
/// narrowing rather than a bug.
///
/// # Layout
///
/// `#[repr(C)]` so the layout is FFI-stable and matches the C-side
/// struct that the post-M1 steering kernel will consume. [`Aabb`] is
/// also `#[repr(C)]`; the composite layout is
/// `{ [f32;3], [f32;3], u32, u32 }` = 32 bytes contiguous, no
/// padding holes on x86-64.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldViewAabbFfi {
    /// Tight axis-aligned bounding box over the role-view's support
    /// set, in the simulation coordinate frame. POD; LBVH-direct-
    /// consumable by the Morton encoder.
    pub aabb: Aabb,
    /// Number of residues in the role-view's support set. Used by
    /// the steering subsystem to weight or threshold steering force
    /// magnitude (small support → low confidence target).
    pub support_count: u32,
    /// MD frame index, narrowed from the engine's `u64`
    /// `FrameIndex`. See type-level docs for headroom calculation.
    pub frame: u32,
}

// ============================================================================
// M1.2.1 — AuditedTransform identity, FFI extern surface, accessor exports
// ============================================================================

/// Stable identity for the M1 producer transform. Appears in every
/// audit record and every violation emitted by the M1 lane's
/// `AuditedTransform::verify` impl. The transform impl lands in a
/// subsequent M1.2.x commit alongside the real CUDA kernel.
pub const TRANSFORM_M1_SPIKE_TO_CLUSTER_4D: TransformId =
    TransformId("m1_spike_to_cluster_4d");

/// M1 algebraic conservation-of-mass law:
///
/// ```text
///     total_attributed + background_count == total_input_spikes
/// ```
///
/// Family A (Algebraic local). Routed `AuditRouting::Abort` on
/// violation: a conservation failure indicates engine-wide accounting
/// corruption and the run dies before any downstream consumer sees
/// the corrupted manifold.
pub const LAW_M1_CONSERVATION_OF_MASS: LawId =
    LawId::new("m1_conservation_of_mass", LawFamily::Algebraic);

/// Full declared law set for the M1 producer transform.
pub const DECLARED_LAWS_M1: &[LawId] = &[LAW_M1_CONSERVATION_OF_MASS];

/// M1 producer transform — the on-device counterpart to the legacy
/// post-MD `clustering_to_clustered_sites::ClusteringToClusteredSites`.
///
/// Zero-sized singleton. State-free — all governance lives in the
/// future `AuditedTransform` trait impl (lands in a subsequent M1.2.x
/// commit alongside the real CUDA kernel; at M1.2.1 this struct is
/// the type-only scaffold that downstream-facing code can already
/// name in signatures).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpikeToCluster4D;

impl SpikeToCluster4D {
    /// Construct the singleton.
    pub const fn new() -> Self {
        SpikeToCluster4D
    }
}

// ============================================================================
// FFI: extern "C" declarations bound to the static archive built from
// crates/prism-nhs/src/cuda/spike_to_cluster_4d.cu (see build.rs's
// `compile_to_static_archive` call).
//
// At M1.2.1 only the link probe is real. `prism_m1_spike_to_cluster_4d_run`
// is declared with its full signature so the symbol shape is locked in
// from day one (changing the signature later would force the static
// archive to be re-built and the linker to re-resolve every callsite).
// The C-side stub returns cudaSuccess immediately without launching
// kernels.
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)] // The full extern "C" surface is declared at M1.2.1
                    // (foundation) but only the link-probe symbol is
                    // currently consumed. The remaining symbols are used
                    // by the AuditedTransform::apply impl that lands in a
                    // subsequent M1.2.x commit. Declaring them here locks
                    // the symbol shape from day one so the static archive
                    // does not need to be re-built / re-resolved later.
mod ffi {
    use super::Aabb;

    /// FFI mirror of `crate::cuda::spike_to_cluster_4d::SpatialHashParams`
    /// per the C++ side `.cuh`. Layout is `#[repr(C)]` and matches
    /// byte-for-byte. The C++ side has a `static_assert` on `sizeof(Aabb)`;
    /// the Rust side has a `mem::size_of::<...>()` test in the unit test
    /// module that pins the layout.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct SpatialHashParams {
        pub bbox_min: [f32; 3],
        pub bbox_max: [f32; 3],
        pub cell_size: f32,
        pub grid_dim: [i32; 3],
        pub num_cells: u32,
    }

    /// `cudaError_t` opaque alias for the FFI return type. The full
    /// CUDA runtime type is `int32_t` in the ABI; cudarc exposes
    /// richer wrappers. M1.2.1 uses the raw int32 form because the
    /// safe wrapper that converts it lands in the same M1.2.x commit
    /// as the real `apply` impl.
    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        /// Link-probe function. Returns `0xC0FFEE`. Used by the
        /// `link_probe_returns_sentinel` unit test to confirm the
        /// static archive linked correctly and the FFI ABI is
        /// round-tripping.
        pub fn prism_m1_link_probe() -> u32;

        /// Run the M1 producer kernel sequence on the given stream.
        /// At M1.2.1 the C-side body is a stub returning
        /// `cudaSuccess` without launching any kernels. The full
        /// signature is locked here so subsequent M1.2.x commits
        /// fill in the body without changing the symbol shape.
        ///
        /// Safety: every pointer must be valid for the lifetime of
        /// the call; `stream` is a valid `cudaStream_t` (or 0 for
        /// the default stream — though the M1.2 contract §4c
        /// forbids default-stream launches once the body lands).
        pub fn prism_m1_spike_to_cluster_4d_run(
            d_spike_positions: *const f32,
            num_spikes: u32,
            h_params: *const SpatialHashParams,
            stream: usize, // cudaStream_t is typedef'd to a pointer
            d_cluster_id_per_spike: *mut u32,
            d_per_cluster_count: *mut u64,
            d_total_attributed_scalar: *mut u64,
            d_background_count_scalar: *mut u64,
            d_per_cluster_aabb: *mut Aabb,
            num_clusters: u32,
        ) -> CudaError;
    }
}

/// Safe Rust wrapper around the FFI link-probe function. Returns
/// the sentinel value the C-side `prism_m1_link_probe` returns; tests
/// pin this at `0xC0FFEE`. Confirms (a) the static archive linked
/// correctly, (b) the FFI ABI is round-tripping, (c) the `gpu`
/// feature is enabled (this function does not exist otherwise).
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    // SAFETY: the FFI function is a pure value-returning probe; no
    // pointer arguments, no global mutable state, no allocations.
    // Calling it from any thread, in any context, is safe.
    unsafe { ffi::prism_m1_link_probe() }
}

// ============================================================================
// extern "C" accessors for steering subsystem (post-M1) consumption
// ============================================================================
//
// Per blueprint §M9 (Mandate #8) FFI surface clause, the steering
// subsystem consumes per-view AABBs through three named extern-C
// accessors. Per-view discrimination at the FFI symbol-name level;
// the underlying Rust-side `Aabb` stays a pure POD geometric type for
// direct LBVH consumption.
//
// These are pure-Rust accessors that read from a Rust-owned
// `EntangledManifold`. They have no CUDA dependency and are available
// regardless of the `gpu` feature flag.
//
// Frame width: every accessor narrows the engine-side `FrameIndex = u64`
// to `u32` at the boundary. The narrowing is safe by construction —
// at typical 4 fs MD step sizes, `u32::MAX` ≈ 4.29 × 10⁹ steps
// corresponds to ≈ 17.16 μs of simulated time, several orders of
// magnitude above any single PRISM run (typical horizon: tens to
// hundreds of nanoseconds). `try_into().unwrap_or(u32::MAX)` is used
// rather than `as u32` so an overflow surfaces as the `u32::MAX`
// sentinel rather than wrapping silently to a small value that
// would alias an early frame.

/// Return code: success.
pub const PRISM_AABB_OK: i32 = 0;
/// Return code: a required pointer argument was null.
pub const PRISM_AABB_ERR_NULL_POINTER: i32 = -1;

/// Internal helper — narrow a `u64` frame to `u32` with overflow
/// surfaced as `u32::MAX` rather than silently wrapping.
#[inline]
fn frame_u64_to_u32_narrow(f: u64) -> u32 {
    // FRAME WIDTH DOWNCAST (per blueprint §M9 / Mandate #8):
    // engine carries FrameIndex = u64; FFI exposes u32. Safe by
    // construction: at typical 4 fs MD step sizes, u32::MAX = 4.29e9
    // steps = 1.72e4 ns = 17.16 us simulated time, several orders
    // of magnitude above any single PRISM run. `try_into` returns
    // Err on overflow; we map that to u32::MAX as the documented
    // overflow sentinel rather than wrapping silently to a
    // small-frame alias. This is intentional narrowing, not a bug.
    u32::try_from(f).unwrap_or(u32::MAX)
}

/// Read the `CausalDriverView` AABB + provenance from an
/// `EntangledManifold` owned by the caller, projecting it into a
/// `ManifoldViewAabbFfi` on the supplied output pointer.
///
/// # Safety
///
/// `manifold` must point to a valid `EntangledManifold` for the
/// duration of the call. `out` must point to writable memory of at
/// least `size_of::<ManifoldViewAabbFfi>()` bytes, properly aligned.
/// Returns `PRISM_AABB_OK` on success, `PRISM_AABB_ERR_NULL_POINTER`
/// if either pointer is null.
#[no_mangle]
pub unsafe extern "C" fn prism_get_causal_driver_aabb(
    manifold: *const EntangledManifold,
    out: *mut ManifoldViewAabbFfi,
) -> i32 {
    if manifold.is_null() || out.is_null() {
        return PRISM_AABB_ERR_NULL_POINTER;
    }
    let m = &*manifold;
    let view = m.driver.data();
    *out = ManifoldViewAabbFfi {
        aabb: view.aabb,
        support_count: view.support.len() as u32,
        frame: frame_u64_to_u32_narrow(view.provenance.frame),
    };
    PRISM_AABB_OK
}

/// Read the `LiningContactView` AABB + provenance from an
/// `EntangledManifold`. See [`prism_get_causal_driver_aabb`] for
/// the full safety contract.
#[no_mangle]
pub unsafe extern "C" fn prism_get_lining_contact_aabb(
    manifold: *const EntangledManifold,
    out: *mut ManifoldViewAabbFfi,
) -> i32 {
    if manifold.is_null() || out.is_null() {
        return PRISM_AABB_ERR_NULL_POINTER;
    }
    let m = &*manifold;
    let view = m.lining.data();
    *out = ManifoldViewAabbFfi {
        aabb: view.aabb,
        support_count: view.support.len() as u32,
        frame: frame_u64_to_u32_narrow(view.provenance.frame),
    };
    PRISM_AABB_OK
}

/// Read the `LocalizedSubclusterView` AABB + provenance from an
/// `EntangledManifold`. See [`prism_get_causal_driver_aabb`] for
/// the full safety contract.
#[no_mangle]
pub unsafe extern "C" fn prism_get_localized_subcluster_aabb(
    manifold: *const EntangledManifold,
    out: *mut ManifoldViewAabbFfi,
) -> i32 {
    if manifold.is_null() || out.is_null() {
        return PRISM_AABB_ERR_NULL_POINTER;
    }
    let m = &*manifold;
    let view = m.localized.data();
    *out = ManifoldViewAabbFfi {
        aabb: view.aabb,
        support_count: view.support.len() as u32,
        frame: frame_u64_to_u32_narrow(view.provenance.frame),
    };
    PRISM_AABB_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalars(input: u64, attributed: u64, background: u64) -> ConservationScalars {
        ConservationScalars {
            total_input_spikes: input,
            total_attributed: attributed,
            background_count: background,
        }
    }

    #[test]
    fn conservation_holds_when_attributed_plus_background_equals_input() {
        // Family-A conservation: 100 spikes in, 60 clustered + 40
        // background = 100. Equality holds.
        let s = scalars(100, 60, 40);
        assert_eq!(s.attributed_plus_background(), Some(100));
        assert!(s.is_conserved());
    }

    #[test]
    fn conservation_violation_lost_spike() {
        // 100 spikes in, 60 clustered + 30 background = 90.
        // 10 spikes lost. Audit must catch this.
        let s = scalars(100, 60, 30);
        assert_eq!(s.attributed_plus_background(), Some(90));
        assert!(!s.is_conserved());
    }

    #[test]
    fn conservation_violation_phantom_spike() {
        // 100 spikes in, 70 clustered + 40 background = 110.
        // 10 phantom spikes counted twice (or fabricated). Audit
        // must catch this regardless of which side is "wrong."
        let s = scalars(100, 70, 40);
        assert_eq!(s.attributed_plus_background(), Some(110));
        assert!(!s.is_conserved());
    }

    #[test]
    fn conservation_violation_overflow_is_treated_as_violation() {
        // Corrupted output: total_attributed + background_count
        // overflows u64. checked_add returns None; is_conserved
        // returns false. Audit must catch the corruption rather
        // than wrap silently.
        let s = scalars(123, u64::MAX, 1);
        assert!(s.attributed_plus_background().is_none());
        assert!(!s.is_conserved());
    }

    #[test]
    fn conservation_zero_spikes_zero_clusters_is_conserved() {
        // Edge case: a frame where the engine emitted no spikes.
        // 0 input == 0 attributed + 0 background. Trivially conserved.
        let s = scalars(0, 0, 0);
        assert!(s.is_conserved());
    }

    #[test]
    fn conservation_all_background_is_conserved() {
        // Edge case: every spike fell into the UNCLUSTERED sentinel.
        // 100 input == 0 attributed + 100 background. Conserved
        // (the audit doesn't care *where* the spikes went, only
        // that they're accounted for).
        let s = scalars(100, 0, 100);
        assert!(s.is_conserved());
    }

    #[test]
    fn ffi_aabb_struct_layout_size_is_32_bytes_on_x86_64() {
        // FFI ABI sanity check: Aabb is 24 bytes (6 × f32),
        // support_count is 4 bytes (u32), frame is 4 bytes (u32);
        // total 32 bytes contiguous with no padding. The post-M1
        // steering kernel's C-side struct definition must agree.
        assert_eq!(std::mem::size_of::<ManifoldViewAabbFfi>(), 32);
        assert_eq!(std::mem::align_of::<ManifoldViewAabbFfi>(), 4);
    }

    // ========================================================================
    // M1.2.1: extern "C" accessor + frame-width-narrowing tests
    // ========================================================================

    use crate::entangled_manifold::{
        CausalDriverView, CausalSignal, EntangledManifold, LiningContactView,
        LocalizedSubclusterView, SelectionPolicy, TieBreakerPolicy,
        ViewProvenance,
    };

    #[allow(deprecated)]
    fn provenance(frame: u64) -> ViewProvenance {
        ViewProvenance {
            signal: CausalSignal::SpikeAttributionCount,
            selection: SelectionPolicy::TopK { k: 3 },
            tie_breaker: TieBreakerPolicy::CausalThenResid,
            frame,
        }
    }

    fn build_manifold(frame: u64) -> EntangledManifold {
        let coords = vec![[0.0f32; 3]; 8];
        let driver = CausalDriverView::new(
            &coords,
            vec![0, 1],
            vec![10.0, 9.0],
            provenance(frame),
        )
        .unwrap();
        let lining = LiningContactView::new(
            &coords,
            vec![2, 3, 4],
            vec![5.0, 4.0, 3.0],
            provenance(frame),
        )
        .unwrap();
        let localized = LocalizedSubclusterView::new(
            &coords,
            vec![5],
            vec![1.0],
            provenance(frame),
        )
        .unwrap();
        EntangledManifold::new(driver, lining, localized).unwrap()
    }

    #[test]
    fn frame_width_narrow_u64_to_u32_below_limit_is_identity() {
        // Per-§M9 headroom calculation: at 4 fs/step, u32::MAX
        // corresponds to ~17.16 us simulated time, vastly above any
        // single PRISM run. For any realistic frame value the narrow
        // is identity.
        for f in [0u64, 1, 100, 4_000_000_000, u32::MAX as u64] {
            assert_eq!(super::frame_u64_to_u32_narrow(f), f as u32);
        }
    }

    #[test]
    fn frame_width_narrow_overflow_yields_u32_max_sentinel() {
        // Beyond u32::MAX, the documented overflow sentinel is
        // u32::MAX rather than a silently-wrapped small value that
        // would alias an early frame.
        let beyond = (u32::MAX as u64) + 1;
        assert_eq!(super::frame_u64_to_u32_narrow(beyond), u32::MAX);
        assert_eq!(super::frame_u64_to_u32_narrow(u64::MAX), u32::MAX);
    }

    #[test]
    fn prism_get_causal_driver_aabb_reads_from_manifold() {
        let m = build_manifold(7);
        let mut out = ManifoldViewAabbFfi {
            aabb: Aabb { min: [-1.0; 3], max: [-1.0; 3] },
            support_count: 999,
            frame: 999,
        };
        let rc = unsafe {
            super::prism_get_causal_driver_aabb(&m as *const _, &mut out as *mut _)
        };
        assert_eq!(rc, super::PRISM_AABB_OK);
        assert_eq!(out.support_count, 2);   // driver: 2 residues
        assert_eq!(out.frame, 7);
    }

    #[test]
    fn prism_get_lining_contact_aabb_reads_from_manifold() {
        let m = build_manifold(7);
        let mut out = ManifoldViewAabbFfi {
            aabb: Aabb { min: [-1.0; 3], max: [-1.0; 3] },
            support_count: 999,
            frame: 999,
        };
        let rc = unsafe {
            super::prism_get_lining_contact_aabb(&m as *const _, &mut out as *mut _)
        };
        assert_eq!(rc, super::PRISM_AABB_OK);
        assert_eq!(out.support_count, 3);   // lining: 3 residues
        assert_eq!(out.frame, 7);
    }

    #[test]
    fn prism_get_localized_subcluster_aabb_reads_from_manifold() {
        let m = build_manifold(7);
        let mut out = ManifoldViewAabbFfi {
            aabb: Aabb { min: [-1.0; 3], max: [-1.0; 3] },
            support_count: 999,
            frame: 999,
        };
        let rc = unsafe {
            super::prism_get_localized_subcluster_aabb(&m as *const _, &mut out as *mut _)
        };
        assert_eq!(rc, super::PRISM_AABB_OK);
        assert_eq!(out.support_count, 1);   // localized: 1 residue
        assert_eq!(out.frame, 7);
    }

    #[test]
    fn prism_get_aabb_accessors_reject_null_pointers() {
        let m = build_manifold(7);
        let mut out = ManifoldViewAabbFfi {
            aabb: Aabb { min: [0.0; 3], max: [0.0; 3] },
            support_count: 0,
            frame: 0,
        };
        // Null manifold → error code, no write to out.
        unsafe {
            assert_eq!(
                super::prism_get_causal_driver_aabb(std::ptr::null(), &mut out),
                super::PRISM_AABB_ERR_NULL_POINTER
            );
            assert_eq!(
                super::prism_get_lining_contact_aabb(std::ptr::null(), &mut out),
                super::PRISM_AABB_ERR_NULL_POINTER
            );
            assert_eq!(
                super::prism_get_localized_subcluster_aabb(std::ptr::null(), &mut out),
                super::PRISM_AABB_ERR_NULL_POINTER
            );
            // Null out → error code on a valid manifold.
            assert_eq!(
                super::prism_get_causal_driver_aabb(&m, std::ptr::null_mut()),
                super::PRISM_AABB_ERR_NULL_POINTER
            );
        }
    }

    #[test]
    fn prism_get_aabb_accessors_narrow_overflowing_frame_to_u32_max() {
        // M1.2.1 contract: frame > u32::MAX is narrowed to u32::MAX
        // rather than wrapping. Build a manifold at frame = u32::MAX + 1
        // and verify all three accessors return u32::MAX.
        let f: u64 = (u32::MAX as u64) + 1;
        let m = build_manifold(f);
        let mut out = ManifoldViewAabbFfi {
            aabb: Aabb { min: [0.0; 3], max: [0.0; 3] },
            support_count: 0,
            frame: 0,
        };
        unsafe {
            super::prism_get_causal_driver_aabb(&m, &mut out);
        }
        assert_eq!(out.frame, u32::MAX);
        unsafe {
            super::prism_get_lining_contact_aabb(&m, &mut out);
        }
        assert_eq!(out.frame, u32::MAX);
        unsafe {
            super::prism_get_localized_subcluster_aabb(&m, &mut out);
        }
        assert_eq!(out.frame, u32::MAX);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Confirms the static archive built from
        // crates/prism-nhs/src/cuda/spike_to_cluster_4d.cu linked
        // correctly and the FFI ABI is round-tripping. Sentinel
        // pinned at 0xC0FFEE on the C++ side.
        assert_eq!(super::link_probe(), 0x00C0_FFEE);
    }

    #[test]
    fn law_m1_conservation_of_mass_identity_is_pinned() {
        assert_eq!(LAW_M1_CONSERVATION_OF_MASS.key, "m1_conservation_of_mass");
        assert_eq!(LAW_M1_CONSERVATION_OF_MASS.family, LawFamily::Algebraic);
    }

    #[test]
    fn transform_id_is_pinned() {
        assert_eq!(TRANSFORM_M1_SPIKE_TO_CLUSTER_4D.0, "m1_spike_to_cluster_4d");
    }

    // ========================================================================
    // M1.2.2 — synthetic-input verification of the per-spike kernel
    // ========================================================================
    //
    // Drives the M1.2.1-locked FFI surface (`prism_m1_spike_to_cluster_4d_run`)
    // with a 100-spike synthetic stream whose CPU-side ground-truth
    // assignment is known by construction. Asserts:
    //   (a) cluster_id_per_spike[i] equals the CPU reference for every i.
    //   (b) per_cluster_count[c] is BitExact equal to the CPU reference
    //       count for every cluster c.
    //   (c) background_count is BitExact equal to the CPU reference
    //       out-of-bbox count.
    //   (d) total_attributed_scalar is the documented M1.2.2 placeholder
    //       (zero); CUB DeviceReduce::Sum fills it in M1.2.3, which will
    //       force this assertion to be updated alongside the CUB wire-in.
    // AABB validation is owned by M1.2.3 per M1.2 contract §6.

    #[cfg(feature = "gpu")]
    #[test]
    fn m1_2_2_assign_clusters_synthetic_100_voxel_grid() {
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[m1_2_2] CUDA context creation failed: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.default_stream();

        // 4×4×4 voxel grid, cell_size = 1 Å, bbox = [0, 4]³. 100 spikes:
        //   - 80 in-bbox at integer-cell centers (cycle through cells)
        //   - 20 out-of-bbox (UNCLUSTERED)
        const GRID_X: i32 = 4;
        const GRID_Y: i32 = 4;
        const GRID_Z: i32 = 4;
        const CELL_SIZE: f32 = 1.0;
        const NUM_CLUSTERS: u32 = (GRID_X * GRID_Y * GRID_Z) as u32; // 64
        const NUM_SPIKES: u32 = 100;
        const NUM_INBOUND: u32 = 80;
        const NUM_BACKGROUND: u32 = 20;

        let bbox_min = [0.0f32, 0.0, 0.0];
        let bbox_max = [GRID_X as f32, GRID_Y as f32, GRID_Z as f32];

        let mut positions: Vec<f32> = Vec::with_capacity(NUM_SPIKES as usize * 3);
        let mut cpu_cluster_ids: Vec<u32> = Vec::with_capacity(NUM_SPIKES as usize);

        for i in 0..NUM_INBOUND as i32 {
            // Cycle through 64 cells; first 16 cells get 2 spikes each
            // (i ∈ {0..16, 64..80}), remaining 48 cells get 1 each.
            let cx = i % GRID_X;
            let cy = (i / GRID_X) % GRID_Y;
            let cz = (i / (GRID_X * GRID_Y)) % GRID_Z;
            positions.push(cx as f32 + 0.5);
            positions.push(cy as f32 + 0.5);
            positions.push(cz as f32 + 0.5);
            let cell_id = (cz * GRID_X * GRID_Y + cy * GRID_X + cx) as u32;
            cpu_cluster_ids.push(cell_id);
        }
        for i in 0..NUM_BACKGROUND {
            // Out-of-bbox: x < 0 forces UNCLUSTERED.
            positions.push(-100.0 - i as f32);
            positions.push(0.5);
            positions.push(0.5);
            cpu_cluster_ids.push(u32::MAX);
        }
        assert_eq!(positions.len(), NUM_SPIKES as usize * 3);
        assert_eq!(cpu_cluster_ids.len(), NUM_SPIKES as usize);

        // CPU ground-truth aggregate counts.
        let mut cpu_per_cluster_count = vec![0u64; NUM_CLUSTERS as usize];
        let mut cpu_background_count: u64 = 0;
        for &cid in &cpu_cluster_ids {
            if cid == u32::MAX {
                cpu_background_count += 1;
            } else {
                cpu_per_cluster_count[cid as usize] += 1;
            }
        }

        // Device-side allocation + H2D.
        let mut d_positions = stream
            .alloc_zeros::<f32>(positions.len())
            .expect("alloc d_positions");
        stream
            .memcpy_htod(&positions, &mut d_positions)
            .expect("htod positions");

        let d_cluster_ids = stream
            .alloc_zeros::<u32>(NUM_SPIKES as usize)
            .expect("alloc d_cluster_ids");
        let d_per_cluster_count = stream
            .alloc_zeros::<u64>(NUM_CLUSTERS as usize)
            .expect("alloc d_per_cluster_count");
        let d_total_attributed = stream
            .alloc_zeros::<u64>(1)
            .expect("alloc d_total_attributed");
        let d_background_count = stream
            .alloc_zeros::<u64>(1)
            .expect("alloc d_background_count");
        // d_per_cluster_aabb is part of the M1.2.1-locked FFI surface
        // but not consumed by M1.2.2. Allocate as a u32 scratch buffer
        // matching the byte size of [Aabb; NUM_CLUSTERS] (24 B/Aabb =
        // 6 × u32) and reinterpret-cast at the FFI boundary.
        let d_aabb_scratch = stream
            .alloc_zeros::<u32>(NUM_CLUSTERS as usize * 6)
            .expect("alloc d_aabb_scratch");

        let params = ffi::SpatialHashParams {
            bbox_min,
            bbox_max,
            cell_size: CELL_SIZE,
            grid_dim: [GRID_X, GRID_Y, GRID_Z],
            num_cells: NUM_CLUSTERS,
        };

        // Raw stream + raw device pointers for the C ABI.
        let raw_stream = stream.cu_stream() as usize;
        let (positions_dev,    _g_pos)  = d_positions.device_ptr(&stream);
        let (cluster_ids_dev,  _g_cid)  = d_cluster_ids.device_ptr(&stream);
        let (count_dev,        _g_cnt)  = d_per_cluster_count.device_ptr(&stream);
        let (total_dev,        _g_tot)  = d_total_attributed.device_ptr(&stream);
        let (bg_dev,           _g_bg)   = d_background_count.device_ptr(&stream);
        let (aabb_dev,         _g_aabb) = d_aabb_scratch.device_ptr(&stream);

        // SAFETY: every pointer was obtained from a live CudaSlice whose
        // _g_* guards are held until end of function; raw_stream is the
        // active default stream; params is a stack-allocated #[repr(C)]
        // struct living for the duration of the call (synchronous FFI
        // entry — kernel launches are queued onto stream and the FFI
        // returns; stream.synchronize() below blocks until completion
        // before any guard drops).
        let rc = unsafe {
            ffi::prism_m1_spike_to_cluster_4d_run(
                positions_dev   as *const f32,
                NUM_SPIKES,
                &params as *const ffi::SpatialHashParams,
                raw_stream,
                cluster_ids_dev as *mut u32,
                count_dev       as *mut u64,
                total_dev       as *mut u64,
                bg_dev          as *mut u64,
                aabb_dev        as *mut Aabb,
                NUM_CLUSTERS,
            )
        };
        assert_eq!(
            rc,
            ffi::CUDA_SUCCESS,
            "prism_m1_spike_to_cluster_4d_run returned non-success: {}",
            rc
        );

        stream.synchronize().expect("stream sync");

        // D2H + assertions.
        let mut out_cluster_ids = vec![0u32; NUM_SPIKES as usize];
        stream
            .memcpy_dtoh(&d_cluster_ids, &mut out_cluster_ids)
            .expect("dtoh cluster_ids");

        let mut out_per_cluster_count = vec![0u64; NUM_CLUSTERS as usize];
        stream
            .memcpy_dtoh(&d_per_cluster_count, &mut out_per_cluster_count)
            .expect("dtoh per_cluster_count");

        let mut out_background_count = vec![0u64; 1];
        stream
            .memcpy_dtoh(&d_background_count, &mut out_background_count)
            .expect("dtoh background_count");

        let mut out_total_attributed = vec![0u64; 1];
        stream
            .memcpy_dtoh(&d_total_attributed, &mut out_total_attributed)
            .expect("dtoh total_attributed");

        // (a) Per-spike cluster_id matches CPU reference exactly.
        assert_eq!(
            out_cluster_ids, cpu_cluster_ids,
            "M1.2.2: cluster_id_per_spike differs from CPU reference"
        );
        // (b) Per-cluster count BitExact.
        assert_eq!(
            out_per_cluster_count, cpu_per_cluster_count,
            "M1.2.2: per_cluster_count differs from CPU reference (BitExact violation)"
        );
        // (c) Background count BitExact.
        assert_eq!(
            out_background_count[0], cpu_background_count,
            "M1.2.2: background_count differs from CPU reference (BitExact violation)"
        );
        // (d) total_attributed placeholder pinned at 0 — forces M1.2.3
        // to update this assertion when CUB DeviceReduce::Sum lands.
        assert_eq!(
            out_total_attributed[0], 0,
            "M1.2.2 placeholder violated: total_attributed_scalar must be 0 \
             until M1.2.3 wires CUB DeviceReduce::Sum"
        );
    }
}
