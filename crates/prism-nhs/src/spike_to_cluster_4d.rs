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
}
