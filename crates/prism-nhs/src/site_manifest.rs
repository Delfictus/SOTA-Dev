//! M1.1 — Per-site canonical container types.
//!
//! This module defines the **per-site** Rust types referenced in the
//! M1 execution contract (B1 §4 binding spec) and the M1 readiness
//! report. It is the type-definitions layer of the M1 lane; FFI
//! declarations, `AuditedTransform` impls, and constructors that take
//! [`crate::entangled_manifold::CausalSortKey`] land in M1.2 / M1.3.
//!
//! # Layering
//!
//! Three distinct types at three distinct layers in the M1
//! architecture:
//!
//! * [`crate::entangled_manifold::EntangledManifold`] — *frame-level*,
//!   M5-conformant, 3 typed role-views (`CausalDriverView`,
//!   `LiningContactView`, `LocalizedSubclusterView`) with LBVH-ready
//!   AABBs. Produced on-device by the M1.2 clustering kernel.
//! * [`SiteManifest`] — *per-site* canonical Rust type. One instance
//!   per detected site. Holds the 8-slot [`CentroidManifold`]
//!   projection plus per-site identity, causal scalars, frame index,
//!   an opaque [`EntangledManifoldId`] reference back to the
//!   frame-level manifold, and the [`CausalSortKey`] used to order
//!   the constituent role-view residues
//!   (`source_manifold_id` and `sort_lineage` together provide the
//!   audit trail back to the producer).
//! * [`CentroidManifold`] — *per-site, 8-slot* projection of the
//!   frame-level role views, named-accessor-only, M5-non-conformant
//!   (point centroids, not AABBs) by design. This is the canonical
//!   per-site centroid container; the legacy
//!   [`crate::spatial_view::CentroidManifold`] (3-slot, point centroids,
//!   `Option<[f32; 3]>` shape) is deprecated as of M1.1 with migration
//!   of the two `persistent_engine.rs` callsites scheduled for the
//!   POST-M1 steering wiring lane.
//!
//! # M1 vs later lanes — what is honest at M1
//!
//! At M1, only the producer's geometric / attribution outputs are
//! honest:
//!
//! * `CentroidManifold::geometric()`, `lining()`, `driver()` are
//!   populated from the corresponding frame-level role-view AABBs
//!   (each AABB contributes its center via `Aabb::center()`).
//! * Phase-dependent slots — `hot_phase`, `cold_phase`, `burst_motion`,
//!   `validation_structural`, `ligand_adjacent_subcluster` — are
//!   `Option::None` at M1. M2 / M3 populate them as their producers
//!   land.
//! * [`CausalScalars::spike_attribution_count`] is honest at M1.
//!   The four floating-point causal fields (`kcc_score`,
//!   `transfer_entropy`, `causal_dg`, `causal_lag`) are initialized
//!   to `f64::NAN`. M2 fills `kcc_score`; M3 fills the rest.
//!
//! No code change is required to "turn on" M2 / M3 fields — the data
//! becoming honest (NaN → finite) is the activation mechanism. This
//! aligns with the role-aware composite causal sort's
//! progressive-population semantics
//! (see [`crate::entangled_manifold::CausalSortKey`]).
//!
//! # Frame-index sentinels
//!
//! [`Centroid3D::frame`] is `u64`. The reserved sentinel value
//! `u64::MAX` means "frame not yet set" (rare; only legitimately
//! used during construction before the frame is known). Frame `0`
//! is a real frame and MUST NOT be used as a placeholder.

use serde::{Deserialize, Serialize};

use crate::entangled_manifold::{Aabb, CausalSortKey, ViewProvenance};
use crate::spatial_view::SpatialView;

// ============================================================================
// Identity newtypes
// ============================================================================

/// Engine-assigned site identifier. Stable within a single MD run;
/// not assumed stable across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteId(pub u32);

/// Engine-assigned cluster identifier. Stable within a single MD run.
/// `u32::MAX` is reserved as the sentinel for *unclustered* spikes per
/// the M1 producer convention (see M1 readiness report §4 CUDA bar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u32);

impl ClusterId {
    /// Sentinel value indicating "this spike was not assigned to any
    /// cluster" / "this site has no cluster yet". Equals `u32::MAX`.
    pub const UNCLUSTERED: ClusterId = ClusterId(u32::MAX);
}

/// Opaque handle from a [`SiteManifest`] back to its source
/// [`crate::entangled_manifold::EntangledManifold`] in the engine's
/// per-frame state. The engine maintains the
/// `(frame, EntangledManifoldId) → EntangledManifold` mapping; this
/// type is intentionally a thin newtype over `u64` to keep
/// [`SiteManifest`] cheap to clone and to avoid coupling the per-site
/// type to the frame-level manifold's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntangledManifoldId(pub u64);

// ============================================================================
// Centroid3D — per-slot 4D-aware centroid value
// ============================================================================

/// Per-slot centroid value with 4D-aware provenance.
///
/// Per the M1 contract (B3): a bare `[f32; 3]` strips both the 4th
/// dimension (frame / time) and the slot provenance (which view
/// produced this centroid). This struct carries both as required
/// fields. Slot-mismatch bugs (e.g. a centroid whose `view ==
/// HotPhase` ending up in the `cold_phase` slot of a
/// [`CentroidManifold`]) are catchable at audit time by comparing
/// `view` to the slot it was placed in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Centroid3D {
    /// Cartesian position in the simulation coordinate frame.
    pub pos: [f32; 3],
    /// Provenance: which [`SpatialView`] produced this centroid.
    /// Used by audit-time slot-consistency checks.
    pub view: SpatialView,
    /// MD frame index this centroid was constructed at. Sentinel
    /// `u64::MAX` means "frame not yet set" (rare, only valid during
    /// construction); never use `0` as a placeholder — frame `0` is
    /// a real frame.
    pub frame: u64,
}

impl Centroid3D {
    /// Sentinel value for [`Self::frame`] meaning "frame not yet set".
    /// Used only during construction before the frame is known;
    /// see the module-level "Frame-index sentinels" section.
    pub const FRAME_UNSET: u64 = u64::MAX;

    /// Construct a centroid with explicit frame and view provenance.
    pub fn new(pos: [f32; 3], view: SpatialView, frame: u64) -> Self {
        Self { pos, view, frame }
    }
}

// ============================================================================
// CentroidManifold — 8-slot per-site centroid projection
// ============================================================================

/// Per-site, 8-slot centroid projection of the frame-level
/// [`crate::entangled_manifold::EntangledManifold`].
///
/// All slots are `Option<Centroid3D>`. Absence is `None`, never a
/// sentinel coordinate or NaN-vector. Fields are private; access is
/// through named slot accessors (`geometric()`, `lining()`, …) and
/// matching `set_*` methods. There is no bare `.centroid()` method
/// and no `pub` field access — this is enforced at the module
/// boundary by visibility, so a caller that asks for a slot must
/// name it explicitly.
///
/// # The eight slots
///
/// | Slot | Honest at | Source |
/// |---|---|---|
/// | `geometric` | M1 | derived from frame-level `CausalDriverView` AABB center |
/// | `lining` | M1 | derived from frame-level `LiningContactView` AABB center |
/// | `driver` | M1 | derived from frame-level `CausalDriverView` AABB center (subset of `geometric`) |
/// | `hot_phase` | M2 | populated by phase-segregation transform |
/// | `cold_phase` | M2 | populated by phase-segregation transform |
/// | `burst_motion` | M3 | populated by burst-motion / TIDE transform |
/// | `validation_structural` | M2 | populated when a reference structure is supplied |
/// | `ligand_adjacent_subcluster` | M2 | populated when ligand context is supplied |
///
/// # Naming reconciliation against the M1 contract
///
/// The contract used `burst_motion_4d` and `ligand_adjacent`; this
/// module aligns to the existing
/// [`SpatialView`] variants `BurstMotion` and
/// `LigandAdjacentSubcluster` (per non-blocking decision (2) in the
/// B1/B2/B3 resolution turn). Zero rename of existing
/// serialization labels.
///
/// # The dropped ninth slot
///
/// The contract initially listed a `localized` slot then dropped it
/// (per B2 in the resolution turn): the `LocalizedSubclusterView`
/// data is frame-level only; per-site, it is redundant with
/// `driver`. Final shape is exactly 8 slots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentroidManifold {
    geometric: Option<Centroid3D>,
    lining: Option<Centroid3D>,
    driver: Option<Centroid3D>,
    hot_phase: Option<Centroid3D>,
    cold_phase: Option<Centroid3D>,
    burst_motion: Option<Centroid3D>,
    validation_structural: Option<Centroid3D>,
    ligand_adjacent_subcluster: Option<Centroid3D>,
}

impl CentroidManifold {
    /// Construct an empty manifold with every slot `None`.
    pub fn new() -> Self {
        Self::default()
    }

    // --- accessors (read) ---

    /// Geometric centroid slot (voxel-mass / intensity²-weighted).
    /// Honest at M1.
    pub fn geometric(&self) -> Option<Centroid3D> {
        self.geometric
    }
    /// Lining-residue centroid slot. Honest at M1.
    pub fn lining(&self) -> Option<Centroid3D> {
        self.lining
    }
    /// Driver-residue centroid slot. Honest at M1.
    pub fn driver(&self) -> Option<Centroid3D> {
        self.driver
    }
    /// Hot-phase centroid slot. `None` at M1; populated by M2.
    pub fn hot_phase(&self) -> Option<Centroid3D> {
        self.hot_phase
    }
    /// Cold-phase centroid slot. `None` at M1; populated by M2.
    pub fn cold_phase(&self) -> Option<Centroid3D> {
        self.cold_phase
    }
    /// Burst-motion (4D-aware) centroid slot. `None` at M1;
    /// populated by M3.
    pub fn burst_motion(&self) -> Option<Centroid3D> {
        self.burst_motion
    }
    /// Structural-validation-aligned centroid slot. `None` at M1;
    /// populated by M2 when a reference structure is supplied.
    pub fn validation_structural(&self) -> Option<Centroid3D> {
        self.validation_structural
    }
    /// Ligand-adjacent sub-cluster centroid slot. `None` at M1;
    /// populated by M2 when ligand context is present.
    pub fn ligand_adjacent_subcluster(&self) -> Option<Centroid3D> {
        self.ligand_adjacent_subcluster
    }

    // --- accessors (write) ---

    /// Populate or overwrite the geometric slot. Caller is
    /// responsible for ensuring `c.view == SpatialView::GeometricVoxelMass`;
    /// audit-time consistency checks land in M1.2 alongside the
    /// `AuditedTransform::verify` impl.
    pub fn set_geometric(&mut self, c: Centroid3D) {
        self.geometric = Some(c);
    }
    /// Populate or overwrite the lining slot.
    pub fn set_lining(&mut self, c: Centroid3D) {
        self.lining = Some(c);
    }
    /// Populate or overwrite the driver slot.
    pub fn set_driver(&mut self, c: Centroid3D) {
        self.driver = Some(c);
    }
    /// Populate or overwrite the hot-phase slot. Typically called
    /// by the M2 phase-segregation transform.
    pub fn set_hot_phase(&mut self, c: Centroid3D) {
        self.hot_phase = Some(c);
    }
    /// Populate or overwrite the cold-phase slot. Typically called
    /// by the M2 phase-segregation transform.
    pub fn set_cold_phase(&mut self, c: Centroid3D) {
        self.cold_phase = Some(c);
    }
    /// Populate or overwrite the burst-motion slot. Typically called
    /// by the M3 burst-motion / TIDE transform.
    pub fn set_burst_motion(&mut self, c: Centroid3D) {
        self.burst_motion = Some(c);
    }
    /// Populate or overwrite the validation-structural slot.
    pub fn set_validation_structural(&mut self, c: Centroid3D) {
        self.validation_structural = Some(c);
    }
    /// Populate or overwrite the ligand-adjacent sub-cluster slot.
    pub fn set_ligand_adjacent_subcluster(&mut self, c: Centroid3D) {
        self.ligand_adjacent_subcluster = Some(c);
    }

    /// Number of slots currently populated. Useful for audits and
    /// progress tracking as M2 / M3 land.
    pub fn populated_count(&self) -> usize {
        let mut n = 0usize;
        for slot in [
            self.geometric.is_some(),
            self.lining.is_some(),
            self.driver.is_some(),
            self.hot_phase.is_some(),
            self.cold_phase.is_some(),
            self.burst_motion.is_some(),
            self.validation_structural.is_some(),
            self.ligand_adjacent_subcluster.is_some(),
        ] {
            if slot {
                n += 1;
            }
        }
        n
    }
}

// ============================================================================
// CausalScalars — per-site causal scalar tensor
// ============================================================================

/// Per-site causal scalar tensor: M1 / M2 / M3 producers each fill in
/// their column. At M1, only [`Self::spike_attribution_count`] is
/// honest; the four floating-point fields are initialized to
/// `f64::NAN` and become finite as M2 / M3 land.
///
/// `f64::NAN` is the canonical "not yet produced" sentinel. The
/// role-aware composite sort
/// (see [`crate::entangled_manifold::CausalSortKey`]) walks priority
/// lists and selects the first field where ALL residues have
/// non-NaN values, so the NaN sentinel doubles as the activation
/// gate for higher-priority sort fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CausalScalars {
    /// Count of spike events for which this site is in the attributed
    /// support set. Honest at M1.
    pub spike_attribution_count: u64,
    /// KCC (kinetic-causal contribution) score. NaN at M1; honest at M2.
    pub kcc_score: f64,
    /// Directed transfer entropy magnitude. NaN at M1; honest at M3.
    pub transfer_entropy: f64,
    /// Causal Δ-G (Gibbs-style energy contribution). NaN at M1;
    /// honest at M3.
    pub causal_dg: f64,
    /// Causal lag (peak-correlation lag in time units). NaN at M1;
    /// honest at M3.
    pub causal_lag: f64,
}

impl CausalScalars {
    /// Construct with M1's `spike_attribution_count` populated; all
    /// downstream causal fields initialized to `f64::NAN` until
    /// M2 / M3 producers land.
    pub fn new_m1(spike_attribution_count: u64) -> Self {
        Self {
            spike_attribution_count,
            kcc_score: f64::NAN,
            transfer_entropy: f64::NAN,
            causal_dg: f64::NAN,
            causal_lag: f64::NAN,
        }
    }

    /// Empty placeholder: no honest fields. Useful as a default
    /// before any producer has filled in even the M1 column.
    pub fn empty() -> Self {
        Self {
            spike_attribution_count: 0,
            kcc_score: f64::NAN,
            transfer_entropy: f64::NAN,
            causal_dg: f64::NAN,
            causal_lag: f64::NAN,
        }
    }
}

// ============================================================================
// SiteIdentity — per-site identity bundle
// ============================================================================

/// Per-site identity bundle. Carries the engine's [`SiteId`] /
/// [`ClusterId`] plus the [`ViewProvenance`] from the role-view
/// that this site was projected from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteIdentity {
    /// Engine-assigned site identifier.
    pub site_id: SiteId,
    /// Engine-assigned cluster identifier (cluster the site's
    /// driver residue belongs to).
    pub cluster_id: ClusterId,
    /// Provenance from the role-view this site was projected from.
    pub provenance: ViewProvenance,
}

// ============================================================================
// SiteManifest — top-level per-site canonical type
// ============================================================================

/// Canonical per-site Rust type. One [`SiteManifest`] instance per
/// detected site. Per the B1 binding spec, `SiteManifest` does NOT
/// own a copy of the frame-level
/// [`crate::entangled_manifold::EntangledManifold`]; it references it
/// through [`Self::source_manifold_id`]. The engine maintains the
/// `(frame, EntangledManifoldId) → EntangledManifold` mapping.
///
/// Construction at M1 is from the M1.2 producer's per-site projection
/// of the frame-level manifold's role-views. Population semantics:
///
/// * [`Self::identity`], [`Self::frame`], [`Self::source_manifold_id`]
///   — populate from M1's clustering kernel output.
/// * [`Self::centroids`] — `geometric`, `lining`, `driver` populated
///   from M1's role-view AABBs (each AABB → derived centroid via
///   `Aabb::center()`). Phase slots are `None` at M1 and progressively
///   populated by M2 / M3.
/// * [`Self::causal_scalars`] — `spike_attribution_count` honest at
///   M1; the floating-point causal fields are `NaN` at M1 and become
///   honest as M2 / M3 land.
/// * [`Self::sort_lineage`] — the [`CausalSortKey`] used to order the
///   constituent role-view residues at view-construction time.
///   Provides the audit trail for which signal was the live sort
///   field at the time the site was projected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteManifest {
    /// Per-site identity bundle.
    pub identity: SiteIdentity,
    /// 8-slot centroid projection of the frame-level role-views.
    pub centroids: CentroidManifold,
    /// Per-site causal scalar tensor; M1 honest column, M2 / M3
    /// columns NaN until those lanes land.
    pub causal_scalars: CausalScalars,
    /// MD frame index this site was projected at.
    pub frame: u64,
    /// Opaque reference back to the frame-level
    /// [`crate::entangled_manifold::EntangledManifold`] this site
    /// was projected from.
    pub source_manifold_id: EntangledManifoldId,
    /// Sort key used to order the constituent role-view residues at
    /// view-construction time. Audit-trail field.
    pub sort_lineage: CausalSortKey,

    // ─── Anti-Greenfield § 3 — Spatiotemporal Interferometer extension ────
    // Surgically additive, all `Option`-typed with `serde(skip_serializing
    // _if = None)`. Legacy consumers (Python reporting, static PDB viewers,
    // existing JSON schema readers) see the same shape they always saw on
    // the legacy clustering path; `None` everywhere ⇒ no key emitted.
    // Populated only when the InterferometricAdjudicator (T0/T2) is wired
    // into the post-clustering path of the captured WHILE graph.

    /// SO(3) geometry-plane power spectrum `C_l` for `l = 0..5`,
    /// rotationally invariant (RECT-3.1.b output). The KL divergence
    /// in the Adjudicator kernel reads these values from the relaxed
    /// and perturbed `ContactShellTile`s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_shell_geo_power_spectrum: Option<[f32; 6]>,

    /// Latest computed Δ_AB KL divergence between the relaxed (P)
    /// and perturbed (Q) manifolds (T2 output). `None` on the
    /// legacy path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_divergence: Option<f32>,

    /// F1 SWITCH selector — [`crate::pre_rank::AdjudicationCode`]-compatible
    /// integer (0=Prune, 1=Construct, 2=Violation). `None` on the
    /// legacy path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_code: Option<u32>,

    /// Adjudicator-kernel elapsed time in nanoseconds, computed as
    /// `stop_clock - start_clock` from the CudaEvent harness (T4).
    /// Gating threshold: the < 5 μs CSR-H invariant. `None` if the
    /// telemetry harness is not active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_elapsed_ns: Option<u64>,
}

impl SiteManifest {
    /// LBVH-3 constructor — build a per-site manifest from the cluster's
    /// LBVH-derived AABB.
    ///
    /// Per the operator's RECT-4 / LBVH-3 mandate (2026-04-29) and
    /// Anti-Greenfield Doctrine §2.3 (extend, don't duplicate): this is
    /// the canonical bridge from the M1 producer's per-cluster AABB
    /// (output of [`crate::lbvh_tree`]'s bottom-up reduce, or
    /// equivalently the per-cluster AABB column from the M1 producer's
    /// CUB SegmentedReduce) to the per-site
    /// [`CentroidManifold`] 8-slot canonical container.
    ///
    /// # Slot population (M1 honest)
    ///
    /// Three slots are populated from the same LBVH AABB at M1 (per
    /// the [`SiteManifest`] type-level docs):
    ///
    /// | Slot | Value | View tag |
    /// |---|---|---|
    /// | `geometric` | `aabb.center()` | [`SpatialView::GeometricVoxelMass`] |
    /// | `lining`    | `aabb.center()` | [`SpatialView::LiningResidues`] |
    /// | `driver`    | `aabb.center()` | [`SpatialView::DriverResidues`] |
    ///
    /// The remaining 5 slots are intentionally `None` and are populated
    /// by later transforms (M2: hot/cold-phase, validation_structural,
    /// ligand_adjacent_subcluster; M3: burst_motion).
    ///
    /// # Spatial provenance preservation
    ///
    /// `source_manifold_id` carries the frame-level
    /// [`crate::entangled_manifold::EntangledManifold`] handle so the
    /// site can be back-referenced during audit. `sort_lineage` carries
    /// the [`CausalSortKey`] used to order the role-view residues at
    /// frame-build time — preserves the LBVH's spatial-provenance chain
    /// from Morton encoding through cluster assignment to site.
    ///
    /// # The 4 RECT-3 / Adjudicator extension fields
    ///
    /// `contact_shell_geo_power_spectrum`, `adjudicator_divergence`,
    /// `adjudicator_code`, `adjudicator_elapsed_ns` default to `None`.
    /// They are populated post-construction by the SO(3) stamping pass
    /// (`stamp_geo_power_spectrum_into_sites` in `so3_project.rs`) and
    /// by the Adjudicator (Claude-2 lane).
    pub fn from_lbvh_cluster_aabb(
        site_id: SiteId,
        cluster_id: ClusterId,
        aabb: &Aabb,
        source_manifold_id: EntangledManifoldId,
        provenance: ViewProvenance,
        sort_lineage: CausalSortKey,
        spike_attribution_count: u64,
        frame: u64,
    ) -> Self {
        let center = aabb.center();
        let mut centroids = CentroidManifold::new();
        centroids.set_geometric(Centroid3D::new(center, SpatialView::GeometricVoxelMass, frame));
        centroids.set_lining   (Centroid3D::new(center, SpatialView::LiningResidues,     frame));
        centroids.set_driver   (Centroid3D::new(center, SpatialView::DriverResidues,     frame));

        Self {
            identity: SiteIdentity {
                site_id,
                cluster_id,
                provenance,
            },
            centroids,
            causal_scalars: CausalScalars::new_m1(spike_attribution_count),
            frame,
            source_manifold_id,
            sort_lineage,
            // RECT-3 extension defaults — populated by the SO(3)
            // stamping pass (`crate::so3_project::stamp_geo_power_spectrum_into_sites`)
            // and the Adjudicator (Claude-2 lane).
            contact_shell_geo_power_spectrum: None,
            adjudicator_divergence: None,
            adjudicator_code: None,
            adjudicator_elapsed_ns: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entangled_manifold::{
        CausalSignal, IdentityTieBreaker, SelectionPolicy, SortField,
        TieBreakerPolicy,
    };

    fn provenance() -> ViewProvenance {
        // Uses the deprecated TieBreakerPolicy variant on purpose;
        // existing producer paths still emit it. New code uses
        // IdentityTieBreaker via CausalSortKey.
        #[allow(deprecated)]
        ViewProvenance {
            signal: CausalSignal::SpikeAttributionCount,
            selection: SelectionPolicy::TopK { k: 3 },
            tie_breaker: TieBreakerPolicy::CausalThenResid,
            frame: 7,
        }
    }

    fn sort_key() -> CausalSortKey {
        CausalSortKey {
            priorities: vec![
                SortField::TransferEntropy,
                SortField::KccScore,
                SortField::SpikeAttributionCount,
            ],
            identity_tiebreaker: IdentityTieBreaker::ChainResidAtom,
        }
    }

    #[test]
    fn cluster_id_unclustered_sentinel_is_u32_max() {
        // M1 producer convention: UINT32_MAX = "this spike has no
        // cluster." Match the sentinel value exactly.
        assert_eq!(ClusterId::UNCLUSTERED.0, u32::MAX);
    }

    #[test]
    fn centroid_3d_frame_unset_sentinel_is_u64_max() {
        // Per M1 contract B3: u64::MAX is the only legitimate
        // "frame not yet set" sentinel. Frame 0 is a real frame.
        assert_eq!(Centroid3D::FRAME_UNSET, u64::MAX);
    }

    #[test]
    fn centroid_manifold_default_has_zero_populated_slots() {
        let m = CentroidManifold::new();
        assert_eq!(m.populated_count(), 0);
        assert!(m.geometric().is_none());
        assert!(m.lining().is_none());
        assert!(m.driver().is_none());
        assert!(m.hot_phase().is_none());
        assert!(m.cold_phase().is_none());
        assert!(m.burst_motion().is_none());
        assert!(m.validation_structural().is_none());
        assert!(m.ligand_adjacent_subcluster().is_none());
    }

    #[test]
    fn centroid_manifold_set_then_get_round_trips() {
        let mut m = CentroidManifold::new();
        let c = Centroid3D::new([1.0, 2.0, 3.0], SpatialView::GeometricVoxelMass, 7);
        m.set_geometric(c);
        assert_eq!(m.populated_count(), 1);
        let got = m.geometric().expect("geometric set above");
        assert_eq!(got.pos, [1.0, 2.0, 3.0]);
        assert_eq!(got.view, SpatialView::GeometricVoxelMass);
        assert_eq!(got.frame, 7);
    }

    #[test]
    fn centroid_manifold_eight_slots_independently_populable() {
        // Set each of the 8 slots once; populated_count must climb
        // monotonically from 0 to 8. This is the M1 contract's
        // "Eight-slot CentroidManifold" guarantee, mechanically.
        let mut m = CentroidManifold::new();
        let v = [SpatialView::GeometricVoxelMass,
                 SpatialView::LiningResidues,
                 SpatialView::DriverResidues,
                 SpatialView::HotPhase,
                 SpatialView::ColdPhase,
                 SpatialView::BurstMotion,
                 SpatialView::ValidationStructural,
                 SpatialView::LigandAdjacentSubcluster];
        let c = |i: usize, view: SpatialView| Centroid3D::new(
            [i as f32, 0.0, 0.0], view, 0);
        m.set_geometric(c(0, v[0]));
        assert_eq!(m.populated_count(), 1);
        m.set_lining(c(1, v[1]));
        assert_eq!(m.populated_count(), 2);
        m.set_driver(c(2, v[2]));
        assert_eq!(m.populated_count(), 3);
        m.set_hot_phase(c(3, v[3]));
        assert_eq!(m.populated_count(), 4);
        m.set_cold_phase(c(4, v[4]));
        assert_eq!(m.populated_count(), 5);
        m.set_burst_motion(c(5, v[5]));
        assert_eq!(m.populated_count(), 6);
        m.set_validation_structural(c(6, v[6]));
        assert_eq!(m.populated_count(), 7);
        m.set_ligand_adjacent_subcluster(c(7, v[7]));
        assert_eq!(m.populated_count(), 8);
    }

    #[test]
    fn causal_scalars_new_m1_has_only_attribution_count_honest() {
        // M1 producer fills spike_attribution_count; all other
        // causal fields are NaN until M2 / M3.
        let s = CausalScalars::new_m1(42);
        assert_eq!(s.spike_attribution_count, 42);
        assert!(s.kcc_score.is_nan());
        assert!(s.transfer_entropy.is_nan());
        assert!(s.causal_dg.is_nan());
        assert!(s.causal_lag.is_nan());
    }

    #[test]
    fn causal_scalars_empty_has_zero_count_and_all_nan_floats() {
        let s = CausalScalars::empty();
        assert_eq!(s.spike_attribution_count, 0);
        assert!(s.kcc_score.is_nan());
        assert!(s.transfer_entropy.is_nan());
        assert!(s.causal_dg.is_nan());
        assert!(s.causal_lag.is_nan());
    }

    #[test]
    fn site_manifest_round_trips_construction() {
        let m = SiteManifest {
            identity: SiteIdentity {
                site_id: SiteId(0),
                cluster_id: ClusterId(7),
                provenance: provenance(),
            },
            centroids: CentroidManifold::new(),
            causal_scalars: CausalScalars::new_m1(100),
            frame: 7,
            source_manifold_id: EntangledManifoldId(42),
            sort_lineage: sort_key(),
            contact_shell_geo_power_spectrum: None,
            adjudicator_divergence: None,
            adjudicator_code: None,
            adjudicator_elapsed_ns: None,
        };
        assert_eq!(m.identity.site_id.0, 0);
        assert_eq!(m.identity.cluster_id.0, 7);
        assert_eq!(m.frame, 7);
        assert_eq!(m.source_manifold_id.0, 42);
        assert_eq!(m.causal_scalars.spike_attribution_count, 100);
        assert_eq!(m.centroids.populated_count(), 0);
    }

    // ────────────────────────────────────────────────────────────────────
    // LBVH-3 — from_lbvh_cluster_aabb constructor pins the M1
    // 3-honest-slot population from a per-cluster AABB.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn lbvh3_from_cluster_aabb_populates_three_m1_honest_slots() {
        let aabb = Aabb { min: [0.0, -2.0, 1.0], max: [4.0, 2.0, 5.0] };
        let m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(11),
            ClusterId(11),
            &aabb,
            EntangledManifoldId(123),
            provenance(),
            sort_key(),
            42,
            7,
        );

        // Identity round-trips.
        assert_eq!(m.identity.site_id.0, 11);
        assert_eq!(m.identity.cluster_id.0, 11);
        assert_eq!(m.frame, 7);
        assert_eq!(m.source_manifold_id.0, 123);
        assert_eq!(m.causal_scalars.spike_attribution_count, 42);

        // 3 of 8 slots honest at M1.
        assert_eq!(m.centroids.populated_count(), 3);

        // Centre of [0, -2, 1]..[4, 2, 5] = (2, 0, 3). All 3 M1 slots
        // share the same value at M1 (per the type-level docs); the
        // per-view discrimination is in the SpatialView tag.
        let expected = [2.0_f32, 0.0, 3.0];
        let g = m.centroids.geometric().unwrap();
        let l = m.centroids.lining().unwrap();
        let d = m.centroids.driver().unwrap();
        assert_eq!(g.pos, expected);
        assert_eq!(l.pos, expected);
        assert_eq!(d.pos, expected);
        assert_eq!(g.view, SpatialView::GeometricVoxelMass);
        assert_eq!(l.view, SpatialView::LiningResidues);
        assert_eq!(d.view, SpatialView::DriverResidues);
        assert_eq!(g.frame, 7);
        assert_eq!(l.frame, 7);
        assert_eq!(d.frame, 7);

        // RECT-3 / Adjudicator extension fields default to None.
        assert!(m.contact_shell_geo_power_spectrum.is_none());
        assert!(m.adjudicator_divergence.is_none());
        assert!(m.adjudicator_code.is_none());
        assert!(m.adjudicator_elapsed_ns.is_none());

        // 5 of 8 slots remain unset — populated by M2 / M3 transforms.
        assert!(m.centroids.hot_phase().is_none());
        assert!(m.centroids.cold_phase().is_none());
        assert!(m.centroids.burst_motion().is_none());
        assert!(m.centroids.validation_structural().is_none());
        assert!(m.centroids.ligand_adjacent_subcluster().is_none());
    }

    #[test]
    fn lbvh3_point_cluster_aabb_round_trips_centroid() {
        // Degenerate point AABB (single-spike cluster) → centroid is
        // the spike position itself.
        let aabb = Aabb { min: [3.5, -1.0, 7.25], max: [3.5, -1.0, 7.25] };
        let m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(0),
            ClusterId(0),
            &aabb,
            EntangledManifoldId(0),
            provenance(),
            sort_key(),
            1,
            0,
        );
        assert_eq!(m.centroids.geometric().unwrap().pos, [3.5, -1.0, 7.25]);
    }
}
