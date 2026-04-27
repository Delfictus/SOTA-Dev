//! Entangled Transform — typed manifold views with strict AABBs.
//!
//! This module is the type-definition layer for the architecture
//! described in `docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md`. It
//! satisfies blueprint mandates **M4** (strict struct typing for
//! 2+2+2+2 graph prep) and **M5** (LBVH-ready, support-only AABBs).
//! It does NOT yet implement the producer (`SpikeToCluster4D`,
//! mandate M1) or any CUDA / FFI code — those land in subsequent
//! commits, layered on top of these types.
//!
//! # Relationship to existing modules
//!
//! * [`crate::spatial_view::CentroidManifold`] — the *legacy* per-site
//!   container that holds **point** centroids per [`crate::spatial_view::SpatialView`].
//!   It is M5-non-conformant by design (point-centroids are
//!   insufficient for LBVH). We retain it for downstream readers that
//!   were migrated in Phase 1; new code SHOULD use [`EntangledManifold`].
//! * [`crate::causome::CausomeBlock`] — the per-site Pillar-2 typed
//!   block (driver residue id + KCC sub-scores). [`CausalDriverView`]
//!   here is the *spatial* counterpart to that scalar block: same
//!   driver semantics, lifted to a deterministically-ordered support
//!   set with an AABB.
//!
//! # What is in this commit (and what is not)
//!
//! Present:
//!
//! * [`Aabb`] — POD, `repr(C)`, support-only constructor.
//! * [`CausalSignal`] — typed enumeration of authoritative selection
//!   signals (M3).
//! * [`SelectionPolicy`], [`TieBreakerPolicy`], [`ViewProvenance`] —
//!   provenance carried on every view (M3 + M6 audit).
//! * [`CausalDriverView`], [`LiningContactView`], [`LocalizedSubclusterView`] —
//!   typed views (M4) with SoA-layout support sets (M4 1:1 tensor
//!   channel mapping).
//! * [`EntangledManifold`] — top-level container of the three role
//!   views for a single MD frame.
//!
//! Deliberately absent (later commits):
//!
//! * Producer logic (`SpikeToCluster4D`, M1) — selection from spike
//!   tensors and KCC tensors lives in a separate commit.
//! * CUDA kernels and FFI shims — same.
//! * Any conversion to / from the legacy [`crate::spatial_view::CentroidManifold`].
//!   That bridge belongs to the producer, not the type module.
//!
//! # Memory layout notes (M4)
//!
//! Each view stores its support set in **structure-of-arrays** form:
//! `support: Vec<i32>`, `causal_values: Vec<f32>`. The two vectors
//! are parallel (same length, same indexing). When the GPU producer
//! lands, each `Vec` maps directly to one device-side tensor channel
//! via a single `cudaMemcpy` — no host-side repack. AoS would force
//! a transpose at upload; we avoid it by construction.
//!
//! [`Aabb`] is `#[repr(C)]` so its layout is FFI-stable and matches
//! the Morton-encoder kernel's expected input. The view structs
//! themselves are not `repr(C)` because they own [`Vec`]s, whose
//! layout (pointer + len + capacity) is not stable across compilers
//! and is irrelevant for tensor mapping — what gets uploaded is the
//! `Vec`'s data pointer, which is well-defined.

use serde::{Deserialize, Serialize};

/// Engine residue identifier. Matches the convention used by
/// [`crate::causome::CausomeBlock`] and the binding-site JSON schema.
pub type ResidueId = i32;

/// MD frame index (engine step / nominal MD step).
pub type FrameIndex = u64;

/// Authoritative causal signal used to select a view's support set.
///
/// Per blueprint **M3**, view construction MUST sort residues by one of
/// these signals; geometric proximity to a legacy scalar centroid is
/// FORBIDDEN. The variant chosen at construction time is recorded on
/// the view via [`ViewProvenance::signal`] so downstream readers can
/// audit the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalSignal {
    /// Count of spike events whose attributed support set includes
    /// this residue. Integer-valued upstream; cast to `f32` at view
    /// construction time so the SoA `causal_values` channel is
    /// uniformly typed.
    SpikeAttributionCount,
    /// KCC (kinetic-causal contribution) score per residue, as
    /// produced by the KCC analysis kernels.
    KccScore,
    /// Directed transfer entropy, residue → site (incoming).
    /// Reserved for the producer-repair / TE lane.
    TransferEntropyIn,
    /// Directed transfer entropy, site → residue (outgoing).
    /// Reserved for the producer-repair / TE lane.
    TransferEntropyOut,
}

impl CausalSignal {
    /// Stable snake_case identifier for provenance / JSON labels.
    /// Pinned by the `causal_signal_label_is_stable` unit test;
    /// downstream auditors depend on the exact strings.
    pub fn as_str(&self) -> &'static str {
        match self {
            CausalSignal::SpikeAttributionCount => "spike_attribution_count",
            CausalSignal::KccScore => "kcc_score",
            CausalSignal::TransferEntropyIn => "te_in",
            CausalSignal::TransferEntropyOut => "te_out",
        }
    }
}

/// How the view's support set was selected from the ranked residue
/// list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SelectionPolicy {
    /// Take the top-k residues by signal value (descending).
    TopK {
        /// Number of residues to retain.
        k: u32,
    },
    /// Take all residues with signal value above (or equal to) the
    /// threshold.
    Threshold {
        /// Inclusive lower-bound on the causal signal value.
        value: f32,
    },
}

/// Tie-breaker policy used when sorting residues by causal signal.
///
/// Per blueprint **M6**, every sort site MUST use a named, deterministic
/// tie-breaker so Morton codes are stable across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TieBreakerPolicy {
    /// Primary: signal value (descending). Secondary: residue id
    /// (ascending). This is the default policy for all single-chain
    /// targets and any merged-topology target whose engine residue
    /// ids are globally unique.
    CausalThenResid,
    /// Primary: signal value (descending). Secondary: residue id
    /// (ascending). Tertiary: chain id (ascending). For multichain
    /// targets where engine residue ids may collide across chains.
    /// Implementations MAY treat this as identical to
    /// [`Self::CausalThenResid`] when the topology has no chain
    /// information.
    CausalThenResidThenChain,
}

impl TieBreakerPolicy {
    /// Stable snake_case identifier for provenance / JSON labels.
    /// Pinned by the `deterministic_tie_breaker_policy_label_is_stable` unit test.
    pub fn as_str(&self) -> &'static str {
        match self {
            TieBreakerPolicy::CausalThenResid => "causal_then_resid",
            TieBreakerPolicy::CausalThenResidThenChain => "causal_then_resid_then_chain",
        }
    }
}

/// Provenance of a view's support set. Stored on every view per
/// blueprint **M4** (compile-time provenance) and **M3** (causal signal
/// must be explicit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewProvenance {
    /// Which causal signal selected the support set (M3).
    pub signal: CausalSignal,
    /// How the support set was cut from the ranked list.
    pub selection: SelectionPolicy,
    /// Tie-breaker used during the sort (M6).
    pub tie_breaker: TieBreakerPolicy,
    /// MD frame index this view was constructed at.
    pub frame: FrameIndex,
}

/// Axis-aligned bounding box in the simulation coordinate frame.
///
/// Per blueprint **M5**, an [`Aabb`] attached to a view MUST be derived
/// **only** from that view's support set — no inflation, no padding,
/// no inclusion of residues outside the support set, no derivation
/// from a legacy scalar centroid.
///
/// `repr(C)` and `[f32; 3]` are chosen so the type is FFI-stable and
/// matches the Morton-encoder kernel's expected layout. `f32` is
/// sufficient for LBVH construction; the AABB is not the place to
/// pay for `f64`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    /// Per-axis minimum coordinate of the box, in the simulation
    /// coordinate frame. `min[d] <= max[d]` for every axis `d` on a
    /// valid box constructed by [`Aabb::from_support_set`].
    pub min: [f32; 3],
    /// Per-axis maximum coordinate of the box. See [`Self::min`].
    pub max: [f32; 3],
}

/// Errors produced when constructing a view or its [`Aabb`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifoldError {
    /// The view's support set was empty. A view with no support is
    /// undefined and MUST NOT be constructed; the engine should halt
    /// with this error rather than fall back to a geometric default
    /// (M3).
    EmptySupport,
    /// `support.len() != causal_values.len()`. The two SoA channels
    /// must be parallel.
    SoaLengthMismatch {
        /// Length of the `support` vector reported by the caller.
        support_len: usize,
        /// Length of the `causal_values` vector reported by the caller.
        values_len: usize,
    },
    /// A residue id in the support set was out of range for the
    /// supplied coordinate slice. The producer is responsible for
    /// passing a coordinate slice that covers every referenced
    /// residue.
    SupportIndexOutOfRange {
        /// The offending residue id.
        resid: ResidueId,
        /// Length of the coordinate slice the producer passed.
        coord_len: usize,
    },
}

impl std::fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifoldError::EmptySupport => {
                write!(f, "view support set is empty (M3: no geometric fallback)")
            }
            ManifoldError::SoaLengthMismatch { support_len, values_len } => {
                write!(
                    f,
                    "SoA length mismatch: support={support_len} values={values_len}"
                )
            }
            ManifoldError::SupportIndexOutOfRange { resid, coord_len } => {
                write!(
                    f,
                    "support residue id {resid} out of range for coord slice of length {coord_len}"
                )
            }
        }
    }
}

impl std::error::Error for ManifoldError {}

impl Aabb {
    /// Construct an AABB from a support set's coordinates.
    ///
    /// `coords` is indexed by residue id (0-based); `support` lists
    /// the residue ids belonging to the view. The returned AABB is
    /// the tight axis-aligned box over `coords[support[i]]` for every
    /// `i` in `0..support.len()`.
    ///
    /// # Errors
    ///
    /// * [`ManifoldError::EmptySupport`] if `support.is_empty()`.
    /// * [`ManifoldError::SupportIndexOutOfRange`] if any `support[i]`
    ///   is negative or `>= coords.len()`.
    ///
    /// # Blueprint M5
    ///
    /// This constructor is the single allowed entry point for
    /// AABB-from-view computation. Callers MUST NOT inflate, pad,
    /// or post-process the returned box.
    pub fn from_support_set(
        coords: &[[f32; 3]],
        support: &[ResidueId],
    ) -> Result<Self, ManifoldError> {
        if support.is_empty() {
            return Err(ManifoldError::EmptySupport);
        }
        let coord_len = coords.len();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &resid in support {
            if resid < 0 || (resid as usize) >= coord_len {
                return Err(ManifoldError::SupportIndexOutOfRange { resid, coord_len });
            }
            let p = coords[resid as usize];
            for d in 0..3 {
                if p[d] < min[d] {
                    min[d] = p[d];
                }
                if p[d] > max[d] {
                    max[d] = p[d];
                }
            }
        }
        Ok(Aabb { min, max })
    }

    /// Geometric centre of the AABB. Convenience accessor for
    /// downstream code that wants a scalar surrogate (e.g., legacy
    /// per-site centroid bridges); explicit so the call site is
    /// auditable.
    #[inline]
    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Edge-length-vector of the AABB.
    #[inline]
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

/// Common shape of every role-typed view: a deterministically-ordered
/// support set, parallel causal values, an AABB, and provenance.
///
/// Each role view (e.g., [`CausalDriverView`]) is a thin newtype over
/// a `ManifoldViewData` so the *type* enforces role separation at
/// compile time (M4) while the *layout* stays uniform across roles —
/// which is what lets the producer use the same SoA upload code path
/// for every view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifoldViewData {
    /// Support set: residue ids in this view, sorted by causal signal
    /// (descending) with the tie-breaker recorded in
    /// [`ViewProvenance::tie_breaker`]. Length matches `causal_values`.
    pub support: Vec<ResidueId>,
    /// Per-residue causal-signal values, parallel to `support`.
    /// Same length, same indexing. The signal that produced these
    /// values is recorded in [`ViewProvenance::signal`].
    pub causal_values: Vec<f32>,
    /// Tight AABB over the coordinates of `support`. Computed once
    /// at construction via [`Aabb::from_support_set`] (M5).
    pub aabb: Aabb,
    /// How this view was selected (M3 + M6 audit).
    pub provenance: ViewProvenance,
}

impl ManifoldViewData {
    /// Number of residues in the support set.
    #[inline]
    pub fn len(&self) -> usize {
        self.support.len()
    }

    /// True iff the support set is empty. Constructed views are
    /// never empty (constructors reject `EmptySupport`); this is
    /// provided for the sake of `clippy::len_without_is_empty`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.support.is_empty()
    }
}

/// Generic constructor used by every role view. Validates SoA
/// invariants and computes the AABB (M5) before returning.
fn build_view_data(
    coords: &[[f32; 3]],
    support: Vec<ResidueId>,
    causal_values: Vec<f32>,
    provenance: ViewProvenance,
) -> Result<ManifoldViewData, ManifoldError> {
    if support.len() != causal_values.len() {
        return Err(ManifoldError::SoaLengthMismatch {
            support_len: support.len(),
            values_len: causal_values.len(),
        });
    }
    let aabb = Aabb::from_support_set(coords, &support)?;
    Ok(ManifoldViewData {
        support,
        causal_values,
        aabb,
        provenance,
    })
}

/// View of residues with the highest causal driver attribution.
///
/// Selection signal: typically [`CausalSignal::SpikeAttributionCount`]
/// or [`CausalSignal::KccScore`] (M3). Compile-time role separation
/// (M4): a function that takes [`CausalDriverView`] cannot accept a
/// [`LiningContactView`].
///
/// Tensor channel role: "driver" — the first "2" of the 2+2+2+2 graph
/// (driver↔lining edge channel pair).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalDriverView(pub ManifoldViewData);

impl CausalDriverView {
    /// Construct from a deterministically-ordered support set and its
    /// parallel causal values. The caller MUST have sorted `support`
    /// and `causal_values` jointly per [`TieBreakerPolicy`] before
    /// calling — this constructor does not re-sort. The provenance
    /// records which tie-breaker was used so downstream code can
    /// audit.
    pub fn new(
        coords: &[[f32; 3]],
        support: Vec<ResidueId>,
        causal_values: Vec<f32>,
        provenance: ViewProvenance,
    ) -> Result<Self, ManifoldError> {
        Ok(Self(build_view_data(coords, support, causal_values, provenance)?))
    }

    /// Borrow the underlying [`ManifoldViewData`] (support set,
    /// causal values, AABB, provenance).
    #[inline]
    pub fn data(&self) -> &ManifoldViewData {
        &self.0
    }
}

/// View of residues forming the lining/contact set around the driver
/// region.
///
/// Per blueprint **M2**, lining residues MUST be selected from causal
/// contact participation, NOT from spatial proximity to the legacy
/// scalar centroid. The signal recorded in [`ViewProvenance::signal`]
/// audits this.
///
/// Tensor channel role: "lining" — paired with [`CausalDriverView`]
/// in the driver↔lining edge channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LiningContactView(pub ManifoldViewData);

impl LiningContactView {
    /// Construct from a deterministically-ordered support set and its
    /// parallel causal values. See [`CausalDriverView::new`] for the
    /// shared SoA / sort / provenance contract.
    pub fn new(
        coords: &[[f32; 3]],
        support: Vec<ResidueId>,
        causal_values: Vec<f32>,
        provenance: ViewProvenance,
    ) -> Result<Self, ManifoldError> {
        Ok(Self(build_view_data(coords, support, causal_values, provenance)?))
    }

    /// Borrow the underlying [`ManifoldViewData`].
    #[inline]
    pub fn data(&self) -> &ManifoldViewData {
        &self.0
    }
}

/// View of residues confined to a localized, causally-coherent
/// subregion (e.g., a cryptic-pocket lining lobe or a coupled-motion
/// subcluster).
///
/// Tensor channel role: "localized" — paired with both driver and
/// lining views in the 2+2+2+2 graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LocalizedSubclusterView(pub ManifoldViewData);

impl LocalizedSubclusterView {
    /// Construct from a deterministically-ordered support set and its
    /// parallel causal values. See [`CausalDriverView::new`] for the
    /// shared SoA / sort / provenance contract.
    pub fn new(
        coords: &[[f32; 3]],
        support: Vec<ResidueId>,
        causal_values: Vec<f32>,
        provenance: ViewProvenance,
    ) -> Result<Self, ManifoldError> {
        Ok(Self(build_view_data(coords, support, causal_values, provenance)?))
    }

    /// Borrow the underlying [`ManifoldViewData`].
    #[inline]
    pub fn data(&self) -> &ManifoldViewData {
        &self.0
    }
}

/// Top-level container of role-typed manifold views for a single MD
/// frame. Produced by `SpikeToCluster4D` (mandate M1, separate
/// commit) on-device and consumed by the LBVH / 2+2+2+2 graph kernels
/// downstream.
///
/// # Naming note
///
/// The blueprint's mandate text refers to "CentroidManifold" as the
/// container of these views. That name is already taken by
/// [`crate::spatial_view::CentroidManifold`], a Phase-1 per-site
/// point-centroid container that is M5-non-conformant by design
/// (point-centroids are insufficient for LBVH). The two types are
/// kept distinct: `EntangledManifold` is the M4/M5-conformant
/// container; the legacy `CentroidManifold` is retained for downstream
/// readers migrated in Phase 1 and SHOULD NOT be used by new code.
///
/// Per blueprint **M2**, NOTHING in this struct may be selected by
/// distance to the legacy scalar centroid. The view constructors
/// enforce this by recording the selection signal on
/// [`ViewProvenance`]; an auditor scanning produced manifolds will
/// see only causal signals (M3), never a "legacy_centroid_distance"
/// variant — there is no such variant in [`CausalSignal`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntangledManifold {
    /// Driver residues (highest causal attribution). M4 typed slot.
    pub driver: CausalDriverView,
    /// Lining / contact residues. M4 typed slot.
    pub lining: LiningContactView,
    /// Localized subcluster residues. M4 typed slot.
    pub localized: LocalizedSubclusterView,
    /// MD frame index this manifold was constructed at. Must equal
    /// `driver.0.provenance.frame == lining.0.provenance.frame ==
    /// localized.0.provenance.frame` — invariant checked by
    /// [`Self::new`].
    pub frame: FrameIndex,
}

impl EntangledManifold {
    /// Assemble a manifold from three already-built views. Verifies
    /// the cross-view frame invariant.
    pub fn new(
        driver: CausalDriverView,
        lining: LiningContactView,
        localized: LocalizedSubclusterView,
    ) -> Result<Self, ManifoldFrameMismatch> {
        let f = driver.0.provenance.frame;
        if lining.0.provenance.frame != f {
            return Err(ManifoldFrameMismatch {
                driver: f,
                lining: lining.0.provenance.frame,
                localized: localized.0.provenance.frame,
            });
        }
        if localized.0.provenance.frame != f {
            return Err(ManifoldFrameMismatch {
                driver: f,
                lining: lining.0.provenance.frame,
                localized: localized.0.provenance.frame,
            });
        }
        Ok(Self {
            driver,
            lining,
            localized,
            frame: f,
        })
    }
}

/// Cross-view frame invariant violation. The three role views in an
/// [`EntangledManifold`] must all describe the same MD frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifoldFrameMismatch {
    /// Frame recorded on the [`CausalDriverView`].
    pub driver: FrameIndex,
    /// Frame recorded on the [`LiningContactView`].
    pub lining: FrameIndex,
    /// Frame recorded on the [`LocalizedSubclusterView`].
    pub localized: FrameIndex,
}

impl std::fmt::Display for ManifoldFrameMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EntangledManifold frame mismatch: driver={} lining={} localized={}",
            self.driver, self.lining, self.localized
        )
    }
}

impl std::error::Error for ManifoldFrameMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(frame: FrameIndex) -> ViewProvenance {
        ViewProvenance {
            signal: CausalSignal::SpikeAttributionCount,
            selection: SelectionPolicy::TopK { k: 3 },
            tie_breaker: TieBreakerPolicy::CausalThenResid,
            frame,
        }
    }

    #[test]
    fn aabb_from_single_residue_is_a_point_box() {
        let coords = vec![[1.0, 2.0, 3.0]];
        let aabb = Aabb::from_support_set(&coords, &[0]).unwrap();
        assert_eq!(aabb.min, [1.0, 2.0, 3.0]);
        assert_eq!(aabb.max, [1.0, 2.0, 3.0]);
        assert_eq!(aabb.extent(), [0.0, 0.0, 0.0]);
        assert_eq!(aabb.center(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn aabb_tight_over_support_set_only() {
        // M5 invariant: AABB sees ONLY support residues, not the
        // outliers that would inflate it.
        let coords = vec![
            [0.0, 0.0, 0.0],   // 0 — support
            [1.0, 1.0, 1.0],   // 1 — support
            [2.0, 2.0, 2.0],   // 2 — support
            [99.0, 99.0, 99.0], // 3 — NOT in support; must not inflate
        ];
        let aabb = Aabb::from_support_set(&coords, &[0, 1, 2]).unwrap();
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn aabb_rejects_empty_support() {
        let coords: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let err = Aabb::from_support_set(&coords, &[]).unwrap_err();
        assert_eq!(err, ManifoldError::EmptySupport);
    }

    #[test]
    fn aabb_rejects_out_of_range_support() {
        let coords = vec![[0.0, 0.0, 0.0]];
        let err = Aabb::from_support_set(&coords, &[5]).unwrap_err();
        assert!(matches!(
            err,
            ManifoldError::SupportIndexOutOfRange { resid: 5, coord_len: 1 }
        ));
        let err = Aabb::from_support_set(&coords, &[-1]).unwrap_err();
        assert!(matches!(err, ManifoldError::SupportIndexOutOfRange { .. }));
    }

    #[test]
    fn driver_view_round_trips_provenance_and_aabb() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let view = CausalDriverView::new(
            &coords,
            vec![1, 2, 3],
            vec![10.0, 9.0, 8.0],
            provenance(7),
        )
        .unwrap();
        assert_eq!(view.data().support, vec![1, 2, 3]);
        assert_eq!(view.data().causal_values, vec![10.0, 9.0, 8.0]);
        assert_eq!(view.data().aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(view.data().aabb.max, [1.0, 2.0, 3.0]);
        assert_eq!(view.data().provenance.signal, CausalSignal::SpikeAttributionCount);
        assert_eq!(view.data().provenance.frame, 7);
    }

    #[test]
    fn view_rejects_soa_length_mismatch() {
        let coords = vec![[0.0; 3]; 4];
        let err = CausalDriverView::new(
            &coords,
            vec![1, 2, 3],
            vec![10.0, 9.0], // length mismatch
            provenance(0),
        )
        .unwrap_err();
        assert_eq!(
            err,
            ManifoldError::SoaLengthMismatch {
                support_len: 3,
                values_len: 2,
            }
        );
    }

    #[test]
    fn view_rejects_empty_support() {
        let coords = vec![[0.0; 3]; 4];
        let err = CausalDriverView::new(&coords, vec![], vec![], provenance(0)).unwrap_err();
        assert_eq!(err, ManifoldError::EmptySupport);
    }

    #[test]
    fn entangled_manifold_enforces_frame_consistency() {
        let coords = vec![[0.0; 3]; 4];
        let driver =
            CausalDriverView::new(&coords, vec![0], vec![1.0], provenance(7)).unwrap();
        let lining =
            LiningContactView::new(&coords, vec![1], vec![1.0], provenance(7)).unwrap();
        let localized =
            LocalizedSubclusterView::new(&coords, vec![2], vec![1.0], provenance(7)).unwrap();

        let manifold = EntangledManifold::new(driver, lining, localized).unwrap();
        assert_eq!(manifold.frame, 7);
    }

    #[test]
    fn entangled_manifold_rejects_frame_mismatch() {
        let coords = vec![[0.0; 3]; 4];
        let driver =
            CausalDriverView::new(&coords, vec![0], vec![1.0], provenance(7)).unwrap();
        let lining =
            LiningContactView::new(&coords, vec![1], vec![1.0], provenance(8)).unwrap();
        let localized =
            LocalizedSubclusterView::new(&coords, vec![2], vec![1.0], provenance(7)).unwrap();
        let err = EntangledManifold::new(driver, lining, localized).unwrap_err();
        assert_eq!(err.driver, 7);
        assert_eq!(err.lining, 8);
        assert_eq!(err.localized, 7);
    }

    #[test]
    fn deterministic_tie_breaker_policy_label_is_stable() {
        // M6 audit: the policy name is part of the provenance and
        // must round-trip stably for downstream auditors.
        assert_eq!(TieBreakerPolicy::CausalThenResid.as_str(), "causal_then_resid");
        assert_eq!(
            TieBreakerPolicy::CausalThenResidThenChain.as_str(),
            "causal_then_resid_then_chain"
        );
    }

    #[test]
    fn causal_signal_label_is_stable() {
        // M3 audit: the signal name is part of the provenance and
        // must round-trip stably.
        assert_eq!(CausalSignal::SpikeAttributionCount.as_str(), "spike_attribution_count");
        assert_eq!(CausalSignal::KccScore.as_str(), "kcc_score");
        assert_eq!(CausalSignal::TransferEntropyIn.as_str(), "te_in");
        assert_eq!(CausalSignal::TransferEntropyOut.as_str(), "te_out");
    }

    #[test]
    fn no_legacy_centroid_signal_variant_exists() {
        // M2 regression guard: future maintainers must NOT add a
        // "LegacyScalarCentroid" or similar geometric-distance variant
        // to CausalSignal. This test pins the current variant set.
        // If you are extending CausalSignal with a new *causal*
        // signal, add it here. If you are tempted to add a geometric
        // signal, re-read blueprint M2 first.
        let all_variants = [
            CausalSignal::SpikeAttributionCount,
            CausalSignal::KccScore,
            CausalSignal::TransferEntropyIn,
            CausalSignal::TransferEntropyOut,
        ];
        for v in all_variants {
            // every variant's name must be a *causal* identifier,
            // never a geometric one
            let s = v.as_str();
            assert!(
                !s.contains("centroid") && !s.contains("distance"),
                "CausalSignal variant {:?} has geometric label '{}' — violates M2",
                v,
                s
            );
        }
    }
}
