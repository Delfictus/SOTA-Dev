//! Phase 3 — Pillar 2: Causome typed block.
//!
//! `CausomeBlock` is the typed projection of the per-site causal data
//! that is currently scattered across `kcc_visualization.json.residues[]`
//! + `kcc_visualization.json.sites[].kcc{}` + `kcc_validation.json.sites[]`
//! + `binding_sites.json sites[i].kcc{}` + `binding_sites.json
//! sites[i].signal_preservation{}` + `gcpid_synergy.json residues[]`.
//!
//! The struct itself is pure data — no behavior, no methods that
//! reach into other Pillars.  The `cluster_to_causome` transform is
//! the only canonical producer; the JSON-emit code is the only
//! canonical consumer.  Future Phase 4 work (`CausomeToTherm`,
//! `CausomeToRankTerms`) will consume this object as `Gpre` for their
//! own typed outputs.
//!
//! # What this module does NOT do
//!
//! * Compute KCC scores — those are computed upstream in the
//!   KCC analysis kernels and arrive as inputs to the transform.
//! * Hold tolerance / determinism state — those live on the
//!   `ClusterToCausome` transform impl per the audit-spine doctrine.
//! * Emit JSON directly — the canonical emitter in `nhs_rt_full.rs`
//!   serializes via serde.
//!
//! # Reserved fields (Phase 4–5)
//!
//! `tide_decomposition: Vec<TideResidueDecomp>` is a documented
//! reserved slot — already emitted in some run modes
//! (`prism_therm.sites[].tide_decomposition[]` per the schema-walker
//! reference) but belongs to Pillar 3 (`ThermBlock`) once
//! `CausomeToTherm` lands.  Phase 3 does not introduce the field
//! here; the comment is the schema reservation marker.

use serde::{Deserialize, Serialize};

/// Typed Pillar-2 block per detected site.  Produced by
/// `crate::transform::cluster_to_causome::ClusterToCausome` from the
/// already-emitted KCC / signal-preservation data, never reconstructed.
///
/// Field semantics map to the existing
/// `binding_sites.json sites[i].kcc{...}` JSON sub-dict (which Phase 3
/// retains for backward compat) — every Phase-2 KCC sub-field has a
/// typed home here, with explicit naming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausomeBlock {
    /// Engine cluster id of the site this block describes.
    pub site_id: i32,

    /// The single residue judged most causally responsible for the
    /// site's pocket dynamics.  Must satisfy
    /// `LAW_L3_DRIVER_SUBSET_OF_LINING` —
    /// `driver_residue_id ∈ lining_residues[].resid` for this site.
    pub driver_residue_id: i32,

    /// Top-K candidate driver residues (typically K=3) considered
    /// during the KCC reduction.  Must each satisfy `LAW_L3` (subset
    /// of lining).  Length matches `candidate_residue_support`.
    pub candidate_residue_ids: Vec<i32>,
    /// Per-candidate causal-support weight.  Length matches
    /// `candidate_residue_ids`.  Values are KCC-derived; expected to
    /// satisfy `RelativeEpsilon { eps: 1e-4 }` reproducibility under
    /// CUDA atomics.
    pub candidate_residue_support: Vec<f32>,

    /// KCC sub-scores aggregated to the driver residue.
    pub burst_motion: f32,
    pub causal_lag: f32,
    pub direction_score: f32,
    pub kcc_confidence: f32,
    pub lag_corr_peak: f32,
    pub local_cov: f32,
    pub motion_efficiency: f32,

    /// KCC sub-scores aggregated to the SITE (vs the driver residue).
    /// Corresponds to the `site_*` keys in the v1 `kcc{}` sub-dict.
    pub site_burst_motion: f32,
    pub site_causal_lag: f32,
    pub site_direction_score: f32,
    pub site_lag_corr_peak: f32,
    pub site_local_cov: f32,
    pub site_motion_efficiency: f32,

    /// Cross-correlation peak between motion and causal signal across
    /// the full site neighborhood (vs just the driver residue's
    /// `lag_corr_peak`).
    pub temporal_corr: f32,

    /// Number of simulation steps in which this site's neighborhood
    /// saw causal UV→LIF coupling (the "active" subset of total_steps).
    pub active_causal_steps: u32,
    /// Total simulation steps the site was tracked across.
    pub total_steps: u32,
    // Reserved (Phase 4–5):
    //   pub tide_decomposition: Vec<TideResidueDecomp>,
    // — TIDE per-residue thermodynamic decomposition belongs to
    // Pillar 3 (`ThermBlock`) once `CausomeToTherm` lands.  Phase 3
    // omits this field; the comment is the schema reservation marker.
}

impl CausomeBlock {
    /// Total number of candidate residues recorded.  Matches
    /// `candidate_residue_ids.len()` and `candidate_residue_support.len()`.
    pub fn candidate_count(&self) -> usize {
        debug_assert_eq!(
            self.candidate_residue_ids.len(),
            self.candidate_residue_support.len(),
            "CausomeBlock candidate_residue_ids/support length mismatch"
        );
        self.candidate_residue_ids.len()
    }
}
