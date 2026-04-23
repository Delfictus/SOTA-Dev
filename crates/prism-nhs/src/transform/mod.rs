//! Phase 2 — Transform Audit Spine.
//!
//! This module introduces the first-class, machine-enforceable vocabulary for
//! PRISM-4D post-MD transforms. It does not define the pipeline's transforms;
//! it defines the **governance surface** they implement.
//!
//! # What this module provides
//!
//! * [`AuditedTransform`] — the canonical trait every governed transform
//!   implements. It carries identity, determinism class, tolerance policy,
//!   a declared law set, a verification hook, and an `apply` entry point
//!   whose return type carries structured audit semantics.
//! * [`DeterminismClass`] — owned by the transform, not the data struct.
//!   Documents whether the transform's output is bit-reproducible, atomics-
//!   affected, or order-dependent under real runtime conditions.
//! * [`TolerancePolicy`] — owned by the transform. Describes how the
//!   transform's output should be compared when a law or a downstream
//!   consumer needs equivalence.
//! * [`TransformViolation`] / [`ViolationEvidence`] / [`AuditRouting`] —
//!   structured, machine-readable law failures. No string-only errors.
//! * [`AuditOutcome`] — typed three-way outcome: `Accepted`, `Quarantined`
//!   (output accepted but carries site-local violations), or `Aborted`
//!   (structural/corruption-class failure — no output produced).
//! * [`AuditRecord`] — lightweight lineage carried with every outcome so
//!   downstream consumers can trace which transform produced an output and
//!   under what governance.
//!
//! # What this module explicitly does NOT provide (Phase 2 non-goals)
//!
//! * No rank-monotonicity / order-theoretic (Family B) laws. Those belong
//!   to Phase 9 population-level governance, not to local transform spine.
//! * No tolerance calibration envelopes. Empirical calibration is Phase 8.
//! * No serialization format for outcomes or violations. JSON/Arrow export
//!   is a downstream concern to be handled when a consumer requires it.
//! * No general graph-runtime machinery. The spine wires one transform at
//!   a time; the DAG shape is implicit in the call graph, not in the
//!   framework.
//!
//! # Scope hook
//!
//! The first transform wrapped under this spine is
//! [`clustering_to_clustered_sites::ClusteringToClusteredSites`], which
//! governs the canonical site-construction boundary and enforces two
//! Family-A local algebraic laws (`GvmPopulated`, `EmissionCompatMatchesGvm`).
//! See that module for the concrete wrapping pattern.

// The concrete first-wrap transform depends on `build_clustered_sites`,
// which is itself feature-gated on `gpu`. Match that gate here so the
// framework types in this file remain feature-agnostic while the
// concrete wrap is only compiled where its dependencies exist.
#[cfg(feature = "gpu")]
pub mod clustering_to_clustered_sites;

use std::error::Error;
use std::fmt;

// ============================================================================
// Identity newtypes
// ============================================================================

/// Stable string identifier for a transform. Stored as `&'static str` so
/// no allocation is required at the spine boundary and identifiers can
/// participate in `match`es and constant comparisons.
///
/// Identifiers are snake_case and map to a single transform family:
/// `"clustering_to_clustered_sites"`, `"spike_to_residue"`, etc. They are
/// forensic anchors: a violation with a given `TransformId` + `LawId`
/// pair identifies the exact enforcement point in the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformId(pub &'static str);

impl fmt::Display for TransformId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Stable string identifier for a single law. Laws are numbered within
/// their transform (`l1_...`, `l2_...`) and named by what they enforce,
/// not by what they forbid. A law id is unique within a transform and
/// stable across releases so violation logs are comparable over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LawId(pub &'static str);

impl fmt::Display for LawId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

// ============================================================================
// Determinism class
// ============================================================================

/// Determinism classification owned by the transform, not by the data
/// struct it produces.
///
/// This is a machine-readable declaration of the reproducibility regime
/// under which the transform operates when invoked on realistic inputs.
/// It is NOT a claim about what the math "could" be under idealized
/// conditions — it is a claim about the code path that actually runs.
///
/// Classes:
///
/// * [`DeterminismClass::BitExact`] — two runs on identical inputs under
///   identical hardware produce byte-identical outputs.
/// * [`DeterminismClass::AtomicsAffected`] — GPU atomic-order dependence
///   or similar non-deterministic reduction causes bounded numerical
///   drift across runs at fixed inputs; outputs are semantically
///   equivalent but not bit-identical.
/// * [`DeterminismClass::OrderDependent`] — output depends on the
///   permutation of the input sequence at the API level; if the caller
///   permutes their input, the output changes in a way that is not
///   covered by the transform's tolerance policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterminismClass {
    BitExact,
    AtomicsAffected,
    OrderDependent,
}

impl fmt::Display for DeterminismClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DeterminismClass::BitExact => "bit_exact",
            DeterminismClass::AtomicsAffected => "atomics_affected",
            DeterminismClass::OrderDependent => "order_dependent",
        })
    }
}

// ============================================================================
// Tolerance policy
// ============================================================================

/// Comparison/tolerance policy owned by the transform. Used when a law
/// or a downstream consumer needs to compare two values the transform
/// produced (or one it produced against a reference).
///
/// The policy describes the semantics of equivalence, not a numeric knob
/// to be tuned ad hoc. Phase 8 will extend this enum with empirically
/// calibrated envelopes; Phase 2 uses only `BitExact` because the first
/// wrapped transform's two laws compare values that must be structurally
/// identical (no floating-point arithmetic intervenes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TolerancePolicy {
    /// Byte-equal / bit-equal comparison. Only valid for values that
    /// were produced without any floating-point arithmetic between the
    /// comparison endpoints — e.g., two fields both set from the same
    /// `[f32; 3]` input.
    BitExact,
    /// Absolute-magnitude L2 tolerance: `‖a - b‖₂ <= epsilon`.
    /// Reserved for later phases; not used by Phase 2 laws.
    AbsL2 { epsilon: f32 },
}

impl fmt::Display for TolerancePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TolerancePolicy::BitExact => f.write_str("bit_exact"),
            TolerancePolicy::AbsL2 { epsilon } => write!(f, "abs_l2(eps={epsilon:e})"),
        }
    }
}

// ============================================================================
// Violation + routing
// ============================================================================

/// Routing recommendation attached to each [`TransformViolation`].
///
/// The routing is a property of the **law**, not of the caller. It
/// answers: "if this law is violated, what is the correct reaction at
/// the spine boundary?"
///
/// * [`AuditRouting::Abort`] — the whole transform invocation is a
///   failure. No partial output is emitted. Reserved for true
///   accounting/corruption class failures (spike-conservation failure,
///   impossible attribution mismatch, structural invariants like
///   R4/R5 that must hold by construction).
/// * [`AuditRouting::Quarantine`] — the specific offending output
///   element is excluded from downstream consumption (ranking,
///   training-export, etc.) but the transform as a whole produced a
///   valid result. Reserved for site-local / branch-local semantic
///   violations (locality failures, weak coherence, thermodynamic
///   incoherence, etc.). The Phase 2 law set does not emit
///   `Quarantine` — it is present here as a typed member of the
///   framework so Phase 3+ transforms can emit it without re-opening
///   this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditRouting {
    Abort,
    Quarantine,
}

impl fmt::Display for AuditRouting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AuditRouting::Abort => "abort",
            AuditRouting::Quarantine => "quarantine",
        })
    }
}

/// Structured, machine-readable evidence describing WHY a law failed.
///
/// Every variant carries the concrete data a forensic auditor needs to
/// reproduce the judgement — not just a string. New variants are added
/// per-phase as new laws are introduced; extending this enum is a
/// source-incompatible change on purpose (every consumer must handle
/// the new variant or opt out explicitly).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ViolationEvidence {
    /// Law `l1_gvm_populated`: the site's
    /// `localization.geometric_voxel_mass` view is `None` in the
    /// transform output.
    GvmViewAbsent {
        /// The `cluster_id` of the site that failed the law.
        cluster_id: i32,
    },
    /// Law `l2_emission_compat_matches_gvm`: the site's
    /// `emission_compat_centroid()` does not equal its
    /// `localization.geometric_voxel_mass` view under the transform's
    /// declared tolerance policy.
    EmissionCompatDivergent {
        cluster_id: i32,
        emission_compat: [f32; 3],
        gvm_view: [f32; 3],
    },
    /// Synthetic test evidence used only by the Phase 2 mock transform
    /// in the test harness. Never emitted by production transforms.
    /// Present as a typed variant (rather than a string) to keep the
    /// `ViolationEvidence` enum uniformly machine-readable even in test
    /// scaffolding.
    #[doc(hidden)]
    SyntheticForTesting { tag: &'static str },
}

/// One structured law failure emitted by a transform.
///
/// Violations are the sole channel through which a transform
/// communicates "this output is not correct." String-only errors and
/// log-only failures are explicitly not permitted by the Phase 2
/// doctrine.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformViolation {
    /// The transform that emitted the violation.
    pub transform: TransformId,
    /// The specific law that was violated.
    pub law: LawId,
    /// How the spine should react to this violation.
    pub routing: AuditRouting,
    /// Structured evidence explaining the failure.
    pub evidence: ViolationEvidence,
}

// ============================================================================
// Audit record + outcome
// ============================================================================

/// Lightweight provenance record carried alongside every audit outcome.
///
/// The record answers "which transform produced this output, under
/// which determinism and tolerance regime, with which laws declared?"
/// It is the Phase 2 form of lineage — deliberately minimal. Phase 5
/// (field propagation matrix) will extend this with input descriptors
/// and consumer annotations; Phase 2 does not.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRecord {
    pub transform: TransformId,
    pub determinism: DeterminismClass,
    pub tolerance: TolerancePolicy,
    pub laws_declared: &'static [LawId],
}

/// Typed three-way outcome of applying an [`AuditedTransform`].
///
/// Every variant carries an [`AuditRecord`] so downstream consumers
/// that want to log or archive provenance do not need a second call to
/// the transform.
///
/// * `Accepted` — no violations; the output is consumable as-is.
/// * `Quarantined` — the output is consumable but one or more
///   site-local violations were observed; consumers that assume clean
///   semantics must filter out the offending elements using the
///   `cluster_id` carried in each violation's evidence.
/// * `Aborted` — a structural / corruption-class violation was
///   observed. The transform produced no output. The caller must
///   treat this as a run failure.
#[derive(Debug, Clone)]
pub enum AuditOutcome<T> {
    Accepted {
        output: T,
        record: AuditRecord,
    },
    Quarantined {
        output: T,
        record: AuditRecord,
        violations: Vec<TransformViolation>,
    },
    Aborted {
        record: AuditRecord,
        violations: Vec<TransformViolation>,
    },
}

impl<T> AuditOutcome<T> {
    /// Convenience: convert to `Result<(T, Vec<TransformViolation>), AbortedAudit>`.
    ///
    /// On `Accepted`, returns `Ok((output, empty))`.
    /// On `Quarantined`, returns `Ok((output, violations))` — the caller
    /// owns the choice of whether to log, filter, or escalate.
    /// On `Aborted`, returns `Err(AbortedAudit)` which implements
    /// `std::error::Error` and composes cleanly with `anyhow`.
    pub fn into_result(self) -> Result<(T, Vec<TransformViolation>), AbortedAudit> {
        match self {
            AuditOutcome::Accepted { output, .. } => Ok((output, Vec::new())),
            AuditOutcome::Quarantined { output, violations, .. } => Ok((output, violations)),
            AuditOutcome::Aborted { record, violations } => {
                Err(AbortedAudit { record, violations })
            }
        }
    }

    pub fn record(&self) -> &AuditRecord {
        match self {
            AuditOutcome::Accepted { record, .. }
            | AuditOutcome::Quarantined { record, .. }
            | AuditOutcome::Aborted { record, .. } => record,
        }
    }
}

/// Error type returned by [`AuditOutcome::into_result`] on the abort path.
///
/// Implements `std::error::Error` so it composes with `anyhow::Result`
/// at the engine-API boundary (`run_and_cluster`, `process_batch`).
#[derive(Debug, Clone)]
pub struct AbortedAudit {
    pub record: AuditRecord,
    pub violations: Vec<TransformViolation>,
}

impl fmt::Display for AbortedAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "transform {} aborted under {} tolerance ({} violation(s)): ",
            self.record.transform, self.record.tolerance, self.violations.len()
        )?;
        for (i, v) in self.violations.iter().enumerate() {
            if i > 0 {
                f.write_str("; ")?;
            }
            write!(f, "[{}:{} → {}] {:?}", v.transform, v.law, v.routing, v.evidence)?;
        }
        Ok(())
    }
}

impl Error for AbortedAudit {}

// ============================================================================
// The trait
// ============================================================================

/// The canonical trait every governed transform in the post-MD pipeline
/// implements.
///
/// The trait carries the full Phase 2 governance surface:
///
/// * **identity** — a stable [`TransformId`] that appears in every
///   violation and in every audit record.
/// * **determinism** — a [`DeterminismClass`] declaration that
///   documents reproducibility under real runtime conditions, not
///   ideal ones.
/// * **tolerance** — a transform-owned [`TolerancePolicy`] used by
///   laws and by downstream consumers when equivalence is required.
/// * **laws** — the full declared law set. Every law the transform's
///   `verify` method can emit must be declared here.
/// * **apply** — forward application. The spine boundary. Returns an
///   [`AuditOutcome`] (never a bare `Self::Output`).
/// * **verify** — pure verification on an already-produced output.
///   Factored out so tests can exercise law detection directly.
///
/// The trait uses a generic associated type (`Input<'a>`) so concrete
/// transforms can borrow from the caller's data (e.g., `&[SpikeEvent]`)
/// without forcing a clone at the spine boundary.
pub trait AuditedTransform {
    /// Caller-supplied input, typically a struct of references.
    type Input<'a>
    where
        Self: 'a;

    /// The successful output type.
    type Output;

    /// Stable identity for this transform. Appears in every audit
    /// record and every violation.
    fn identity(&self) -> TransformId;

    /// Determinism class under realistic runtime conditions.
    fn determinism(&self) -> DeterminismClass;

    /// Tolerance policy used by this transform's laws and by
    /// downstream consumers.
    fn tolerance(&self) -> TolerancePolicy;

    /// Full declared law set. Every law a `verify` call may emit must
    /// be listed here; extending this list is a deliberate API change.
    fn laws(&self) -> &'static [LawId];

    /// Pure verification on an already-produced output. Must not
    /// mutate, allocate on the happy path beyond violation collection,
    /// or re-run the transform.
    fn verify(&self, output: &Self::Output) -> Vec<TransformViolation>;

    /// Forward application. Concrete implementations produce the
    /// output and then call [`AuditedTransform::adjudicate`] to route
    /// the result through the standard accept/quarantine/abort logic.
    fn apply<'a>(&self, input: Self::Input<'a>) -> AuditOutcome<Self::Output>;

    /// Route an already-produced output through verify-and-adjudicate.
    ///
    /// Default implementation. Concrete transforms should call this
    /// from their `apply` method rather than building `AuditOutcome`
    /// variants directly.
    fn adjudicate(&self, output: Self::Output) -> AuditOutcome<Self::Output> {
        let violations = self.verify(&output);
        let record = AuditRecord {
            transform: self.identity(),
            determinism: self.determinism(),
            tolerance: self.tolerance(),
            laws_declared: self.laws(),
        };
        if violations.is_empty() {
            AuditOutcome::Accepted { output, record }
        } else if violations.iter().any(|v| v.routing == AuditRouting::Abort) {
            AuditOutcome::Aborted { record, violations }
        } else {
            AuditOutcome::Quarantined { output, record, violations }
        }
    }
}
