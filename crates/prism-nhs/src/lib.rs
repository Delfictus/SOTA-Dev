//! # PRISM-4D Neuromorphic Holographic Stream (NHS)
//!
//! Real-time cryptic binding site detection through the "holographic negative" principle:
//! instead of simulating water (expensive), we map what EXCLUDES water (hydrophobic atoms)
//! and infer water's presence from its absence.
//!
//! ## Core Components
//!
//! - **Hydrophobic Exclusion Mapping (HEM)** - 3D field showing where water CAN'T exist
//! - **Holographic Water Inference** - Infer water density from exclusion + polar fields
//! - **Neuromorphic Dewetting Detection** - Spike-based detection of pocket opening
//! - **UV Bias Perturbation** - Targeted aromatic excitation for causal validation
//!
//! ## Performance Targets
//!
//! - **30,000×** faster than explicit solvent MD
//! - **85-90%** accuracy vs explicit solvent ground truth
//! - **<2ms** per frame on RTX 3060
//! - **Real-time** streaming detection
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use prism_nhs::{NhsPipeline, NhsConfig};
//!
//! // Create pipeline with default config
//! let config = NhsConfig::default();
//! let mut pipeline = NhsPipeline::new(config);
//!
//! // Initialize with protein structure
//! pipeline.initialize(
//!     &positions,
//!     &elements,
//!     &charges,
//!     &residue_names,
//!     &atom_names,      // IUPAC atom names for ring detection
//!     &atom_residues,
//! )?;
//!
//! // Process trajectory frames
//! for frame_positions in trajectory {
//!     let (events, perturbation) = pipeline.process_frame(&frame_positions)?;
//!
//!     for event in events {
//!         println!("Cryptic site detected: {:?}", event);
//!     }
//!
//!     // Apply UV perturbation to velocities if provided
//!     if let Some(p) = perturbation {
//!         p.apply_to_velocities(&mut velocities);
//!     }
//! }
//!
//! // Print statistics
//! println!("{}", pipeline.stats());
//! ```
//!
//! ## The Holographic Negative Principle
//!
//! Traditional approaches:
//! - **Explicit MD**: Simulate millions of water molecules → O(N²) → SLOW
//! - **Implicit MD**: Approximate water as dielectric → Fast but loses cavity effects
//!
//! NHS approach:
//! - Map hydrophobic atoms (~30-40% of protein)
//! - Compute exclusion field → where water CAN'T go
//! - Infer water from the "negative" → O(N) → FAST AND ACCURATE
//!
//! ## UV Bias: The Pump-Probe Innovation
//!
//! Aromatic residues (Trp, Tyr, Phe) absorb UV at 280nm.
//! Water is TRANSPARENT at 280nm.
//!
//! This enables causal inference:
//! - **PUMP**: UV burst to aromatics → local perturbation
//! - **PROBE**: Detect dewetting spikes → pocket opening
//! - **CORRELATE**: Establish causal link → validated cryptic site
//!
//! ## Membrane Protein Support
//!
//! NHS naturally extends to membrane proteins because:
//! - Lipid tails are "super-hydrophobic" → strong exclusion
//! - Lipid heads form boundary layer
//! - Cryptic sites at membrane interface are detectable
//!
//! This is the **first tool** to detect cryptic sites in membrane proteins
//! on consumer hardware.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod adaptive;
pub mod aromatic_proximity;
pub mod avalanche;
pub mod config;
#[cfg(feature = "gpu")]
pub mod coupled_md;
pub mod exclusion;
#[cfg(feature = "gpu")]
pub mod fused_engine;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod graph_capture;
#[cfg(feature = "gpu")]
pub mod persistent_engine;
#[cfg(feature = "gpu")]
pub mod protocol_state;
#[cfg(feature = "gpu")]
pub mod rt_analysis;
#[cfg(feature = "gpu")]
pub mod rt_probe;
pub mod rt_targets;
/// PRISM-Therm: SDST thermodynamic analysis bridge.
/// Requires the `gpu` feature (SDST is CUDA-only).
#[cfg(feature = "gpu")]
pub mod sdst_bridge;
/// PRISM-Therm: output pipeline (JSON reports, druggability PDB, ranked table).
#[cfg(feature = "gpu")]
pub mod sdst_report;
pub mod solvate;
/// Spike Thermodynamic Integration: Jarzynski/Crooks/BAR free energy from spike events.
#[cfg(feature = "gpu")]
pub mod spike_thermodynamic_integration;
#[cfg(feature = "gpu")]
pub mod twin_detection;
#[cfg(feature = "gpu")]
pub mod twin_kernels;
// TIER 7 (2026-05-03) — CUDA 13.x cuGraphAddNode wrappers for the
// monolithic-splice migration; see crates/prism-nhs/src/graph_node.rs
// for status + rationale.
#[cfg(feature = "gpu")]
pub mod active_sensing;
/// DAG-COND-WIRE / IGNITION — self-contained captured-pipeline LEGO
/// brick that wraps SO(3) projection (Node B), Adjudicator step
/// (Node C), trampoline (Node C'), explicit conditional node
/// (Node D), and the cross-stream ghost-pipe DMA in a single
/// `cuStreamCaptureModeGlobal` block. Instantiates a verifiable
/// `CUgraphExec`; structural tests cover G18 (capture integrity),
/// G19 (predicate stability), G20 (telemetry overlap). Production
/// wire-in into `nhs_rt_full.rs` is a follow-up commit.
#[cfg(feature = "gpu")]
pub mod captured_pipeline;
/// Phase 3 — Pillar 2 typed block.  See [`causome::CausomeBlock`].
/// The producer is `transform::cluster_to_causome::ClusterToCausome`.
pub mod causome;
/// Diagnostic / informational modules — opt-in via the `diagnostic`
/// Cargo feature. Per the Blackwell Convergence mandate the M1.2.5b
/// typed-producer differential is no longer a closure gate; it
/// survives here as an informational tool only. See
/// [`diagnostic`] module docs for the full deprecation rationale.
#[cfg(feature = "diagnostic")]
pub mod diagnostic;
/// PRISM-4D → DSTW Air-Gap dispatcher.  Implements Option A: rigid-backbone
/// Δq / ΔV projection of variant perturbations onto the WT thermodynamic
/// tensors, emitting the vectorial [delta_P_active, delta_P_lock,
/// delta_P_ensemble] DSTW expects in its BALD loop.  Spec:
/// `crates/prism-nhs/spec/option_a_variant_dispatcher.md`.
pub mod dstw_dispatch;
/// Ensemble teacher manifest contracts and validation gates for the
/// v004 distillation pipeline.
pub mod ensemble;
/// Entangled Transform — typed manifold views with strict AABBs.
/// Implements blueprint mandates M4 (strict struct typing for the
/// 2+2+2+2 graph) and M5 (LBVH-ready, support-only AABBs). See
/// [`docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md`](../docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md)
/// for the binding contract. This module is type-definitions-only;
/// the on-device producer (`SpikeToCluster4D`, M1) and CUDA / FFI
/// shims land in subsequent commits.
pub mod entangled_manifold;
/// G26 Chronometric Gearbox — Wave B.1 foundation: __constant__
/// d_gearbox_table[16] + ChronometricStateTensor + PointerSwap kernel.
/// B.2 lands the predicate bridge + VelocityRescale + Berendsen guard;
/// B.3 lands the PTX-trap kernel + 1k-step transition test.
#[cfg(feature = "gpu")]
pub mod gearbox;
/// CLA-2 — Ghost Telemetry Pipeline. Asynchronous, transparent
/// device→host exfiltration via a triple-buffered pinned-host ring on
/// a dedicated non-blocking telemetry stream. Eliminates the
/// Host-Sync Fallacy (synchronous `cuMemcpyDtoH` on the critical
/// path) per the operator directive 2026-04-29 Part 2. The host
/// observes the simulation's metadata at a 2-frame lag without
/// gating in-flight physics or the F1 SWITCH adjudication.
#[cfg(feature = "gpu")]
pub mod ghost_telemetry;
/// GPU spatial-hash CCL backend for post-MD clustering. Replaces the
/// CPU `fallback_grid_cluster` on SM120+. See `gpu_cluster_backend.rs`
/// module docs.
#[cfg(feature = "gpu")]
pub mod gpu_cluster_backend;
/// Rectification Phase 1 — hard-trap GPU invariant enforcement.
/// Provides the `gpu_hard_assert` device helper (PTX `trap`
/// instruction) and the M1 Conservation-of-Mass audit kernel. On
/// invariant violation the device terminates the warp and the
/// driver returns a non-recoverable CUDA error from the host's
/// stream-synchronize. See module docs for the §9 ESCALATION
/// posture and verification contract.
pub mod gpu_invariant;
#[cfg(feature = "gpu")]
pub mod graph_node;
#[cfg(feature = "gpu")]
pub mod grid_debug_backend;
/// T0/T1/T2/T3/T4 — InterferometricAdjudicator (Blackwell Convergence
/// directive). FFI struct (128 B / align(128)), Quantum-Photonic
/// Bridge LUT (μ_01² derived from `crate::config` extinction
/// coefficients), KL-divergence Adjudicator with PTX NaN/le0 guards
/// (Total Function over all 32-bit f32 inputs), ASC Boundary
/// Repulsion Tensor (T3, atomicAdds into the existing
/// `fused_engine.rs::d_forces` buffer per Anti-Greenfield § 2.1),
/// and clock64-based pipeline timing bookends (T4). Adjudication
/// codes mirror [`crate::pre_rank::AdjudicationCode`] (no parallel
/// type). See module docs for the F1 SWITCH routing contract.
pub mod interferometric_adjudicator;
/// LBVH lane Phase 1 — Morton 30-bit encoder + (subsequent commits)
/// Karras 2012 parallel binary radix-tree construction + per-node
/// AABB bottom-up reduce. The output of this lane replaces the
/// M1.2.5a placeholder synthetic support set in
/// [`spike_to_cluster_4d::SpikeToCluster4D::apply`] with native LBVH
/// AABBs populated into the 8-slot
/// [`site_manifest::CentroidManifold`]. Per Mandate §M5 the AABB is
/// derived only from the support set; LBVH gives that derivation a
/// real algorithmic backing.
pub mod lbvh;
/// LBVH Phase 2 — Karras 2012 radix-tree builder + bottom-up AABB
/// reduce via the last-arrival atomic-flag pattern. Consumes the
/// sorted 64-bit composite keys (Morton + residue_id) produced by
/// the upstream sort node and the leaf positions; produces a tree
/// of `n_leaves - 1` 64-byte cache-line-aligned [`LBVHNode`]s with
/// per-node AABBs propagated from the leaves to the root. See
/// module docs for the Karras encoding and AABB reduce contracts.
pub mod lbvh_tree;
/// Gate G1 — Multi-seed Arrow file merger. Concatenates per-seed
/// `<prefix>.topology.spike_events.arrow` files into a single
/// multi-seed stream while preserving per-spike `replica_seed`
/// provenance through the existing schema column. Pure Rust IPC
/// reader/writer; no Python in the data plane.
pub mod multi_seed_merger;
/// Rectification Phase 2 — shift-left MAR pre-rank adjudicator.
/// Computes per-cluster Pruning Observables (energy density,
/// solvation flux) and writes a 3-way `AdjudicationCode` (Prune /
/// Construct / Violation) that the cudaGraphConditionalNode (F1)
/// SWITCH variant consumes as its branch selector. Replaces
/// late-stage adjudication with early-stage classification, dropping
/// noise clusters before any expensive TIDE / KCC kernel runs.
pub mod pre_rank;
/// Continuous Learning Architecture Phase 1 — RichSpike 64-byte
/// cache-line-aligned event record. Replaces the legacy 16-byte
/// spike with a four-plane schema (spatiotemporal / thermodynamic
/// / causal-neuromorphic / provenance) that captures the full
/// emission state for the closed-loop knowledge-distillation
/// pipeline. See module docs for layout + composite-sort-key
/// design.
pub mod rich_spike;
#[cfg(feature = "gpu")]
pub mod rt_clustering;
/// RECT-3.1 — Spherical Harmonics Y_lm evaluator (Lmax=5).
/// Straight-line PTX evaluator (no recursion, no per-thread
/// branching) that produces 36 real Y_lm values per (θ, φ) input.
/// Feeds the SO(3) projection kernel (RECT-3.1.b) which accumulates
/// the per-spike `Y_lm * intensity` contributions into `a_lm`
/// coefficients via WMMA Tensor Core fragments. The
/// rotationally-invariant power spectrum `C_l = Σ_m |a_lm|²` is the
/// F1 SWITCH selector for the Kabsch-purged adjudicator.
pub mod sh_basis;
/// M1.1 — Per-site canonical container types: [`site_manifest::SiteManifest`],
/// [`site_manifest::CentroidManifold`] (8-slot, named-accessor-only),
/// [`site_manifest::Centroid3D`] (4D-aware), [`site_manifest::CausalScalars`],
/// and the supporting identity newtypes. See module docs for the
/// per-site / frame-level / 8-slot-projection layering.
pub mod site_manifest;
/// RECT-3.1.b — SO(3) projection kernel + [`so3_project::ContactShellTile`].
/// Consumes RichSpike clusters, evaluates Y_lm per spike (relative
/// to the cluster centroid), accumulates per-cluster `a_lm` via
/// warp-shuffle reduce, and writes a 384-byte, 128-byte-aligned
/// hardware execution tile with the rotation-invariant power
/// spectrum `C_l = Σ_m |a_lm|²`. The `g11_rotation_invariance`
/// test pins the SO(3)-invariance contract within 1e-3 relative
/// tolerance over 10 random rotations. RECT-3.1.c will refactor
/// the inner accumulator to `nvcuda::wmma::fragment` matmul without
/// changing the on-tile layout.
pub mod so3_project;
pub mod spatial_index;
/// Typed multi-view localization layer (Phase 1 of the
/// post-05dbc3dc canonical architecture lane). Introduces
/// [`spatial_view::SpatialView`] + [`spatial_view::CentroidManifold`]
/// so that canonical site/localization access must name the physical
/// representation. No hidden default. Module is always compiled
/// (no `gpu` gate) because the types are pure data.
pub mod spatial_view;
/// Phase 2 audit spine — [`transform::AuditedTransform`] trait plus
/// the typed vocabulary (`DeterminismClass`, `TolerancePolicy`,
/// `TransformViolation`, `AuditOutcome`) every governed transform
/// uses. The framework module is feature-agnostic; individual
/// concrete transforms may be feature-gated on their dependencies.
pub mod transform;
#[cfg(feature = "gpu")]
pub mod ultimate_engine;
/// F2 lane (Blackwell Convergence mandate §3) — stream-ordered
/// memory pool wrapper around `cudaMemPool_t` + VRAM audit telemetry
/// struct. Replaces static "worst-case" pre-allocation with
/// `cudaMallocFromPoolAsync` / `cudaFreeAsync` and a 24-byte
/// device-side audit (`current_allocated_bytes`,
/// `peak_high_water_mark`, `pool_exhaustion_flag`) that the F1
/// Adjudicator's SWITCH node consumes as the Case-2 Violation
/// trigger. See module docs for the caller contract.
pub mod vram_pool;
/// ZSTR — Zero-Stall Telemetry Ring.
/// Phase 1 (G21): `ZstrRing` pinned triple-buffer + alignment gate.
/// Phase 2 (G22): `spawn_zstr_consumer` O_DIRECT NVMe writer thread.
/// CUDA kernels: `zstr_signal_completion_kernel` (fence) +
/// `asc_inject_repulsion_v4_kernel` (vectorized ASC force injection).
#[cfg(feature = "gpu")]
pub mod zstr;

pub mod boltzmann_weights;
pub mod cubical_ph;
#[cfg(feature = "gpu")]
pub mod gpu_knn;
#[cfg(feature = "gpu")]
pub mod hierarchical_clustering;
#[cfg(feature = "gpu")]
pub mod parallel_replica;
#[cfg(feature = "gpu")]
pub mod pharmacophore_gpu;
/// Gate G3 — `phase_bits: u32` (per-spike Arrow column) layout decoder.
/// Documents and decodes the bit-packed CCNS phase information emitted
/// per-spike. Layout: bits 0–9 = 10-bit CCNS phase index (0–1023);
/// bits 10–31 = reserved/unused. Schema doc: `docs/phase_bits_schema.md`.
pub mod phase_bits;
#[cfg(feature = "gpu")]
pub mod rt_utils;
/// Per-spike Apache Arrow IPC writer for training-grade output (Stage 1B-1).
/// Produces a single columnar `.arrow` file per target with full per-spike
/// tagging (30 columns) including stratified background spike preservation.
#[cfg(feature = "gpu")]
pub mod spike_arrow_writer;
#[cfg(feature = "gpu")]
pub mod spike_density;
/// M1.1 — `SpikeToCluster4D` transform type-definitions:
/// [`spike_to_cluster_4d::SpikeToCluster4DInput`],
/// [`spike_to_cluster_4d::SpikeToCluster4DOutput`],
/// [`spike_to_cluster_4d::ConservationScalars`] (Family-A audit payload),
/// [`spike_to_cluster_4d::ManifoldViewAabbFfi`] (`#[repr(C)]` FFI sibling).
/// FFI extern declarations and the `AuditedTransform` impl land in M1.2.
pub mod spike_to_cluster_4d;

/// Bayesian Online Changepoint Detection (Adams & MacKay, 2007) for
/// physics-driven dynamic chunking. Stage 1B-2: opt-in alternative to the
/// magic `chunk_size = 500` constant in the autonomous chunk loop.
pub mod bocpd;

/// Tokenized ranker (v4 LOTO) — learned site-ranking lookup baked into
/// the binary. Replaces legacy druggability-weighted ranking. Source:
/// `/mnt/storage/spike-audit/ranker-loto-v4/` (SR@1 36.4%, SR@3 81.1%,
/// SR@5 94.7% via LOTO on 302 targets).
pub mod tokenized_ranker;

/// XGBoost v3 site ranker — ONNX-backed gradient-boosted model with 13
/// continuous features and graded LOTO labels. Beats tokenized v4 by
/// +11.4 pts SR@1 (47.83% vs 36.42%). Loaded from embedded ONNX via the
/// `ort` crate. Source: `/mnt/storage/spike-audit/ranker-xgb-v3/`.
pub mod xgb_ranker;

/// Gaussian-Copula Partial Information Decomposition (Ince 2017) for
/// per-residue synergy fraction estimation. Stage 1B-3: principled,
/// parameter-free steering weights for the Stage 2 closed-loop ASC
/// writeback (`weight = synergy_fraction(target=future_spike_rate,
/// source_A=scout_group_rate, source_B=observer_group_rate)` per residue).
pub mod gcpid;

/// Control-plane trace schema and NDJSON writer helper.
///
/// Pure Rust data-plane types for tagged control records; this module does
/// not install runtime wiring or touch CUDA graph topology.
pub mod control_trace;
/// F2 evidence-plane sidecar schema.
///
/// Pure Rust data-plane types for ring status, write commits, and artifact
/// completeness; runtime emission remains separately gated.
pub mod f2_evidence;
/// PATH-A mid-chunk stall watchdog and crash-safe evidence-exit fallback.
///
/// Per-stream heartbeats, polling watchdog, async-signal-safe SIGINT
/// handler, and a deadman path that emits minimal path_a_completion.json
/// then process::exit(2)s when normal teardown cannot return.
pub mod path_a_watchdog;
/// Tier-2 ASC rescue controller (Adams & MacKay 2007 BOCPD, Ince 2017 GC-PID).
/// Data-driven, continuous-magnitude rescue decisions when the engine falls
/// into a zero-spike regime. Decision-only (v1); emits telemetry for V2-only
/// actions requiring engine kernel work.
pub mod rescue_controller;

pub mod batch_scheduler;
pub mod clustering;
pub mod composition;
pub mod input;
pub mod mapping;
pub mod neuromorphic;
pub mod pipeline;
pub mod rmsf;
pub mod trajectory;
pub mod uv_bias;

// Re-exports for convenience
#[cfg(feature = "gpu")]
pub use active_sensing::{
    ActiveSensingBuilder, ActiveSensingConfig, ActiveSensingEngine, ActiveSensingMode,
    ActiveSensingResults, AromaticGroup, CoherentProbe, CrypticSiteCandidate, Float3,
    ProbeResponse, ProbeType, ResonancePeak, SpikeSequenceDetector,
};
pub use adaptive::{
    AdaptiveGridProtocol, AdaptiveNhsEngine, AdaptiveStepResult, AdaptiveSummary, CascadeDetector,
    CascadeEvent, ExplorationPhase, GridPhase, JitterConfig, JitterDetector, JitterSignal,
    QuietBaseline, UvStrategy,
};
pub use aromatic_proximity::{
    AromaticProximityAnalysis, AromaticProximityAnalyzer, CrypticSite, ProximityBin,
    ProximitySummary, SiteProximityResult,
};
pub use avalanche::{AvalancheDetector, CrypticSiteEvent};
pub use batch_scheduler::{
    BatchExecutionResult, BatchExecutionSummary, BatchScheduler, BatchSchedulerConfig,
    ExecutionSchedule, ScheduleStats, ScheduledBatch,
};
pub use clustering::{
    ClusteringConfig, ClusteringResults, RepresentativeFrame, TrajectoryClusterer,
};
pub use composition::{
    group_for_batch, BatchCompatibility, BatchGroup, ChainInfo, ComplexityTier, MemoryProfile,
    MemoryTier, ResidueInfo, ResidueKey, SpikeDensityTier, StructureComposition,
};
pub use config::UvSpectroscopyConfig;
pub use config::{HydrophobicityThresholds, NhsConfig, UvBiasConfig};
pub use exclusion::{ClassifiedAtom, ExclusionComputer, ExclusionGrid};
#[cfg(feature = "gpu")]
#[allow(deprecated)]
pub use fused_engine::{
    compute_alignment_quality,
    // RMSD utilities
    compute_rmsd_subset,
    find_atoms_near_position,
    CryoPhase,
    // Unified cryo-UV protocol (canonical method)
    CryoUvProtocol,
    EnsembleSnapshot,
    FlexibilityClass,
    GpuSpikeEvent, // Full spike event from GPU with timestamps and residues
    NhsAmberFusedEngine,
    RunSummary,
    // Spike events
    SpikeEvent as FusedSpikeEvent,
    SpikePersistenceTracker,
    SpikeQualityCategory,
    // Quality scoring types
    SpikeQualityScore,
    StepResult,
    // Deprecated (use CryoUvProtocol instead)
    TemperatureProtocol,
    UvProbeConfig,
};
#[cfg(feature = "gpu")]
pub use gpu::{FrameResult, NhsGpuEngine, DEFAULT_GRID_DIM, DEFAULT_GRID_SPACING};
#[cfg(feature = "gpu")]
pub use hierarchical_clustering::{
    HierarchicalConfig, HierarchicalResult, HierarchicalRtClustering, PersistentCluster,
};
pub use input::{NhsAtomType, NhsPreparedInput, PrismPrepTopology};
pub use mapping::{
    compare_conditions, CalibrationMetrics, ComparativeAnalysis, ExperimentalCondition,
    MappedHotspot, NearbyResidue, NhsSiteId, NhsSiteMapper, ProtocolType, RobustSite,
};
pub use neuromorphic::{DewettingNetwork, DewettingNeuron, NetworkStats, Synapse};
#[cfg(feature = "gpu")]
pub use parallel_replica::{ParallelReplicaEngine, ParallelReplicaResult, ParallelSpikeEvent};
#[cfg(feature = "gpu")]
pub use persistent_engine::{
    enhance_sites_with_aromatics,
    write_binding_site_visualizations,
    AromaticProximityInfo,
    BatchProcessor,
    BindingSiteFormatter,
    ClusteredBindingSite,
    ClusteringStats,
    DruggabilityScore,
    LiningResidue,
    MergedCluster,
    MultiScaleClusteringResult,
    PersistenceAnalysis,
    PersistentBatchConfig,
    PersistentEngineStats,
    PersistentNhsEngine,
    // Multi-scale clustering
    ScaleCluster,
    SiteClassification,
    SitePersistenceTracker,
    StructureResult,
    TrackedSite,
};
pub use pipeline::{NhsPipeline, NhsStats};
pub use rmsf::{RmsfAnalysis, RmsfCalculator};
#[cfg(feature = "gpu")]
pub use rt_analysis::{
    RtAnalysisConfig, RtAnalysisResults, RtProbeAnalyzer, SolvationDisruptionEvent,
    VoidFormationEvent,
};
#[cfg(feature = "gpu")]
pub use rt_clustering::{
    find_optixir_path, RtClusteringConfig, RtClusteringEngine, RtClusteringResult,
};
#[cfg(feature = "gpu")]
pub use rt_probe::{RtProbeConfig, RtProbeEngine, RtProbeSnapshot};
#[cfg(feature = "gpu")]
pub use rt_utils::{get_architecture_name, has_rt_cores, is_optix_available};
pub use trajectory::{
    load_ensemble_pdb, write_ensemble_pdb, TrajectoryConfig, TrajectoryFrame, TrajectoryStats,
    TrajectoryWriter,
};
#[cfg(feature = "gpu")]
pub use ultimate_engine::{
    OptimizationLevel, SimulationParams, UltimateEngine, UltimateEngineConfig, UltimateStepResult,
};
pub use uv_bias::{
    AromaticTarget,
    AromaticType,
    BurstEvent,
    CausalCorrelation,
    // Enhanced UV spectroscopy types
    ChromophoreType,
    DisulfideTarget,
    FrequencyHoppingProtocol,
    LocalTempRecord,
    PerturbationResult,
    SpectroscopyStats,
    SpikeCategory,
    SpikeEvent,
    UvBiasEngine,
    UvBiasStats,
    UvSpectroscopyResults,
    WavelengthAwareSpike,
};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if NHS was compiled with GPU support
#[cfg(feature = "gpu")]
pub const GPU_ENABLED: bool = true;

#[cfg(not(feature = "gpu"))]
pub const GPU_ENABLED: bool = false;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = NhsConfig::default();
        assert!(config.grid_spacing > 0.0);
        assert!(config.uv_bias_enabled);
    }
}
pub mod simd_batch_integration;

// M1.2.19.B / Amendment 3.13 — Asynchronous Manifold Sequencer Channel-B.
// Pinned-host, device-mapped GhostTileFrame ring + capture-kernel FFI.
#[cfg(feature = "gpu")]
pub mod ghost_tile;

/// GhostPhaseLattice4D — physically-constrained spatiotemporal connected
/// components for Ghost v2 records. Replaces the legacy O(N²) DBSCAN on
/// post-MD spike clouds with a 4D phase lattice (spatial × protocol-phase ×
/// step bucket) gated by AABB overlap, monotone protocol-phase transitions,
/// and 4-plane SO(3) cosine-similarity scoring. See
/// `crates/prism-nhs/src/cuda/ghost_lattice_kernel.cu` for the kernel and
/// `crates/prism-nhs/src/cuda/ghost_lattice_kernel.cuh` for the FFI.
#[cfg(feature = "gpu")]
pub mod ghost_phase_lattice;

/// Materializer that bridges a `GhostPhaseLattice4D` outcome into the
/// per-site `Option<...>` extension blocks on
/// `crate::site_manifest::SiteManifest` (provenance + phase manifold +
/// therm/ccns lifecycle + so3 manifold).
#[cfg(feature = "gpu")]
pub mod ghost_phase_materializer;
