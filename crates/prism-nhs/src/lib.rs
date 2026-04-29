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
pub mod solvate;
pub mod rt_targets;
pub mod exclusion;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod fused_engine;
/// PRISM-Therm: SDST thermodynamic analysis bridge.
/// Requires the `gpu` feature (SDST is CUDA-only).
#[cfg(feature = "gpu")]
pub mod sdst_bridge;
/// Spike Thermodynamic Integration: Jarzynski/Crooks/BAR free energy from spike events.
#[cfg(feature = "gpu")]
pub mod spike_thermodynamic_integration;
/// PRISM-Therm: output pipeline (JSON reports, druggability PDB, ranked table).
#[cfg(feature = "gpu")]
pub mod sdst_report;
#[cfg(feature = "gpu")]
pub mod rt_probe;
#[cfg(feature = "gpu")]
pub mod rt_analysis;
#[cfg(feature = "gpu")]
pub mod persistent_engine;
#[cfg(feature = "gpu")]
pub mod coupled_md;
#[cfg(feature = "gpu")]
pub mod twin_detection;
#[cfg(feature = "gpu")]
pub mod twin_kernels;
#[cfg(feature = "gpu")]
pub mod protocol_state;
#[cfg(feature = "gpu")]
pub mod graph_capture;
#[cfg(feature = "gpu")]
pub mod active_sensing;
#[cfg(feature = "gpu")]
pub mod ultimate_engine;
#[cfg(feature = "gpu")]
pub mod rt_clustering;
pub mod spatial_index;
#[cfg(feature = "gpu")]
pub mod grid_debug_backend;
/// GPU spatial-hash CCL backend for post-MD clustering. Replaces the
/// CPU `fallback_grid_cluster` on SM120+. See `gpu_cluster_backend.rs`
/// module docs.
#[cfg(feature = "gpu")]
pub mod gpu_cluster_backend;
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
/// Phase 3 — Pillar 2 typed block.  See [`causome::CausomeBlock`].
/// The producer is `transform::cluster_to_causome::ClusterToCausome`.
pub mod causome;
/// Entangled Transform — typed manifold views with strict AABBs.
/// Implements blueprint mandates M4 (strict struct typing for the
/// 2+2+2+2 graph) and M5 (LBVH-ready, support-only AABBs). See
/// [`docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md`](../docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md)
/// for the binding contract. This module is type-definitions-only;
/// the on-device producer (`SpikeToCluster4D`, M1) and CUDA / FFI
/// shims land in subsequent commits.
pub mod entangled_manifold;
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
/// Diagnostic / informational modules — opt-in via the `diagnostic`
/// Cargo feature. Per the Blackwell Convergence mandate the M1.2.5b
/// typed-producer differential is no longer a closure gate; it
/// survives here as an informational tool only. See
/// [`diagnostic`] module docs for the full deprecation rationale.
#[cfg(feature = "diagnostic")]
pub mod diagnostic;
/// M1.1 — Per-site canonical container types: [`site_manifest::SiteManifest`],
/// [`site_manifest::CentroidManifold`] (8-slot, named-accessor-only),
/// [`site_manifest::Centroid3D`] (4D-aware), [`site_manifest::CausalScalars`],
/// and the supporting identity newtypes. See module docs for the
/// per-site / frame-level / 8-slot-projection layering.
pub mod site_manifest;
/// Gate G1 — Multi-seed Arrow file merger. Concatenates per-seed
/// `<prefix>.topology.spike_events.arrow` files into a single
/// multi-seed stream while preserving per-spike `replica_seed`
/// provenance through the existing schema column. Pure Rust IPC
/// reader/writer; no Python in the data plane.
pub mod multi_seed_merger;

/// Gate G3 — `phase_bits: u32` (per-spike Arrow column) layout decoder.
/// Documents and decodes the bit-packed CCNS phase information emitted
/// per-spike. Layout: bits 0–9 = 10-bit CCNS phase index (0–1023);
/// bits 10–31 = reserved/unused. Schema doc: `docs/phase_bits_schema.md`.
pub mod phase_bits;
/// M1.1 — `SpikeToCluster4D` transform type-definitions:
/// [`spike_to_cluster_4d::SpikeToCluster4DInput`],
/// [`spike_to_cluster_4d::SpikeToCluster4DOutput`],
/// [`spike_to_cluster_4d::ConservationScalars`] (Family-A audit payload),
/// [`spike_to_cluster_4d::ManifoldViewAabbFfi`] (`#[repr(C)]` FFI sibling).
/// FFI extern declarations and the `AuditedTransform` impl land in M1.2.
pub mod spike_to_cluster_4d;
#[cfg(feature = "gpu")]
pub mod spike_density;
#[cfg(feature = "gpu")]
pub mod hierarchical_clustering;
#[cfg(feature = "gpu")]
pub mod rt_utils;
#[cfg(feature = "gpu")]
pub mod parallel_replica;
#[cfg(feature = "gpu")]
pub mod pharmacophore_gpu;
#[cfg(feature = "gpu")]
pub mod gpu_knn;
pub mod cubical_ph;
pub mod boltzmann_weights;
/// Per-spike Apache Arrow IPC writer for training-grade output (Stage 1B-1).
/// Produces a single columnar `.arrow` file per target with full per-spike
/// tagging (30 columns) including stratified background spike preservation.
#[cfg(feature = "gpu")]
pub mod spike_arrow_writer;

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

/// Tier-2 ASC rescue controller (Adams & MacKay 2007 BOCPD, Ince 2017 GC-PID).
/// Data-driven, continuous-magnitude rescue decisions when the engine falls
/// into a zero-spike regime. Decision-only (v1); emits telemetry for V2-only
/// actions requiring engine kernel work.
pub mod rescue_controller;

pub mod composition;
pub mod batch_scheduler;
pub mod input;
pub mod mapping;
pub mod neuromorphic;
pub mod pipeline;
pub mod trajectory;
pub mod rmsf;
pub mod clustering;
pub mod uv_bias;

// Re-exports for convenience
pub use avalanche::{AvalancheDetector, CrypticSiteEvent};
pub use config::{HydrophobicityThresholds, NhsConfig, UvBiasConfig};
pub use exclusion::{ClassifiedAtom, ExclusionComputer, ExclusionGrid};
pub use neuromorphic::{DewettingNetwork, DewettingNeuron, NetworkStats, Synapse};
pub use pipeline::{NhsPipeline, NhsStats};
pub use uv_bias::{
    AromaticTarget, AromaticType, BurstEvent, CausalCorrelation, PerturbationResult,
    SpikeEvent, UvBiasEngine, UvBiasStats,
    // Enhanced UV spectroscopy types
    ChromophoreType, DisulfideTarget, FrequencyHoppingProtocol, LocalTempRecord,
    SpikeCategory, WavelengthAwareSpike, UvSpectroscopyResults, SpectroscopyStats,
};
pub use config::UvSpectroscopyConfig;
pub use aromatic_proximity::{
    AromaticProximityAnalyzer, AromaticProximityAnalysis, CrypticSite,
    ProximityBin, SiteProximityResult, ProximitySummary,
};
#[cfg(feature = "gpu")]
pub use gpu::{NhsGpuEngine, FrameResult, DEFAULT_GRID_DIM, DEFAULT_GRID_SPACING};
#[cfg(feature = "gpu")]
#[allow(deprecated)]
pub use fused_engine::{
    NhsAmberFusedEngine,
    // Unified cryo-UV protocol (canonical method)
    CryoUvProtocol, CryoPhase, FlexibilityClass,
    // Deprecated (use CryoUvProtocol instead)
    TemperatureProtocol,
    UvProbeConfig,
    // Spike events
    SpikeEvent as FusedSpikeEvent, EnsembleSnapshot, StepResult, RunSummary,
    GpuSpikeEvent,  // Full spike event from GPU with timestamps and residues
    // Quality scoring types
    SpikeQualityScore, SpikeQualityCategory, SpikePersistenceTracker,
    // RMSD utilities
    compute_rmsd_subset, find_atoms_near_position, compute_alignment_quality,
};
#[cfg(feature = "gpu")]
pub use rt_probe::{
    RtProbeEngine, RtProbeConfig, RtProbeSnapshot,
};
#[cfg(feature = "gpu")]
pub use rt_analysis::{
    RtProbeAnalyzer, RtAnalysisConfig, RtAnalysisResults,
    VoidFormationEvent, SolvationDisruptionEvent,
};
#[cfg(feature = "gpu")]
pub use active_sensing::{
    ActiveSensingEngine, ActiveSensingConfig, ActiveSensingMode, ActiveSensingResults,
    ActiveSensingBuilder, CoherentProbe, AromaticGroup, ProbeResponse, ResonancePeak,
    SpikeSequenceDetector, CrypticSiteCandidate, Float3, ProbeType,
};
#[cfg(feature = "gpu")]
pub use persistent_engine::{
    PersistentNhsEngine, PersistentBatchConfig, BatchProcessor,
    StructureResult, PersistentEngineStats,
    ClusteredBindingSite, ClusteringStats, DruggabilityScore, SiteClassification,
    AromaticProximityInfo, enhance_sites_with_aromatics,
    SitePersistenceTracker, TrackedSite, PersistenceAnalysis,
    BindingSiteFormatter, write_binding_site_visualizations,
    LiningResidue,
    // Multi-scale clustering
    ScaleCluster, MergedCluster, MultiScaleClusteringResult,
};
#[cfg(feature = "gpu")]
pub use ultimate_engine::{
    UltimateEngine, UltimateEngineConfig, UltimateStepResult,
    OptimizationLevel, SimulationParams,
};
#[cfg(feature = "gpu")]
pub use rt_clustering::{
    RtClusteringEngine, RtClusteringConfig, RtClusteringResult,
    find_optixir_path,
};
#[cfg(feature = "gpu")]
pub use rt_utils::{has_rt_cores, is_optix_available, get_architecture_name};
#[cfg(feature = "gpu")]
pub use hierarchical_clustering::{
    HierarchicalRtClustering, HierarchicalConfig, HierarchicalResult, PersistentCluster,
};
#[cfg(feature = "gpu")]
pub use parallel_replica::{
    ParallelReplicaEngine, ParallelReplicaResult, ParallelSpikeEvent,
};
pub use input::{NhsAtomType, NhsPreparedInput, PrismPrepTopology};
pub use adaptive::{
    AdaptiveGridProtocol, AdaptiveNhsEngine, AdaptiveStepResult, AdaptiveSummary,
    CascadeDetector, CascadeEvent, ExplorationPhase, GridPhase, JitterConfig,
    JitterDetector, JitterSignal, QuietBaseline, UvStrategy,
};
pub use mapping::{
    CalibrationMetrics, ComparativeAnalysis, ExperimentalCondition, MappedHotspot,
    NearbyResidue, NhsSiteId, NhsSiteMapper, ProtocolType, RobustSite, compare_conditions,
};
pub use trajectory::{
    TrajectoryConfig, TrajectoryFrame, TrajectoryWriter, TrajectoryStats,
    write_ensemble_pdb, load_ensemble_pdb,
};
pub use rmsf::{RmsfAnalysis, RmsfCalculator};
pub use clustering::{
    ClusteringConfig, ClusteringResults, RepresentativeFrame, TrajectoryClusterer,
};
pub use composition::{
    StructureComposition, ResidueKey, ResidueInfo, ChainInfo,
    MemoryTier, ComplexityTier, SpikeDensityTier, BatchCompatibility,
    MemoryProfile, BatchGroup, group_for_batch,
};
pub use batch_scheduler::{
    BatchScheduler, BatchSchedulerConfig, ExecutionSchedule, ScheduledBatch,
    ScheduleStats, BatchExecutionResult, BatchExecutionSummary,
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
