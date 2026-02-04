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
#[cfg(feature = "gpu")]
pub mod rt_probe;
#[cfg(feature = "gpu")]
pub mod rt_analysis;
#[cfg(feature = "gpu")]
pub mod persistent_engine;
#[cfg(feature = "gpu")]
pub mod active_sensing;
#[cfg(feature = "gpu")]
pub mod ultimate_engine;
#[cfg(feature = "gpu")]
pub mod rt_clustering;
#[cfg(feature = "gpu")]
pub mod rt_utils;
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
    CryoUvProtocol,
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
