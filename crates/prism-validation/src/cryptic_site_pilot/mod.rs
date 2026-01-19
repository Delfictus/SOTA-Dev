//! PRISM-4D Cryptic Binding Site Detection - Pilot Module
//!
//! Production-quality cryptic pocket detection for pharmaceutical pilots.
//!
//! # Deliverables
//!
//! | Deliverable | Format | Purpose |
//! |-------------|--------|---------|
//! | Pocket report | PDF/HTML | Executive summary for pharma partner |
//! | Trajectory file | Multi-MODEL PDB | Kabsch-aligned conformations |
//! | Pocket structures | PDB ensemble | Top 5 open conformations per cryptic site |
//! | RMSF heatmap | CSV | Per-residue flexibility |
//! | Volume time series | CSV | Pocket breathing dynamics |
//! | Residue contact list | CSV | Defines binding site for docking |
//!
//! # Validation Targets
//!
//! Must detect ≥4/5 known cryptic sites:
//! - TEM-1 β-lactamase (1BTL) - Omega loop pocket
//! - p38 MAP kinase (1A9U) - DFG-out pocket
//! - IL-2 (1M47) - Composite groove
//! - BCL-xL (1MAZ) - BH3 groove extension
//! - PDK1 (1H1W) - PIF pocket
//!
//! # Usage
//!
//! ```rust,ignore
//! use prism_validation::cryptic_site_pilot::{CrypticPilotConfig, CrypticPilotPipeline};
//!
//! let config = CrypticPilotConfig::default();
//! let pipeline = CrypticPilotPipeline::new(config)?;
//! let result = pipeline.analyze("path/to/structure.pdb")?;
//! result.write_all_outputs("output_dir/")?;
//! ```

pub mod config;
pub mod pipeline;
pub mod volume_tracker;
pub mod druggability;
pub mod outputs;
pub mod utils;
pub mod topology_loader;

// MD-based pipeline (Langevin dynamics + Jaccard matching)
pub mod md_cryptic_pipeline;

// Re-exports for convenience
pub use config::CrypticPilotConfig;
pub use pipeline::{CrypticPilotPipeline, CrypticPilotResult};
pub use volume_tracker::{VolumeFrame, VolumeTimeSeries};
pub use druggability::{DruggabilityScore, DruggabilityScorer};
pub use outputs::{MultiModelPdbWriter, ReportGenerator};

// MD pipeline exports
#[cfg(feature = "cryptic-gpu")]
pub use md_cryptic_pipeline::{
    MdCrypticConfig, MdCrypticPipeline, MdCrypticResult,
    CrypticSite, ResidueBasedVolumeTracker, TrackerDiagnostics,
};
