//! Report pipeline orchestration (DEPRECATED)
//!
//! # Deprecation Notice
//!
//! `ReportPipeline` is DEPRECATED. Use `FinalizeStage` instead.
//!
//! The correct production path is:
//! 1. `prism4d run` - Runs the GPU engine which produces `events.jsonl`
//! 2. Engine writes real spike detections to `events.jsonl` (no synthetic fallback)
//! 3. `FinalizeStage` runs automatically after successful engine completion
//! 4. `FinalizeStage` consumes `events.jsonl` and produces the evidence pack
//!
//! To run the finalize stage standalone (from existing events.jsonl):
//! ```bash
//! prism4d finalize --events events.jsonl --pdb input.pdb --out results/
//! ```
//!
//! # Why This Change?
//!
//! - The old `ReportPipeline` had fallback paths that could fabricate test data
//! - This made it impossible to distinguish real MD results from fabricated data
//! - `FinalizeStage` has NO fabricated data paths - it requires real `events.jsonl`
//! - All PocketEvent creation in production code is now ONLY in the GPU engine

use crate::config::ReportConfig;
use crate::outputs::OutputContract;
use anyhow::Result;

/// Summary of voxelization results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoxelizationSummary {
    /// Grid dimensions [nx, ny, nz]
    pub dims: [usize; 3],
    /// Voxel spacing (Å)
    pub spacing: f32,
    /// Total estimated pocket volume (Å³)
    pub total_volume: f32,
    /// Number of voxels above threshold
    pub voxels_above_threshold: usize,
    /// Number of events processed
    pub n_events: usize,
}

/// Pipeline result
#[derive(Debug)]
pub struct PipelineResult {
    /// Output directory
    pub output_dir: std::path::PathBuf,
    /// Number of sites detected
    pub n_sites: usize,
    /// Number of druggable sites
    pub n_druggable: usize,
    /// Ablation significant
    pub cryo_significant: bool,
    pub uv_significant: bool,
    /// Voxelization results
    pub voxelization: Option<VoxelizationSummary>,
    /// Holo correlation (if holo provided)
    pub holo_overlap: Option<crate::alignment::VoxelLigandOverlap>,
    /// Files generated
    pub files_generated: Vec<String>,
}

/// Main report pipeline (DEPRECATED - use FinalizeStage instead)
///
/// # Deprecation
///
/// This struct is deprecated. Use `FinalizeStage` from `prism_report::finalize` instead.
///
/// The production pipeline is:
/// 1. `prism4d run` - Runs GPU engine, produces `events.jsonl`
/// 2. `FinalizeStage` - Consumes `events.jsonl`, produces evidence pack
///
/// Example:
/// ```bash
/// # Complete pipeline
/// prism4d run --topology prep.json --pdb input.pdb --out results/
///
/// # Finalize only (existing events.jsonl)
/// prism4d finalize --events events.jsonl --pdb input.pdb --out results/
/// ```
#[deprecated(since = "1.2.0", note = "Use FinalizeStage instead. See module docs.")]
pub struct ReportPipeline {
    #[allow(dead_code)]
    config: ReportConfig,
    #[allow(dead_code)]
    output: OutputContract,
}

#[allow(deprecated)]
impl ReportPipeline {
    /// Create new pipeline (DEPRECATED)
    #[deprecated(since = "1.2.0", note = "Use FinalizeStage::new instead")]
    pub fn new(config: ReportConfig) -> Result<Self> {
        let output = OutputContract::new(&config.output_dir)?;
        Ok(Self { config, output })
    }

    /// Run the complete pipeline (DEPRECATED - always fails)
    ///
    /// This method always returns an error directing you to use `FinalizeStage`.
    /// There is no fallback path for fabricated data - real events.jsonl is required.
    #[deprecated(since = "1.2.0", note = "Use FinalizeStage::run instead")]
    pub fn run(&self) -> Result<PipelineResult> {
        anyhow::bail!(
            "ReportPipeline is DEPRECATED and no longer functional.\n\n\
             Use FinalizeStage instead:\n\n\
             1. Run the complete pipeline:\n\
                prism4d run --topology prep.json --pdb input.pdb --out results/\n\n\
             2. Or run finalize stage from existing events.jsonl:\n\
                prism4d finalize --events events.jsonl --pdb input.pdb --out results/\n\n\
             The prism4d pipeline:\n\
             - Requires real GPU engine execution (no fabricated data)\n\
             - Validates events.jsonl exists and is non-empty\n\
             - Runs FinalizeStage automatically after engine completion\n\n\
             See prism_report::finalize::FinalizeStage for the production code path."
        );
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    #[allow(deprecated)]
    fn test_pipeline_creation() {
        let tmp = TempDir::new().unwrap();
        let config = ReportConfig {
            output_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };

        let pipeline = ReportPipeline::new(config).unwrap();

        // Running should fail with deprecation message
        let result = pipeline.run();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("DEPRECATED"), "Should mention deprecation");
        assert!(err.contains("FinalizeStage"), "Should mention FinalizeStage");
    }
}
