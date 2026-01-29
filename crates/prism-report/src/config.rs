//! Configuration structures for the report pipeline

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Input PDB file (apo structure)
    pub input_pdb: PathBuf,

    /// Output directory
    pub output_dir: PathBuf,

    /// Number of replicates
    pub replicates: usize,

    /// UV wavelengths to scan (nm)
    pub wavelengths: Vec<f32>,

    /// Optional holo structure for Tier 1 correlation
    pub holo_pdb: Option<PathBuf>,

    /// Optional truth residues for Tier 2 correlation
    pub truth_residues: Option<PathBuf>,

    /// Contact distance cutoff for auto-extracting truth residues from holo (Angstroms)
    /// Used when --holo is provided but --truth-residues is not
    /// Default: 4.5 Å (standard binding site definition)
    pub contact_cutoff: Option<f32>,

    /// Temperature protocol
    pub temperature_protocol: TemperatureProtocol,

    /// Ablation configuration
    pub ablation: AblationConfig,

    /// Site detection parameters
    pub site_detection: SiteDetectionConfig,

    /// Output format options
    pub output_formats: OutputFormats,

    /// CUDA device ID
    pub device_id: i32,

    /// Verbose output
    pub verbose: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            input_pdb: PathBuf::new(),
            output_dir: PathBuf::from("results"),
            replicates: 5,
            wavelengths: vec![258.0, 274.0, 280.0],
            holo_pdb: None,
            truth_residues: None,
            contact_cutoff: None,  // Uses default 4.5Å in load_truth()
            temperature_protocol: TemperatureProtocol::default(),
            ablation: AblationConfig::default(),
            site_detection: SiteDetectionConfig::default(),
            output_formats: OutputFormats::default(),
            device_id: 0,
            verbose: false,
        }
    }
}

/// Temperature protocol configuration
///
/// The cryo-UV protocol has three distinct phases:
/// 1. Cold hold: system equilibrates at cryogenic temperature (start_temp)
/// 2. Ramp: temperature transitions from start_temp to end_temp
/// 3. Warm hold: system equilibrates at physiological temperature (end_temp)
///
/// Each phase has an explicit step count - no inference from ratios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureProtocol {
    /// Start temperature (K) - cryogenic
    pub start_temp: f32,
    /// End temperature (K) - physiological
    pub end_temp: f32,
    /// Steps to hold at cold (cryogenic) temperature before ramp
    pub cold_hold_steps: i32,
    /// Steps for temperature ramp from start_temp to end_temp
    pub ramp_steps: i32,
    /// Steps to hold at warm (physiological) temperature after ramp
    pub warm_hold_steps: i32,
}

impl Default for TemperatureProtocol {
    fn default() -> Self {
        Self {
            start_temp: 50.0,   // Cryogenic temperature for maximum contrast
            end_temp: 300.0,    // Physiological temperature
            cold_hold_steps: 20000,  // Equilibrate at cryo
            ramp_steps: 30000,       // Temperature transition
            warm_hold_steps: 50000,  // Equilibrate at physiological
        }
    }
}

impl TemperatureProtocol {
    /// Total steps for the complete protocol
    pub fn total_steps(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps + self.warm_hold_steps
    }

    /// Get the step boundary where cold phase ends and ramp begins
    pub fn cold_end_step(&self) -> i32 {
        self.cold_hold_steps
    }

    /// Get the step boundary where ramp phase ends and warm begins
    pub fn ramp_end_step(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps
    }
}

/// Ablation configuration - MANDATORY for all runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Run baseline (no cryo + UV off)
    pub run_baseline: bool,
    /// Run cryo-only (cryo on + UV off)
    pub run_cryo_only: bool,
    /// Run cryo+UV (cryo on + UV on)
    pub run_cryo_uv: bool,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            // Ablation is MANDATORY - all three must run
            run_baseline: true,
            run_cryo_only: true,
            run_cryo_uv: true,
        }
    }
}

impl AblationConfig {
    /// Validate ablation configuration
    ///
    /// NOTE: Ablation validation is now OPTIONAL (production mode).
    /// The pipeline will run regardless of which modes are enabled.
    /// Ablation metrics are computed as interpretive outputs only,
    /// not as gating/blocking requirements.
    pub fn validate(&self) -> Result<(), String> {
        // Production mode: ablation is optional/interpretive only
        // Always return Ok to allow pipeline to run
        Ok(())
    }

    /// Check if all ablation modes are enabled (for informational purposes)
    pub fn is_complete(&self) -> bool {
        self.run_baseline && self.run_cryo_only && self.run_cryo_uv
    }
}

/// Site detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteDetectionConfig {
    /// Minimum spike cluster size
    pub min_cluster_size: usize,
    /// Spatial clustering threshold (Å)
    pub cluster_threshold: f32,
    /// Minimum druggable volume (Å³)
    pub min_volume: f32,
    /// Minimum persistence (fraction of frames)
    pub min_persistence: f32,
    /// Minimum confidence score
    pub min_confidence: f32,
    /// Residue query radius for mapping (Å) - expanded from 5.0 to capture broader context
    #[serde(default = "default_residue_query_radius")]
    pub residue_query_radius_a: f32,
    /// Minimum replica agreement (fraction, 0.0-1.0) - site must appear in this fraction of replicates
    /// Set to 0.0 to disable replica agreement filtering
    #[serde(default = "default_min_replica_agreement")]
    pub min_replica_agreement: f32,
}

fn default_residue_query_radius() -> f32 {
    8.0
}

fn default_min_replica_agreement() -> f32 {
    0.5  // Default: site must appear in 50% of replicates
}

impl Default for SiteDetectionConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 100,  // Increased from 5 to reduce noise
            cluster_threshold: 5.0,
            min_volume: 100.0,
            min_persistence: 0.002, // 0.2% - require more persistent sites
            min_confidence: 0.3,
            residue_query_radius_a: 8.0,  // Expanded from 5.0 for broader binding site context
            min_replica_agreement: 0.5,   // Default: 50% replica agreement
        }
    }
}

/// Output format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFormats {
    /// Generate HTML report
    pub html: bool,
    /// Generate PDF report
    pub pdf: bool,
    /// Generate JSON summary
    pub json: bool,
    /// Generate CSV correlation
    pub csv: bool,
    /// Generate PyMOL sessions
    pub pymol: bool,
    /// Generate ChimeraX sessions
    pub chimerax: bool,
    /// Generate figures
    pub figures: bool,
    /// Generate MRC volume files
    pub mrc_volumes: bool,
}

impl Default for OutputFormats {
    fn default() -> Self {
        Self {
            html: true,
            pdf: true,
            json: true,
            csv: true,
            pymol: true,
            chimerax: true,
            figures: true,
            mrc_volumes: true,
        }
    }
}

/// Ranking score weights (must sum to 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingWeights {
    /// Weight for persistence score
    pub persistence: f32,
    /// Weight for volume score
    pub volume: f32,
    /// Weight for UV response (delta SASA/volume)
    pub uv_response: f32,
    /// Weight for hydrophobicity
    pub hydrophobicity: f32,
    /// Weight for replica agreement
    pub replica_agreement: f32,
}

impl Default for RankingWeights {
    fn default() -> Self {
        Self {
            persistence: 0.20,         // Reduced from 0.25
            volume: 0.15,              // Reduced from 0.20
            uv_response: 0.45,         // INCREASED from 0.25 (UV-LIF is primary validation)
            hydrophobicity: 0.10,      // Reduced from 0.15
            replica_agreement: 0.10,   // Reduced from 0.15
        }
    }
}

impl RankingWeights {
    /// Validate weights sum to 1.0
    pub fn validate(&self) -> Result<(), String> {
        let sum = self.persistence
            + self.volume
            + self.uv_response
            + self.hydrophobicity
            + self.replica_agreement;
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!(
                "Ranking weights must sum to 1.0, got {:.3}",
                sum
            ));
        }
        Ok(())
    }
}
