//! Unified Hybrid Configuration for PRISM-LBS
//!
//! Configuration structures for benchmark-compliant detection with provenance tracking.
//!
//! ## Detection Modes
//!
//! | Mode        | Kempe | Power | Use Case            | Speed      |
//! |-------------|-------|-------|---------------------|------------|
//! | Turbo       | 3     | 5     | High-throughput     | ~50 str/s  |
//! | Standard    | 8     | 12    | Balanced quality    | ~20 str/s  |
//! | Publication | 12    | 20    | Benchmark submission| ~10 str/s  |

use serde::{Deserialize, Serialize};

/// Main configuration for unified hybrid detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedHybridConfig {
    // --- Quality parameters ---
    /// Kempe refinement iterations (higher = better pocket boundary quality)
    pub kempe_iterations: u32,
    /// Power iterations for eigenvalue approximation (higher = better stability)
    pub power_iterations: u32,
    /// Grid spacing for alpha sphere generation (Angstroms)
    pub grid_spacing: f32,
    /// Minimum pocket volume threshold (cubic Angstroms)
    pub min_volume: f32,
    /// Maximum pocket volume threshold (cubic Angstroms)
    pub max_volume: f32,

    // --- Scoring weights ---
    /// Druggability scoring component weights
    pub druggability_weights: DruggabilityWeights,

    // --- Output configuration ---
    /// What level of provenance to include
    pub provenance_level: ProvenanceLevel,
    /// Enable LIGYSIS-compatible output fields
    pub enable_ligysis_compat: bool,
    /// Enable CryptoBench per-residue scoring
    pub enable_cryptobench_compat: bool,

    // --- GPU optimization ---
    /// Use GPU for detection kernels
    pub use_gpu: bool,
    /// Batch size for GPU operations
    pub gpu_batch_size: usize,
    /// Use pure GPU mode (bypass CPU geometry)
    pub pure_gpu_mode: bool,

    // --- Pocket filtering ---
    /// Maximum number of pockets to return
    pub max_pockets: usize,
    /// Minimum druggability score threshold
    pub min_druggability: f32,
    /// Enable pocket merging for fragmented sites
    pub enable_pocket_merging: bool,
    /// Merge distance threshold (Angstroms)
    pub merge_distance: f32,
}

impl Default for UnifiedHybridConfig {
    fn default() -> Self {
        Self {
            // Standard quality mode (balanced)
            kempe_iterations: 8,
            power_iterations: 12,
            grid_spacing: 1.0,
            min_volume: 50.0,
            max_volume: 5000.0,

            // Validated druggability weights
            druggability_weights: DruggabilityWeights::default(),

            // Full provenance for benchmark submission
            provenance_level: ProvenanceLevel::Full,
            enable_ligysis_compat: true,
            enable_cryptobench_compat: true,

            // GPU enabled by default
            use_gpu: true,
            gpu_batch_size: 1024,
            pure_gpu_mode: false,

            // Pocket filtering
            max_pockets: 20,
            min_druggability: 0.20,
            enable_pocket_merging: true,
            merge_distance: 12.0,
        }
    }
}

impl UnifiedHybridConfig {
    /// Create configuration from a quality preset
    pub fn from_preset(preset: QualityPreset) -> Self {
        let mut config = Self::default();
        match preset {
            QualityPreset::Turbo => {
                config.kempe_iterations = 3;
                config.power_iterations = 5;
                config.provenance_level = ProvenanceLevel::Minimal;
                config.enable_ligysis_compat = false;
                config.enable_cryptobench_compat = false;
            }
            QualityPreset::Standard => {
                // Default values are already Standard
            }
            QualityPreset::Publication => {
                config.kempe_iterations = 12;
                config.power_iterations = 20;
                config.provenance_level = ProvenanceLevel::Full;
                config.enable_ligysis_compat = true;
                config.enable_cryptobench_compat = true;
            }
            QualityPreset::UltraRefined => {
                config.kempe_iterations = 16;
                config.power_iterations = 30;
                config.provenance_level = ProvenanceLevel::Debug;
                config.enable_ligysis_compat = true;
                config.enable_cryptobench_compat = true;
            }
        }
        config
    }

    /// Create a DetectionMode from this config
    pub fn detection_mode(&self) -> DetectionMode {
        if self.kempe_iterations <= 3 && self.power_iterations <= 5 {
            DetectionMode::Turbo
        } else if self.kempe_iterations <= 8 && self.power_iterations <= 12 {
            DetectionMode::Standard
        } else if self.kempe_iterations <= 12 && self.power_iterations <= 20 {
            DetectionMode::Publication
        } else {
            DetectionMode::UltraRefined
        }
    }

    /// Get the mode name as a string (for provenance)
    pub fn mode_name(&self) -> &'static str {
        match self.detection_mode() {
            DetectionMode::Turbo => "turbo",
            DetectionMode::Standard => "standard",
            DetectionMode::Publication => "publication",
            DetectionMode::UltraRefined => "ultra_refined",
        }
    }
}

/// Detection mode presets
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMode {
    /// High-throughput screening (~50 str/s)
    Turbo,
    /// Balanced quality and speed (~20 str/s)
    Standard,
    /// Benchmark submission quality (~10 str/s)
    Publication,
    /// Maximum quality for difficult targets (~5 str/s)
    UltraRefined,
}

impl Default for DetectionMode {
    fn default() -> Self {
        Self::Standard
    }
}

impl std::fmt::Display for DetectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Turbo => write!(f, "turbo"),
            Self::Standard => write!(f, "standard"),
            Self::Publication => write!(f, "publication"),
            Self::UltraRefined => write!(f, "ultra_refined"),
        }
    }
}

/// Quality preset for configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityPreset {
    /// Fast screening mode
    Turbo,
    /// Balanced quality
    Standard,
    /// Benchmark submission
    Publication,
    /// Maximum quality
    UltraRefined,
}

/// Level of provenance detail to include
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceLevel {
    /// No provenance (fastest)
    None,
    /// Basic version and timing only
    Minimal,
    /// Standard provenance (recommended)
    Standard,
    /// Full provenance for benchmark submission
    Full,
    /// Debug-level provenance with all internal state
    Debug,
}

impl Default for ProvenanceLevel {
    fn default() -> Self {
        Self::Full
    }
}

impl ProvenanceLevel {
    /// Whether to include timing information
    pub fn include_timing(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether to include hardware information
    pub fn include_hardware(&self) -> bool {
        matches!(self, Self::Standard | Self::Full | Self::Debug)
    }

    /// Whether to include per-pocket genesis
    pub fn include_pocket_genesis(&self) -> bool {
        matches!(self, Self::Full | Self::Debug)
    }

    /// Whether to include per-residue assignment
    pub fn include_residue_assignment(&self) -> bool {
        matches!(self, Self::Full | Self::Debug)
    }

    /// Whether to include score breakdowns
    pub fn include_score_breakdown(&self) -> bool {
        matches!(self, Self::Full | Self::Debug)
    }

    /// Whether to include debug information
    pub fn include_debug(&self) -> bool {
        matches!(self, Self::Debug)
    }
}

/// Druggability scoring weights
///
/// These weights have been validated against PDBBind and DUD-E benchmarks
/// to achieve optimal AUC-ROC and AUPRC metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DruggabilityWeights {
    /// Weight for hydrophobicity contribution (aromatic/aliphatic coverage)
    pub hydrophobicity: f32,
    /// Weight for enclosure ratio (pocket burial)
    pub enclosure: f32,
    /// Weight for volume contribution (size appropriateness)
    pub volume: f32,
    /// Weight for H-bond network strength
    pub hbond: f32,
    /// Weight for electrostatic complementarity
    pub electrostatic: f32,
    /// Weight for depth/accessibility balance
    pub depth: f32,
    /// Weight for flexibility (B-factor based)
    pub flexibility: f32,
    /// Weight for conservation score (if available)
    pub conservation: f32,
}

impl Default for DruggabilityWeights {
    fn default() -> Self {
        // Validated weights from benchmark optimization
        Self {
            hydrophobicity: 0.25,
            enclosure: 0.20,
            volume: 0.15,
            hbond: 0.15,
            electrostatic: 0.10,
            depth: 0.08,
            flexibility: 0.04,
            conservation: 0.03,
        }
    }
}

impl DruggabilityWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.hydrophobicity
            + self.enclosure
            + self.volume
            + self.hbond
            + self.electrostatic
            + self.depth
            + self.flexibility
            + self.conservation;

        if sum > 0.0 {
            self.hydrophobicity /= sum;
            self.enclosure /= sum;
            self.volume /= sum;
            self.hbond /= sum;
            self.electrostatic /= sum;
            self.depth /= sum;
            self.flexibility /= sum;
            self.conservation /= sum;
        }
    }

    /// Create weights optimized for cryptic site detection
    pub fn cryptic_optimized() -> Self {
        Self {
            hydrophobicity: 0.20,
            enclosure: 0.15,
            volume: 0.10,
            hbond: 0.10,
            electrostatic: 0.10,
            depth: 0.05,
            flexibility: 0.20, // Higher weight for flexible regions
            conservation: 0.10,
        }
    }

    /// Create weights optimized for allosteric site detection
    pub fn allosteric_optimized() -> Self {
        Self {
            hydrophobicity: 0.20,
            enclosure: 0.25, // Higher enclosure for buried sites
            volume: 0.15,
            hbond: 0.10,
            electrostatic: 0.10,
            depth: 0.10,
            flexibility: 0.05,
            conservation: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = UnifiedHybridConfig::default();
        assert_eq!(config.kempe_iterations, 8);
        assert_eq!(config.power_iterations, 12);
        assert_eq!(config.detection_mode(), DetectionMode::Standard);
    }

    #[test]
    fn test_presets() {
        let turbo = UnifiedHybridConfig::from_preset(QualityPreset::Turbo);
        assert_eq!(turbo.kempe_iterations, 3);
        assert_eq!(turbo.detection_mode(), DetectionMode::Turbo);

        let publication = UnifiedHybridConfig::from_preset(QualityPreset::Publication);
        assert_eq!(publication.kempe_iterations, 12);
        assert_eq!(publication.detection_mode(), DetectionMode::Publication);
    }

    #[test]
    fn test_weight_normalization() {
        let mut weights = DruggabilityWeights {
            hydrophobicity: 1.0,
            enclosure: 1.0,
            volume: 1.0,
            hbond: 1.0,
            electrostatic: 1.0,
            depth: 1.0,
            flexibility: 1.0,
            conservation: 1.0,
        };
        weights.normalize();

        let sum = weights.hydrophobicity
            + weights.enclosure
            + weights.volume
            + weights.hbond
            + weights.electrostatic
            + weights.depth
            + weights.flexibility
            + weights.conservation;

        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_provenance_levels() {
        assert!(!ProvenanceLevel::None.include_timing());
        assert!(ProvenanceLevel::Minimal.include_timing());
        assert!(ProvenanceLevel::Full.include_score_breakdown());
        assert!(ProvenanceLevel::Debug.include_debug());
    }
}
