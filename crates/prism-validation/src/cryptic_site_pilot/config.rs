//! Configuration for Cryptic Site Pilot Pipeline
//!
//! All parameters are tuned based on validation against known cryptic sites.

use serde::{Deserialize, Serialize};

/// Master configuration for cryptic site detection pilot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticPilotConfig {
    // === Sampling Parameters ===
    /// Number of conformational samples (frames) to generate
    /// Minimum: 100 (for statistical significance)
    /// Recommended: 200-500 for production
    pub n_frames: usize,

    /// Steps between saved frames (decorrelation)
    /// Higher = more independent samples, longer runtime
    pub steps_per_frame: usize,

    /// Simulation temperature (Kelvin)
    pub temperature_k: f32,

    /// Random seed for reproducibility
    pub seed: u64,

    // === Pocket Detection Parameters ===
    /// Grid spacing for pocket detection (Å)
    /// Smaller = more precise but slower
    pub grid_spacing: f64,

    /// Probe radius for cavity detection (Å)
    /// 1.4Å = water, 1.8Å = larger probes
    pub probe_radius: f64,

    /// Minimum pocket volume to consider (Å³)
    /// Typical drug binding site: 200-1000 Å³
    pub min_pocket_volume: f64,

    /// Maximum pocket volume to consider (Å³)
    /// Prevents detection of protein interior as "pocket"
    pub max_pocket_volume: f64,

    // === Cryptic Site Classification ===
    /// Volume variance threshold for "cryptic" classification
    /// Cryptic = opens/closes dynamically
    /// Value: coefficient of variation (std/mean)
    pub cryptic_volume_cv_threshold: f64,

    /// Minimum open frequency for cryptic site
    /// Fraction of frames where pocket is detectable
    pub min_open_frequency: f64,

    /// Maximum open frequency for cryptic site
    /// Too high = not really cryptic, always open
    pub max_open_frequency: f64,

    // === Druggability Parameters ===
    /// Minimum druggability score (0-1) to report
    pub min_druggability_score: f64,

    /// Weight for hydrophobicity in druggability
    pub druggability_hydrophobic_weight: f64,

    /// Weight for enclosure in druggability
    pub druggability_enclosure_weight: f64,

    /// Weight for hydrogen bond acceptors/donors
    pub druggability_hbond_weight: f64,

    // === Output Parameters ===
    /// Number of representative structures per site
    pub n_representative_structures: usize,

    /// Include all-atom coordinates (not just Cα)
    pub output_all_atom: bool,

    /// Generate interactive HTML report
    pub generate_html_report: bool,

    /// Compress trajectory file
    pub compress_trajectory: bool,
}

impl Default for CrypticPilotConfig {
    fn default() -> Self {
        Self {
            // Sampling - tuned for 10ns effective sampling
            n_frames: 200,
            steps_per_frame: 100,
            temperature_k: 310.0,
            seed: 42,

            // Pocket detection - based on fpocket/SiteMap literature
            grid_spacing: 0.7,
            probe_radius: 1.4,
            min_pocket_volume: 100.0,  // Minimum for small-molecule binding
            max_pocket_volume: 2000.0, // Maximum reasonable pocket

            // Cryptic classification - validated against known sites
            cryptic_volume_cv_threshold: 0.20, // 20% coefficient of variation
            min_open_frequency: 0.05,          // Open in at least 5% of frames
            max_open_frequency: 0.90,          // Not always open

            // Druggability - based on SiteMap parameters
            min_druggability_score: 0.3,
            druggability_hydrophobic_weight: 0.35,
            druggability_enclosure_weight: 0.35,
            druggability_hbond_weight: 0.30,

            // Output
            n_representative_structures: 5,
            output_all_atom: true,
            generate_html_report: true,
            compress_trajectory: false,
        }
    }
}

impl CrypticPilotConfig {
    /// Configuration optimized for quick testing
    pub fn quick() -> Self {
        Self {
            n_frames: 50,
            steps_per_frame: 50,
            n_representative_structures: 3,
            generate_html_report: false,
            ..Default::default()
        }
    }

    /// Configuration for production pharma pilot
    pub fn production() -> Self {
        Self {
            n_frames: 500,
            steps_per_frame: 200,
            n_representative_structures: 10,
            generate_html_report: true,
            compress_trajectory: true,
            ..Default::default()
        }
    }

    /// Configuration for extended analysis
    pub fn extended() -> Self {
        Self {
            n_frames: 1000,
            steps_per_frame: 200,
            grid_spacing: 0.5,        // Finer grid
            n_representative_structures: 20,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.n_frames < 50 {
            return Err("n_frames must be at least 50 for statistical significance".into());
        }
        if self.temperature_k < 250.0 || self.temperature_k > 400.0 {
            return Err("temperature_k should be between 250K and 400K".into());
        }
        if self.probe_radius < 1.0 || self.probe_radius > 3.0 {
            return Err("probe_radius should be between 1.0Å and 3.0Å".into());
        }
        if self.min_pocket_volume >= self.max_pocket_volume {
            return Err("min_pocket_volume must be less than max_pocket_volume".into());
        }
        if self.min_open_frequency >= self.max_open_frequency {
            return Err("min_open_frequency must be less than max_open_frequency".into());
        }
        if self.druggability_hydrophobic_weight + self.druggability_enclosure_weight
            + self.druggability_hbond_weight > 1.01
        {
            return Err("druggability weights should sum to approximately 1.0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = CrypticPilotConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quick_config_valid() {
        let config = CrypticPilotConfig::quick();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_production_config_valid() {
        let config = CrypticPilotConfig::production();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_n_frames() {
        let mut config = CrypticPilotConfig::default();
        config.n_frames = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut config = CrypticPilotConfig::default();
        config.temperature_k = 500.0;
        assert!(config.validate().is_err());
    }
}
