//! Precision filtering for pocket detection
//!
//! This module provides configurable filtering to reduce false positives
//! while maintaining high recall in binding site prediction.
//!
//! # Design Rationale
//!
//! False positives in pocket detection typically share these characteristics:
//! - Small volume (< 150 Å³) - too small for meaningful drug binding
//! - Low druggability (< 0.35) - surface exposed or hydrophilic
//! - Low burial score (< 0.2) - not sufficiently enclosed
//! - Few residues (< 5) - likely noise or artifacts
//!
//! By filtering based on these properties after pocket detection,
//! we can significantly reduce false positives without losing true binding sites.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::pocket::properties::Pocket;

/// Precision mode for pocket filtering
///
/// Controls the stringency of pocket filtering:
/// - `HighRecall`: Keep more pockets, accept more false positives
/// - `Balanced`: Default mode, good trade-off between precision and recall
/// - `HighPrecision`: Aggressive filtering, may lose some true positives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    /// Keep more pockets, minimize false negatives
    /// FP tolerance: high, FN tolerance: very low
    HighRecall,

    /// Balanced filtering (default)
    /// Targets AUC > 0.6, F1 > 0.6
    #[default]
    Balanced,

    /// Aggressive filtering, prioritize precision
    /// May lose ~10% of true positives
    HighPrecision,
}

impl PrecisionMode {
    /// Parse precision mode from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "high_recall" | "highrecall" | "recall" => Ok(Self::HighRecall),
            "balanced" | "default" => Ok(Self::Balanced),
            "high_precision" | "highprecision" | "precision" => Ok(Self::HighPrecision),
            _ => anyhow::bail!(
                "Invalid precision mode '{}'. Valid options: high_recall, balanced, high_precision",
                s
            ),
        }
    }
}

impl std::fmt::Display for PrecisionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighRecall => write!(f, "high_recall"),
            Self::Balanced => write!(f, "balanced"),
            Self::HighPrecision => write!(f, "high_precision"),
        }
    }
}

/// Configuration for precision-based pocket filtering
///
/// Each threshold can be tuned independently. Pockets must meet ALL
/// criteria to pass the filter (conjunctive filtering).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionFilterConfig {
    /// Minimum pocket volume in Å³ (default: 150-200)
    /// Filters out tiny cavities that can't accommodate ligands
    pub min_volume: f64,

    /// Minimum mean hydrophobicity score (default: -0.3 to -0.1)
    /// More negative = more hydrophobic = better drug binding
    /// Note: Some binding sites are polar, so this is relaxed
    pub min_hydrophobicity: f64,

    /// Minimum burial/enclosure score (default: 0.2-0.3)
    /// Higher = more enclosed = true cavity vs surface groove
    pub min_burial_score: f64,

    /// Minimum druggability score 0-1 (default: 0.35-0.45)
    /// Composite score from physicochemical properties
    pub min_druggability: f64,

    /// Minimum number of lining residues (default: 5-8)
    /// Very small pockets with few residues are likely noise
    pub min_residues: usize,

    /// Maximum number of pockets to return (default: 10-15)
    /// After sorting by druggability, keep only top N
    pub max_pockets: usize,

    /// Minimum number of alpha spheres (default: 3-5)
    /// Pockets with very few alpha spheres are unreliable
    pub min_alpha_spheres: usize,

    /// Whether to enable the filter (for bypass/debug)
    pub enabled: bool,
}

impl Default for PrecisionFilterConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl PrecisionFilterConfig {
    /// Create configuration for high recall mode
    /// Relaxed thresholds to minimize false negatives
    pub fn high_recall() -> Self {
        Self {
            min_volume: 80.0,           // Lower threshold
            min_hydrophobicity: -0.5,   // Very relaxed
            min_burial_score: 0.0,      // Disabled - keep all
            min_druggability: 0.10,     // Very low - catch borderline cryptic sites
            min_residues: 3,            // Smaller pockets OK
            max_pockets: 20,            // Keep more pockets
            min_alpha_spheres: 2,       // Fewer spheres OK
            enabled: true,
        }
    }

    /// Create configuration for balanced mode (default)
    /// Good trade-off between precision and recall
    ///
    /// min_druggability = 0.25 (lowered for cryptic binding site detection)
    /// Cryptic sites often have lower initial druggability scores due to their
    /// transient/hidden nature. Higher scoring/ranking handles final selection.
    /// Tuned for optimal AUC/AUPRC/MCC/F1 metrics on CryptoBench.
    pub fn balanced() -> Self {
        Self {
            min_volume: 100.0,          // Allow smaller cryptic sites
            min_hydrophobicity: -0.5,   // Relaxed - cryptic sites may be polar
            min_burial_score: 0.01,     // Very relaxed for Voronoi-based detection
            min_druggability: 0.25,     // Lowered threshold for cryptic sites
            min_residues: 4,            // Allow smaller pocket residue counts
            max_pockets: 12,            // More pockets for cryptic site detection
            min_alpha_spheres: 3,       // Allow smaller alpha sphere clusters
            enabled: true,
        }
    }

    /// Create configuration for high precision mode
    /// Aggressive filtering to minimize false positives
    pub fn high_precision() -> Self {
        Self {
            min_volume: 300.0,          // Larger minimum for high-confidence cavities
            min_hydrophobicity: -0.3,   // More hydrophobic required
            min_burial_score: 0.05,     // Moderate enclosure (Voronoi values may be low)
            min_druggability: 0.50,     // Higher threshold - only high-confidence pockets
            min_residues: 8,            // More residues required
            max_pockets: 5,             // Only top 5 pockets
            min_alpha_spheres: 5,       // More alpha spheres
            enabled: true,
        }
    }

    /// Create configuration from precision mode
    pub fn from_mode(mode: PrecisionMode) -> Self {
        match mode {
            PrecisionMode::HighRecall => Self::high_recall(),
            PrecisionMode::Balanced => Self::balanced(),
            PrecisionMode::HighPrecision => Self::high_precision(),
        }
    }

    /// Create a disabled filter (pass-through)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::balanced()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        ensure!(self.min_volume >= 0.0, "min_volume must be non-negative");
        ensure!(self.min_volume <= 5000.0, "min_volume too large (max 5000)");
        ensure!(
            self.min_burial_score >= 0.0 && self.min_burial_score <= 1.0,
            "min_burial_score must be in [0, 1]"
        );
        ensure!(
            self.min_druggability >= 0.0 && self.min_druggability <= 1.0,
            "min_druggability must be in [0, 1]"
        );
        ensure!(self.min_residues > 0, "min_residues must be positive");
        ensure!(self.max_pockets > 0, "max_pockets must be positive");
        Ok(())
    }
}

/// Statistics from precision filtering
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Number of pockets before filtering
    pub input_count: usize,
    /// Number of pockets after filtering
    pub output_count: usize,
    /// Pockets removed due to volume threshold
    pub removed_by_volume: usize,
    /// Pockets removed due to druggability threshold
    pub removed_by_druggability: usize,
    /// Pockets removed due to burial score threshold
    pub removed_by_burial: usize,
    /// Pockets removed due to residue count threshold
    pub removed_by_residues: usize,
    /// Pockets removed due to alpha sphere threshold
    pub removed_by_alpha_spheres: usize,
    /// Pockets removed due to max_pockets limit
    pub removed_by_limit: usize,
}

impl FilterStats {
    /// Total number of pockets removed
    pub fn total_removed(&self) -> usize {
        self.input_count.saturating_sub(self.output_count)
    }

    /// Retention rate as a percentage
    pub fn retention_rate(&self) -> f64 {
        if self.input_count == 0 {
            100.0
        } else {
            (self.output_count as f64 / self.input_count as f64) * 100.0
        }
    }
}

/// Filter pockets based on precision configuration
///
/// # Arguments
/// * `pockets` - Input pockets from detection
/// * `config` - Filtering configuration
///
/// # Returns
/// * Filtered pockets sorted by druggability (descending)
/// * Filter statistics for diagnostics
pub fn filter_pockets_for_precision(
    pockets: Vec<Pocket>,
    config: &PrecisionFilterConfig,
) -> (Vec<Pocket>, FilterStats) {
    let mut stats = FilterStats {
        input_count: pockets.len(),
        ..Default::default()
    };

    // If filtering is disabled, return all pockets
    if !config.enabled {
        stats.output_count = pockets.len();
        return (pockets, stats);
    }

    log::debug!(
        "Precision filter: input={} pockets, mode={}",
        pockets.len(),
        if config.min_druggability >= 0.40 {
            "high_precision"
        } else if config.min_druggability <= 0.15 {
            "high_recall"
        } else {
            "balanced"
        }
    );

    // Apply individual filters and track removal reasons
    let filtered: Vec<Pocket> = pockets
        .into_iter()
        .filter(|p| {
            // Volume check
            if p.volume < config.min_volume {
                stats.removed_by_volume += 1;
                log::trace!(
                    "Pocket filtered: volume {:.1} < {:.1}",
                    p.volume,
                    config.min_volume
                );
                return false;
            }

            // Druggability check
            if p.druggability_score.total < config.min_druggability {
                stats.removed_by_druggability += 1;
                log::trace!(
                    "Pocket filtered: druggability {:.3} < {:.3}",
                    p.druggability_score.total,
                    config.min_druggability
                );
                return false;
            }

            // Burial/enclosure check
            if p.enclosure_ratio < config.min_burial_score {
                stats.removed_by_burial += 1;
                log::trace!(
                    "Pocket filtered: burial {:.3} < {:.3}",
                    p.enclosure_ratio,
                    config.min_burial_score
                );
                return false;
            }

            // Residue count check
            let residue_count = p.residue_indices.len();
            if residue_count < config.min_residues {
                stats.removed_by_residues += 1;
                log::trace!(
                    "Pocket filtered: residues {} < {}",
                    residue_count,
                    config.min_residues
                );
                return false;
            }

            // Alpha sphere check (use atom_indices as proxy if alpha_spheres not available)
            let sphere_count = p.atom_indices.len();
            if sphere_count < config.min_alpha_spheres {
                stats.removed_by_alpha_spheres += 1;
                log::trace!(
                    "Pocket filtered: alpha_spheres {} < {}",
                    sphere_count,
                    config.min_alpha_spheres
                );
                return false;
            }

            true
        })
        .collect();

    // Sort by druggability (descending) for max_pockets selection
    let mut sorted = filtered;
    sorted.sort_by(|a, b| {
        b.druggability_score
            .total
            .partial_cmp(&a.druggability_score.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply max_pockets limit
    let final_count = sorted.len().min(config.max_pockets);
    if sorted.len() > config.max_pockets {
        stats.removed_by_limit = sorted.len() - config.max_pockets;
        log::debug!(
            "Precision filter: limiting to top {} of {} pockets",
            config.max_pockets,
            sorted.len()
        );
    }
    sorted.truncate(final_count);

    stats.output_count = sorted.len();

    log::info!(
        "Precision filter: {} -> {} pockets ({:.1}% retained)",
        stats.input_count,
        stats.output_count,
        stats.retention_rate()
    );

    (sorted, stats)
}

/// Filter pockets using a specific precision mode
pub fn filter_by_mode(pockets: Vec<Pocket>, mode: PrecisionMode) -> (Vec<Pocket>, FilterStats) {
    let config = PrecisionFilterConfig::from_mode(mode);
    filter_pockets_for_precision(pockets, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_pocket(_id: u32, volume: f64, druggability: f64, enclosure: f64) -> Pocket {
        use crate::scoring::{DrugabilityClass, DruggabilityScore, Components};
        Pocket {
            volume,
            druggability_score: DruggabilityScore {
                total: druggability,
                classification: DrugabilityClass::Druggable,
                components: Components::default(),
            },
            enclosure_ratio: enclosure,
            mean_hydrophobicity: -0.2,
            residue_indices: (0..10).collect(),
            atom_indices: (0..20).collect(),
            centroid: [0.0, 0.0, 0.0],
            ..Default::default()
        }
    }

    #[test]
    fn test_precision_mode_parsing() {
        assert_eq!(
            PrecisionMode::from_str("balanced").unwrap(),
            PrecisionMode::Balanced
        );
        assert_eq!(
            PrecisionMode::from_str("high_recall").unwrap(),
            PrecisionMode::HighRecall
        );
        assert_eq!(
            PrecisionMode::from_str("HIGH_PRECISION").unwrap(),
            PrecisionMode::HighPrecision
        );
        assert!(PrecisionMode::from_str("invalid").is_err());
    }

    #[test]
    fn test_config_validation() {
        let config = PrecisionFilterConfig::balanced();
        assert!(config.validate().is_ok());

        let invalid = PrecisionFilterConfig {
            min_druggability: 1.5, // Invalid
            ..config.clone()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_filter_volume() {
        let pockets = vec![
            make_test_pocket(1, 50.0, 0.8, 0.5),   // Too small
            make_test_pocket(2, 200.0, 0.8, 0.5),  // OK
            make_test_pocket(3, 500.0, 0.8, 0.5),  // OK
        ];

        let config = PrecisionFilterConfig::balanced();
        let (filtered, stats) = filter_pockets_for_precision(pockets, &config);

        assert_eq!(filtered.len(), 2);
        assert_eq!(stats.removed_by_volume, 1);
    }

    #[test]
    fn test_filter_druggability() {
        let pockets = vec![
            make_test_pocket(1, 300.0, 0.1, 0.5),  // Low druggability (below 0.25)
            make_test_pocket(2, 300.0, 0.35, 0.5), // OK (above 0.25)
            make_test_pocket(3, 300.0, 0.9, 0.5),  // High druggability
        ];

        let config = PrecisionFilterConfig::balanced();
        let (filtered, stats) = filter_pockets_for_precision(pockets, &config);

        assert_eq!(filtered.len(), 2);
        assert_eq!(stats.removed_by_druggability, 1);
    }

    #[test]
    fn test_filter_disabled() {
        let pockets = vec![
            make_test_pocket(1, 10.0, 0.01, 0.01),  // Would fail all checks
        ];

        let config = PrecisionFilterConfig::disabled();
        let (filtered, _stats) = filter_pockets_for_precision(pockets, &config);

        assert_eq!(filtered.len(), 1); // Passes because disabled
    }

    #[test]
    fn test_max_pockets_limit() {
        let pockets: Vec<Pocket> = (0..20)
            .map(|i| make_test_pocket(i, 400.0, 0.35 + (i as f64) * 0.01, 0.5))
            .collect();

        let config = PrecisionFilterConfig::balanced(); // max_pockets = 12
        let (filtered, stats) = filter_pockets_for_precision(pockets, &config);

        assert_eq!(filtered.len(), 12);
        assert_eq!(stats.removed_by_limit, 8);
        // Verify sorted by druggability (highest first)
        assert!(filtered[0].druggability_score.total > filtered[11].druggability_score.total);
    }

    #[test]
    fn test_mode_presets() {
        let high_recall = PrecisionFilterConfig::high_recall();
        let balanced = PrecisionFilterConfig::balanced();
        let high_precision = PrecisionFilterConfig::high_precision();

        // High recall should have lowest thresholds
        assert!(high_recall.min_volume < balanced.min_volume);
        assert!(high_recall.min_druggability < balanced.min_druggability);

        // High precision should have highest thresholds
        assert!(high_precision.min_volume > balanced.min_volume);
        assert!(high_precision.min_druggability > balanced.min_druggability);
    }
}
