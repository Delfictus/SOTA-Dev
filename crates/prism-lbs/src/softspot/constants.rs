//! Tunable constants for soft-spot detection
//!
//! These parameters control the sensitivity and specificity of cryptic
//! binding site detection. Values are based on literature (CryptoSite, FTMap)
//! and empirical tuning on benchmark datasets.

//=============================================================================
// FLEXIBILITY ANALYSIS (B-factor based)
//=============================================================================

/// B-factor z-score threshold to consider a residue "flexible"
/// Residues with z-score > this value are flagged as flexible
/// Literature: CryptoSite uses ~0.8 standard deviations
pub const BFACTOR_ZSCORE_THRESHOLD: f64 = 0.8;

/// Minimum B-factor z-score to consider at all
/// Residues below this are excluded from analysis
pub const BFACTOR_ZSCORE_MINIMUM: f64 = 0.3;

/// Weight of flexibility component in final cryptic score
pub const FLEXIBILITY_WEIGHT: f64 = 0.35;

//=============================================================================
// PACKING ANALYSIS (Local density)
//=============================================================================

/// Radius (Angstroms) for local packing density calculation
/// 8 Angstroms captures the first and second shell neighbors
pub const PACKING_RADIUS: f64 = 8.0;

/// Expected packing density for well-packed protein interior
/// Expressed as atoms per cubic Angstrom
/// Derived from average protein density ~1.35 g/cm^3
pub const EXPECTED_PACKING_DENSITY: f64 = 0.008;

/// Packing density ratio below this threshold indicates "loose" packing
/// Value of 0.7 means 30% below expected density
pub const PACKING_DEFICIT_THRESHOLD: f64 = 0.7;

/// Weight of packing deficit component in final cryptic score
pub const PACKING_WEIGHT: f64 = 0.35;

//=============================================================================
// HYDROPHOBICITY ANALYSIS
//=============================================================================

/// Minimum average hydrophobicity for a drug-like pocket
/// Based on observation that druggable pockets are typically hydrophobic
/// Value of 0.4 on normalized Kyte-Doolittle scale
pub const HYDROPHOBICITY_THRESHOLD: f64 = 0.4;

/// Weight of hydrophobicity component in final cryptic score
pub const HYDROPHOBICITY_WEIGHT: f64 = 0.20;

//=============================================================================
// CLUSTERING
//=============================================================================

/// Maximum distance (Angstroms) between flexible residue centroids to cluster
/// 8 Angstroms allows residues within ~2 amino acids in sequence
pub const CLUSTER_DISTANCE: f64 = 8.0;

/// Minimum residues to form a viable cryptic site candidate
/// 4 residues is roughly minimum for small-molecule binding
pub const MIN_CLUSTER_SIZE: usize = 4;

/// Maximum residues in a single cluster
/// Prevents whole-protein clusters in highly flexible structures
pub const MAX_CLUSTER_SIZE: usize = 30;

//=============================================================================
// SCORING
//=============================================================================

/// Minimum combined score to report as a cryptic site candidate
/// Lower values increase sensitivity but may increase false positives
pub const MIN_CRYPTIC_SCORE: f64 = 0.35;

/// Weight of spatial coherence in final score
/// Coherent clusters (compact shape) score higher
pub const COHERENCE_WEIGHT: f64 = 0.10;

//=============================================================================
// VOLUME ESTIMATION
//=============================================================================

/// Average volume per residue in an open pocket (Angstroms cubed)
/// Used to estimate potential pocket volume from residue count
/// Derived from analysis of known binding sites
pub const VOLUME_PER_RESIDUE: f64 = 45.0;

//=============================================================================
// DRUGGABILITY PREDICTION
//=============================================================================

/// Optimal pocket volume for druggability (Angstroms cubed)
/// Pockets near this size receive higher druggability scores
pub const OPTIMAL_POCKET_VOLUME: f64 = 600.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_sum() {
        // Weights should sum to approximately 1.0
        let total = FLEXIBILITY_WEIGHT + PACKING_WEIGHT + HYDROPHOBICITY_WEIGHT + COHERENCE_WEIGHT;
        assert!((total - 1.0).abs() < 0.01, "Weights sum to {}, expected ~1.0", total);
    }

    #[test]
    fn test_thresholds_reasonable() {
        assert!(BFACTOR_ZSCORE_THRESHOLD > 0.0);
        assert!(BFACTOR_ZSCORE_MINIMUM < BFACTOR_ZSCORE_THRESHOLD);
        assert!(PACKING_DEFICIT_THRESHOLD > 0.0 && PACKING_DEFICIT_THRESHOLD < 1.0);
        assert!(MIN_CLUSTER_SIZE < MAX_CLUSTER_SIZE);
    }
}
