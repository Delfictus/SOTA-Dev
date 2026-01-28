//! Data structures for soft-spot (cryptic site) detection
//!
//! This module defines types for representing flexible residues and
//! cryptic binding site candidates detected through B-factor, packing,
//! and hydrophobicity analysis.

use serde::{Deserialize, Serialize};

/// A residue flagged as potentially part of a cryptic site
#[derive(Debug, Clone)]
pub struct FlexibleResidue {
    /// Chain identifier
    pub chain_id: char,

    /// Residue sequence number (PDB RESSEQ)
    pub residue_seq: i32,

    /// Residue name (3-letter code, e.g., "ALA", "GLY")
    pub residue_name: String,

    /// Normalized B-factor (z-score relative to structure mean)
    /// Higher values indicate greater flexibility
    pub bfactor_zscore: f64,

    /// Local packing density ratio (actual / expected)
    /// Values < 1.0 indicate loose packing
    pub packing_density: f64,

    /// Average hydrophobicity of residue (0.0-1.0 normalized Kyte-Doolittle)
    pub hydrophobicity: f64,

    /// Centroid of residue atoms [x, y, z] in Angstroms
    pub centroid: [f64; 3],
}

/// A candidate cryptic binding site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticCandidate {
    /// Candidate ID (1-indexed)
    pub id: usize,

    /// Residue sequence numbers forming this candidate (PDB RESSEQ)
    pub residue_indices: Vec<i32>,

    /// Centroid of the candidate site [x, y, z] in Angstroms
    pub centroid: [f64; 3],

    /// Estimated volume if pocket were to open (Angstroms cubed)
    pub estimated_volume: f64,

    /// Flexibility score [0, 1] - higher = more flexible
    pub flexibility_score: f64,

    /// Packing deficit [0, 1] - higher = more room to open
    pub packing_deficit: f64,

    /// Hydrophobic propensity [0, 1] - higher = more hydrophobic
    pub hydrophobic_score: f64,

    /// Combined cryptic site score [0, 1]
    pub cryptic_score: f64,

    /// Predicted druggability if site opens [0, 1]
    pub predicted_druggability: f64,

    /// Confidence level based on signal strength
    pub confidence: CrypticConfidence,

    /// Human-readable rationale explaining the detection
    pub rationale: String,
}

/// Confidence level for cryptic site detection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CrypticConfidence {
    /// Strong signals from multiple indicators (score >= 0.7)
    High,
    /// Moderate signals (score >= 0.5)
    Medium,
    /// Weak signals (score < 0.5)
    Low,
}

impl CrypticConfidence {
    /// Derive confidence level from a cryptic score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.7 {
            Self::High
        } else if score >= 0.5 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

impl Default for CrypticCandidate {
    fn default() -> Self {
        Self {
            id: 0,
            residue_indices: Vec::new(),
            centroid: [0.0, 0.0, 0.0],
            estimated_volume: 0.0,
            flexibility_score: 0.0,
            packing_deficit: 0.0,
            hydrophobic_score: 0.0,
            cryptic_score: 0.0,
            predicted_druggability: 0.0,
            confidence: CrypticConfidence::Low,
            rationale: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_from_score() {
        assert_eq!(CrypticConfidence::from_score(0.8), CrypticConfidence::High);
        assert_eq!(CrypticConfidence::from_score(0.7), CrypticConfidence::High);
        assert_eq!(CrypticConfidence::from_score(0.6), CrypticConfidence::Medium);
        assert_eq!(CrypticConfidence::from_score(0.5), CrypticConfidence::Medium);
        assert_eq!(CrypticConfidence::from_score(0.4), CrypticConfidence::Low);
        assert_eq!(CrypticConfidence::from_score(0.0), CrypticConfidence::Low);
    }

    #[test]
    fn test_cryptic_candidate_default() {
        let candidate = CrypticCandidate::default();
        assert_eq!(candidate.id, 0);
        assert!(candidate.residue_indices.is_empty());
        assert_eq!(candidate.confidence, CrypticConfidence::Low);
    }
}
