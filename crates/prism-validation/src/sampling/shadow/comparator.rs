//! Shadow Comparator
//!
//! Compares outputs from NOVA and AMBER paths to validate equivalence.

use serde::{Deserialize, Serialize};

use crate::sampling::result::SamplingResult;

/// Result of shadow comparison
#[derive(Debug)]
pub enum ShadowResult {
    /// Both paths ran, results compared
    Compared {
        /// Primary result (NOVA)
        primary: SamplingResult,
        /// Shadow result (AMBER)
        shadow: SamplingResult,
        /// Divergence metrics
        divergence: DivergenceMetrics,
    },
    /// Structure too large for NOVA, only AMBER ran
    SkippedLargeStructure(SamplingResult),
}

impl ShadowResult {
    /// Get the primary result
    pub fn primary(&self) -> &SamplingResult {
        match self {
            ShadowResult::Compared { primary, .. } => primary,
            ShadowResult::SkippedLargeStructure(result) => result,
        }
    }

    /// Get divergence if comparison was performed
    pub fn divergence(&self) -> Option<&DivergenceMetrics> {
        match self {
            ShadowResult::Compared { divergence, .. } => Some(divergence),
            ShadowResult::SkippedLargeStructure(_) => None,
        }
    }

    /// Check if comparison was performed
    pub fn was_compared(&self) -> bool {
        matches!(self, ShadowResult::Compared { .. })
    }
}

/// Divergence metrics between two sampling runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceMetrics {
    /// Mean RMSD between conformations (Angstroms)
    pub mean_rmsd: f32,
    /// Maximum RMSD between any pair of conformations
    pub max_rmsd: f32,
    /// Pearson correlation of energies
    pub energy_correlation: f32,
}

impl DivergenceMetrics {
    /// Get verdict on divergence level
    pub fn verdict(&self) -> DivergenceVerdict {
        if self.mean_rmsd < 0.5 && self.energy_correlation > 0.95 {
            DivergenceVerdict::Equivalent
        } else if self.mean_rmsd < 1.5 && self.energy_correlation > 0.8 {
            DivergenceVerdict::MinorDivergence
        } else if self.mean_rmsd < 3.0 && self.energy_correlation > 0.5 {
            DivergenceVerdict::ModerateDivergence
        } else {
            DivergenceVerdict::MajorDivergence
        }
    }
}

/// Verdict on divergence between paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceVerdict {
    /// Results are equivalent (RMSD < 0.5A, correlation > 0.95)
    Equivalent,
    /// Minor differences (RMSD < 1.5A, correlation > 0.8)
    MinorDivergence,
    /// Moderate differences (RMSD < 3.0A, correlation > 0.5)
    ModerateDivergence,
    /// Major differences (investigation required)
    MajorDivergence,
}

impl std::fmt::Display for DivergenceVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceVerdict::Equivalent => write!(f, "Equivalent"),
            DivergenceVerdict::MinorDivergence => write!(f, "Minor Divergence"),
            DivergenceVerdict::ModerateDivergence => write!(f, "Moderate Divergence"),
            DivergenceVerdict::MajorDivergence => write!(f, "Major Divergence"),
        }
    }
}

/// Shadow comparator for running both paths
pub struct ShadowComparator;

impl ShadowComparator {
    /// Create new comparator
    pub fn new() -> Self {
        Self
    }
}

impl Default for ShadowComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_verdict_equivalent() {
        let metrics = DivergenceMetrics {
            mean_rmsd: 0.3,
            max_rmsd: 0.5,
            energy_correlation: 0.98,
        };
        assert_eq!(metrics.verdict(), DivergenceVerdict::Equivalent);
    }

    #[test]
    fn test_divergence_verdict_minor() {
        let metrics = DivergenceMetrics {
            mean_rmsd: 1.0,
            max_rmsd: 1.5,
            energy_correlation: 0.85,
        };
        assert_eq!(metrics.verdict(), DivergenceVerdict::MinorDivergence);
    }

    #[test]
    fn test_divergence_verdict_major() {
        let metrics = DivergenceMetrics {
            mean_rmsd: 5.0,
            max_rmsd: 10.0,
            energy_correlation: 0.2,
        };
        assert_eq!(metrics.verdict(), DivergenceVerdict::MajorDivergence);
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", DivergenceVerdict::Equivalent), "Equivalent");
    }
}
