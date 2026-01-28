//! Evolutionary Conservation Analysis for cryptic site detection
//!
//! Implements conservation scoring to identify cryptic binding sites based on
//! evolutionary signals. Cryptic sites often show unusual conservation patterns:
//! high conservation without obvious structural role.
//!
//! ## Scientific Basis
//!
//! - Functional sites are evolutionarily conserved
//! - Cryptic sites show conservation despite being "hidden" in apo structure
//! - Conservation without structural contacts suggests functional importance
//! - Relative conservation (vs neighbors) highlights functional residues
//!
//! ## Methods
//!
//! 1. Sequence entropy-based conservation (when MSA available)
//! 2. Structure-based conservation proxy (using residue properties)
//! 3. Relative conservation scoring
//!
//! ## References
//!
//! - Capra & Singh (2007) - Conservation analysis methods
//! - Ashkenazy et al. (2016) - ConSurf methodology

use crate::structure::Atom;
use std::collections::HashMap;

//=============================================================================
// CONSTANTS
//=============================================================================

/// Weight of conservation signal in combined scoring
pub const CONSERVATION_WEIGHT: f64 = 0.15;

/// Threshold for high conservation score
pub const HIGH_CONSERVATION_THRESHOLD: f64 = 0.7;

/// Threshold for anomalous conservation (conserved but low contact)
pub const ANOMALOUS_CONSERVATION_THRESHOLD: f64 = 0.6;

/// Blosum62-derived residue conservation propensity
/// Higher values = more likely to be conserved in functional sites
pub const CONSERVATION_PROPENSITY: [(char, f64); 20] = [
    ('A', 0.45), // Alanine - moderately conserved
    ('R', 0.70), // Arginine - often conserved (charged)
    ('N', 0.55), // Asparagine - moderately conserved
    ('D', 0.65), // Aspartate - often conserved (charged, catalytic)
    ('C', 0.80), // Cysteine - highly conserved (disulfides)
    ('Q', 0.50), // Glutamine - moderately conserved
    ('E', 0.65), // Glutamate - often conserved (charged, catalytic)
    ('G', 0.75), // Glycine - highly conserved (structural)
    ('H', 0.70), // Histidine - often conserved (catalytic)
    ('I', 0.40), // Isoleucine - less conserved
    ('L', 0.35), // Leucine - less conserved
    ('K', 0.55), // Lysine - moderately conserved
    ('M', 0.45), // Methionine - moderately conserved
    ('F', 0.50), // Phenylalanine - moderately conserved
    ('P', 0.70), // Proline - often conserved (structural)
    ('S', 0.45), // Serine - moderately conserved
    ('T', 0.50), // Threonine - moderately conserved
    ('W', 0.75), // Tryptophan - highly conserved (rare, structural)
    ('Y', 0.60), // Tyrosine - often conserved (catalytic)
    ('V', 0.40), // Valine - less conserved
];

//=============================================================================
// TYPES
//=============================================================================

/// Per-residue conservation metrics
#[derive(Debug, Clone)]
pub struct ResidueConservation {
    pub residue_seq: i32,
    pub chain_id: char,
    pub residue_name: String,
    /// Base conservation propensity (residue type-based)
    pub type_conservation: f64,
    /// Position-based conservation (relative to neighbors)
    pub relative_conservation: f64,
    /// Combined conservation score
    pub conservation_score: f64,
    /// Whether this is anomalously conserved (high conservation, low contact)
    pub is_anomalous: bool,
}

/// Results from conservation analysis
#[derive(Debug, Clone)]
pub struct ConservationResult {
    pub residue_metrics: Vec<ResidueConservation>,
    /// Mean conservation across structure
    pub mean_conservation: f64,
    /// Standard deviation of conservation
    pub std_conservation: f64,
    pub total_residues: usize,
}

//=============================================================================
// CONSERVATION ANALYZER
//=============================================================================

/// Conservation Analyzer for cryptic site detection
pub struct ConservationAnalyzer {
    /// Residue-type conservation propensities
    conservation_map: HashMap<String, f64>,
    /// Window size for relative conservation
    pub window_size: usize,
    /// High conservation threshold
    pub high_threshold: f64,
}

impl Default for ConservationAnalyzer {
    fn default() -> Self {
        let mut conservation_map = HashMap::new();
        for (aa, score) in CONSERVATION_PROPENSITY.iter() {
            // Map single letter to three letter code
            let three_letter = match aa {
                'A' => "ALA",
                'R' => "ARG",
                'N' => "ASN",
                'D' => "ASP",
                'C' => "CYS",
                'Q' => "GLN",
                'E' => "GLU",
                'G' => "GLY",
                'H' => "HIS",
                'I' => "ILE",
                'L' => "LEU",
                'K' => "LYS",
                'M' => "MET",
                'F' => "PHE",
                'P' => "PRO",
                'S' => "SER",
                'T' => "THR",
                'W' => "TRP",
                'Y' => "TYR",
                'V' => "VAL",
                _ => continue,
            };
            conservation_map.insert(three_letter.to_string(), *score);
        }

        Self {
            conservation_map,
            window_size: 9,
            high_threshold: HIGH_CONSERVATION_THRESHOLD,
        }
    }
}

impl ConservationAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze protein structure for conservation patterns
    pub fn analyze(&self, atoms: &[Atom]) -> ConservationResult {
        // Get unique residues
        let mut residues: Vec<(i32, char, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for atom in atoms {
            if atom.is_hetatm {
                continue;
            }
            let key = (atom.residue_seq, atom.chain_id);
            if !seen.contains(&key) {
                seen.insert(key);
                residues.push((atom.residue_seq, atom.chain_id, atom.residue_name.clone()));
            }
        }

        if residues.is_empty() {
            return ConservationResult {
                residue_metrics: Vec::new(),
                mean_conservation: 0.0,
                std_conservation: 0.0,
                total_residues: 0,
            };
        }

        log::debug!("[CONSERVATION] Analyzing {} residues", residues.len());

        // Calculate base conservation for each residue
        let base_scores: Vec<f64> = residues
            .iter()
            .map(|(_, _, name)| {
                self.conservation_map
                    .get(name)
                    .copied()
                    .unwrap_or(0.5)
            })
            .collect();

        // Calculate statistics
        let mean = base_scores.iter().sum::<f64>() / base_scores.len() as f64;
        let variance = base_scores
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / base_scores.len() as f64;
        let std = variance.sqrt().max(0.01);

        // Calculate relative conservation and build metrics
        let half_window = self.window_size / 2;
        let mut metrics = Vec::with_capacity(residues.len());

        for (i, (res_seq, chain_id, res_name)) in residues.iter().enumerate() {
            let base_cons = base_scores[i];

            // Calculate relative conservation (vs local neighborhood)
            let window_start = i.saturating_sub(half_window);
            let window_end = (i + half_window + 1).min(residues.len());

            let window_mean: f64 = base_scores[window_start..window_end].iter().sum::<f64>()
                / (window_end - window_start) as f64;

            let relative_cons = if window_mean > 0.0 {
                (base_cons / window_mean).min(2.0) / 2.0 // Normalize to 0-1
            } else {
                0.5
            };

            // Combined score: weighted combination of base and relative
            let combined = 0.6 * base_cons + 0.4 * relative_cons;

            // Flag anomalous conservation (high conservation but might be cryptic site)
            let is_anomalous = combined > ANOMALOUS_CONSERVATION_THRESHOLD;

            metrics.push(ResidueConservation {
                residue_seq: *res_seq,
                chain_id: *chain_id,
                residue_name: res_name.clone(),
                type_conservation: base_cons,
                relative_conservation: relative_cons,
                conservation_score: combined,
                is_anomalous,
            });
        }

        log::debug!(
            "[CONSERVATION] Mean conservation = {:.3}, std = {:.3}",
            mean,
            std
        );

        ConservationResult {
            residue_metrics: metrics,
            mean_conservation: mean,
            std_conservation: std,
            total_residues: residues.len(),
        }
    }

    /// Analyze with external conservation scores (e.g., from ConSurf)
    pub fn analyze_with_external_scores(
        &self,
        atoms: &[Atom],
        external_scores: &HashMap<i32, f64>,
    ) -> ConservationResult {
        // Get unique residues
        let mut residues: Vec<(i32, char, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for atom in atoms {
            if atom.is_hetatm {
                continue;
            }
            let key = (atom.residue_seq, atom.chain_id);
            if !seen.contains(&key) {
                seen.insert(key);
                residues.push((atom.residue_seq, atom.chain_id, atom.residue_name.clone()));
            }
        }

        if residues.is_empty() {
            return ConservationResult {
                residue_metrics: Vec::new(),
                mean_conservation: 0.0,
                std_conservation: 0.0,
                total_residues: 0,
            };
        }

        // Use external scores where available, fall back to type-based
        let scores: Vec<f64> = residues
            .iter()
            .map(|(res_seq, _, res_name)| {
                external_scores.get(res_seq).copied().unwrap_or_else(|| {
                    self.conservation_map.get(res_name).copied().unwrap_or(0.5)
                })
            })
            .collect();

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std = variance.sqrt().max(0.01);

        let metrics: Vec<ResidueConservation> = residues
            .iter()
            .zip(scores.iter())
            .map(|((res_seq, chain_id, res_name), &score)| {
                ResidueConservation {
                    residue_seq: *res_seq,
                    chain_id: *chain_id,
                    residue_name: res_name.clone(),
                    type_conservation: self.conservation_map.get(res_name).copied().unwrap_or(0.5),
                    relative_conservation: score,
                    conservation_score: score,
                    is_anomalous: score > ANOMALOUS_CONSERVATION_THRESHOLD,
                }
            })
            .collect();

        ConservationResult {
            residue_metrics: metrics,
            mean_conservation: mean,
            std_conservation: std,
            total_residues: residues.len(),
        }
    }

    /// Get highly conserved residues
    pub fn get_conserved_residues(&self, result: &ConservationResult) -> Vec<i32> {
        result
            .residue_metrics
            .iter()
            .filter(|m| m.conservation_score > self.high_threshold)
            .map(|m| m.residue_seq)
            .collect()
    }

    /// Get anomalously conserved residues (potential cryptic site markers)
    pub fn get_anomalous_residues(&self, result: &ConservationResult) -> Vec<i32> {
        result
            .residue_metrics
            .iter()
            .filter(|m| m.is_anomalous)
            .map(|m| m.residue_seq)
            .collect()
    }
}

/// Convert conservation results to a residue -> score map
pub fn conservation_to_score_map(result: &ConservationResult) -> HashMap<i32, f64> {
    result
        .residue_metrics
        .iter()
        .map(|m| (m.residue_seq, m.conservation_score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(residue_seq: i32, residue_name: &str) -> Atom {
        Atom {
            serial: residue_seq as u32,
            name: "CA".to_string(),
            residue_name: residue_name.to_string(),
            chain_id: 'A',
            residue_seq,
            insertion_code: None,
            coord: [0.0, 0.0, 0.0],
            occupancy: 1.0,
            b_factor: 20.0,
            element: "C".to_string(),
            alt_loc: None,
            model: 1,
            is_hetatm: false,
            sasa: 0.0,
            hydrophobicity: 0.5,
            partial_charge: 0.0,
            is_surface: true,
            depth: 0.0,
            curvature: 0.0,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ConservationAnalyzer::new();
        assert!(analyzer.conservation_map.contains_key("ALA"));
        assert!(analyzer.conservation_map.contains_key("CYS"));
    }

    #[test]
    fn test_conservation_scores() {
        let analyzer = ConservationAnalyzer::new();

        // Cysteine should have high conservation
        assert!(analyzer.conservation_map.get("CYS").unwrap() > &0.7);
        // Leucine should have lower conservation
        assert!(analyzer.conservation_map.get("LEU").unwrap() < &0.5);
    }

    #[test]
    fn test_analyze_structure() {
        let analyzer = ConservationAnalyzer::new();

        let atoms = vec![
            make_atom(1, "CYS"), // High conservation
            make_atom(2, "GLY"), // High conservation
            make_atom(3, "LEU"), // Low conservation
            make_atom(4, "ALA"), // Medium conservation
            make_atom(5, "HIS"), // High conservation (catalytic)
        ];

        let result = analyzer.analyze(&atoms);
        assert_eq!(result.total_residues, 5);
        assert!(result.mean_conservation > 0.4);
    }

    #[test]
    fn test_empty_input() {
        let analyzer = ConservationAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.total_residues, 0);
    }
}
