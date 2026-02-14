//! Antibody Binding Site Validation
//!
//! Validates cryptic site predictions against known antibody epitopes,
//! such as the m102.4 antibody binding site on Nipah virus G protein.
//!
//! # Validation Approach
//!
//! For each known antibody epitope:
//! 1. **Epitope Recall**: What fraction of epitope residues are predicted as cryptic?
//! 2. **Epitope Precision**: Of residues predicted cryptic, what fraction are in the epitope?
//! 3. **Rank Analysis**: How highly ranked are epitope residues in the predictions?
//! 4. **Overlap Score**: Combined metric for therapeutic relevance.
//!
//! # Example
//!
//! ```rust,ignore
//! use prism_validation::antibody_validation::*;
//!
//! // Load m102.4 epitope
//! let epitope = AntibodyEpitope::m102_4_nipah();
//!
//! // Validate predictions
//! let metrics = validate_against_epitope(&predictions, &epitope);
//! println!("Epitope recall: {:.1}%", metrics.epitope_recall * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Antibody epitope definition for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntibodyEpitope {
    /// Antibody name (e.g., "m102.4")
    pub name: String,

    /// Target protein (e.g., "NiV G")
    pub target_protein: String,

    /// PDB ID of the antibody-antigen complex
    pub complex_pdb_id: Option<String>,

    /// PDB ID of the apo target
    pub apo_pdb_id: String,

    /// Core epitope residues (directly contacted by CDRs)
    pub core_residues: Vec<i32>,

    /// Extended epitope residues (within 5Å of antibody)
    pub extended_residues: Vec<i32>,

    /// All epitope residues (core + extended)
    pub all_residues: Vec<i32>,

    /// Binding affinity (Kd in nM, lower = stronger)
    pub kd_nm: Option<f64>,

    /// Is this a neutralizing antibody?
    pub neutralizing: bool,

    /// Literature reference
    pub reference: String,
}

impl AntibodyEpitope {
    /// Create m102.4 epitope for Nipah virus G protein
    ///
    /// m102.4 is a broadly neutralizing human monoclonal antibody that
    /// targets the central cavity of Nipah/Hendra virus G proteins.
    /// It blocks receptor binding by occupying the ephrin binding pocket.
    ///
    /// Reference: Xu K, et al. (2008) PNAS 105(29):9953-9958
    pub fn m102_4_nipah() -> Self {
        Self {
            name: "m102.4".to_string(),
            target_protein: "NiV G (Attachment Glycoprotein)".to_string(),
            complex_pdb_id: Some("3D11".to_string()),
            apo_pdb_id: "2VWD".to_string(),

            // Core epitope - directly contacted by m102.4 CDRs
            core_residues: vec![
                507, 508, 509, 510, 511, 512,
                529, 530, 531, 532, 533,
            ],

            // Extended epitope - within 5Å of antibody
            extended_residues: vec![
                504, 505, 506, 513, 514,
                527, 528, 534, 535,
            ],

            // All epitope residues
            all_residues: vec![
                504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514,
                527, 528, 529, 530, 531, 532, 533, 534, 535,
            ],

            kd_nm: Some(0.04), // 40 pM - extremely potent
            neutralizing: true,
            reference: "Xu K, et al. (2008) PNAS 105(29):9953-9958".to_string(),
        }
    }

    /// Get all epitope residues as a HashSet
    pub fn residue_set(&self) -> HashSet<i32> {
        self.all_residues.iter().copied().collect()
    }

    /// Get core epitope residues as a HashSet
    pub fn core_residue_set(&self) -> HashSet<i32> {
        self.core_residues.iter().copied().collect()
    }
}

/// Metrics for antibody epitope validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntibodyValidationMetrics {
    /// Antibody name
    pub antibody_name: String,

    /// Total number of epitope residues
    pub n_epitope_residues: usize,

    /// Number of epitope residues predicted as cryptic
    pub n_epitope_predicted: usize,

    /// Epitope recall: fraction of epitope in predictions
    pub epitope_recall: f64,

    /// Epitope precision: fraction of predictions that are epitope
    pub epitope_precision: f64,

    /// F1 score
    pub f1_score: f64,

    /// Rank of first epitope residue in predictions
    pub first_epitope_rank: Option<usize>,

    /// Mean rank of epitope residues
    pub mean_epitope_rank: f64,

    /// Median rank of epitope residues
    pub median_epitope_rank: f64,

    /// Mean cryptic score of epitope residues
    pub mean_epitope_score: f64,

    /// Mean cryptic score of all predictions
    pub mean_prediction_score: f64,

    /// Score enrichment: epitope scores vs all scores
    pub score_enrichment: f64,

    /// Which epitope residues were found
    pub found_epitope_residues: Vec<i32>,

    /// Which epitope residues were missed
    pub missed_epitope_residues: Vec<i32>,
}

impl AntibodyValidationMetrics {
    /// Check if validation passes (epitope recall ≥ 70%)
    pub fn passes_threshold(&self, min_recall: f64) -> bool {
        self.epitope_recall >= min_recall
    }

    /// Get overall quality grade
    pub fn grade(&self) -> &'static str {
        match self.epitope_recall {
            r if r >= 0.90 => "A (Excellent)",
            r if r >= 0.70 => "B (Good)",
            r if r >= 0.50 => "C (Moderate)",
            r if r >= 0.30 => "D (Poor)",
            _ => "F (Failed)",
        }
    }

    /// Generate summary text
    pub fn summary(&self) -> String {
        format!(
            "{}: Recall={:.1}%, Precision={:.1}%, F1={:.3}, MeanRank={:.1} [{}]",
            self.antibody_name,
            self.epitope_recall * 100.0,
            self.epitope_precision * 100.0,
            self.f1_score,
            self.mean_epitope_rank,
            self.grade()
        )
    }
}

/// Validate cryptic site predictions against an antibody epitope
///
/// # Arguments
/// * `predictions` - Ranked list of (residue_id, cryptic_score) tuples, highest score first
/// * `epitope` - Antibody epitope to validate against
///
/// # Returns
/// Validation metrics showing how well predictions overlap with the epitope
pub fn validate_against_epitope(
    predictions: &[(i32, f64)],
    epitope: &AntibodyEpitope,
) -> AntibodyValidationMetrics {
    let epitope_set = epitope.residue_set();

    if predictions.is_empty() {
        return AntibodyValidationMetrics {
            antibody_name: epitope.name.clone(),
            n_epitope_residues: epitope.all_residues.len(),
            n_epitope_predicted: 0,
            epitope_recall: 0.0,
            epitope_precision: 0.0,
            f1_score: 0.0,
            first_epitope_rank: None,
            mean_epitope_rank: f64::INFINITY,
            median_epitope_rank: f64::INFINITY,
            mean_epitope_score: 0.0,
            mean_prediction_score: 0.0,
            score_enrichment: 0.0,
            found_epitope_residues: Vec::new(),
            missed_epitope_residues: epitope.all_residues.clone(),
        };
    }

    // Get predictions as a set
    let pred_set: HashSet<i32> = predictions.iter().map(|(r, _)| *r).collect();

    // Overlap analysis
    let found: Vec<i32> = epitope_set.intersection(&pred_set).copied().collect();
    let missed: Vec<i32> = epitope_set.difference(&pred_set).copied().collect();

    let n_epitope_predicted = found.len();
    let epitope_recall = n_epitope_predicted as f64 / epitope.all_residues.len() as f64;
    let epitope_precision = if predictions.is_empty() {
        0.0
    } else {
        n_epitope_predicted as f64 / predictions.len() as f64
    };

    let f1_score = if epitope_recall + epitope_precision > 0.0 {
        2.0 * epitope_recall * epitope_precision / (epitope_recall + epitope_precision)
    } else {
        0.0
    };

    // Rank analysis
    let mut epitope_ranks: Vec<usize> = Vec::new();
    let mut epitope_scores: Vec<f64> = Vec::new();
    let mut first_epitope_rank: Option<usize> = None;

    for (rank, (res_id, score)) in predictions.iter().enumerate() {
        if epitope_set.contains(res_id) {
            epitope_ranks.push(rank + 1); // 1-indexed
            epitope_scores.push(*score);
            if first_epitope_rank.is_none() {
                first_epitope_rank = Some(rank + 1);
            }
        }
    }

    let mean_epitope_rank = if epitope_ranks.is_empty() {
        f64::INFINITY
    } else {
        epitope_ranks.iter().sum::<usize>() as f64 / epitope_ranks.len() as f64
    };

    let median_epitope_rank = if epitope_ranks.is_empty() {
        f64::INFINITY
    } else {
        let mut sorted = epitope_ranks.clone();
        sorted.sort();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[mid] as f64
        }
    };

    let mean_epitope_score = if epitope_scores.is_empty() {
        0.0
    } else {
        epitope_scores.iter().sum::<f64>() / epitope_scores.len() as f64
    };

    let mean_prediction_score = predictions.iter().map(|(_, s)| s).sum::<f64>()
        / predictions.len() as f64;

    let score_enrichment = if mean_prediction_score > 0.0 {
        mean_epitope_score / mean_prediction_score
    } else {
        0.0
    };

    AntibodyValidationMetrics {
        antibody_name: epitope.name.clone(),
        n_epitope_residues: epitope.all_residues.len(),
        n_epitope_predicted,
        epitope_recall,
        epitope_precision,
        f1_score,
        first_epitope_rank,
        mean_epitope_rank,
        median_epitope_rank,
        mean_epitope_score,
        mean_prediction_score,
        score_enrichment,
        found_epitope_residues: found,
        missed_epitope_residues: missed,
    }
}

/// Validate cryptic scores (HashMap) against an antibody epitope
pub fn validate_scores_against_epitope(
    cryptic_scores: &HashMap<i32, f64>,
    threshold: f64,
    epitope: &AntibodyEpitope,
) -> AntibodyValidationMetrics {
    // Convert to sorted predictions
    let mut predictions: Vec<(i32, f64)> = cryptic_scores
        .iter()
        .filter(|(_, &score)| score >= threshold)
        .map(|(&res, &score)| (res, score))
        .collect();

    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    validate_against_epitope(&predictions, epitope)
}

/// Validate full CrypticSiteResultV2 against an antibody epitope
pub fn validate_result_against_epitope(
    result: &crate::ensemble_pocket_detector_v2::CrypticSiteResultV2,
    epitope: &AntibodyEpitope,
) -> AntibodyValidationMetrics {
    validate_scores_against_epitope(
        &result.cryptic_scores,
        result.adaptive_threshold,
        epitope,
    )
}

/// Summary of antibody validation for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntibodyValidationSummary {
    /// Structure tested (e.g., "2VWD")
    pub pdb_id: String,

    /// All antibody validations performed
    pub validations: Vec<AntibodyValidationMetrics>,

    /// Mean recall across all antibodies
    pub mean_recall: f64,

    /// Mean precision across all antibodies
    pub mean_precision: f64,

    /// Mean F1 across all antibodies
    pub mean_f1: f64,

    /// Number of antibodies that passed (recall ≥ 70%)
    pub n_passed: usize,

    /// Total number of antibodies tested
    pub n_total: usize,
}

impl AntibodyValidationSummary {
    /// Create summary from individual validations
    pub fn from_validations(pdb_id: &str, validations: Vec<AntibodyValidationMetrics>) -> Self {
        let n_total = validations.len();

        if n_total == 0 {
            return Self {
                pdb_id: pdb_id.to_string(),
                validations: Vec::new(),
                mean_recall: 0.0,
                mean_precision: 0.0,
                mean_f1: 0.0,
                n_passed: 0,
                n_total: 0,
            };
        }

        let mean_recall = validations.iter().map(|v| v.epitope_recall).sum::<f64>() / n_total as f64;
        let mean_precision = validations.iter().map(|v| v.epitope_precision).sum::<f64>() / n_total as f64;
        let mean_f1 = validations.iter().map(|v| v.f1_score).sum::<f64>() / n_total as f64;
        let n_passed = validations.iter().filter(|v| v.passes_threshold(0.7)).count();

        Self {
            pdb_id: pdb_id.to_string(),
            validations,
            mean_recall,
            mean_precision,
            mean_f1,
            n_passed,
            n_total,
        }
    }

    /// Generate report text
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("═══════════════════════════════════════════════════════════════════"));
        lines.push(format!("  ANTIBODY VALIDATION SUMMARY: {}", self.pdb_id));
        lines.push(format!("═══════════════════════════════════════════════════════════════════"));
        lines.push(String::new());

        for v in &self.validations {
            lines.push(v.summary());
        }

        lines.push(String::new());
        lines.push(format!("───────────────────────────────────────────────────────────────────"));
        lines.push(format!(
            "  Overall: {}/{} passed (≥70% recall), Mean Recall={:.1}%, Mean F1={:.3}",
            self.n_passed, self.n_total,
            self.mean_recall * 100.0,
            self.mean_f1
        ));
        lines.push(format!("═══════════════════════════════════════════════════════════════════"));

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m102_4_epitope() {
        let epitope = AntibodyEpitope::m102_4_nipah();

        assert_eq!(epitope.name, "m102.4");
        assert_eq!(epitope.apo_pdb_id, "2VWD");
        assert!(epitope.neutralizing);
        assert!(epitope.kd_nm.unwrap() < 1.0); // Sub-nanomolar

        // Core should be subset of all
        let core = epitope.core_residue_set();
        let all = epitope.residue_set();
        assert!(core.is_subset(&all));
    }

    #[test]
    fn test_validation_perfect() {
        let epitope = AntibodyEpitope::m102_4_nipah();

        // Perfect predictions: all epitope residues with high scores
        let predictions: Vec<(i32, f64)> = epitope.all_residues
            .iter()
            .enumerate()
            .map(|(i, &res)| (res, 1.0 - i as f64 * 0.01))
            .collect();

        let metrics = validate_against_epitope(&predictions, &epitope);

        assert!((metrics.epitope_recall - 1.0).abs() < 0.001);
        assert!((metrics.epitope_precision - 1.0).abs() < 0.001);
        assert!((metrics.f1_score - 1.0).abs() < 0.001);
        assert_eq!(metrics.first_epitope_rank, Some(1));
    }

    #[test]
    fn test_validation_partial() {
        let epitope = AntibodyEpitope::m102_4_nipah();

        // Partial predictions: half of epitope plus some extras
        let mut predictions: Vec<(i32, f64)> = epitope.all_residues
            .iter()
            .take(10)
            .enumerate()
            .map(|(i, &res)| (res, 0.9 - i as f64 * 0.01))
            .collect();

        // Add some non-epitope predictions
        predictions.push((100, 0.8));
        predictions.push((200, 0.7));

        let metrics = validate_against_epitope(&predictions, &epitope);

        assert!(metrics.epitope_recall > 0.4 && metrics.epitope_recall < 0.6);
        assert!(metrics.epitope_precision > 0.7); // Most predictions are epitope
    }

    #[test]
    fn test_validation_empty() {
        let epitope = AntibodyEpitope::m102_4_nipah();
        let predictions: Vec<(i32, f64)> = Vec::new();

        let metrics = validate_against_epitope(&predictions, &epitope);

        assert!((metrics.epitope_recall - 0.0).abs() < 0.001);
        assert_eq!(metrics.first_epitope_rank, None);
    }

    #[test]
    fn test_grade() {
        let epitope = AntibodyEpitope::m102_4_nipah();

        // Create metrics with different recall levels
        for (recall, expected_grade) in [
            (0.95, "A (Excellent)"),
            (0.75, "B (Good)"),
            (0.55, "C (Moderate)"),
            (0.35, "D (Poor)"),
            (0.15, "F (Failed)"),
        ] {
            let n_to_find = (epitope.all_residues.len() as f64 * recall) as usize;
            let predictions: Vec<(i32, f64)> = epitope.all_residues
                .iter()
                .take(n_to_find)
                .enumerate()
                .map(|(i, &res)| (res, 0.9 - i as f64 * 0.01))
                .collect();

            let metrics = validate_against_epitope(&predictions, &epitope);
            assert_eq!(metrics.grade(), expected_grade,
                "Expected {} for recall {:.2}, got {}",
                expected_grade, recall, metrics.grade());
        }
    }
}
