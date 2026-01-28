//! Cryptic Site Validation Metrics
//!
//! Standard metrics for benchmarking cryptic site detectors:
//! - PR AUC (Precision-Recall Area Under Curve)
//! - Success Rate (Detection Rate)
//! - Ranking Accuracy (Top-N hit rate)
//! - ROC AUC (Receiver Operating Characteristic)
//!
//! ## SOTA Baselines
//!
//! | Method      | Success% | ROC AUC | PR AUC | Top-3% |
//! |-------------|----------|---------|--------|--------|
//! | PocketMiner | -        | 0.87    | -      | -      |
//! | CryptoBank  | -        | 0.74    | 0.17   | -      |
//! | CrypToth    | 88.9%    | -       | -      | 78%    |
//! | Schrödinger | 83%      | -       | -      | -      |

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Single residue prediction with score
#[derive(Debug, Clone)]
pub struct ResiduePrediction {
    pub residue_id: i32,
    pub chain_id: Option<String>,
    pub score: f64,
}

/// Cryptic site candidate (cluster of residues)
#[derive(Debug, Clone)]
pub struct CrypticCandidate {
    pub residues: Vec<i32>,
    pub centroid: [f64; 3],
    pub score: f64,
    pub confidence: f64,
}

/// Ground truth cryptic site definition
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// Residues within distance cutoff of ligand
    pub cryptic_residues: HashSet<i32>,
    /// Ligand atom coordinates for overlap calculation
    pub ligand_coords: Vec<[f64; 3]>,
    /// Centroid of the cryptic pocket
    pub pocket_centroid: [f64; 3],
}

/// Validation result for a single protein
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleProteinResult {
    pub pdb_id: String,
    pub detected: bool,
    pub overlap_fraction: f64,
    pub top1_hit: bool,
    pub top3_hit: bool,
    pub top5_hit: bool,
    pub best_rank: Option<usize>,
    pub n_predictions: usize,
    pub n_ground_truth: usize,
}

// =============================================================================
// PR AUC (Precision-Recall Area Under Curve)
// =============================================================================

/// Compute Precision-Recall AUC for imbalanced cryptic site detection
///
/// Critical for publication claims - CryptoBank reports PR AUC = 0.17 on OOD test.
/// PR AUC is preferred over ROC AUC when classes are highly imbalanced
/// (cryptic residues are rare compared to non-cryptic).
///
/// # Arguments
/// * `predictions` - (residue_id, score) pairs, will be sorted by score descending
/// * `ground_truth` - Set of cryptic residue IDs
///
/// # Returns
/// PR AUC value in range [0, 1]
pub fn compute_pr_auc(
    predictions: &[(i32, f64)],
    ground_truth: &HashSet<i32>,
) -> f64 {
    if predictions.is_empty() || ground_truth.is_empty() {
        return 0.0;
    }

    // Sort by score descending
    let mut sorted: Vec<_> = predictions.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_positives = ground_truth.len() as f64;
    let mut true_positives = 0.0;
    let mut false_positives = 0.0;

    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    // Add initial point (recall=0, precision=1)
    precisions.push(1.0);
    recalls.push(0.0);

    for (res_id, _score) in &sorted {
        if ground_truth.contains(res_id) {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
        }

        let precision = true_positives / (true_positives + false_positives);
        let recall = true_positives / total_positives;

        precisions.push(precision);
        recalls.push(recall);
    }

    // Compute AUC using trapezoid rule
    let mut auc = 0.0;
    for i in 1..recalls.len() {
        let delta_recall = recalls[i] - recalls[i - 1];
        let avg_precision = (precisions[i] + precisions[i - 1]) / 2.0;
        auc += delta_recall * avg_precision;
    }

    auc
}

/// Compute ROC AUC (Receiver Operating Characteristic Area Under Curve)
///
/// Standard metric for binary classification. Less informative than PR AUC
/// for imbalanced data but included for comparison with published baselines.
///
/// # Returns
/// ROC AUC value in range [0, 1] where 0.5 = random, 1.0 = perfect
pub fn compute_roc_auc(
    predictions: &[(i32, f64)],
    ground_truth: &HashSet<i32>,
) -> f64 {
    if predictions.is_empty() || ground_truth.is_empty() {
        return 0.5; // Random baseline
    }

    // Sort by score descending
    let mut sorted: Vec<_> = predictions.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_positives = ground_truth.len() as f64;
    let total_negatives = (predictions.len() - ground_truth.len()) as f64;

    if total_negatives <= 0.0 {
        return 1.0; // All are positive
    }

    let mut true_positives = 0.0;
    let mut false_positives = 0.0;

    let mut tpr_values = vec![0.0];
    let mut fpr_values = vec![0.0];

    for (res_id, _score) in &sorted {
        if ground_truth.contains(res_id) {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
        }

        let tpr = true_positives / total_positives;
        let fpr = false_positives / total_negatives;

        tpr_values.push(tpr);
        fpr_values.push(fpr);
    }

    // Compute AUC using trapezoid rule
    let mut auc = 0.0;
    for i in 1..fpr_values.len() {
        let delta_fpr = fpr_values[i] - fpr_values[i - 1];
        let avg_tpr = (tpr_values[i] + tpr_values[i - 1]) / 2.0;
        auc += delta_fpr * avg_tpr;
    }

    auc
}

// =============================================================================
// Success Rate (Detection Rate)
// =============================================================================

/// Check if a cryptic site is "detected" based on residue overlap
///
/// A cryptic site is considered detected if ≥overlap_threshold of predicted
/// residues are within distance_cutoff of ligand atoms.
///
/// SOTA Reference: Schrödinger achieves 83% on 61 targets
///
/// # Arguments
/// * `predicted_residues` - Predicted cryptic residue IDs and their positions
/// * `ligand_coords` - 3D coordinates of ligand atoms
/// * `overlap_threshold` - Fraction of residues that must overlap (default 0.5)
/// * `distance_cutoff` - Distance in Å to consider "overlapping" (default 4.5)
pub fn compute_detection(
    predicted_residues: &[(i32, [f64; 3])],
    ligand_coords: &[[f64; 3]],
    overlap_threshold: f64,
    distance_cutoff: f64,
) -> (bool, f64) {
    if predicted_residues.is_empty() || ligand_coords.is_empty() {
        return (false, 0.0);
    }

    let cutoff_sq = distance_cutoff * distance_cutoff;
    let mut overlapping = 0;

    for (_res_id, res_pos) in predicted_residues {
        let mut near_ligand = false;
        for lig_pos in ligand_coords {
            let dx = res_pos[0] - lig_pos[0];
            let dy = res_pos[1] - lig_pos[1];
            let dz = res_pos[2] - lig_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq {
                near_ligand = true;
                break;
            }
        }

        if near_ligand {
            overlapping += 1;
        }
    }

    let overlap_fraction = overlapping as f64 / predicted_residues.len() as f64;
    let detected = overlap_fraction >= overlap_threshold;

    (detected, overlap_fraction)
}

/// Compute aggregate success rate across multiple proteins
pub fn aggregate_success_rate(results: &[SingleProteinResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let detected = results.iter().filter(|r| r.detected).count();
    detected as f64 / results.len() as f64
}

// =============================================================================
// Ranking Accuracy
// =============================================================================

/// Check if ground truth pocket ranks in top-N predictions
///
/// A prediction is considered a "hit" if its centroid is within distance_threshold
/// of the ground truth pocket centroid.
///
/// SOTA Reference: CrypToth achieves 78% top-1, 89% top-3
///
/// # Arguments
/// * `predictions` - Sorted by score (highest first)
/// * `ground_truth_centroid` - 3D position of true pocket center
/// * `distance_threshold` - Max distance for a match (default 8Å)
/// * `top_n` - Check top-N predictions (1, 3, or 5)
pub fn compute_ranking_accuracy(
    predictions: &[CrypticCandidate],
    ground_truth_centroid: [f64; 3],
    distance_threshold: f64,
    top_n: usize,
) -> (bool, Option<usize>) {
    let threshold_sq = distance_threshold * distance_threshold;

    for (rank, pred) in predictions.iter().take(top_n).enumerate() {
        let dx = pred.centroid[0] - ground_truth_centroid[0];
        let dy = pred.centroid[1] - ground_truth_centroid[1];
        let dz = pred.centroid[2] - ground_truth_centroid[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq < threshold_sq {
            return (true, Some(rank + 1)); // 1-indexed rank
        }
    }

    (false, None)
}

/// Compute aggregate ranking accuracy (Top-N hit rate)
pub fn aggregate_ranking_accuracy(results: &[SingleProteinResult], top_n: usize) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let hits = results.iter().filter(|r| {
        match top_n {
            1 => r.top1_hit,
            3 => r.top3_hit,
            5 => r.top5_hit,
            _ => r.best_rank.map(|r| r <= top_n).unwrap_or(false),
        }
    }).count();

    hits as f64 / results.len() as f64
}

// =============================================================================
// Aggregate Metrics
// =============================================================================

/// Aggregate metrics across a benchmark dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub success_rate: f64,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub top1_accuracy: f64,
    pub top3_accuracy: f64,
    pub top5_accuracy: f64,
    pub mean_overlap: f64,
    pub n_structures: usize,
    pub n_detected: usize,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from per-protein results
    pub fn from_results(
        results: &[SingleProteinResult],
        all_predictions: &[(i32, f64)],
        all_ground_truth: &HashSet<i32>,
    ) -> Self {
        let n_structures = results.len();
        let n_detected = results.iter().filter(|r| r.detected).count();

        let success_rate = aggregate_success_rate(results);
        let top1_accuracy = aggregate_ranking_accuracy(results, 1);
        let top3_accuracy = aggregate_ranking_accuracy(results, 3);
        let top5_accuracy = aggregate_ranking_accuracy(results, 5);

        let mean_overlap = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.overlap_fraction).sum::<f64>() / results.len() as f64
        };

        let roc_auc = compute_roc_auc(all_predictions, all_ground_truth);
        let pr_auc = compute_pr_auc(all_predictions, all_ground_truth);

        Self {
            success_rate,
            roc_auc,
            pr_auc,
            top1_accuracy,
            top3_accuracy,
            top5_accuracy,
            mean_overlap,
            n_structures,
            n_detected,
        }
    }
}

/// SOTA comparison baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SotaBaselines {
    pub pocketminer_roc_auc: f64,
    pub cryptobank_pr_auc: f64,
    pub cryptobank_roc_auc: f64,
    pub schrodinger_success: f64,
    pub cryptoth_top1: f64,
    pub cryptoth_top3: f64,
}

impl Default for SotaBaselines {
    fn default() -> Self {
        Self {
            pocketminer_roc_auc: 0.87,
            cryptobank_pr_auc: 0.17,  // OOD test set
            cryptobank_roc_auc: 0.74,
            schrodinger_success: 0.83,
            cryptoth_top1: 0.78,
            cryptoth_top3: 0.889,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pr_auc_perfect() {
        // Perfect predictions
        let predictions = vec![(1, 0.9), (2, 0.8), (3, 0.1), (4, 0.05)];
        let ground_truth: HashSet<i32> = [1, 2].into_iter().collect();

        let pr_auc = compute_pr_auc(&predictions, &ground_truth);
        assert!(pr_auc > 0.95, "Perfect predictions should have high PR AUC");
    }

    #[test]
    fn test_pr_auc_random() {
        // Random predictions
        let predictions = vec![(3, 0.9), (4, 0.8), (1, 0.1), (2, 0.05)];
        let ground_truth: HashSet<i32> = [1, 2].into_iter().collect();

        let pr_auc = compute_pr_auc(&predictions, &ground_truth);
        assert!(pr_auc < 0.5, "Random predictions should have lower PR AUC");
    }

    #[test]
    fn test_roc_auc_perfect() {
        let predictions = vec![(1, 0.9), (2, 0.8), (3, 0.1), (4, 0.05)];
        let ground_truth: HashSet<i32> = [1, 2].into_iter().collect();

        let roc_auc = compute_roc_auc(&predictions, &ground_truth);
        assert!(roc_auc > 0.95, "Perfect predictions should have high ROC AUC");
    }

    #[test]
    fn test_detection() {
        let predicted = vec![
            (1, [0.0, 0.0, 0.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [10.0, 10.0, 10.0]), // Far from ligand
        ];
        let ligand = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];

        let (detected, overlap) = compute_detection(&predicted, &ligand, 0.5, 4.5);

        assert!(detected, "Should detect when >50% overlap");
        assert!((overlap - 0.666).abs() < 0.01, "2/3 residues should overlap");
    }

    #[test]
    fn test_ranking() {
        let predictions = vec![
            CrypticCandidate {
                residues: vec![1, 2],
                centroid: [5.0, 5.0, 5.0],
                score: 0.9,
                confidence: 0.8,
            },
            CrypticCandidate {
                residues: vec![3, 4],
                centroid: [0.0, 0.0, 0.0], // This matches ground truth
                score: 0.7,
                confidence: 0.6,
            },
        ];

        let (hit, rank) = compute_ranking_accuracy(&predictions, [0.0, 0.0, 0.0], 8.0, 3);

        assert!(hit, "Should find match in top-3");
        assert_eq!(rank, Some(2), "Should rank as #2");
    }

    #[test]
    fn test_sota_baselines() {
        let baselines = SotaBaselines::default();
        assert_eq!(baselines.pocketminer_roc_auc, 0.87);
        assert_eq!(baselines.cryptobank_pr_auc, 0.17);
    }
}
