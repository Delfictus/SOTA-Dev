//! Pocket-level metrics for benchmark evaluation
//!
//! Implements top-N+2 recall as proposed by LIGYSIS (2024)
//! and other pocket-level success metrics.

use super::dvo::{calculate_dvo_simple, DVO_SUCCESS};
use serde::{Deserialize, Serialize};

/// Standard DCC thresholds used in literature
pub const DCC_STRICT: f64 = 4.0;   // Traditional strict threshold
pub const DCC_LIGYSIS: f64 = 12.0; // LIGYSIS benchmark threshold
pub const DCC_RELAXED: f64 = 15.0; // Relaxed threshold for large sites

/// Result of matching a prediction to ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketMatch {
    pub ground_truth_idx: usize,
    pub predicted_idx: Option<usize>,
    pub dcc: f64,
    pub dvo: f64,
    pub is_success_dcc: bool,
    pub is_success_dvo: bool,
}

/// Recall metrics at various rank thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallAtRanks {
    pub top_1: f64,
    pub top_3: f64,
    pub top_5: f64,
    pub top_n: f64,
    pub top_n_plus_2: f64,
    pub all: f64,
}

/// Calculate top-N recall
///
/// N = number of ground truth binding sites
///
/// # Arguments
/// * `predicted_centroids` - Ranked list of predicted pocket centroids
/// * `ground_truth_centers` - List of ground truth binding site centers
/// * `dcc_threshold` - DCC threshold for success (default 12Å for LIGYSIS)
///
/// # Returns
/// Fraction of ground truth sites matched within top-N predictions
pub fn calculate_top_n_recall(
    predicted_centroids: &[[f64; 3]],
    ground_truth_centers: &[[f64; 3]],
    dcc_threshold: f64,
) -> f64 {
    let n = ground_truth_centers.len();
    if n == 0 || predicted_centroids.is_empty() {
        return 0.0;
    }

    calculate_recall_at_rank(predicted_centroids, ground_truth_centers, n, dcc_threshold)
}

/// Calculate top-(N+2) recall (LIGYSIS standard)
///
/// Allows 2 extra predictions to account for method variability.
/// This is the recommended metric per LIGYSIS (2024).
pub fn calculate_top_n_plus_2_recall(
    predicted_centroids: &[[f64; 3]],
    ground_truth_centers: &[[f64; 3]],
    dcc_threshold: f64,
) -> f64 {
    let n = ground_truth_centers.len();
    if n == 0 || predicted_centroids.is_empty() {
        return 0.0;
    }

    calculate_recall_at_rank(predicted_centroids, ground_truth_centers, n + 2, dcc_threshold)
}

/// Calculate recall at a specific rank threshold
pub fn calculate_recall_at_rank(
    predicted_centroids: &[[f64; 3]],
    ground_truth_centers: &[[f64; 3]],
    rank: usize,
    dcc_threshold: f64,
) -> f64 {
    let n_gt = ground_truth_centers.len();
    if n_gt == 0 || predicted_centroids.is_empty() {
        return 0.0;
    }

    // Take top-k predictions
    let top_k: Vec<_> = predicted_centroids.iter().take(rank).collect();

    // Greedy matching: for each ground truth, find best unmatched prediction
    let mut matched = 0;
    let mut used_predictions: Vec<bool> = vec![false; top_k.len()];

    for gt in ground_truth_centers {
        let mut best_dist = f64::INFINITY;
        let mut best_idx = None;

        for (idx, pred) in top_k.iter().enumerate() {
            if used_predictions[idx] {
                continue;
            }
            let dist = euclidean_distance(pred, gt);
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        }

        if best_dist <= dcc_threshold {
            matched += 1;
            if let Some(idx) = best_idx {
                used_predictions[idx] = true;
            }
        }
    }

    matched as f64 / n_gt as f64
}

/// Calculate recall at various rank thresholds
pub fn calculate_recall_at_ranks(
    predicted_centroids: &[[f64; 3]],
    ground_truth_centers: &[[f64; 3]],
    dcc_threshold: f64,
) -> RecallAtRanks {
    let n = ground_truth_centers.len();

    RecallAtRanks {
        top_1: calculate_recall_at_rank(predicted_centroids, ground_truth_centers, 1, dcc_threshold),
        top_3: calculate_recall_at_rank(predicted_centroids, ground_truth_centers, 3, dcc_threshold),
        top_5: calculate_recall_at_rank(predicted_centroids, ground_truth_centers, 5, dcc_threshold),
        top_n: calculate_recall_at_rank(predicted_centroids, ground_truth_centers, n, dcc_threshold),
        top_n_plus_2: calculate_recall_at_rank(predicted_centroids, ground_truth_centers, n + 2, dcc_threshold),
        all: calculate_recall_at_rank(
            predicted_centroids,
            ground_truth_centers,
            predicted_centroids.len(),
            dcc_threshold,
        ),
    }
}

/// Prediction result for matching
#[derive(Debug, Clone)]
pub struct PocketPrediction {
    pub centroid: [f64; 3],
    pub volume: f64,
    pub rank: usize,
    pub confidence: f64,
}

/// Ground truth binding site
#[derive(Debug, Clone)]
pub struct GroundTruthSite {
    pub center: [f64; 3],
    pub volume: f64,
    pub residues: Vec<i32>,
}

/// Match predictions to ground truth with detailed results
pub fn match_predictions_to_ground_truth(
    predicted_pockets: &[PocketPrediction],
    ground_truth_sites: &[GroundTruthSite],
    dcc_threshold: f64,
) -> Vec<PocketMatch> {
    let mut matches = Vec::new();
    let mut used_predictions: Vec<bool> = vec![false; predicted_pockets.len()];

    for (gt_idx, gt) in ground_truth_sites.iter().enumerate() {
        let mut best_pred_idx = None;
        let mut best_dcc = f64::INFINITY;

        for (pred_idx, pred) in predicted_pockets.iter().enumerate() {
            if used_predictions[pred_idx] {
                continue;
            }

            let dcc = euclidean_distance(&pred.centroid, &gt.center);
            if dcc < best_dcc {
                best_dcc = dcc;
                best_pred_idx = Some(pred_idx);
            }
        }

        // Calculate DVO if we have a match
        let dvo = if let Some(pred_idx) = best_pred_idx {
            let pred = &predicted_pockets[pred_idx];
            let dvo_result = calculate_dvo_simple(
                &pred.centroid,
                pred.volume,
                &gt.center,
                gt.volume,
            );
            dvo_result.jaccard
        } else {
            0.0
        };

        let is_success_dcc = best_dcc <= dcc_threshold;
        let is_success_dvo = dvo >= DVO_SUCCESS;

        matches.push(PocketMatch {
            ground_truth_idx: gt_idx,
            predicted_idx: best_pred_idx,
            dcc: best_dcc,
            dvo,
            is_success_dcc,
            is_success_dvo,
        });

        // Mark prediction as used if successful
        if let Some(idx) = best_pred_idx {
            if is_success_dcc {
                used_predictions[idx] = true;
            }
        }
    }

    matches
}

/// Aggregate pocket-level metrics across multiple structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPocketMetrics {
    pub num_structures: usize,
    pub num_ground_truth_sites: usize,
    pub num_matched_dcc: usize,
    pub num_matched_dvo: usize,
    pub success_rate_dcc: f64,
    pub success_rate_dvo: f64,
    pub mean_dcc: f64,
    pub median_dcc: f64,
    pub mean_dvo: f64,
    pub recall_at_ranks: RecallAtRanks,
}

impl AggregatedPocketMetrics {
    /// Compute aggregated metrics from multiple match results
    pub fn from_matches(
        all_matches: &[Vec<PocketMatch>],
        all_predictions: &[Vec<[f64; 3]>],
        all_ground_truth: &[Vec<[f64; 3]>],
        dcc_threshold: f64,
    ) -> Self {
        let num_structures = all_matches.len();
        let mut all_dccs = Vec::new();
        let mut all_dvos = Vec::new();
        let mut num_matched_dcc = 0;
        let mut num_matched_dvo = 0;
        let mut num_ground_truth_sites = 0;

        for matches in all_matches {
            for m in matches {
                num_ground_truth_sites += 1;
                if m.dcc.is_finite() {
                    all_dccs.push(m.dcc);
                }
                all_dvos.push(m.dvo);
                if m.is_success_dcc {
                    num_matched_dcc += 1;
                }
                if m.is_success_dvo {
                    num_matched_dvo += 1;
                }
            }
        }

        // Calculate aggregated recall
        let all_pred_centroids: Vec<[f64; 3]> = all_predictions.iter().flatten().copied().collect();
        let all_gt_centers: Vec<[f64; 3]> = all_ground_truth.iter().flatten().copied().collect();
        let recall_at_ranks = calculate_recall_at_ranks(&all_pred_centroids, &all_gt_centers, dcc_threshold);

        let mean_dcc = if all_dccs.is_empty() {
            f64::INFINITY
        } else {
            all_dccs.iter().sum::<f64>() / all_dccs.len() as f64
        };

        let median_dcc = if all_dccs.is_empty() {
            f64::INFINITY
        } else {
            let mut sorted = all_dccs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        let mean_dvo = if all_dvos.is_empty() {
            0.0
        } else {
            all_dvos.iter().sum::<f64>() / all_dvos.len() as f64
        };

        Self {
            num_structures,
            num_ground_truth_sites,
            num_matched_dcc,
            num_matched_dvo,
            success_rate_dcc: if num_ground_truth_sites == 0 {
                0.0
            } else {
                num_matched_dcc as f64 / num_ground_truth_sites as f64
            },
            success_rate_dvo: if num_ground_truth_sites == 0 {
                0.0
            } else {
                num_matched_dvo as f64 / num_ground_truth_sites as f64
            },
            mean_dcc,
            median_dcc,
            mean_dvo,
            recall_at_ranks,
        }
    }

    /// Generate formatted report
    pub fn report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║              POCKET-LEVEL BENCHMARK METRICS                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Structures:       {:>4}                                          ║
║  Ground Truth:     {:>4} binding sites                            ║
╠══════════════════════════════════════════════════════════════════╣
║                    DCC (< {}Å)                                    ║
║  ─────────────────────────────────────                            ║
║  Success Rate:     {:>5.1}%  ({}/{})                              ║
║  Mean DCC:         {:>6.2} Å                                      ║
║  Median DCC:       {:>6.2} Å                                      ║
╠══════════════════════════════════════════════════════════════════╣
║                    DVO (> 0.2)                                    ║
║  ─────────────────────────────────────                            ║
║  Success Rate:     {:>5.1}%  ({}/{})                              ║
║  Mean DVO:         {:>6.3}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║                    RECALL AT RANKS                                ║
║  ─────────────────────────────────────                            ║
║  Top-1:            {:>5.1}%                                        ║
║  Top-3:            {:>5.1}%                                        ║
║  Top-5:            {:>5.1}%                                        ║
║  Top-N:            {:>5.1}%                                        ║
║  Top-N+2:          {:>5.1}%  ← LIGYSIS standard                   ║
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.num_structures,
            self.num_ground_truth_sites,
            DCC_LIGYSIS as i32,
            self.success_rate_dcc * 100.0,
            self.num_matched_dcc,
            self.num_ground_truth_sites,
            self.mean_dcc,
            self.median_dcc,
            self.success_rate_dvo * 100.0,
            self.num_matched_dvo,
            self.num_ground_truth_sites,
            self.mean_dvo,
            self.recall_at_ranks.top_1 * 100.0,
            self.recall_at_ranks.top_3 * 100.0,
            self.recall_at_ranks.top_5 * 100.0,
            self.recall_at_ranks.top_n * 100.0,
            self.recall_at_ranks.top_n_plus_2 * 100.0,
        )
    }
}

fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_n_perfect() {
        let predicted = vec![
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ];
        let ground_truth = vec![
            [10.5, 20.5, 30.5], // Close to first prediction
            [40.5, 50.5, 60.5], // Close to second prediction
        ];

        let recall = calculate_top_n_recall(&predicted, &ground_truth, 4.0);
        assert!((recall - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_top_n_partial() {
        let predicted = vec![
            [0.0, 0.0, 0.0],    // No match
            [10.0, 20.0, 30.0], // Matches first GT
        ];
        let ground_truth = vec![
            [10.5, 20.5, 30.5], // Close to second prediction
            [100.0, 100.0, 100.0], // No match
        ];

        let recall = calculate_top_n_recall(&predicted, &ground_truth, 4.0);
        assert!((recall - 0.5).abs() < 0.01); // 1 of 2 matched
    }

    #[test]
    fn test_recall_at_ranks() {
        let predicted = vec![
            [100.0, 100.0, 100.0], // Wrong, rank 1
            [200.0, 200.0, 200.0], // Wrong, rank 2
            [10.0, 20.0, 30.0],    // Correct, rank 3
        ];
        let ground_truth = vec![
            [10.5, 20.5, 30.5], // Should match rank 3
        ];

        let recalls = calculate_recall_at_ranks(&predicted, &ground_truth, 4.0);
        assert!((recalls.top_1 - 0.0).abs() < 0.01);
        assert!((recalls.top_3 - 1.0).abs() < 0.01);
        assert!((recalls.top_5 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_inputs() {
        let recall = calculate_top_n_recall(&[], &[[0.0, 0.0, 0.0]], 4.0);
        assert!((recall - 0.0).abs() < 0.01);

        let recall = calculate_top_n_recall(&[[0.0, 0.0, 0.0]], &[], 4.0);
        assert!((recall - 0.0).abs() < 0.01);
    }
}
