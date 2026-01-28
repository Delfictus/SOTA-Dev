//! Residue-level classification metrics
//!
//! For CryptoBench and residue-centric evaluation:
//! - AUC (Area Under ROC Curve)
//! - AUPRC (Area Under Precision-Recall Curve)
//! - MCC (Matthews Correlation Coefficient)
//! - F1 Score

use serde::{Deserialize, Serialize};

/// Binary classification results for a single residue
#[derive(Debug, Clone)]
pub struct ResiduePrediction {
    pub residue_id: i32,
    pub predicted_score: f64, // 0.0 to 1.0 probability
    pub is_binding: bool,     // ground truth label
}

/// Confusion matrix counts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub tp: usize, // True Positives
    pub fp: usize, // False Positives
    pub tn: usize, // True Negatives
    pub fn_count: usize, // False Negatives (fn_ to avoid keyword)
}

impl ConfusionMatrix {
    /// Build confusion matrix from predictions at given threshold
    pub fn from_predictions(predictions: &[ResiduePrediction], threshold: f64) -> Self {
        let mut cm = ConfusionMatrix::default();

        for pred in predictions {
            let predicted_positive = pred.predicted_score >= threshold;
            match (predicted_positive, pred.is_binding) {
                (true, true) => cm.tp += 1,
                (true, false) => cm.fp += 1,
                (false, true) => cm.fn_count += 1,
                (false, false) => cm.tn += 1,
            }
        }

        cm
    }

    /// Precision = TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }

    /// Recall (Sensitivity) = TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_count == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fn_count) as f64
        }
    }

    /// Specificity = TN / (TN + FP)
    pub fn specificity(&self) -> f64 {
        if self.tn + self.fp == 0 {
            0.0
        } else {
            self.tn as f64 / (self.tn + self.fp) as f64
        }
    }

    /// F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Matthews Correlation Coefficient
    /// MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_ = self.fn_count as f64;

        let numerator = tp * tn - fp * fn_;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Accuracy = (TP + TN) / (TP + TN + FP + FN)
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_count;
        if total == 0 {
            0.0
        } else {
            (self.tp + self.tn) as f64 / total as f64
        }
    }

    /// False Positive Rate = FP / (FP + TN)
    pub fn fpr(&self) -> f64 {
        if self.fp + self.tn == 0 {
            0.0
        } else {
            self.fp as f64 / (self.fp + self.tn) as f64
        }
    }

    /// True Positive Rate (same as recall)
    pub fn tpr(&self) -> f64 {
        self.recall()
    }
}

/// Calculate F1 score at given threshold
pub fn calculate_f1(predictions: &[ResiduePrediction], threshold: f64) -> f64 {
    ConfusionMatrix::from_predictions(predictions, threshold).f1()
}

/// Calculate MCC at given threshold
pub fn calculate_mcc(predictions: &[ResiduePrediction], threshold: f64) -> f64 {
    ConfusionMatrix::from_predictions(predictions, threshold).mcc()
}

/// Calculate AUC-ROC using trapezoidal integration
pub fn calculate_auc(predictions: &[ResiduePrediction]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    // Get ROC curve points
    let roc_points = calculate_roc_curve(predictions);

    // Trapezoidal integration
    let mut auc = 0.0;
    for i in 1..roc_points.len() {
        let (fpr1, tpr1) = roc_points[i - 1];
        let (fpr2, tpr2) = roc_points[i];
        auc += (fpr2 - fpr1) * (tpr1 + tpr2) / 2.0;
    }

    auc.clamp(0.0, 1.0)
}

/// Calculate ROC curve points (FPR, TPR) at various thresholds
pub fn calculate_roc_curve(predictions: &[ResiduePrediction]) -> Vec<(f64, f64)> {
    let mut sorted: Vec<_> = predictions.iter().collect();
    sorted.sort_by(|a, b| b.predicted_score.partial_cmp(&a.predicted_score).unwrap());

    let total_positive = predictions.iter().filter(|p| p.is_binding).count() as f64;
    let total_negative = predictions.iter().filter(|p| !p.is_binding).count() as f64;

    if total_positive == 0.0 || total_negative == 0.0 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }

    let mut points = vec![(0.0, 0.0)];
    let mut tp = 0.0;
    let mut fp = 0.0;

    for pred in sorted {
        if pred.is_binding {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / total_positive;
        let fpr = fp / total_negative;
        points.push((fpr, tpr));
    }

    points
}

/// Calculate AUPRC (Area Under Precision-Recall Curve)
pub fn calculate_auprc(predictions: &[ResiduePrediction]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    let pr_points = calculate_pr_curve(predictions);

    // Trapezoidal integration
    let mut auprc = 0.0;
    for i in 1..pr_points.len() {
        let (r1, p1) = pr_points[i - 1];
        let (r2, p2) = pr_points[i];
        auprc += (r2 - r1) * (p1 + p2) / 2.0;
    }

    auprc.clamp(0.0, 1.0)
}

/// Calculate Precision-Recall curve points
pub fn calculate_pr_curve(predictions: &[ResiduePrediction]) -> Vec<(f64, f64)> {
    let mut sorted: Vec<_> = predictions.iter().collect();
    sorted.sort_by(|a, b| b.predicted_score.partial_cmp(&a.predicted_score).unwrap());

    let total_positive = predictions.iter().filter(|p| p.is_binding).count() as f64;

    if total_positive == 0.0 {
        return vec![(0.0, 1.0), (1.0, 0.0)];
    }

    let mut points = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;

    for pred in sorted {
        if pred.is_binding {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let precision = tp / (tp + fp);
        let recall = tp / total_positive;
        points.push((recall, precision));
    }

    // Add start point if needed
    if points.first().map(|(r, _)| *r) != Some(0.0) {
        points.insert(0, (0.0, 1.0));
    }

    points
}

/// Find optimal threshold based on F1 score
pub fn find_optimal_f1_threshold(predictions: &[ResiduePrediction]) -> (f64, f64) {
    let thresholds: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();

    let mut best_threshold = 0.5;
    let mut best_f1 = 0.0;

    for &threshold in &thresholds {
        let f1 = calculate_f1(predictions, threshold);
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }

    (best_threshold, best_f1)
}

/// Find optimal threshold based on MCC
pub fn find_optimal_mcc_threshold(predictions: &[ResiduePrediction]) -> (f64, f64) {
    let thresholds: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();

    let mut best_threshold = 0.5;
    let mut best_mcc = -1.0;

    for &threshold in &thresholds {
        let mcc = calculate_mcc(predictions, threshold);
        if mcc > best_mcc {
            best_mcc = mcc;
            best_threshold = threshold;
        }
    }

    (best_threshold, best_mcc)
}

/// Comprehensive residue-level metrics at optimal threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueMetrics {
    pub auc: f64,
    pub auprc: f64,
    pub mcc: f64,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub specificity: f64,
    pub accuracy: f64,
    pub optimal_threshold: f64,
    pub confusion_matrix: ConfusionMatrix,
}

/// Calculate all residue-level metrics at optimal threshold
pub fn calculate_all_residue_metrics(predictions: &[ResiduePrediction]) -> ResidueMetrics {
    let (optimal_threshold, _) = find_optimal_f1_threshold(predictions);
    let cm = ConfusionMatrix::from_predictions(predictions, optimal_threshold);

    ResidueMetrics {
        auc: calculate_auc(predictions),
        auprc: calculate_auprc(predictions),
        mcc: cm.mcc(),
        f1: cm.f1(),
        precision: cm.precision(),
        recall: cm.recall(),
        specificity: cm.specificity(),
        accuracy: cm.accuracy(),
        optimal_threshold,
        confusion_matrix: cm,
    }
}

/// Convert pocket residue predictions to ResiduePrediction format
///
/// # Arguments
/// * `predicted_residues` - Residues predicted to be in binding site
/// * `predicted_scores` - Per-residue scores (if available)
/// * `ground_truth_residues` - Known binding site residues
/// * `all_residues` - All residues in the protein
pub fn predictions_from_residue_lists(
    predicted_residues: &[i32],
    predicted_scores: Option<&std::collections::HashMap<String, f64>>,
    ground_truth_residues: &[i32],
    all_residues: &[i32],
) -> Vec<ResiduePrediction> {
    use std::collections::HashSet;

    let pred_set: HashSet<i32> = predicted_residues.iter().copied().collect();
    let gt_set: HashSet<i32> = ground_truth_residues.iter().copied().collect();

    all_residues
        .iter()
        .map(|&res_id| {
            let predicted_score = if let Some(scores) = predicted_scores {
                scores.get(&res_id.to_string()).copied().unwrap_or(0.0)
            } else if pred_set.contains(&res_id) {
                1.0
            } else {
                0.0
            };

            ResiduePrediction {
                residue_id: res_id,
                predicted_score,
                is_binding: gt_set.contains(&res_id),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions() {
        let predictions = vec![
            ResiduePrediction {
                residue_id: 1,
                predicted_score: 0.9,
                is_binding: true,
            },
            ResiduePrediction {
                residue_id: 2,
                predicted_score: 0.8,
                is_binding: true,
            },
            ResiduePrediction {
                residue_id: 3,
                predicted_score: 0.1,
                is_binding: false,
            },
            ResiduePrediction {
                residue_id: 4,
                predicted_score: 0.2,
                is_binding: false,
            },
        ];

        let metrics = calculate_all_residue_metrics(&predictions);
        assert!(metrics.auc > 0.99);
        assert!(metrics.f1 > 0.99);
        assert!(metrics.mcc > 0.99);
    }

    #[test]
    fn test_random_predictions() {
        let predictions = vec![
            ResiduePrediction {
                residue_id: 1,
                predicted_score: 0.5,
                is_binding: true,
            },
            ResiduePrediction {
                residue_id: 2,
                predicted_score: 0.5,
                is_binding: false,
            },
        ];

        let auc = calculate_auc(&predictions);
        assert!((auc - 0.5).abs() < 0.2); // Should be around 0.5
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = vec![
            ResiduePrediction {
                residue_id: 1,
                predicted_score: 1.0,
                is_binding: true,
            }, // TP
            ResiduePrediction {
                residue_id: 2,
                predicted_score: 1.0,
                is_binding: false,
            }, // FP
            ResiduePrediction {
                residue_id: 3,
                predicted_score: 0.0,
                is_binding: true,
            }, // FN
            ResiduePrediction {
                residue_id: 4,
                predicted_score: 0.0,
                is_binding: false,
            }, // TN
        ];

        let cm = ConfusionMatrix::from_predictions(&predictions, 0.5);
        assert_eq!(cm.tp, 1);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_count, 1);
        assert_eq!(cm.tn, 1);
        assert!((cm.accuracy() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mcc_perfect() {
        let predictions = vec![
            ResiduePrediction {
                residue_id: 1,
                predicted_score: 1.0,
                is_binding: true,
            },
            ResiduePrediction {
                residue_id: 2,
                predicted_score: 0.0,
                is_binding: false,
            },
        ];
        let mcc = calculate_mcc(&predictions, 0.5);
        assert!((mcc - 1.0).abs() < 0.01);
    }
}
