//! PRISM-Bench: Unified Comprehensive Benchmark Suite
//!
//! A complete validation framework for protein dynamics prediction,
//! comparing against all major published benchmarks.
//!
//! # Benchmark Categories
//!
//! ## 1. Flexibility Validation
//! - B-factor correlation (crystallographic)
//! - pLDDT anti-correlation (AlphaFold disorder)
//! - MD RMSF correlation (molecular dynamics)
//!
//! ## 2. Ensemble Quality
//! - Pairwise RMSD distribution
//! - Ensemble diversity metrics
//! - Distance Correlation Coefficient (DCC)
//!
//! ## 3. Functional Site Detection
//! - Cryptic binding sites (CryptoSite benchmark)
//! - Allosteric sites (ASD database)
//! - Ligand binding sites (LBS)
//!
//! ## 4. Conformational Changes
//! - Apo-Holo RMSD prediction
//! - Hinge detection
//! - Domain motion capture
//!
//! ## 5. Classification Metrics
//! - ROC AUC
//! - F1 Score
//! - Precision / Recall
//! - Matthews Correlation Coefficient (MCC)

use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================================
// PUBLISHED LEADERBOARD BASELINES
// ============================================================================

/// Published benchmark results for comparison
pub struct Leaderboards;

impl Leaderboards {
    /// ATLAS RMSF Pearson correlation (Jing et al. 2024)
    pub const ALPHAFLOW_RMSF: f64 = 0.62;
    pub const ESMFLOW_RMSF: f64 = 0.58;
    pub const MD_REFERENCE: f64 = 1.00;

    /// CryptoSite benchmark (Cimermancic et al. 2016)
    pub const CRYPTOSITE_AUC: f64 = 0.83;
    pub const FPOCKET_AUC: f64 = 0.72;

    /// Allosteric site detection (ASD database)
    pub const ALLOSITE_AUC: f64 = 0.78;

    /// B-factor correlation baselines
    pub const GNMR_BFACTOR: f64 = 0.59;  // Gaussian Network Model
    pub const ANM_BFACTOR: f64 = 0.56;   // Anisotropic Network Model
    pub const NMA_BFACTOR: f64 = 0.54;   // Normal Mode Analysis

    /// Apo-Holo RMSD prediction
    pub const DYNAMINE_APOHOLO: f64 = 0.65;
}

// ============================================================================
// FLEXIBILITY METRICS
// ============================================================================

/// Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Spearman rank correlation
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);
    pearson_correlation(&rank_x, &rank_y)
}

fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as f64 + 1.0;
    }
    ranks
}

/// B-factor to RMSF conversion (Debye-Waller)
/// RMSF = sqrt(3 * B / (8 * π²))
pub fn bfactor_to_rmsf(b: f64) -> f64 {
    if b <= 0.0 {
        return 0.0;
    }
    (3.0 * b / (8.0 * PI * PI)).sqrt()
}

/// Distance Correlation Coefficient (DCC)
/// Measures non-linear dependence between variables
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len();

    // Compute distance matrices
    let mut dx = vec![vec![0.0; n]; n];
    let mut dy = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            dx[i][j] = (x[i] - x[j]).abs();
            dy[i][j] = (y[i] - y[j]).abs();
        }
    }

    // Double center the matrices
    let ax = double_center(&dx);
    let ay = double_center(&dy);

    // Compute distance covariance and variances
    let dcov_xy = distance_covariance(&ax, &ay);
    let dcov_xx = distance_covariance(&ax, &ax);
    let dcov_yy = distance_covariance(&ay, &ay);

    if dcov_xx < 1e-10 || dcov_yy < 1e-10 {
        return 0.0;
    }

    dcov_xy / (dcov_xx.sqrt() * dcov_yy.sqrt())
}

fn double_center(d: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = d.len();
    let mut a = vec![vec![0.0; n]; n];

    // Row means
    let row_means: Vec<f64> = d.iter().map(|row| row.iter().sum::<f64>() / n as f64).collect();
    // Column means
    let col_means: Vec<f64> = (0..n)
        .map(|j| d.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();
    // Grand mean
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

    for i in 0..n {
        for j in 0..n {
            a[i][j] = d[i][j] - row_means[i] - col_means[j] + grand_mean;
        }
    }
    a
}

fn distance_covariance(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += a[i][j] * b[i][j];
        }
    }
    sum / (n * n) as f64
}

// ============================================================================
// ENSEMBLE QUALITY METRICS
// ============================================================================

/// Pairwise RMSD statistics for ensemble
#[derive(Debug, Clone)]
pub struct EnsembleRmsdStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub count: usize,
}

/// Compute pairwise RMSD distribution from CA coordinates
pub fn compute_pairwise_rmsd_stats(
    ensembles: &[Vec<[f64; 3]>],  // Vec of conformations, each is Vec of CA coords
) -> EnsembleRmsdStats {
    let mut rmsds = Vec::new();

    for i in 0..ensembles.len() {
        for j in (i + 1)..ensembles.len() {
            let rmsd = compute_rmsd(&ensembles[i], &ensembles[j]);
            rmsds.push(rmsd);
        }
    }

    if rmsds.is_empty() {
        return EnsembleRmsdStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            count: 0,
        };
    }

    rmsds.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = rmsds.iter().sum::<f64>() / rmsds.len() as f64;
    let variance = rmsds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rmsds.len() as f64;
    let std = variance.sqrt();
    let median = rmsds[rmsds.len() / 2];

    EnsembleRmsdStats {
        mean,
        std,
        min: rmsds[0],
        max: *rmsds.last().unwrap(),
        median,
        count: rmsds.len(),
    }
}

fn compute_rmsd(coords1: &[[f64; 3]], coords2: &[[f64; 3]]) -> f64 {
    if coords1.len() != coords2.len() || coords1.is_empty() {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    for (c1, c2) in coords1.iter().zip(coords2.iter()) {
        let dx = c1[0] - c2[0];
        let dy = c1[1] - c2[1];
        let dz = c1[2] - c2[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }

    (sum_sq / coords1.len() as f64).sqrt()
}

/// Ensemble diversity (average deviation from mean structure)
pub fn compute_ensemble_diversity(ensembles: &[Vec<[f64; 3]>]) -> f64 {
    if ensembles.is_empty() {
        return 0.0;
    }

    let n_atoms = ensembles[0].len();
    let n_conf = ensembles.len();

    // Compute mean structure
    let mut mean_coords = vec![[0.0; 3]; n_atoms];
    for conf in ensembles {
        for (i, coord) in conf.iter().enumerate() {
            mean_coords[i][0] += coord[0];
            mean_coords[i][1] += coord[1];
            mean_coords[i][2] += coord[2];
        }
    }
    for coord in &mut mean_coords {
        coord[0] /= n_conf as f64;
        coord[1] /= n_conf as f64;
        coord[2] /= n_conf as f64;
    }

    // Compute average RMSD to mean
    let mut total_rmsd = 0.0;
    for conf in ensembles {
        total_rmsd += compute_rmsd(conf, &mean_coords);
    }

    total_rmsd / n_conf as f64
}

// ============================================================================
// CLASSIFICATION METRICS
// ============================================================================

/// Binary classification results
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    pub tp: usize,  // True positives
    pub fp: usize,  // False positives
    pub tn: usize,  // True negatives
    pub fn_: usize, // False negatives
}

impl ClassificationMetrics {
    pub fn new(predictions: &[bool], ground_truth: &[bool]) -> Self {
        let mut metrics = Self::default();

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            match (*pred, *truth) {
                (true, true) => metrics.tp += 1,
                (true, false) => metrics.fp += 1,
                (false, true) => metrics.fn_ += 1,
                (false, false) => metrics.tn += 1,
            }
        }

        metrics
    }

    /// Precision = TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { 0.0 } else { self.tp as f64 / denom as f64 }
    }

    /// Recall = TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 { 0.0 } else { self.tp as f64 / denom as f64 }
    }

    /// F1 = 2 * (precision * recall) / (precision + recall)
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r < 1e-10 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    /// Accuracy = (TP + TN) / total
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.fp + self.tn + self.fn_;
        if total == 0 { 0.0 } else { (self.tp + self.tn) as f64 / total as f64 }
    }

    /// Matthews Correlation Coefficient
    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let fp = self.fp as f64;
        let tn = self.tn as f64;
        let fn_ = self.fn_ as f64;

        let numer = tp * tn - fp * fn_;
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

        if denom < 1e-10 { 0.0 } else { numer / denom }
    }
}

/// ROC AUC computation
pub fn compute_roc_auc(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.5;
    }

    // Sort by score descending
    let mut indexed: Vec<(f64, bool)> = scores.iter().cloned().zip(labels.iter().cloned()).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = labels.iter().filter(|&&x| x).count() as f64;
    let n_neg = labels.iter().filter(|&&x| !x).count() as f64;

    if n_pos < 1.0 || n_neg < 1.0 {
        return 0.5;
    }

    // Compute AUC using trapezoidal rule
    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;

    for (_, label) in &indexed {
        if *label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        // Trapezoidal integration
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        prev_tp = tp;
        prev_fp = fp;
    }

    auc / (n_pos * n_neg)
}

// ============================================================================
// BINDING SITE DETECTION
// ============================================================================

/// Cryptic binding site detection result
#[derive(Debug, Clone)]
pub struct CrypticSiteResult {
    pub pdb_id: String,
    pub residue_scores: Vec<f64>,      // Per-residue cryptic site probability
    pub predicted_sites: Vec<usize>,   // Residue indices of predicted sites
    pub known_sites: Vec<usize>,       // Known cryptic site residues
    pub auc: f64,
    pub f1: f64,
}

/// Allosteric site detection result
#[derive(Debug, Clone)]
pub struct AllostericSiteResult {
    pub pdb_id: String,
    pub residue_scores: Vec<f64>,
    pub predicted_sites: Vec<usize>,
    pub known_sites: Vec<usize>,
    pub auc: f64,
    pub f1: f64,
}

/// Detect cryptic binding sites from ensemble RMSF
/// High RMSF + pocket formation = cryptic site candidate
pub fn detect_cryptic_sites(
    rmsf: &[f64],
    pocket_scores: &[f64],  // From fpocket or similar
    threshold_percentile: f64,
) -> Vec<usize> {
    let mut combined: Vec<f64> = rmsf
        .iter()
        .zip(pocket_scores.iter())
        .map(|(r, p)| r * p)  // Combined score
        .collect();

    // Find threshold at percentile
    let mut sorted = combined.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_idx = ((threshold_percentile / 100.0) * sorted.len() as f64) as usize;
    let threshold = sorted.get(threshold_idx).copied().unwrap_or(0.0);

    combined
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= threshold)
        .map(|(idx, _)| idx)
        .collect()
}

// ============================================================================
// APO-HOLO VALIDATION
// ============================================================================

/// Apo-Holo pair validation result
#[derive(Debug, Clone)]
pub struct ApoHoloResult {
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub experimental_rmsd: f64,        // Actual apo-holo RMSD
    pub predicted_rmsd: f64,           // From ensemble
    pub per_residue_correlation: f64,  // Correlation of per-residue changes
    pub binding_site_captured: bool,   // Did ensemble sample the binding site?
}

// ============================================================================
// PRISM-BENCH COMPREHENSIVE RESULT
// ============================================================================

/// Complete PRISM-Bench evaluation result
#[derive(Debug, Clone)]
pub struct PrismBenchResult {
    // Dataset info
    pub n_proteins: usize,
    pub dataset_name: String,

    // Flexibility metrics
    pub bfactor_pearson: f64,
    pub bfactor_spearman: f64,
    pub plddt_anticorr: f64,
    pub md_rmsf_pearson: f64,

    // Ensemble quality
    pub mean_pairwise_rmsd: f64,
    pub ensemble_diversity: f64,
    pub dcc: f64,

    // Binding site detection
    pub cryptic_auc: f64,
    pub cryptic_f1: f64,
    pub allosteric_auc: f64,
    pub allosteric_f1: f64,
    pub lbs_auc: f64,

    // Conformational change
    pub apoholo_rmsd_corr: f64,
    pub apoholo_site_recall: f64,

    // Per-protein results
    pub protein_results: Vec<ProteinBenchResult>,
}

#[derive(Debug, Clone)]
pub struct ProteinBenchResult {
    pub pdb_id: String,
    pub n_residues: usize,

    // Flexibility
    pub bfactor_corr: f64,
    pub plddt_corr: f64,
    pub md_rmsf_corr: Option<f64>,

    // Ensemble
    pub pairwise_rmsd: f64,
    pub diversity: f64,

    // Binding sites
    pub cryptic_score: Option<f64>,
    pub allosteric_score: Option<f64>,

    // Classification
    pub passed: bool,
}

impl PrismBenchResult {
    /// Generate comparison table against published baselines
    pub fn comparison_table(&self) -> String {
        let mut table = String::new();

        table.push_str("╔═══════════════════════════════════════════════════════════════════════════╗\n");
        table.push_str("║                    PRISM-BENCH COMPREHENSIVE RESULTS                      ║\n");
        table.push_str("╠═══════════════════════════════════════════════════════════════════════════╣\n");
        table.push_str(&format!("║  Dataset: {:<64} ║\n", self.dataset_name));
        table.push_str(&format!("║  Proteins: {:<63} ║\n", self.n_proteins));
        table.push_str("╠═══════════════════════════════════════════════════════════════════════════╣\n");
        table.push_str("║  FLEXIBILITY VALIDATION                                                   ║\n");
        table.push_str("╠───────────────────────────┬────────────┬────────────┬─────────────────────╣\n");
        table.push_str("║  Metric                   │ PRISM-Δ    │ Baseline   │ Improvement         ║\n");
        table.push_str("╠───────────────────────────┼────────────┼────────────┼─────────────────────╣\n");
        table.push_str(&format!(
            "║  B-factor ρ (Pearson)     │ {:>10.3} │ {:>10.3} │ {:>+18.1}% ║\n",
            self.bfactor_pearson,
            Leaderboards::GNMR_BFACTOR,
            (self.bfactor_pearson / Leaderboards::GNMR_BFACTOR - 1.0) * 100.0
        ));
        table.push_str(&format!(
            "║  MD RMSF ρ (AlphaFlow)    │ {:>10.3} │ {:>10.3} │ {:>+18.1}% ║\n",
            self.md_rmsf_pearson,
            Leaderboards::ALPHAFLOW_RMSF,
            (self.md_rmsf_pearson / Leaderboards::ALPHAFLOW_RMSF - 1.0) * 100.0
        ));
        table.push_str("╠═══════════════════════════════════════════════════════════════════════════╣\n");
        table.push_str("║  BINDING SITE DETECTION                                                   ║\n");
        table.push_str("╠───────────────────────────┬────────────┬────────────┬─────────────────────╣\n");
        table.push_str(&format!(
            "║  Cryptic Sites (AUC)      │ {:>10.3} │ {:>10.3} │ {:>+18.1}% ║\n",
            self.cryptic_auc,
            Leaderboards::CRYPTOSITE_AUC,
            (self.cryptic_auc / Leaderboards::CRYPTOSITE_AUC - 1.0) * 100.0
        ));
        table.push_str(&format!(
            "║  Allosteric Sites (AUC)   │ {:>10.3} │ {:>10.3} │ {:>+18.1}% ║\n",
            self.allosteric_auc,
            Leaderboards::ALLOSITE_AUC,
            (self.allosteric_auc / Leaderboards::ALLOSITE_AUC - 1.0) * 100.0
        ));
        table.push_str("╠═══════════════════════════════════════════════════════════════════════════╣\n");
        table.push_str("║  ENSEMBLE QUALITY                                                         ║\n");
        table.push_str("╠───────────────────────────┬────────────────────────────────────────────────╣\n");
        table.push_str(&format!(
            "║  Pairwise RMSD            │ {:>10.2} Å                                     ║\n",
            self.mean_pairwise_rmsd
        ));
        table.push_str(&format!(
            "║  Ensemble Diversity       │ {:>10.2} Å                                     ║\n",
            self.ensemble_diversity
        ));
        table.push_str(&format!(
            "║  Distance Corr. (DCC)     │ {:>10.3}                                       ║\n",
            self.dcc
        ));
        table.push_str("╠═══════════════════════════════════════════════════════════════════════════╣\n");
        table.push_str("║  CONFORMATIONAL CHANGE                                                    ║\n");
        table.push_str("╠───────────────────────────┬────────────┬────────────┬─────────────────────╣\n");
        table.push_str(&format!(
            "║  Apo-Holo RMSD ρ          │ {:>10.3} │ {:>10.3} │ {:>+18.1}% ║\n",
            self.apoholo_rmsd_corr,
            Leaderboards::DYNAMINE_APOHOLO,
            (self.apoholo_rmsd_corr / Leaderboards::DYNAMINE_APOHOLO - 1.0) * 100.0
        ));
        table.push_str(&format!(
            "║  Binding Site Recall      │ {:>10.3} │        n/a │                n/a ║\n",
            self.apoholo_site_recall
        ));
        table.push_str("╚═══════════════════════════════════════════════════════════════════════════╝\n");

        table
    }
}

// ============================================================================
// TEST SET CURATORS
// ============================================================================

/// Known apo-holo pairs for validation
pub static APO_HOLO_PAIRS: &[(&str, &str, f64)] = &[
    // (apo_pdb, holo_pdb, expected_rmsd)
    ("1AKE", "4AKE", 7.1),   // Adenylate kinase - large conformational change
    ("1GGG", "1WDN", 3.5),   // Glutamine binding protein
    ("2LZM", "3LZM", 0.8),   // Lysozyme
    ("1BPG", "2PGK", 6.7),   // Phosphoglycerate kinase
    ("1OMP", "1ANF", 4.2),   // Maltodextrin binding protein
    ("3CLN", "1CTR", 1.5),   // Calmodulin
    ("1URN", "1URJ", 2.1),   // Ribonuclease inhibitor
];

/// Known cryptic binding site proteins (from CryptoSite benchmark)
pub static CRYPTIC_SITE_PROTEINS: &[(&str, &[usize])] = &[
    // (pdb_id, cryptic_site_residues)
    ("1FKB", &[26, 27, 28, 29, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]),  // FKBP12
    ("1N2C", &[115, 116, 117, 118, 119, 120, 121, 122, 123]),  // TEM-1 β-lactamase
    ("2P5E", &[39, 40, 41, 42, 43, 44, 45]),  // IL-2
    ("2BX2", &[201, 202, 203, 204, 205, 206, 207, 208, 209, 210]),  // p38 MAPK
];

/// Known allosteric site proteins (from ASD database)
pub static ALLOSTERIC_PROTEINS: &[(&str, &[usize])] = &[
    ("1F3U", &[153, 154, 155, 156, 157, 158, 159]),  // HIV protease
    ("1M17", &[752, 753, 754, 755, 756]),  // EGFR
    ("3K5V", &[233, 234, 235, 236, 237]),  // PDK1
];

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001);

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!((r_neg - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_roc_auc() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        let labels = vec![true, true, true, true, false, false, false, false];
        let auc = compute_roc_auc(&scores, &labels);
        assert!((auc - 1.0).abs() < 0.001, "Perfect separation should give AUC=1");
    }

    #[test]
    fn test_classification_metrics() {
        let pred = vec![true, true, false, false];
        let truth = vec![true, false, true, false];
        let m = ClassificationMetrics::new(&pred, &truth);

        assert_eq!(m.tp, 1);
        assert_eq!(m.fp, 1);
        assert_eq!(m.fn_, 1);
        assert_eq!(m.tn, 1);
        assert!((m.accuracy() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_bfactor_conversion() {
        let b = 30.0;  // Typical B-factor
        let rmsf = bfactor_to_rmsf(b);
        assert!(rmsf > 0.5 && rmsf < 1.5);  // Should be in reasonable range
    }
}
