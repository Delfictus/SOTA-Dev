//! DCC/DCA Validation Metrics for Ligand Binding Site Prediction
//!
//! Industry-standard metrics used in PDBBind, scPDB, and DUD-E benchmarks:
//! - DCC (Distance to Closest Contact): Min distance from any pocket atom to any ligand atom
//! - DCA (Distance to Center of Active site): Distance from pocket centroid to ligand centroid
//! - Top-N Success Rates: Percentage of cases where top-N pockets contain binding site
//! - Volume Overlap: Jaccard and Dice coefficients for spatial overlap
//!
//! Success criteria (industry standard):
//! - DCC < 4.0 Å = successful prediction
//! - DCA < 4.0 Å = successful prediction

use crate::pocket::Pocket;
use crate::structure::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Standard success threshold in Angstroms
pub const DEFAULT_SUCCESS_THRESHOLD: f64 = 4.0;

/// Comprehensive validation metrics for a single pocket-ligand pair
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// DCC: Distance to Closest Contact (min pocket atom to ligand atom distance)
    pub dcc: f64,
    /// DCA: Distance to Center of Active site (pocket centroid to ligand centroid)
    pub dca: f64,
    /// Ligand coverage: fraction of ligand atoms within threshold of pocket
    pub ligand_coverage: f64,
    /// Pocket precision: fraction of pocket atoms near ligand
    pub pocket_precision: f64,
    /// Volume overlap (Jaccard coefficient)
    pub volume_overlap: f64,
    /// Success flag (DCC < threshold)
    pub dcc_success: bool,
    /// Success flag (DCA < threshold)
    pub dca_success: bool,
    /// Pocket rank (1-indexed)
    pub pocket_rank: usize,
}

impl ValidationMetrics {
    /// Backwards compatibility: alias for dcc
    #[inline]
    pub fn center_distance(&self) -> f64 {
        self.dcc
    }

    /// Backwards compatibility: returns 1.0 if dcc_success, 0.0 otherwise
    #[inline]
    pub fn success_rate(&self) -> f64 {
        if self.dcc_success { 1.0 } else { 0.0 }
    }

    /// Compute aggregated metrics for multiple pockets (backwards compatibility)
    ///
    /// Returns metrics for the best pocket (lowest DCC) from the list.
    /// This is the old API preserved for FluxNet training.
    pub fn compute_batch(
        pockets: &[Pocket],
        ligand_coords: &[[f64; 3]],
        threshold: f64,
    ) -> Self {
        if pockets.is_empty() || ligand_coords.is_empty() {
            return Self::default();
        }

        // Find best pocket by DCC
        let mut best_metrics = Self::default();
        let mut best_dcc = f64::INFINITY;

        for (i, pocket) in pockets.iter().enumerate() {
            // Use pocket's own coordinates for computing DCC
            let pocket_coords: Vec<[f64; 3]> = pocket
                .residue_indices
                .iter()
                .map(|_| pocket.centroid) // Simplified: use centroid as representative
                .collect();

            let dcc = Self::compute_dcc_from_coords(&pocket_coords, ligand_coords, &pocket.centroid);

            if dcc < best_dcc {
                best_dcc = dcc;
                best_metrics = Self {
                    dcc,
                    dca: Self::point_distance(&pocket.centroid, &Self::compute_centroid(ligand_coords)),
                    ligand_coverage: Self::compute_coverage_simple(&pocket.centroid, pocket.volume, ligand_coords, threshold),
                    pocket_precision: 0.5, // Estimated for batch
                    volume_overlap: 0.0,
                    dcc_success: dcc < threshold,
                    dca_success: Self::point_distance(&pocket.centroid, &Self::compute_centroid(ligand_coords)) < threshold,
                    pocket_rank: i + 1,
                };
            }
        }

        best_metrics
    }

    /// Simplified DCC computation using pocket centroid
    fn compute_dcc_from_coords(pocket_coords: &[[f64; 3]], ligand_coords: &[[f64; 3]], pocket_center: &[f64; 3]) -> f64 {
        if ligand_coords.is_empty() {
            return f64::INFINITY;
        }
        // Use centroid-to-nearest-ligand as DCC estimate
        ligand_coords
            .iter()
            .map(|l| Self::point_distance(pocket_center, l))
            .fold(f64::INFINITY, f64::min)
    }

    /// Simplified coverage for batch mode
    fn compute_coverage_simple(pocket_center: &[f64; 3], pocket_volume: f64, ligand_coords: &[[f64; 3]], threshold: f64) -> f64 {
        if ligand_coords.is_empty() {
            return 0.0;
        }
        let pocket_radius = (3.0 * pocket_volume / (4.0 * std::f64::consts::PI)).cbrt();
        let coverage_radius = pocket_radius + threshold;

        let covered = ligand_coords
            .iter()
            .filter(|l| Self::point_distance(pocket_center, l) < coverage_radius)
            .count();

        covered as f64 / ligand_coords.len() as f64
    }

    /// Compute all metrics for a pocket against ligand coordinates
    pub fn compute(
        pocket: &Pocket,
        pocket_atoms: &[Atom],
        ligand_coords: &[[f64; 3]],
        threshold: f64,
    ) -> Self {
        if ligand_coords.is_empty() {
            return Self::default();
        }

        // Get pocket atom coordinates
        let pocket_coords: Vec<[f64; 3]> = pocket
            .atom_indices
            .iter()
            .filter_map(|&idx| pocket_atoms.get(idx).map(|a| a.coord))
            .collect();

        // DCC: minimum distance from any pocket atom to any ligand atom
        let dcc = Self::compute_dcc(&pocket_coords, ligand_coords);

        // DCA: distance from pocket centroid to ligand centroid
        let ligand_centroid = Self::compute_centroid(ligand_coords);
        let dca = Self::point_distance(&pocket.centroid, &ligand_centroid);

        // Coverage: fraction of ligand atoms within threshold of any pocket atom
        let ligand_coverage = Self::compute_coverage(&pocket_coords, ligand_coords, threshold);

        // Precision: fraction of pocket atoms within threshold of any ligand atom
        let pocket_precision = Self::compute_coverage(ligand_coords, &pocket_coords, threshold);

        // Volume overlap (grid-based Jaccard)
        let volume_overlap = Self::compute_volume_overlap(
            &pocket.centroid,
            pocket.volume,
            &ligand_centroid,
            ligand_coords,
        );

        Self {
            dcc,
            dca,
            ligand_coverage,
            pocket_precision,
            volume_overlap,
            dcc_success: dcc < threshold,
            dca_success: dca < threshold,
            pocket_rank: 0,
        }
    }

    /// Compute DCC (Distance to Closest Contact)
    fn compute_dcc(pocket_coords: &[[f64; 3]], ligand_coords: &[[f64; 3]]) -> f64 {
        if pocket_coords.is_empty() || ligand_coords.is_empty() {
            return f64::INFINITY;
        }

        let mut min_dist = f64::INFINITY;
        for p in pocket_coords {
            for l in ligand_coords {
                let dist = Self::point_distance(p, l);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        min_dist
    }

    /// Compute centroid of a set of coordinates
    fn compute_centroid(coords: &[[f64; 3]]) -> [f64; 3] {
        if coords.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let n = coords.len() as f64;
        let sum: [f64; 3] = coords.iter().fold([0.0, 0.0, 0.0], |acc, c| {
            [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Euclidean distance between two points
    fn point_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute coverage: fraction of target atoms within threshold of any source atom
    fn compute_coverage(source: &[[f64; 3]], target: &[[f64; 3]], threshold: f64) -> f64 {
        if target.is_empty() || source.is_empty() {
            return 0.0;
        }

        let hits = target
            .iter()
            .filter(|t| {
                source
                    .iter()
                    .any(|s| Self::point_distance(s, t) < threshold)
            })
            .count();

        hits as f64 / target.len() as f64
    }

    /// Compute volume overlap using grid-based Jaccard coefficient
    fn compute_volume_overlap(
        pocket_center: &[f64; 3],
        pocket_volume: f64,
        ligand_center: &[f64; 3],
        ligand_coords: &[[f64; 3]],
    ) -> f64 {
        // Estimate pocket radius from volume (assuming sphere)
        let pocket_radius = (3.0 * pocket_volume / (4.0 * std::f64::consts::PI)).cbrt();

        // Estimate ligand radius from bounding box
        let ligand_radius = Self::estimate_ligand_radius(ligand_coords);

        // Distance between centers
        let center_dist = Self::point_distance(pocket_center, ligand_center);

        // Calculate sphere overlap (simplified)
        if center_dist >= pocket_radius + ligand_radius {
            0.0 // No overlap
        } else if center_dist <= (pocket_radius - ligand_radius).abs() {
            // One inside the other
            let smaller_vol =
                (4.0 / 3.0) * std::f64::consts::PI * ligand_radius.min(pocket_radius).powi(3);
            let larger_vol =
                (4.0 / 3.0) * std::f64::consts::PI * ligand_radius.max(pocket_radius).powi(3);
            smaller_vol / larger_vol
        } else {
            // Partial overlap - use lens formula
            let r1 = pocket_radius;
            let r2 = ligand_radius;
            let d = center_dist;

            let overlap_vol = (std::f64::consts::PI / (12.0 * d))
                * (r1 + r2 - d).powi(2)
                * (d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1 - r2).powi(2));

            let vol1 = (4.0 / 3.0) * std::f64::consts::PI * r1.powi(3);
            let vol2 = (4.0 / 3.0) * std::f64::consts::PI * r2.powi(3);
            let union = vol1 + vol2 - overlap_vol;

            if union > 0.0 {
                (overlap_vol / union).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }

    /// Estimate ligand radius from coordinates
    fn estimate_ligand_radius(coords: &[[f64; 3]]) -> f64 {
        if coords.is_empty() {
            return 0.0;
        }
        let centroid = Self::compute_centroid(coords);
        let max_dist = coords
            .iter()
            .map(|c| Self::point_distance(c, &centroid))
            .fold(0.0, f64::max);
        max_dist + 1.5 // Add van der Waals radius
    }
}

/// Top-N success rate calculator
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopNMetrics {
    /// Top-1 success rate (best pocket)
    pub top1_dcc: f64,
    pub top1_dca: f64,
    /// Top-3 success rate
    pub top3_dcc: f64,
    pub top3_dca: f64,
    /// Top-5 success rate
    pub top5_dcc: f64,
    pub top5_dca: f64,
    /// Top-10 success rate
    pub top10_dcc: f64,
    pub top10_dca: f64,
    /// Mean DCC across all pockets
    pub mean_dcc: f64,
    /// Mean DCA across all pockets
    pub mean_dca: f64,
    /// Best DCC found
    pub best_dcc: f64,
    /// Best DCA found
    pub best_dca: f64,
}

impl TopNMetrics {
    /// Compute Top-N metrics for ranked pockets
    pub fn compute(
        pockets: &[Pocket],
        atoms: &[Atom],
        ligand_coords: &[[f64; 3]],
        threshold: f64,
    ) -> Self {
        if pockets.is_empty() || ligand_coords.is_empty() {
            return Self::default();
        }

        let mut metrics: Vec<ValidationMetrics> = pockets
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut m = ValidationMetrics::compute(p, atoms, ligand_coords, threshold);
                m.pocket_rank = i + 1;
                m
            })
            .collect();

        // Sort by DCC (best first)
        metrics.sort_by(|a, b| a.dcc.partial_cmp(&b.dcc).unwrap());

        let n = metrics.len();
        let best_dcc = metrics.first().map(|m| m.dcc).unwrap_or(f64::INFINITY);
        let best_dca = metrics
            .iter()
            .map(|m| m.dca)
            .fold(f64::INFINITY, f64::min);

        let mean_dcc = metrics.iter().map(|m| m.dcc).sum::<f64>() / n as f64;
        let mean_dca = metrics.iter().map(|m| m.dca).sum::<f64>() / n as f64;

        // Calculate Top-N success rates
        let top1_dcc = if metrics.first().map(|m| m.dcc_success).unwrap_or(false) {
            1.0
        } else {
            0.0
        };
        let top1_dca = if metrics
            .iter()
            .min_by(|a, b| a.dca.partial_cmp(&b.dca).unwrap())
            .map(|m| m.dca_success)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };

        let top3_dcc = Self::topn_success(&metrics, 3, |m| m.dcc_success);
        let top3_dca = Self::topn_success_dca(&metrics, 3, threshold);

        let top5_dcc = Self::topn_success(&metrics, 5, |m| m.dcc_success);
        let top5_dca = Self::topn_success_dca(&metrics, 5, threshold);

        let top10_dcc = Self::topn_success(&metrics, 10, |m| m.dcc_success);
        let top10_dca = Self::topn_success_dca(&metrics, 10, threshold);

        Self {
            top1_dcc,
            top1_dca,
            top3_dcc,
            top3_dca,
            top5_dcc,
            top5_dca,
            top10_dcc,
            top10_dca,
            mean_dcc,
            mean_dca,
            best_dcc,
            best_dca,
        }
    }

    fn topn_success<F>(metrics: &[ValidationMetrics], n: usize, success_fn: F) -> f64
    where
        F: Fn(&ValidationMetrics) -> bool,
    {
        let top_n = metrics.iter().take(n);
        if top_n.clone().any(|m| success_fn(m)) {
            1.0
        } else {
            0.0
        }
    }

    fn topn_success_dca(metrics: &[ValidationMetrics], n: usize, threshold: f64) -> f64 {
        let mut by_dca = metrics.to_vec();
        by_dca.sort_by(|a, b| a.dca.partial_cmp(&b.dca).unwrap());
        if by_dca.iter().take(n).any(|m| m.dca < threshold) {
            1.0
        } else {
            0.0
        }
    }
}

/// Benchmark case describing a structure and its known binding site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    /// PDB ID or structure name
    pub name: String,
    /// Known ligand coordinates (from co-crystallized ligand)
    pub ligand_coords: Vec<[f64; 3]>,
    /// Ligand residue name (e.g., "ATP", "NAD")
    pub ligand_name: Option<String>,
    /// Known binding residues (if available)
    pub binding_residues: Vec<i32>,
    /// Success threshold (default 4.0 Å)
    pub threshold: f64,
    /// Source dataset (PDBBind, scPDB, DUD-E, etc.)
    pub source: Option<String>,
}

impl BenchmarkCase {
    /// Create a new benchmark case
    pub fn new(name: impl Into<String>, ligand_coords: Vec<[f64; 3]>) -> Self {
        Self {
            name: name.into(),
            ligand_coords,
            ligand_name: None,
            binding_residues: Vec::new(),
            threshold: DEFAULT_SUCCESS_THRESHOLD,
            source: None,
        }
    }

    /// Load from XYZ file (simple coordinate format)
    pub fn from_xyz(name: impl Into<String>, path: &Path, threshold: f64) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut coords = Vec::new();
        for line in content.lines() {
            let cols: Vec<_> = line
                .split_whitespace()
                .filter_map(|c| c.parse::<f64>().ok())
                .collect();
            if cols.len() >= 3 {
                coords.push([cols[0], cols[1], cols[2]]);
            }
        }
        Ok(Self {
            name: name.into(),
            ligand_coords: coords,
            ligand_name: None,
            binding_residues: Vec::new(),
            threshold,
            source: Some("xyz".into()),
        })
    }

    /// Load all benchmark cases from a directory
    pub fn load_dir(dir: &Path, threshold: f64) -> std::io::Result<Vec<Self>> {
        let mut cases = Vec::new();
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let path = entry.path();
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_lowercase();

                    if ext == "xyz" {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("case");
                        cases.push(Self::from_xyz(name.to_string(), &path, threshold)?);
                    }
                }
            }
        }
        Ok(cases)
    }
}

/// Aggregate benchmark results across multiple cases
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Total number of benchmark cases
    pub total_cases: usize,
    /// Number of successful predictions (DCC < threshold)
    pub dcc_successes: usize,
    /// Number of successful predictions (DCA < threshold)
    pub dca_successes: usize,
    /// Overall DCC success rate
    pub dcc_success_rate: f64,
    /// Overall DCA success rate
    pub dca_success_rate: f64,
    /// Top-1 DCC success rate
    pub top1_dcc_rate: f64,
    /// Top-3 DCC success rate
    pub top3_dcc_rate: f64,
    /// Top-5 DCC success rate
    pub top5_dcc_rate: f64,
    /// Top-1 DCA success rate
    pub top1_dca_rate: f64,
    /// Top-3 DCA success rate
    pub top3_dca_rate: f64,
    /// Top-5 DCA success rate
    pub top5_dca_rate: f64,
    /// Mean DCC across all cases
    pub mean_dcc: f64,
    /// Mean DCA across all cases
    pub mean_dca: f64,
    /// Standard deviation of DCC
    pub std_dcc: f64,
    /// Standard deviation of DCA
    pub std_dca: f64,
    /// Per-case detailed results
    pub case_results: Vec<CaseResult>,
}

/// Result for a single benchmark case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub name: String,
    pub best_dcc: f64,
    pub best_dca: f64,
    pub dcc_success: bool,
    pub dca_success: bool,
    pub pockets_found: usize,
    pub top_n: TopNMetrics,
}

impl BenchmarkSummary {
    /// Evaluate multiple benchmark cases
    pub fn evaluate(
        cases: &[BenchmarkCase],
        predict_fn: impl Fn(&str) -> (Vec<Pocket>, Vec<Atom>),
    ) -> Self {
        if cases.is_empty() {
            return Self::default();
        }

        let mut case_results = Vec::new();
        let mut dcc_values = Vec::new();
        let mut dca_values = Vec::new();
        let mut top1_dcc_sum = 0.0;
        let mut top3_dcc_sum = 0.0;
        let mut top5_dcc_sum = 0.0;
        let mut top1_dca_sum = 0.0;
        let mut top3_dca_sum = 0.0;
        let mut top5_dca_sum = 0.0;

        for case in cases {
            let (pockets, atoms) = predict_fn(&case.name);

            let top_n =
                TopNMetrics::compute(&pockets, &atoms, &case.ligand_coords, case.threshold);

            dcc_values.push(top_n.best_dcc);
            dca_values.push(top_n.best_dca);

            top1_dcc_sum += top_n.top1_dcc;
            top3_dcc_sum += top_n.top3_dcc;
            top5_dcc_sum += top_n.top5_dcc;
            top1_dca_sum += top_n.top1_dca;
            top3_dca_sum += top_n.top3_dca;
            top5_dca_sum += top_n.top5_dca;

            case_results.push(CaseResult {
                name: case.name.clone(),
                best_dcc: top_n.best_dcc,
                best_dca: top_n.best_dca,
                dcc_success: top_n.best_dcc < case.threshold,
                dca_success: top_n.best_dca < case.threshold,
                pockets_found: pockets.len(),
                top_n,
            });
        }

        let n = cases.len() as f64;
        let dcc_successes = case_results.iter().filter(|r| r.dcc_success).count();
        let dca_successes = case_results.iter().filter(|r| r.dca_success).count();

        let mean_dcc = dcc_values.iter().sum::<f64>() / n;
        let mean_dca = dca_values.iter().sum::<f64>() / n;

        let std_dcc = (dcc_values.iter().map(|&x| (x - mean_dcc).powi(2)).sum::<f64>() / n).sqrt();
        let std_dca = (dca_values.iter().map(|&x| (x - mean_dca).powi(2)).sum::<f64>() / n).sqrt();

        Self {
            total_cases: cases.len(),
            dcc_successes,
            dca_successes,
            dcc_success_rate: dcc_successes as f64 / n,
            dca_success_rate: dca_successes as f64 / n,
            top1_dcc_rate: top1_dcc_sum / n,
            top3_dcc_rate: top3_dcc_sum / n,
            top5_dcc_rate: top5_dcc_sum / n,
            top1_dca_rate: top1_dca_sum / n,
            top3_dca_rate: top3_dca_sum / n,
            top5_dca_rate: top5_dca_sum / n,
            mean_dcc,
            mean_dca,
            std_dcc,
            std_dca,
            case_results,
        }
    }

    /// Generate a formatted report
    pub fn report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║              PRISM-LBS VALIDATION BENCHMARK REPORT               ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Cases: {:>4}                                               ║
╠══════════════════════════════════════════════════════════════════╣
║                        SUCCESS RATES                             ║
╠──────────────────────────────────────────────────────────────────╣
║  Metric       │  DCC (< 4Å)  │  DCA (< 4Å)                       ║
║  ─────────────┼──────────────┼─────────────                      ║
║  Top-1        │    {:>5.1}%    │    {:>5.1}%                         ║
║  Top-3        │    {:>5.1}%    │    {:>5.1}%                         ║
║  Top-5        │    {:>5.1}%    │    {:>5.1}%                         ║
╠══════════════════════════════════════════════════════════════════╣
║                       DISTANCE METRICS                           ║
╠──────────────────────────────────────────────────────────────────╣
║  Mean DCC: {:>6.2} Å  (σ = {:>5.2} Å)                              ║
║  Mean DCA: {:>6.2} Å  (σ = {:>5.2} Å)                              ║
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.total_cases,
            self.top1_dcc_rate * 100.0,
            self.top1_dca_rate * 100.0,
            self.top3_dcc_rate * 100.0,
            self.top3_dca_rate * 100.0,
            self.top5_dcc_rate * 100.0,
            self.top5_dca_rate * 100.0,
            self.mean_dcc,
            self.std_dcc,
            self.mean_dca,
            self.std_dca,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcc_computation() {
        let pocket_coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let ligand_coords = vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];

        let dcc = ValidationMetrics::compute_dcc(&pocket_coords, &ligand_coords);
        assert!((dcc - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_centroid() {
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let centroid = ValidationMetrics::compute_centroid(&coords);
        assert!((centroid[0] - 1.0).abs() < 0.001);
        assert!((centroid[1] - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_coverage() {
        let source = vec![[0.0, 0.0, 0.0]];
        let target = vec![[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]];

        let coverage = ValidationMetrics::compute_coverage(&source, &target, 2.0);
        assert!((coverage - 0.5).abs() < 0.001); // 1 of 2 within threshold
    }
}
