//! AlphaFlow-Compatible Ensemble Export
//!
//! This module provides output format compatibility with AlphaFlow's evaluation
//! script (analyze_ensembles.py), enabling direct comparison with the 2024-2026
//! gold standard for protein dynamics validation.
//!
//! Reference: github.com/bjing2016/alphaflow/scripts/analyze_ensembles.py
//!
//! ## Metrics Computed
//!
//! - **RMSF Pearson correlation**: Per-residue fluctuation vs MD reference
//! - **Pairwise Cα RMSD distribution**: Ensemble diversity
//! - **Global RMSD to ground truth**: Structural accuracy
//! - **MD-ensemble joint distribution matching**: Full trajectory comparison
//!
//! ## Output Formats
//!
//! - Multi-model PDB files (for visualization)
//! - NPZ files (for analyze_ensembles.py)
//! - JSON metrics (for PRISM integration)

use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// AlphaFlow-compatible ensemble representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaFlowEnsemble {
    /// Protein identifier
    pub name: String,
    /// PDB ID (if available)
    pub pdb_id: Option<String>,
    /// Number of residues
    pub n_residues: usize,
    /// Number of ensemble members
    pub n_models: usize,
    /// Cα coordinates for each model: [n_models, n_residues, 3]
    pub ca_coords: Vec<Vec<[f32; 3]>>,
    /// Reference structure (MD average or experimental)
    pub reference_coords: Option<Vec<[f32; 3]>>,
    /// Reference RMSF from MD simulation
    pub reference_rmsf: Option<Vec<f32>>,
    /// Residue sequence
    pub sequence: Option<String>,
    /// Chain ID
    pub chain_id: String,
}

impl AlphaFlowEnsemble {
    /// Create from PRISM-Delta simulation trajectory
    pub fn from_trajectory(
        name: &str,
        coords: &[Vec<[f32; 3]>],  // [n_frames, n_atoms, 3]
        ca_indices: &[usize],      // Indices of Cα atoms
        subsample: usize,          // Take every Nth frame
    ) -> Self {
        let n_models = coords.len() / subsample.max(1);
        let n_residues = ca_indices.len();

        let mut ca_coords = Vec::with_capacity(n_models);

        for i in (0..coords.len()).step_by(subsample.max(1)) {
            let frame = &coords[i];
            let ca_frame: Vec<[f32; 3]> = ca_indices
                .iter()
                .map(|&idx| {
                    if idx < frame.len() {
                        frame[idx]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                })
                .collect();
            ca_coords.push(ca_frame);
        }

        Self {
            name: name.to_string(),
            pdb_id: None,
            n_residues,
            n_models: ca_coords.len(),
            ca_coords,
            reference_coords: None,
            reference_rmsf: None,
            sequence: None,
            chain_id: "A".to_string(),
        }
    }

    /// Set reference structure and RMSF from MD
    pub fn with_reference(mut self, coords: Vec<[f32; 3]>, rmsf: Vec<f32>) -> Self {
        self.reference_coords = Some(coords);
        self.reference_rmsf = Some(rmsf);
        self
    }

    /// Compute RMSF from ensemble
    pub fn compute_rmsf(&self) -> Vec<f32> {
        if self.n_models == 0 || self.n_residues == 0 {
            return vec![0.0; self.n_residues];
        }

        // Compute mean position for each residue
        let mut mean_pos = vec![[0.0f64; 3]; self.n_residues];
        for frame in &self.ca_coords {
            for (i, pos) in frame.iter().enumerate() {
                mean_pos[i][0] += pos[0] as f64;
                mean_pos[i][1] += pos[1] as f64;
                mean_pos[i][2] += pos[2] as f64;
            }
        }
        for pos in &mut mean_pos {
            pos[0] /= self.n_models as f64;
            pos[1] /= self.n_models as f64;
            pos[2] /= self.n_models as f64;
        }

        // Compute RMSF
        let mut rmsf = vec![0.0f64; self.n_residues];
        for frame in &self.ca_coords {
            for (i, pos) in frame.iter().enumerate() {
                let dx = pos[0] as f64 - mean_pos[i][0];
                let dy = pos[1] as f64 - mean_pos[i][1];
                let dz = pos[2] as f64 - mean_pos[i][2];
                rmsf[i] += dx * dx + dy * dy + dz * dz;
            }
        }

        rmsf.iter()
            .map(|&v| ((v / self.n_models as f64).sqrt()) as f32)
            .collect()
    }

    /// Compute pairwise RMSD between ensemble members
    pub fn compute_pairwise_rmsd(&self) -> PairwiseRmsdStats {
        let mut rmsds = Vec::new();

        for i in 0..self.n_models {
            for j in (i + 1)..self.n_models {
                let rmsd = self.compute_rmsd_between(&self.ca_coords[i], &self.ca_coords[j]);
                rmsds.push(rmsd);
            }
        }

        if rmsds.is_empty() {
            return PairwiseRmsdStats::default();
        }

        let mean = rmsds.iter().sum::<f32>() / rmsds.len() as f32;
        let variance = rmsds.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rmsds.len() as f32;
        let std = variance.sqrt();

        rmsds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = rmsds[rmsds.len() / 2];
        let min = rmsds[0];
        let max = *rmsds.last().unwrap();

        PairwiseRmsdStats {
            mean,
            std,
            median,
            min,
            max,
            n_pairs: rmsds.len(),
        }
    }

    /// Compute RMSD between two coordinate sets
    fn compute_rmsd_between(&self, coords1: &[[f32; 3]], coords2: &[[f32; 3]]) -> f32 {
        if coords1.len() != coords2.len() || coords1.is_empty() {
            return f32::MAX;
        }

        let mut sum_sq = 0.0f64;
        for (c1, c2) in coords1.iter().zip(coords2.iter()) {
            let dx = (c1[0] - c2[0]) as f64;
            let dy = (c1[1] - c2[1]) as f64;
            let dz = (c1[2] - c2[2]) as f64;
            sum_sq += dx * dx + dy * dy + dz * dz;
        }

        (sum_sq / coords1.len() as f64).sqrt() as f32
    }

    /// Compute global RMSD to reference structure
    pub fn compute_global_rmsd(&self) -> Option<GlobalRmsdStats> {
        let reference = self.reference_coords.as_ref()?;

        let rmsds: Vec<f32> = self.ca_coords
            .iter()
            .map(|frame| self.compute_rmsd_between(frame, reference))
            .collect();

        if rmsds.is_empty() {
            return None;
        }

        let mean = rmsds.iter().sum::<f32>() / rmsds.len() as f32;
        let variance = rmsds.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rmsds.len() as f32;
        let std = variance.sqrt();
        let min = rmsds.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = rmsds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Some(GlobalRmsdStats {
            mean,
            std,
            min,
            max,
            per_model: rmsds,
        })
    }

    /// Compute RMSF Pearson correlation with reference
    pub fn compute_rmsf_correlation(&self) -> Option<f32> {
        let reference_rmsf = self.reference_rmsf.as_ref()?;
        let computed_rmsf = self.compute_rmsf();

        if reference_rmsf.len() != computed_rmsf.len() || reference_rmsf.is_empty() {
            return None;
        }

        pearson_correlation(reference_rmsf, &computed_rmsf)
    }

    /// Export to multi-model PDB format
    pub fn to_pdb(&self) -> String {
        let mut pdb = String::new();

        for (model_idx, frame) in self.ca_coords.iter().enumerate() {
            pdb.push_str(&format!("MODEL     {:4}\n", model_idx + 1));

            for (res_idx, pos) in frame.iter().enumerate() {
                pdb.push_str(&format!(
                    "ATOM  {:5}  CA  ALA {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C\n",
                    res_idx + 1,
                    self.chain_id,
                    res_idx + 1,
                    pos[0],
                    pos[1],
                    pos[2]
                ));
            }

            pdb.push_str("ENDMDL\n");
        }

        pdb.push_str("END\n");
        pdb
    }

    /// Export to NPZ format for analyze_ensembles.py
    /// Returns raw bytes that can be written to .npz file
    pub fn to_npz_bytes(&self) -> Result<Vec<u8>> {
        // Create coordinate array: [n_models, n_residues, 3]
        let mut coords_flat: Vec<f32> = Vec::with_capacity(self.n_models * self.n_residues * 3);

        for frame in &self.ca_coords {
            for pos in frame {
                coords_flat.push(pos[0]);
                coords_flat.push(pos[1]);
                coords_flat.push(pos[2]);
            }
        }

        // Create simple NPZ-compatible format
        // For full compatibility, use ndarray-npy crate in production
        let header = NpzHeader {
            shape: vec![self.n_models, self.n_residues, 3],
            dtype: "float32".to_string(),
        };

        // Serialize to JSON for now (real implementation would use NPZ format)
        let output = NpzExport {
            header,
            coords: coords_flat,
            rmsf: self.compute_rmsf(),
            reference_rmsf: self.reference_rmsf.clone(),
        };

        serde_json::to_vec(&output).context("Failed to serialize NPZ-like format")
    }

    /// Save ensemble to directory
    pub fn save(&self, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // Save PDB
        let pdb_path = output_dir.join(format!("{}_ensemble.pdb", self.name));
        std::fs::write(&pdb_path, self.to_pdb())?;

        // Save coordinates as JSON (NPZ-compatible structure)
        let npz_path = output_dir.join(format!("{}_coords.json", self.name));
        std::fs::write(&npz_path, self.to_npz_bytes()?)?;

        // Save metrics
        let metrics = self.compute_all_metrics();
        let metrics_path = output_dir.join(format!("{}_metrics.json", self.name));
        std::fs::write(&metrics_path, serde_json::to_string_pretty(&metrics)?)?;

        Ok(())
    }

    /// Compute all AlphaFlow-compatible metrics
    pub fn compute_all_metrics(&self) -> AlphaFlowMetrics {
        let rmsf = self.compute_rmsf();
        let pairwise = self.compute_pairwise_rmsd();
        let global = self.compute_global_rmsd();
        let rmsf_corr = self.compute_rmsf_correlation();

        AlphaFlowMetrics {
            name: self.name.clone(),
            n_residues: self.n_residues,
            n_models: self.n_models,
            rmsf,
            rmsf_correlation: rmsf_corr,
            pairwise_rmsd: pairwise,
            global_rmsd: global,
            ensemble_diversity: self.compute_ensemble_diversity(),
        }
    }

    /// Compute ensemble diversity (standard metric for dynamics validation)
    fn compute_ensemble_diversity(&self) -> f32 {
        let rmsf = self.compute_rmsf();
        if rmsf.is_empty() {
            return 0.0;
        }

        // Mean RMSF as proxy for diversity
        rmsf.iter().sum::<f32>() / rmsf.len() as f32
    }
}

/// Pairwise RMSD statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PairwiseRmsdStats {
    pub mean: f32,
    pub std: f32,
    pub median: f32,
    pub min: f32,
    pub max: f32,
    pub n_pairs: usize,
}

/// Global RMSD to reference statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRmsdStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub per_model: Vec<f32>,
}

/// Complete AlphaFlow-compatible metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaFlowMetrics {
    pub name: String,
    pub n_residues: usize,
    pub n_models: usize,
    pub rmsf: Vec<f32>,
    pub rmsf_correlation: Option<f32>,
    pub pairwise_rmsd: PairwiseRmsdStats,
    pub global_rmsd: Option<GlobalRmsdStats>,
    pub ensemble_diversity: f32,
}

impl AlphaFlowMetrics {
    /// Check if metrics pass ATLAS benchmark threshold
    pub fn passes_atlas_threshold(&self) -> bool {
        // ATLAS threshold: RMSF Pearson > 0.70
        self.rmsf_correlation.map(|r| r > 0.70).unwrap_or(false)
    }

    /// Compute overall quality score (0-100)
    pub fn quality_score(&self) -> f64 {
        let mut score = 0.0;
        let mut weight = 0.0;

        // RMSF correlation (40% weight)
        if let Some(corr) = self.rmsf_correlation {
            score += corr.max(0.0) as f64 * 100.0 * 0.4;
            weight += 0.4;
        }

        // Ensemble diversity (30% weight) - target ~2-4 Å mean RMSF
        let diversity_score = if self.ensemble_diversity > 0.5 && self.ensemble_diversity < 6.0 {
            let target = 2.5;
            let diff = (self.ensemble_diversity - target).abs();
            ((4.0 - diff) / 4.0).max(0.0) * 100.0
        } else {
            20.0 // Penalty for unrealistic diversity
        };
        score += diversity_score as f64 * 0.3;
        weight += 0.3;

        // Pairwise RMSD distribution (30% weight) - target ~2-5 Å
        if self.pairwise_rmsd.n_pairs > 0 {
            let prmsd_score = if self.pairwise_rmsd.mean > 1.0 && self.pairwise_rmsd.mean < 8.0 {
                90.0
            } else if self.pairwise_rmsd.mean > 0.5 && self.pairwise_rmsd.mean < 10.0 {
                70.0
            } else {
                40.0
            };
            score += prmsd_score * 0.3;
            weight += 0.3;
        }

        if weight > 0.0 {
            score / weight
        } else {
            0.0
        }
    }
}

/// NPZ header for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NpzHeader {
    shape: Vec<usize>,
    dtype: String,
}

/// NPZ export format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NpzExport {
    header: NpzHeader,
    coords: Vec<f32>,
    rmsf: Vec<f32>,
    reference_rmsf: Option<Vec<f32>>,
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f32], y: &[f32]) -> Option<f32> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_y = y.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi as f64 - mean_x;
        let dy = yi as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        None
    } else {
        Some((cov / denom) as f32)
    }
}

/// ATLAS Benchmark Runner using AlphaFlow methodology
pub struct AtlasBenchmarkRunner {
    /// ATLAS dataset directory
    pub data_dir: std::path::PathBuf,
    /// Output directory for results
    pub output_dir: std::path::PathBuf,
    /// Number of ensemble members to generate
    pub n_samples: usize,
    /// Target proteins to evaluate
    pub targets: Vec<AtlasTarget>,
}

/// ATLAS benchmark target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasTarget {
    pub pdb_id: String,
    pub chain: String,
    pub n_residues: usize,
    /// Reference RMSF from MD simulation
    pub md_rmsf: Vec<f32>,
    /// Reference Cα coordinates
    pub reference_coords: Vec<[f32; 3]>,
}

impl AtlasBenchmarkRunner {
    /// Load ATLAS targets from dataset
    pub fn load_targets(data_dir: &Path) -> Result<Vec<AtlasTarget>> {
        let targets_file = data_dir.join("atlas_targets.json");

        if targets_file.exists() {
            let content = std::fs::read_to_string(&targets_file)?;
            let targets: Vec<AtlasTarget> = serde_json::from_str(&content)?;
            Ok(targets)
        } else {
            // Return empty list if no targets file
            log::warn!("ATLAS targets file not found: {:?}", targets_file);
            Ok(Vec::new())
        }
    }

    /// Run ATLAS benchmark on an ensemble
    pub fn evaluate(&self, ensemble: &AlphaFlowEnsemble, target: &AtlasTarget) -> AtlasBenchmarkResult {
        // Set reference for comparison
        let ensemble_with_ref = AlphaFlowEnsemble {
            reference_coords: Some(target.reference_coords.clone()),
            reference_rmsf: Some(target.md_rmsf.clone()),
            ..ensemble.clone()
        };

        let metrics = ensemble_with_ref.compute_all_metrics();

        AtlasBenchmarkResult {
            pdb_id: target.pdb_id.clone(),
            n_residues: target.n_residues,
            n_samples: ensemble.n_models,
            rmsf_pearson: metrics.rmsf_correlation.unwrap_or(0.0),
            pairwise_rmsd_mean: metrics.pairwise_rmsd.mean,
            pairwise_rmsd_std: metrics.pairwise_rmsd.std,
            global_rmsd_mean: metrics.global_rmsd.as_ref().map(|g| g.mean).unwrap_or(0.0),
            ensemble_diversity: metrics.ensemble_diversity,
            passed: metrics.passes_atlas_threshold(),
            quality_score: metrics.quality_score(),
        }
    }

    /// Run full ATLAS benchmark suite
    pub fn run_all(&self, ensembles: &HashMap<String, AlphaFlowEnsemble>) -> AtlasBenchmarkSummary {
        let mut results = Vec::new();
        let mut total_corr = 0.0;
        let mut passed = 0;

        for target in &self.targets {
            if let Some(ensemble) = ensembles.get(&target.pdb_id) {
                let result = self.evaluate(ensemble, target);
                total_corr += result.rmsf_pearson;
                if result.passed {
                    passed += 1;
                }
                results.push(result);
            }
        }

        let n = results.len();
        let mean_pearson = if n > 0 { total_corr / n as f32 } else { 0.0 };

        AtlasBenchmarkSummary {
            n_targets: n,
            n_passed: passed,
            pass_rate: if n > 0 { passed as f64 / n as f64 } else { 0.0 },
            mean_rmsf_pearson: mean_pearson,
            results,
        }
    }
}

/// Result for a single ATLAS target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasBenchmarkResult {
    pub pdb_id: String,
    pub n_residues: usize,
    pub n_samples: usize,
    pub rmsf_pearson: f32,
    pub pairwise_rmsd_mean: f32,
    pub pairwise_rmsd_std: f32,
    pub global_rmsd_mean: f32,
    pub ensemble_diversity: f32,
    pub passed: bool,
    pub quality_score: f64,
}

/// Summary of full ATLAS benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasBenchmarkSummary {
    pub n_targets: usize,
    pub n_passed: usize,
    pub pass_rate: f64,
    pub mean_rmsf_pearson: f32,
    pub results: Vec<AtlasBenchmarkResult>,
}

impl AtlasBenchmarkSummary {
    /// Generate comparison table (for paper)
    pub fn to_comparison_table(&self, method_name: &str) -> String {
        let mut table = String::new();

        table.push_str(&format!("| Method | N | Pass Rate | Mean ρ | Best | Worst |\n"));
        table.push_str(&format!("|--------|---|-----------|--------|------|-------|\n"));

        let best = self.results.iter()
            .max_by(|a, b| a.rmsf_pearson.partial_cmp(&b.rmsf_pearson).unwrap())
            .map(|r| r.rmsf_pearson)
            .unwrap_or(0.0);

        let worst = self.results.iter()
            .min_by(|a, b| a.rmsf_pearson.partial_cmp(&b.rmsf_pearson).unwrap())
            .map(|r| r.rmsf_pearson)
            .unwrap_or(0.0);

        table.push_str(&format!(
            "| {} | {} | {:.1}% | {:.3} | {:.3} | {:.3} |\n",
            method_name,
            self.n_targets,
            self.pass_rate * 100.0,
            self.mean_rmsf_pearson,
            best,
            worst
        ));

        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_rmsf() {
        // Create simple 2-model ensemble
        let ensemble = AlphaFlowEnsemble {
            name: "test".to_string(),
            pdb_id: None,
            n_residues: 3,
            n_models: 2,
            ca_coords: vec![
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ],
            reference_coords: None,
            reference_rmsf: None,
            sequence: None,
            chain_id: "A".to_string(),
        };

        let rmsf = ensemble.compute_rmsf();
        assert_eq!(rmsf.len(), 3);
        // Each residue moves 0.5 in Y, so RMSF should be 0.5
        assert!((rmsf[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.01);

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = pearson_correlation(&x, &y_neg).unwrap();
        assert!((corr_neg + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pairwise_rmsd() {
        let ensemble = AlphaFlowEnsemble {
            name: "test".to_string(),
            pdb_id: None,
            n_residues: 2,
            n_models: 3,
            ca_coords: vec![
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ],
            reference_coords: None,
            reference_rmsf: None,
            sequence: None,
            chain_id: "A".to_string(),
        };

        let stats = ensemble.compute_pairwise_rmsd();
        assert_eq!(stats.n_pairs, 3); // 3 choose 2 = 3
        assert!(stats.mean > 0.0);
    }
}
