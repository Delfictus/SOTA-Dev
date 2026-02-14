//! PDBBind/scPDB Benchmark Runner
//!
//! Automated validation against standard binding site prediction datasets:
//! - PDBBind refined set (high-quality protein-ligand complexes)
//! - scPDB (druggable binding sites)
//! - DUD-E (drug-like molecules for docking)
//!
//! Generates industry-standard validation reports with DCC/DCA metrics.

use super::ligand_parser::{Ligand, LigandParser};
use super::metrics::{BenchmarkCase, BenchmarkSummary, TopNMetrics, DEFAULT_SUCCESS_THRESHOLD};
use crate::pocket::Pocket;
use crate::structure::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// PDBBind benchmark runner
pub struct PDBBindBenchmark {
    /// Root directory containing PDBBind data
    pub data_dir: PathBuf,
    /// Success threshold (default 4.0 Å)
    pub threshold: f64,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Maximum structures to evaluate (None = all)
    pub max_structures: Option<usize>,
}

impl PDBBindBenchmark {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            threshold: DEFAULT_SUCCESS_THRESHOLD,
            num_workers: 4,
            max_structures: None,
        }
    }

    /// Set success threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Limit number of structures
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_structures = Some(limit);
        self
    }

    /// Discover available PDB IDs in the dataset
    pub fn discover_structures(&self) -> std::io::Result<Vec<String>> {
        let mut pdb_ids = Vec::new();

        if !self.data_dir.exists() {
            return Ok(pdb_ids);
        }

        for entry in fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // PDB IDs are 4 characters
                if name_str.len() == 4 && name_str.chars().all(|c| c.is_alphanumeric()) {
                    pdb_ids.push(name_str.to_lowercase());
                }
            }
        }

        // Apply limit if set
        if let Some(limit) = self.max_structures {
            pdb_ids.truncate(limit);
        }

        Ok(pdb_ids)
    }

    /// Load a benchmark case from PDBBind directory structure
    pub fn load_case(&self, pdb_id: &str) -> std::io::Result<Option<BenchmarkCase>> {
        let case_dir = self.data_dir.join(pdb_id);
        if !case_dir.exists() {
            return Ok(None);
        }

        // Try different ligand file naming conventions
        let ligand_paths = [
            case_dir.join(format!("{}_ligand.mol2", pdb_id)),
            case_dir.join(format!("{}_ligand.sdf", pdb_id)),
            case_dir.join("ligand.mol2"),
            case_dir.join("ligand.sdf"),
        ];

        let mut ligand: Option<Ligand> = None;
        for path in &ligand_paths {
            if path.exists() {
                if let Ok(ligands) = LigandParser::parse(path) {
                    ligand = ligands.into_iter().next();
                    if ligand.is_some() {
                        break;
                    }
                }
            }
        }

        // Also try to extract from protein PDB
        if ligand.is_none() {
            let pdb_paths = [
                case_dir.join(format!("{}_protein.pdb", pdb_id)),
                case_dir.join(format!("{}_pocket.pdb", pdb_id)),
                case_dir.join("protein.pdb"),
            ];

            for path in &pdb_paths {
                if path.exists() {
                    if let Ok(ligands) = LigandParser::parse(path) {
                        ligand = ligands.into_iter().next();
                        if ligand.is_some() {
                            break;
                        }
                    }
                }
            }
        }

        match ligand {
            Some(l) => Ok(Some(BenchmarkCase {
                name: pdb_id.to_uppercase(),
                ligand_coords: l.heavy_atom_coordinates(),
                ligand_name: Some(l.name),
                binding_residues: Vec::new(),
                threshold: self.threshold,
                source: Some("PDBBind".to_string()),
            })),
            None => Ok(None),
        }
    }

    /// Run benchmark on all discovered structures
    pub fn run<F>(&self, predict_fn: F) -> std::io::Result<BenchmarkReport>
    where
        F: Fn(&Path) -> Option<(Vec<Pocket>, Vec<Atom>)>,
    {
        let start = Instant::now();
        let pdb_ids = self.discover_structures()?;

        log::info!(
            "[BENCHMARK] Starting PDBBind validation on {} structures",
            pdb_ids.len()
        );

        let mut results = Vec::new();
        let mut failed = Vec::new();

        for (i, pdb_id) in pdb_ids.iter().enumerate() {
            log::debug!(
                "[{}/{}] Processing {}",
                i + 1,
                pdb_ids.len(),
                pdb_id.to_uppercase()
            );

            // Load benchmark case
            let case = match self.load_case(pdb_id)? {
                Some(c) => c,
                None => {
                    failed.push((pdb_id.clone(), "No ligand found".to_string()));
                    continue;
                }
            };

            // Find protein PDB file
            let case_dir = self.data_dir.join(pdb_id);
            let pdb_paths = [
                case_dir.join(format!("{}_protein.pdb", pdb_id)),
                case_dir.join("protein.pdb"),
            ];

            let pdb_path = pdb_paths.iter().find(|p| p.exists());
            if pdb_path.is_none() {
                failed.push((pdb_id.clone(), "No protein PDB found".to_string()));
                continue;
            }

            // Run prediction
            let prediction = predict_fn(pdb_path.unwrap());
            if prediction.is_none() {
                failed.push((pdb_id.clone(), "Prediction failed".to_string()));
                continue;
            }

            let (pockets, atoms) = prediction.unwrap();

            // Calculate metrics
            let top_n = TopNMetrics::compute(&pockets, &atoms, &case.ligand_coords, self.threshold);

            results.push(CaseEvaluation {
                pdb_id: pdb_id.to_uppercase(),
                ligand_name: case.ligand_name.clone(),
                num_pockets: pockets.len(),
                best_dcc: top_n.best_dcc,
                best_dca: top_n.best_dca,
                dcc_success: top_n.best_dcc < self.threshold,
                dca_success: top_n.best_dca < self.threshold,
                top_n,
            });
        }

        let elapsed = start.elapsed();

        // Calculate summary statistics
        let total = results.len();
        let dcc_successes = results.iter().filter(|r| r.dcc_success).count();
        let dca_successes = results.iter().filter(|r| r.dca_success).count();

        let mean_dcc = results.iter().map(|r| r.best_dcc).sum::<f64>() / total.max(1) as f64;
        let mean_dca = results.iter().map(|r| r.best_dca).sum::<f64>() / total.max(1) as f64;

        let top1_dcc =
            results.iter().map(|r| r.top_n.top1_dcc).sum::<f64>() / total.max(1) as f64;
        let top3_dcc =
            results.iter().map(|r| r.top_n.top3_dcc).sum::<f64>() / total.max(1) as f64;
        let top5_dcc =
            results.iter().map(|r| r.top_n.top5_dcc).sum::<f64>() / total.max(1) as f64;

        Ok(BenchmarkReport {
            dataset: "PDBBind".to_string(),
            total_structures: pdb_ids.len(),
            successful_predictions: total,
            failed_predictions: failed.len(),
            threshold: self.threshold,
            dcc_success_rate: dcc_successes as f64 / total.max(1) as f64,
            dca_success_rate: dca_successes as f64 / total.max(1) as f64,
            top1_dcc_rate: top1_dcc,
            top3_dcc_rate: top3_dcc,
            top5_dcc_rate: top5_dcc,
            mean_dcc,
            mean_dca,
            elapsed_time: elapsed,
            case_results: results,
            failures: failed,
        })
    }
}

/// Evaluation result for a single case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseEvaluation {
    pub pdb_id: String,
    pub ligand_name: Option<String>,
    pub num_pockets: usize,
    pub best_dcc: f64,
    pub best_dca: f64,
    pub dcc_success: bool,
    pub dca_success: bool,
    pub top_n: TopNMetrics,
}

/// Complete benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Dataset name
    pub dataset: String,
    /// Total structures attempted
    pub total_structures: usize,
    /// Successfully processed
    pub successful_predictions: usize,
    /// Failed to process
    pub failed_predictions: usize,
    /// Success threshold used
    pub threshold: f64,
    /// DCC success rate (DCC < threshold)
    pub dcc_success_rate: f64,
    /// DCA success rate (DCA < threshold)
    pub dca_success_rate: f64,
    /// Top-1 DCC success rate
    pub top1_dcc_rate: f64,
    /// Top-3 DCC success rate
    pub top3_dcc_rate: f64,
    /// Top-5 DCC success rate
    pub top5_dcc_rate: f64,
    /// Mean DCC across all cases
    pub mean_dcc: f64,
    /// Mean DCA across all cases
    pub mean_dca: f64,
    /// Total time elapsed
    #[serde(with = "duration_serde")]
    pub elapsed_time: Duration,
    /// Per-case results
    pub case_results: Vec<CaseEvaluation>,
    /// Failed cases
    pub failures: Vec<(String, String)>,
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

impl BenchmarkReport {
    /// Generate a formatted text report
    pub fn report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRISM-LBS BENCHMARK VALIDATION REPORT                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dataset: {:<20}  Threshold: {:.1} Å                                ║
║  Structures: {:>4} total │ {:>4} success │ {:>4} failed                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              SUCCESS RATES                                   ║
╠──────────────────────────────────────────────────────────────────────────────╣
║                   │    DCC < {:.1}Å    │    DCA < {:.1}Å                       ║
║  ─────────────────┼─────────────────┼─────────────────                       ║
║  Overall          │     {:>6.1}%      │     {:>6.1}%                           ║
║  Top-1 Pocket     │     {:>6.1}%      │       --                             ║
║  Top-3 Pockets    │     {:>6.1}%      │       --                             ║
║  Top-5 Pockets    │     {:>6.1}%      │       --                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            DISTANCE METRICS                                  ║
╠──────────────────────────────────────────────────────────────────────────────╣
║  Mean DCC: {:>6.2} Å        Mean DCA: {:>6.2} Å                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Time: {:.2}s ({:.2} structures/sec)                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"#,
            self.dataset,
            self.threshold,
            self.total_structures,
            self.successful_predictions,
            self.failed_predictions,
            self.threshold,
            self.threshold,
            self.dcc_success_rate * 100.0,
            self.dca_success_rate * 100.0,
            self.top1_dcc_rate * 100.0,
            self.top3_dcc_rate * 100.0,
            self.top5_dcc_rate * 100.0,
            self.mean_dcc,
            self.mean_dca,
            self.elapsed_time.as_secs_f64(),
            self.successful_predictions as f64 / self.elapsed_time.as_secs_f64().max(0.001),
        )
    }

    /// Generate JSON report
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Save report to file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        fs::write(path, json)
    }

    /// Compare with another benchmark (e.g., fpocket baseline)
    pub fn compare(&self, baseline: &BenchmarkReport) -> BenchmarkComparison {
        BenchmarkComparison {
            prism_dcc_rate: self.dcc_success_rate,
            baseline_dcc_rate: baseline.dcc_success_rate,
            dcc_improvement: self.dcc_success_rate - baseline.dcc_success_rate,
            prism_dca_rate: self.dca_success_rate,
            baseline_dca_rate: baseline.dca_success_rate,
            dca_improvement: self.dca_success_rate - baseline.dca_success_rate,
            prism_mean_dcc: self.mean_dcc,
            baseline_mean_dcc: baseline.mean_dcc,
            mean_dcc_improvement: baseline.mean_dcc - self.mean_dcc, // Lower is better
        }
    }
}

/// Comparison between PRISM and baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub prism_dcc_rate: f64,
    pub baseline_dcc_rate: f64,
    pub dcc_improvement: f64,
    pub prism_dca_rate: f64,
    pub baseline_dca_rate: f64,
    pub dca_improvement: f64,
    pub prism_mean_dcc: f64,
    pub baseline_mean_dcc: f64,
    pub mean_dcc_improvement: f64,
}

impl BenchmarkComparison {
    pub fn report(&self) -> String {
        let dcc_status = if self.dcc_improvement > 0.0 { "+" } else { "" };
        let dca_status = if self.dca_improvement > 0.0 { "+" } else { "" };
        let mean_status = if self.mean_dcc_improvement > 0.0 {
            "+"
        } else {
            ""
        };

        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║                   BENCHMARK COMPARISON REPORT                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Metric          │  PRISM   │  Baseline  │  Improvement           ║
║  ────────────────┼──────────┼────────────┼────────────            ║
║  DCC Success     │  {:>5.1}%  │   {:>5.1}%   │  {}{:>5.1}%              ║
║  DCA Success     │  {:>5.1}%  │   {:>5.1}%   │  {}{:>5.1}%              ║
║  Mean DCC (Å)    │  {:>6.2}  │   {:>6.2}   │  {}{:>5.2} Å            ║
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.prism_dcc_rate * 100.0,
            self.baseline_dcc_rate * 100.0,
            dcc_status,
            self.dcc_improvement * 100.0,
            self.prism_dca_rate * 100.0,
            self.baseline_dca_rate * 100.0,
            dca_status,
            self.dca_improvement * 100.0,
            self.prism_mean_dcc,
            self.baseline_mean_dcc,
            mean_status,
            self.mean_dcc_improvement,
        )
    }
}

/// Quick validation on a single structure
pub fn validate_single(
    pdb_path: &Path,
    ligand_path: Option<&Path>,
    pockets: &[Pocket],
    atoms: &[Atom],
    threshold: f64,
) -> std::io::Result<TopNMetrics> {
    // Extract ligand coordinates
    let ligand_coords = if let Some(lig_path) = ligand_path {
        let ligands = LigandParser::parse(lig_path)?;
        ligands
            .first()
            .map(|l| l.heavy_atom_coordinates())
            .unwrap_or_default()
    } else {
        // Try to extract from PDB
        let ligands = LigandParser::parse_pdb(pdb_path)?;
        ligands
            .first()
            .map(|l| l.heavy_atom_coordinates())
            .unwrap_or_default()
    };

    if ligand_coords.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No ligand coordinates found",
        ));
    }

    Ok(TopNMetrics::compute(pockets, atoms, &ligand_coords, threshold))
}
