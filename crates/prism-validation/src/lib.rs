//! PRISM-4D Validation Framework
//!
//! Multi-tier validation for dynamics and cryptic pocket discovery,
//! specifically designed to demonstrate capabilities beyond AlphaFold3.
//!
//! ## Validation Tiers
//!
//! 1. **ATLAS Ensemble Recovery**: Validate conformational sampling against NMR/MD ensembles
//! 2. **Apo-Holo Transition**: Predict cryptic pocket opening from apo structures
//! 3. **Retrospective Blind**: Validate against real drug discovery cases
//! 4. **Novel Cryptic Benchmark**: New benchmark for dynamics-based pocket discovery
//!
//! ## Design Philosophy
//!
//! - **No mock data**: All validation against real experimental data
//! - **Blind protocols**: Test data never seen during development
//! - **Head-to-head AF3 comparison**: Demonstrate clear differentiation
//! - **Pharma-relevant metrics**: Focus on what matters for drug discovery

pub mod benchmarks;
pub mod metrics;
pub mod targets;
pub mod comparison;
pub mod reports;
pub mod data_curation;
pub mod pipeline;
pub mod benchmark_integration;
pub mod alphaflow_compat;
pub mod prism_bench;

// GNM-enhanced cryptic site detection
pub mod cryptic_sites;

// Scientific integrity and audit trail (BLAKE3 hashing)
pub mod integrity;

// Cryptic site validation metrics (PR AUC, Success Rate, Ranking)
pub mod cryptic_metrics;

// Native physics-based druggability scoring
pub mod druggability;

// PocketMiner benchmark dataset handling
pub mod pocketminer_dataset;

// ANM-based conformational ensemble generation
pub mod anm_ensemble;

// ANM-based conformational ensemble generation v2 (enhanced for cryptic detection)
pub mod anm_ensemble_v2;

// Ensemble pocket detector for cryptic site identification
pub mod ensemble_pocket_detector;

// Ensemble pocket detector v2 (adaptive threshold, dynamic prior, graph clustering)
pub mod ensemble_pocket_detector_v2;

// HMC-refined ensemble generation (Phase 2.1)
// Requires prism-physics for AmberSimulator
#[cfg(feature = "cryptic")]
pub mod hmc_refined_ensemble;

// PRISM-ZrO cryptic site scorer (Phase 2.2)
// Requires prism-gpu for DendriticSNNReservoir (GPU-accelerated 512-neuron reservoir)
#[cfg(feature = "cryptic-gpu")]
pub mod prism_zro_cryptic_scorer;

// Runtime observability for benchmark reports
pub mod observability;

// Kabsch alignment for accurate displacement computation (Phase 4.1)
pub mod kabsch_alignment;

// Escape resistance scoring for cryptic binding sites (Phase 4.2)
pub mod escape_resistance_scorer;

// Blind validation pipeline for 6VXX and 2VWD (Phase 4)
pub mod blind_validation_pipeline;

// Oligomer topology: biological assembly parsing and interface detection
pub mod oligomer_topology;

// Antibody binding site validation (m102.4 and others)
pub mod antibody_validation;

// Production-quality cryptic detection (requires 'cryptic' feature)
#[cfg(feature = "cryptic")]
pub mod cryptic_production;

// GPU-accelerated spatial clustering for site merging
pub mod spatial_clustering_gpu;

// TDA-guided conformational sampling for cryptic site detection (Phase 5.1)
// Uses void proxies (burial variance, neighbor fluctuations) instead of full Betti-2
pub mod tda_guided_sampling;

// PRISM-ZrO SNN-based adaptive cryptic scoring (Phase 5.3)
// Uses reservoir computing with RLS readout for online learning
pub mod zro_cryptic_integration;

// Phase 6: Cryptic site feature vectors (16-dim + velocity)
// Foundation for GPU-accelerated ZrO scorer
pub mod cryptic_features;

// Phase 6: GPU-accelerated ZrO scorer (512-neuron reservoir + RLS)
// Requires CUDA - will NOT fall back to CPU
#[cfg(feature = "cryptic-gpu")]
pub mod gpu_zro_cryptic_scorer;

// Heterogeneous data acquisition (ATLAS, NMR, MISATO)
pub mod data_acquisition;

#[cfg(feature = "simulation")]
pub mod simulation_runner;

// Re-export AlphaFlow compatibility for easy access
pub use alphaflow_compat::{
    AlphaFlowEnsemble, AlphaFlowMetrics, AtlasBenchmarkRunner,
    AtlasBenchmarkResult, AtlasBenchmarkSummary, AtlasTarget,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use chrono::{DateTime, Utc};

/// Result from running a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub benchmark: String,
    /// Target protein name
    pub target: String,
    /// PDB ID (apo for transitions)
    pub pdb_id: String,
    /// When the benchmark was run
    pub timestamp: DateTime<Utc>,
    /// Time taken in seconds
    pub duration_secs: f64,
    /// Number of simulation steps
    pub steps: usize,
    /// Raw metrics from the run
    pub metrics: BenchmarkMetrics,
    /// Whether the benchmark passed
    pub passed: bool,
    /// Pass/fail reason
    pub reason: String,
}

/// Metrics computed during benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    // Structural metrics
    /// RMSD to target (if available)
    pub rmsd_to_target: Option<f32>,
    /// RMSD of pocket region only
    pub pocket_rmsd: Option<f32>,
    /// SASA of binding site
    pub pocket_sasa: Option<f32>,
    /// SASA gain from starting structure
    pub sasa_gain: Option<f32>,

    // Topological metrics (TDA)
    /// Betti-0: connected components
    pub betti_0: Option<f32>,
    /// Betti-1: cycles
    pub betti_1: Option<f32>,
    /// Betti-2: voids (POCKETS!)
    pub betti_2: Option<f32>,
    /// Persistence entropy
    pub persistence_entropy: Option<f32>,
    /// Pocket signature from TDA
    pub pocket_signature: Option<f32>,

    // Dynamics metrics
    /// Root mean square fluctuation
    pub rmsf: Option<Vec<f32>>,
    /// RMSF correlation with experiment
    pub rmsf_correlation: Option<f32>,
    /// Pairwise RMSD distribution
    pub pairwise_rmsd_mean: Option<f32>,
    pub pairwise_rmsd_std: Option<f32>,
    /// Principal component overlap
    pub pc_overlap: Option<f32>,

    // Transition metrics
    /// Steps until pocket first opened
    pub steps_to_opening: Option<usize>,
    /// Fraction of trajectory with pocket open
    pub pocket_stability: Option<f32>,
    /// Did pocket open and close (reversible)?
    pub reversible: Option<bool>,

    // Active Inference metrics
    /// Expected free energy at end
    pub final_efe: Option<f32>,
    /// Goal prior (druggability belief)
    pub final_goal_prior: Option<f32>,

    // Energy metrics
    /// HMC acceptance rate
    pub acceptance_rate: Option<f32>,
    /// Final system energy
    pub final_energy: Option<f32>,

    // Custom metrics (extensible)
    pub custom: std::collections::HashMap<String, f64>,
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            rmsd_to_target: None,
            pocket_rmsd: None,
            pocket_sasa: None,
            sasa_gain: None,
            betti_0: None,
            betti_1: None,
            betti_2: None,
            persistence_entropy: None,
            pocket_signature: None,
            rmsf: None,
            rmsf_correlation: None,
            pairwise_rmsd_mean: None,
            pairwise_rmsd_std: None,
            pc_overlap: None,
            steps_to_opening: None,
            pocket_stability: None,
            reversible: None,
            final_efe: None,
            final_goal_prior: None,
            acceptance_rate: None,
            final_energy: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// Validation score (0-100)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationScore {
    /// Overall score (0-100)
    pub overall: f64,
    /// Component scores
    pub components: Vec<ScoreComponent>,
    /// Grade (A/B/C/D/F)
    pub grade: String,
    /// Comparison to expected performance
    pub vs_expected: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComponent {
    pub name: String,
    pub score: f64,
    pub weight: f64,
    pub description: String,
}

impl ValidationScore {
    pub fn compute(components: Vec<ScoreComponent>) -> Self {
        let total_weight: f64 = components.iter().map(|c| c.weight).sum();
        let weighted_sum: f64 = components.iter().map(|c| c.score * c.weight).sum();
        let overall = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let grade = match overall {
            s if s >= 90.0 => "A",
            s if s >= 80.0 => "B",
            s if s >= 70.0 => "C",
            s if s >= 60.0 => "D",
            _ => "F",
        }
        .to_string();

        let vs_expected = match overall {
            s if s >= 80.0 => "Exceeds expectations",
            s if s >= 60.0 => "Meets expectations",
            s if s >= 40.0 => "Below expectations",
            _ => "Significantly below expectations",
        }
        .to_string();

        Self {
            overall,
            components,
            grade,
            vs_expected,
        }
    }
}

/// Comparison between PRISM-NOVA and AlphaFold3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Af3Comparison {
    /// Target protein
    pub target: String,
    /// PRISM-NOVA result
    pub prism_metrics: BenchmarkMetrics,
    /// AlphaFold3 result (if available)
    pub af3_metrics: Option<BenchmarkMetrics>,
    /// Comparison summary
    pub comparison: Vec<ComparisonItem>,
    /// Overall winner
    pub winner: String,
    /// Advantage magnitude
    pub advantage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonItem {
    pub metric: String,
    pub prism_value: f64,
    pub af3_value: Option<f64>,
    pub winner: String,
    pub significance: String,
}

/// Trait for all validation benchmarks
pub trait ValidationBenchmark: Send + Sync {
    /// Benchmark name
    fn name(&self) -> &str;

    /// Run the benchmark on a target
    fn run(&self, target: &targets::Target) -> Result<BenchmarkResult>;

    /// Score the result
    fn score(&self, result: &BenchmarkResult) -> ValidationScore;

    /// Compare with AF3 result
    fn compare_af3(&self, result: &BenchmarkResult, af3_result: Option<&BenchmarkMetrics>) -> Af3Comparison;
}

/// Configuration for validation runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Path to validation data
    pub data_dir: PathBuf,
    /// Output directory for results
    pub output_dir: PathBuf,
    /// Number of simulation steps per target
    pub steps_per_target: usize,
    /// Temperature for dynamics
    pub temperature: f32,
    /// Whether to run AF3 comparison
    pub compare_af3: bool,
    /// Benchmarks to run
    pub benchmarks: Vec<String>,
    /// GPU device index
    pub gpu_device: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data/validation"),
            output_dir: PathBuf::from("validation_results"),
            steps_per_target: 10000,
            temperature: 310.0,
            compare_af3: true,
            benchmarks: vec![
                "atlas".to_string(),
                "apo_holo".to_string(),
                "retrospective".to_string(),
            ],
            gpu_device: 0,
        }
    }
}

/// Summary of validation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// When validation started
    pub started: DateTime<Utc>,
    /// When validation finished
    pub finished: DateTime<Utc>,
    /// Configuration used
    pub config: ValidationConfig,
    /// Results per benchmark
    pub benchmark_summaries: Vec<BenchmarkSummary>,
    /// Overall pass rate
    pub overall_pass_rate: f64,
    /// Overall score
    pub overall_score: f64,
    /// AF3 comparison summary (if run)
    pub af3_summary: Option<Af3ComparisonSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub benchmark: String,
    pub targets_run: usize,
    pub targets_passed: usize,
    pub pass_rate: f64,
    pub mean_score: f64,
    pub std_score: f64,
    pub best_target: String,
    pub worst_target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Af3ComparisonSummary {
    pub targets_compared: usize,
    pub prism_wins: usize,
    pub af3_wins: usize,
    pub ties: usize,
    pub prism_advantage_mean: f64,
    pub key_differentiators: Vec<String>,
}

/// Main validation runner
pub struct ValidationRunner {
    config: ValidationConfig,
    benchmarks: Vec<Box<dyn ValidationBenchmark>>,
}

impl ValidationRunner {
    /// Create a new validation runner
    pub fn new(config: ValidationConfig) -> Result<Self> {
        let mut benchmarks: Vec<Box<dyn ValidationBenchmark>> = Vec::new();

        for name in &config.benchmarks {
            match name.as_str() {
                "atlas" => benchmarks.push(Box::new(benchmarks::AtlasBenchmark::new(&config)?)),
                "apo_holo" => benchmarks.push(Box::new(benchmarks::ApoHoloBenchmark::new(&config)?)),
                "retrospective" => benchmarks.push(Box::new(benchmarks::RetrospectiveBenchmark::new(&config)?)),
                "novel" => benchmarks.push(Box::new(benchmarks::NovelCrypticBenchmark::new(&config)?)),
                _ => log::warn!("Unknown benchmark: {}", name),
            }
        }

        Ok(Self { config, benchmarks })
    }

    /// Run all configured benchmarks
    pub fn run_all(&self) -> Result<ValidationSummary> {
        let started = Utc::now();
        let mut benchmark_summaries = Vec::new();
        let mut all_results = Vec::new();

        for benchmark in &self.benchmarks {
            log::info!("Running benchmark: {}", benchmark.name());

            let targets = self.load_targets_for_benchmark(benchmark.name())?;
            let mut passed = 0;
            let mut scores = Vec::new();
            let mut best_score = 0.0;
            let mut worst_score = 100.0;
            let mut best_target = String::new();
            let mut worst_target = String::new();

            for target in &targets {
                log::info!("  Target: {}", target.name);

                match benchmark.run(target) {
                    Ok(result) => {
                        let score = benchmark.score(&result);

                        if result.passed {
                            passed += 1;
                        }

                        if score.overall > best_score {
                            best_score = score.overall;
                            best_target = target.name.clone();
                        }
                        if score.overall < worst_score {
                            worst_score = score.overall;
                            worst_target = target.name.clone();
                        }

                        scores.push(score.overall);
                        all_results.push(result);
                    }
                    Err(e) => {
                        log::error!("  Failed: {}", e);
                    }
                }
            }

            let mean_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            };

            let std_score = if scores.len() > 1 {
                let variance: f64 = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>()
                    / (scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            benchmark_summaries.push(BenchmarkSummary {
                benchmark: benchmark.name().to_string(),
                targets_run: targets.len(),
                targets_passed: passed,
                pass_rate: if targets.is_empty() {
                    0.0
                } else {
                    passed as f64 / targets.len() as f64
                },
                mean_score,
                std_score,
                best_target,
                worst_target,
            });
        }

        let finished = Utc::now();

        let overall_pass_rate = if all_results.is_empty() {
            0.0
        } else {
            all_results.iter().filter(|r| r.passed).count() as f64 / all_results.len() as f64
        };

        let overall_score = if benchmark_summaries.is_empty() {
            0.0
        } else {
            benchmark_summaries.iter().map(|s| s.mean_score).sum::<f64>()
                / benchmark_summaries.len() as f64
        };

        Ok(ValidationSummary {
            started,
            finished,
            config: self.config.clone(),
            benchmark_summaries,
            overall_pass_rate,
            overall_score,
            af3_summary: None, // TODO: Implement AF3 comparison
        })
    }

    fn load_targets_for_benchmark(&self, benchmark_name: &str) -> Result<Vec<targets::Target>> {
        let targets_file = self.config.data_dir.join(benchmark_name).join("targets.json");

        if targets_file.exists() {
            let content = std::fs::read_to_string(&targets_file)?;
            let targets: Vec<targets::Target> = serde_json::from_str(&content)?;
            Ok(targets)
        } else {
            log::warn!("No targets file found for {}: {:?}", benchmark_name, targets_file);
            Ok(Vec::new())
        }
    }

    /// Save results to output directory
    pub fn save_results(&self, summary: &ValidationSummary) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.config.output_dir)?;

        let filename = format!(
            "validation_{}_{}.json",
            summary.started.format("%Y%m%d_%H%M%S"),
            summary.overall_score as i32
        );
        let path = self.config.output_dir.join(filename);

        let content = serde_json::to_string_pretty(summary)?;
        std::fs::write(&path, content)?;

        log::info!("Saved validation results to {:?}", path);
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_score_compute() {
        let components = vec![
            ScoreComponent {
                name: "RMSF Correlation".to_string(),
                score: 85.0,
                weight: 1.0,
                description: "Correlation with experimental RMSF".to_string(),
            },
            ScoreComponent {
                name: "Pocket RMSD".to_string(),
                score: 75.0,
                weight: 1.0,
                description: "RMSD of pocket region".to_string(),
            },
        ];

        let score = ValidationScore::compute(components);
        assert!((score.overall - 80.0).abs() < 0.01);
        assert_eq!(score.grade, "B");
    }

    #[test]
    fn test_benchmark_metrics_default() {
        let metrics = BenchmarkMetrics::default();
        assert!(metrics.rmsd_to_target.is_none());
        assert!(metrics.betti_2.is_none());
    }
}
