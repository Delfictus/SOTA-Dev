//! Apo-Holo Conformational Change Benchmark
//!
//! Tests PRISM's ability to predict conformational changes by:
//! 1. Starting from apo (ligand-free) structure
//! 2. Sampling with HybridSampler (NOVA or AMBER backend)
//! 3. Checking if any conformation approaches holo (ligand-bound) state
//!
//! This is the "killer demo" - proving PRISM can predict conformational changes
//! that reveal cryptic binding sites.
//!
//! # Success Criteria
//!
//! Success thresholds vary by motion type:
//! - SmallRotation: <1.5Å
//! - LoopMotion: <2.0Å
//! - HingeMotion: <2.5Å
//! - DomainRotation: <3.0Å
//! - DomainClosure: <3.5Å
//!
//! # Usage
//!
//! ```ignore
//! use prism_validation::apo_holo_benchmark::{ApoHoloBenchmark, APO_HOLO_PAIRS};
//!
//! let mut benchmark = ApoHoloBenchmark::new("data/benchmarks/apo_holo");
//! let summary = benchmark.run_all()?;
//! println!("Success rate: {:.0}%", summary.success_rate * 100.0);
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::pdb_sanitizer::{sanitize_pdb, SanitizedStructure};
use crate::sampling::{HybridSampler, RoutingStrategy, SamplingConfig};

/// Classic apo-holo pairs with known conformational changes
///
/// These 15 pairs represent well-characterized conformational changes
/// in protein structures, covering different motion types.
pub const APO_HOLO_PAIRS: &[ApoHoloPair] = &[
    ApoHoloPair {
        apo: "1AKE",
        holo: "4AKE",
        name: "Adenylate kinase",
        motion: MotionType::DomainClosure,
    },
    ApoHoloPair {
        apo: "2LAO",
        holo: "1LST",
        name: "Lysine-binding protein",
        motion: MotionType::HingeMotion,
    },
    ApoHoloPair {
        apo: "1GGG",
        holo: "1WDN",
        name: "Calmodulin",
        motion: MotionType::DomainRotation,
    },
    ApoHoloPair {
        apo: "1OMP",
        holo: "1ANF",
        name: "Maltose-binding protein",
        motion: MotionType::DomainClosure,
    },
    ApoHoloPair {
        apo: "1RX2",
        holo: "1RX4",
        name: "Ribonuclease",
        motion: MotionType::LoopMotion,
    },
    ApoHoloPair {
        apo: "3CHY",
        holo: "2CHE",
        name: "CheY",
        motion: MotionType::SmallRotation,
    },
    ApoHoloPair {
        apo: "1EX6",
        holo: "1EX7",
        name: "Galectin",
        motion: MotionType::LoopMotion,
    },
    ApoHoloPair {
        apo: "1STP",
        holo: "1SWB",
        name: "Streptavidin",
        motion: MotionType::LoopMotion,
    },
    ApoHoloPair {
        apo: "1AJJ",
        holo: "1AJK",
        name: "Guanylate kinase",
        motion: MotionType::DomainClosure,
    },
    ApoHoloPair {
        apo: "1PHP",
        holo: "1PHN",
        name: "Phosphotransferase",
        motion: MotionType::HingeMotion,
    },
    ApoHoloPair {
        apo: "1BTL",
        holo: "1BTM",
        name: "Beta-lactamase",
        motion: MotionType::SmallRotation,
    },
    ApoHoloPair {
        apo: "2CPL",
        holo: "1CWA",
        name: "Cyclophilin",
        motion: MotionType::LoopMotion,
    },
    ApoHoloPair {
        apo: "1BMD",
        holo: "1BMC",
        name: "Biotin-binding",
        motion: MotionType::LoopMotion,
    },
    ApoHoloPair {
        apo: "1URN",
        holo: "1URP",
        name: "Ubiquitin",
        motion: MotionType::SmallRotation,
    },
    ApoHoloPair {
        apo: "1HOE",
        holo: "1HOF",
        name: "Alpha-amylase inhibitor",
        motion: MotionType::LoopMotion,
    },
];

/// An apo-holo pair definition
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ApoHoloPair {
    /// PDB ID of apo (ligand-free) structure
    pub apo: &'static str,
    /// PDB ID of holo (ligand-bound) structure
    pub holo: &'static str,
    /// Protein name
    pub name: &'static str,
    /// Type of conformational motion
    pub motion: MotionType,
}

/// Type of conformational motion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionType {
    /// Large hinge-bending motion (>5Å displacement)
    DomainClosure,
    /// Rigid body rotation between domains
    DomainRotation,
    /// Classic hinge movement
    HingeMotion,
    /// Flexible loop rearrangement
    LoopMotion,
    /// Minor conformational shifts (<2Å)
    SmallRotation,
}

impl MotionType {
    /// Success threshold (min RMSD to holo) for this motion type
    ///
    /// Larger motions have more lenient thresholds since they're
    /// harder to predict accurately.
    pub fn success_threshold(&self) -> f32 {
        match self {
            MotionType::SmallRotation => 1.5,
            MotionType::LoopMotion => 2.0,
            MotionType::HingeMotion => 2.5,
            MotionType::DomainRotation => 3.0,
            MotionType::DomainClosure => 3.5,
        }
    }

    /// Display name for the motion type
    pub fn display_name(&self) -> &'static str {
        match self {
            MotionType::SmallRotation => "Small Rotation",
            MotionType::LoopMotion => "Loop Motion",
            MotionType::HingeMotion => "Hinge Motion",
            MotionType::DomainRotation => "Domain Rotation",
            MotionType::DomainClosure => "Domain Closure",
        }
    }
}

impl std::fmt::Display for MotionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Result for a single apo-holo validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApoHoloResult {
    /// PDB ID of apo structure
    pub apo_pdb: String,
    /// PDB ID of holo structure
    pub holo_pdb: String,
    /// Protein name
    pub name: String,
    /// Type of conformational motion
    pub motion_type: MotionType,

    /// Starting RMSD (apo vs holo)
    pub apo_holo_rmsd: f32,

    /// Best (minimum) RMSD to holo achieved during sampling
    pub min_rmsd_to_holo: f32,

    /// Sample index that achieved best RMSD
    pub best_sample_idx: usize,

    /// Improvement: apo_holo_rmsd - min_rmsd_to_holo
    pub rmsd_improvement: f32,

    /// Did we approach holo state? (min_rmsd < threshold)
    pub success: bool,

    /// Sample index of first significant improvement (30% closer to holo)
    pub time_to_open: Option<usize>,

    /// Number of residues compared
    pub n_residues: usize,

    /// All RMSD values to holo (trajectory)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rmsd_trajectory: Vec<f32>,
}

impl ApoHoloResult {
    /// Check if this result shows significant improvement
    pub fn shows_improvement(&self) -> bool {
        self.rmsd_improvement > 0.5
    }

    /// Get fraction of trajectory showing improvement
    pub fn fraction_improved(&self) -> f32 {
        if self.rmsd_trajectory.is_empty() {
            return 0.0;
        }
        let improved = self.rmsd_trajectory.iter()
            .filter(|&&r| r < self.apo_holo_rmsd)
            .count();
        improved as f32 / self.rmsd_trajectory.len() as f32
    }
}

/// Summary of apo-holo benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApoHoloBenchmarkSummary {
    /// Total number of pairs tested
    pub n_total: usize,
    /// Number of successful predictions
    pub n_success: usize,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    /// Mean RMSD improvement across all pairs
    pub mean_rmsd_improvement: f32,
    /// Mean minimum RMSD to holo achieved
    pub mean_min_rmsd_to_holo: f32,
    /// Individual results for each pair
    pub results: Vec<ApoHoloResult>,
    /// Results grouped by motion type
    pub by_motion_type: MotionTypeStats,
}

/// Statistics grouped by motion type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionTypeStats {
    pub small_rotation: MotionStats,
    pub loop_motion: MotionStats,
    pub hinge_motion: MotionStats,
    pub domain_rotation: MotionStats,
    pub domain_closure: MotionStats,
}

/// Statistics for a single motion type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionStats {
    pub n_total: usize,
    pub n_success: usize,
    pub mean_improvement: f32,
}

impl ApoHoloBenchmarkSummary {
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Apo-Holo Benchmark Results\n\n");
        s.push_str(&format!(
            "**Success Rate**: {}/{} ({:.0}%)\n\n",
            self.n_success,
            self.n_total,
            self.success_rate * 100.0
        ));
        s.push_str(&format!(
            "**Mean Improvement**: {:.2} \u{212B}\n",
            self.mean_rmsd_improvement
        ));
        s.push_str(&format!(
            "**Mean Min RMSD**: {:.2} \u{212B}\n\n",
            self.mean_min_rmsd_to_holo
        ));

        s.push_str("## Results by Motion Type\n\n");
        s.push_str("| Motion Type | Success | Mean Improvement |\n");
        s.push_str("|-------------|---------|------------------|\n");
        self.add_motion_row(&mut s, "Small Rotation", &self.by_motion_type.small_rotation);
        self.add_motion_row(&mut s, "Loop Motion", &self.by_motion_type.loop_motion);
        self.add_motion_row(&mut s, "Hinge Motion", &self.by_motion_type.hinge_motion);
        self.add_motion_row(&mut s, "Domain Rotation", &self.by_motion_type.domain_rotation);
        self.add_motion_row(&mut s, "Domain Closure", &self.by_motion_type.domain_closure);
        s.push('\n');

        s.push_str("## Individual Results\n\n");
        s.push_str("| Protein | Motion | Apo\u{2192}Holo | Best | \u{0394} | Status |\n");
        s.push_str("|---------|--------|----------|------|-----|--------|\n");
        for r in &self.results {
            s.push_str(&format!(
                "| {} | {} | {:.2}\u{212B} | {:.2}\u{212B} | {:.2}\u{212B} | {} |\n",
                r.name,
                r.motion_type.display_name(),
                r.apo_holo_rmsd,
                r.min_rmsd_to_holo,
                r.rmsd_improvement,
                if r.success { "\u{2713}" } else { "\u{2717}" }
            ));
        }
        s
    }

    fn add_motion_row(&self, s: &mut String, name: &str, stats: &MotionStats) {
        if stats.n_total > 0 {
            s.push_str(&format!(
                "| {} | {}/{} | {:.2}\u{212B} |\n",
                name, stats.n_success, stats.n_total, stats.mean_improvement
            ));
        }
    }

    /// Generate LaTeX table
    pub fn to_latex(&self) -> String {
        let mut s = String::new();
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\caption{Apo-Holo Conformational Change Prediction}\n");
        s.push_str("\\label{tab:apo_holo}\n");
        s.push_str("\\begin{tabular}{lcccc}\n\\toprule\n");
        s.push_str("Protein & Apo$\\to$Holo & Min RMSD & $\\Delta$ & Success \\\\\n");
        s.push_str("\\midrule\n");

        for r in &self.results {
            let name_escaped = r.name.replace('_', "\\_");
            s.push_str(&format!(
                "{} & {:.2}\\AA & {:.2}\\AA & {:.2}\\AA & {} \\\\\n",
                name_escaped,
                r.apo_holo_rmsd,
                r.min_rmsd_to_holo,
                r.rmsd_improvement,
                if r.success { "\\checkmark" } else { "$\\times$" }
            ));
        }

        s.push_str("\\midrule\n");
        s.push_str(&format!(
            "\\textbf{{Total}} & & {:.2}\\AA & {:.2}\\AA & {:.0}\\% \\\\\n",
            self.mean_min_rmsd_to_holo,
            self.mean_rmsd_improvement,
            self.success_rate * 100.0
        ));
        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }

    /// Save results to JSON file
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize benchmark summary")?;
        std::fs::write(path, json).context("Failed to write JSON file")?;
        Ok(())
    }
}

/// Configuration for apo-holo benchmark
#[derive(Debug, Clone)]
pub struct ApoHoloBenchmarkConfig {
    /// Directory containing apo/holo PDB files
    pub data_dir: String,
    /// Sampling configuration
    pub sampling_config: SamplingConfig,
    /// Routing strategy (Auto, ForceNova, ForceAmber)
    pub routing_strategy: RoutingStrategy,
    /// Whether to store full RMSD trajectories
    pub store_trajectories: bool,
}

impl Default for ApoHoloBenchmarkConfig {
    fn default() -> Self {
        Self {
            data_dir: "data/benchmarks/apo_holo".to_string(),
            sampling_config: SamplingConfig::default(),
            routing_strategy: RoutingStrategy::Auto,
            store_trajectories: true,
        }
    }
}

/// Apo-Holo Benchmark Runner
///
/// Runs conformational sampling from apo structures and measures
/// how close the sampled conformations get to the holo state.
pub struct ApoHoloBenchmark {
    config: ApoHoloBenchmarkConfig,
    results: Vec<ApoHoloResult>,
}

impl ApoHoloBenchmark {
    /// Create new benchmark runner with default configuration
    pub fn new(data_dir: &str) -> Self {
        Self {
            config: ApoHoloBenchmarkConfig {
                data_dir: data_dir.to_string(),
                ..Default::default()
            },
            results: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ApoHoloBenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Set sampling configuration
    pub fn with_sampling_config(mut self, config: SamplingConfig) -> Self {
        self.config.sampling_config = config;
        self
    }

    /// Set routing strategy
    pub fn with_routing_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.config.routing_strategy = strategy;
        self
    }

    /// Run benchmark on all pairs (mock version for testing)
    ///
    /// This version uses the mock sampler and doesn't require GPU.
    pub fn run_all_mock(&mut self) -> Result<ApoHoloBenchmarkSummary> {
        log::info!(
            "Starting apo-holo benchmark (mock) on {} pairs",
            APO_HOLO_PAIRS.len()
        );

        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);

        for pair in APO_HOLO_PAIRS {
            match self.run_pair_with_sampler(pair, &mut sampler) {
                Ok(result) => {
                    log::info!(
                        "  {} {}: {:.2}\u{212B} \u{2192} {:.2}\u{212B} ({})",
                        if result.success { "\u{2713}" } else { "\u{2717}" },
                        pair.name,
                        result.apo_holo_rmsd,
                        result.min_rmsd_to_holo,
                        if result.success { "SUCCESS" } else { "FAILED" }
                    );
                    self.results.push(result);
                }
                Err(e) => {
                    log::error!("  \u{2717} {} FAILED: {}", pair.name, e);
                }
            }
        }

        Ok(self.summarize())
    }

    /// Run benchmark on all pairs with GPU sampler
    #[cfg(feature = "cryptic-gpu")]
    pub fn run_all(&mut self, context: std::sync::Arc<cudarc::driver::CudaContext>) -> Result<ApoHoloBenchmarkSummary> {
        log::info!(
            "Starting apo-holo benchmark on {} pairs",
            APO_HOLO_PAIRS.len()
        );

        let mut sampler = HybridSampler::new(context)?
            .with_strategy(self.config.routing_strategy);

        for pair in APO_HOLO_PAIRS {
            match self.run_pair_with_sampler(pair, &mut sampler) {
                Ok(result) => {
                    log::info!(
                        "  {} {}: {:.2}\u{212B} \u{2192} {:.2}\u{212B} ({})",
                        if result.success { "\u{2713}" } else { "\u{2717}" },
                        pair.name,
                        result.apo_holo_rmsd,
                        result.min_rmsd_to_holo,
                        if result.success { "SUCCESS" } else { "FAILED" }
                    );
                    self.results.push(result);
                }
                Err(e) => {
                    log::error!("  \u{2717} {} FAILED: {}", pair.name, e);
                }
            }

            // Reset sampler for next structure
            sampler.reset()?;
        }

        Ok(self.summarize())
    }

    /// Run benchmark on a single pair
    pub fn run_single_mock(&mut self, pair: &ApoHoloPair) -> Result<ApoHoloResult> {
        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);
        self.run_pair_with_sampler(pair, &mut sampler)
    }

    /// Run a single pair with provided sampler
    fn run_pair_with_sampler(
        &self,
        pair: &ApoHoloPair,
        sampler: &mut HybridSampler,
    ) -> Result<ApoHoloResult> {
        // Load PDB files
        let apo_path = format!("{}/{}_apo.pdb", self.config.data_dir, pair.apo);
        let holo_path = format!("{}/{}_holo.pdb", self.config.data_dir, pair.holo);

        let apo_content =
            std::fs::read_to_string(&apo_path).context(format!("Failed to read {}", apo_path))?;
        let holo_content = std::fs::read_to_string(&holo_path)
            .context(format!("Failed to read {}", holo_path))?;

        // Sanitize structures
        let apo_struct = sanitize_pdb(&apo_content, pair.apo)?;
        let holo_struct = sanitize_pdb(&holo_content, pair.holo)?;

        // Get Cα coordinates
        let apo_ca = apo_struct.get_ca_coords();
        let holo_ca = holo_struct.get_ca_coords();

        // Align lengths (use minimum common residues)
        let n = apo_ca.len().min(holo_ca.len());
        if n < 10 {
            anyhow::bail!(
                "Too few common residues ({}) for {} vs {}",
                n,
                pair.apo,
                pair.holo
            );
        }

        let apo_ca: Vec<[f32; 3]> = apo_ca.into_iter().take(n).collect();
        let holo_ca: Vec<[f32; 3]> = holo_ca.into_iter().take(n).collect();

        // Baseline RMSD (apo vs holo)
        let apo_holo_rmsd = compute_rmsd(&apo_ca, &holo_ca);

        // Load structure into sampler and sample
        sampler.load_structure(&apo_struct)?;
        let sampling_result = sampler.sample(&self.config.sampling_config)?;

        // Compute RMSD to holo for each sampled conformation
        let rmsd_trajectory: Vec<f32> = sampling_result
            .conformations
            .iter()
            .map(|conf| {
                // Take only the first n atoms (Cα aligned)
                let trimmed: Vec<[f32; 3]> = conf.iter().take(n).copied().collect();
                if trimmed.len() == n {
                    compute_rmsd(&trimmed, &holo_ca)
                } else {
                    apo_holo_rmsd // fallback if mismatch
                }
            })
            .collect();

        // Find best (minimum) RMSD to holo
        let (best_idx, min_rmsd) = rmsd_trajectory
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &r)| (i, r))
            .unwrap_or((0, apo_holo_rmsd));

        let rmsd_improvement = apo_holo_rmsd - min_rmsd;
        let success = min_rmsd < pair.motion.success_threshold();

        // Find time to first significant opening (30% closer to holo)
        let opening_threshold = apo_holo_rmsd * 0.7;
        let time_to_open = rmsd_trajectory.iter().position(|&r| r < opening_threshold);

        Ok(ApoHoloResult {
            apo_pdb: pair.apo.to_string(),
            holo_pdb: pair.holo.to_string(),
            name: pair.name.to_string(),
            motion_type: pair.motion,
            apo_holo_rmsd,
            min_rmsd_to_holo: min_rmsd,
            best_sample_idx: best_idx,
            rmsd_improvement,
            success,
            time_to_open,
            n_residues: n,
            rmsd_trajectory: if self.config.store_trajectories {
                rmsd_trajectory
            } else {
                Vec::new()
            },
        })
    }

    /// Generate summary from results
    pub fn summarize(&self) -> ApoHoloBenchmarkSummary {
        let n_total = self.results.len();
        let n_success = self.results.iter().filter(|r| r.success).count();

        let mean_improvement = if n_total > 0 {
            self.results.iter().map(|r| r.rmsd_improvement).sum::<f32>() / n_total as f32
        } else {
            0.0
        };

        let mean_min_rmsd = if n_total > 0 {
            self.results
                .iter()
                .map(|r| r.min_rmsd_to_holo)
                .sum::<f32>()
                / n_total as f32
        } else {
            0.0
        };

        // Group by motion type
        let mut by_motion_type = MotionTypeStats::default();
        for r in &self.results {
            let stats = match r.motion_type {
                MotionType::SmallRotation => &mut by_motion_type.small_rotation,
                MotionType::LoopMotion => &mut by_motion_type.loop_motion,
                MotionType::HingeMotion => &mut by_motion_type.hinge_motion,
                MotionType::DomainRotation => &mut by_motion_type.domain_rotation,
                MotionType::DomainClosure => &mut by_motion_type.domain_closure,
            };
            stats.n_total += 1;
            if r.success {
                stats.n_success += 1;
            }
            stats.mean_improvement += r.rmsd_improvement;
        }

        // Finalize means
        if by_motion_type.small_rotation.n_total > 0 {
            by_motion_type.small_rotation.mean_improvement /=
                by_motion_type.small_rotation.n_total as f32;
        }
        if by_motion_type.loop_motion.n_total > 0 {
            by_motion_type.loop_motion.mean_improvement /=
                by_motion_type.loop_motion.n_total as f32;
        }
        if by_motion_type.hinge_motion.n_total > 0 {
            by_motion_type.hinge_motion.mean_improvement /=
                by_motion_type.hinge_motion.n_total as f32;
        }
        if by_motion_type.domain_rotation.n_total > 0 {
            by_motion_type.domain_rotation.mean_improvement /=
                by_motion_type.domain_rotation.n_total as f32;
        }
        if by_motion_type.domain_closure.n_total > 0 {
            by_motion_type.domain_closure.mean_improvement /=
                by_motion_type.domain_closure.n_total as f32;
        }

        ApoHoloBenchmarkSummary {
            n_total,
            n_success,
            success_rate: n_success as f32 / n_total.max(1) as f32,
            mean_rmsd_improvement: mean_improvement,
            mean_min_rmsd_to_holo: mean_min_rmsd,
            results: self.results.clone(),
            by_motion_type,
        }
    }

    /// Get current results
    pub fn results(&self) -> &[ApoHoloResult] {
        &self.results
    }

    /// Clear results for reuse
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

/// Compute RMSD between two conformations (Cα coordinates)
///
/// Uses standard RMSD formula without superposition.
/// For proper comparison, structures should be pre-aligned.
fn compute_rmsd(conf1: &[[f32; 3]], conf2: &[[f32; 3]]) -> f32 {
    if conf1.len() != conf2.len() || conf1.is_empty() {
        return 0.0;
    }

    let n = conf1.len() as f32;
    let sum_sq: f32 = conf1
        .iter()
        .zip(conf2.iter())
        .map(|(a, b)| {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    (sum_sq / n).sqrt()
}

/// Compute RMSD with Kabsch alignment (superposition)
///
/// This provides a more accurate comparison by first
/// optimally superposing the structures.
#[allow(dead_code)]
fn compute_rmsd_aligned(conf1: &[[f32; 3]], conf2: &[[f32; 3]]) -> f32 {
    if conf1.len() != conf2.len() || conf1.is_empty() {
        return 0.0;
    }

    // Center both conformations
    let center1 = compute_centroid(conf1);
    let center2 = compute_centroid(conf2);

    let centered1: Vec<[f32; 3]> = conf1
        .iter()
        .map(|p| [p[0] - center1[0], p[1] - center1[1], p[2] - center1[2]])
        .collect();

    let centered2: Vec<[f32; 3]> = conf2
        .iter()
        .map(|p| [p[0] - center2[0], p[1] - center2[1], p[2] - center2[2]])
        .collect();

    // For now, use simple RMSD after centering
    // Full Kabsch would require SVD which adds complexity
    compute_rmsd(&centered1, &centered2)
}

/// Compute centroid of a conformation
fn compute_centroid(conf: &[[f32; 3]]) -> [f32; 3] {
    if conf.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = conf.len() as f32;
    let sum: [f32; 3] = conf.iter().fold([0.0, 0.0, 0.0], |acc, p| {
        [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
    });

    [sum[0] / n, sum[1] / n, sum[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_type_thresholds() {
        assert_eq!(MotionType::SmallRotation.success_threshold(), 1.5);
        assert_eq!(MotionType::LoopMotion.success_threshold(), 2.0);
        assert_eq!(MotionType::HingeMotion.success_threshold(), 2.5);
        assert_eq!(MotionType::DomainRotation.success_threshold(), 3.0);
        assert_eq!(MotionType::DomainClosure.success_threshold(), 3.5);
    }

    #[test]
    fn test_motion_type_display() {
        assert_eq!(MotionType::DomainClosure.display_name(), "Domain Closure");
        assert_eq!(format!("{}", MotionType::LoopMotion), "Loop Motion");
    }

    #[test]
    fn test_apo_holo_pairs_count() {
        assert_eq!(APO_HOLO_PAIRS.len(), 15);
    }

    #[test]
    fn test_apo_holo_pairs_unique() {
        let apos: Vec<_> = APO_HOLO_PAIRS.iter().map(|p| p.apo).collect();
        let holos: Vec<_> = APO_HOLO_PAIRS.iter().map(|p| p.holo).collect();

        // All apo IDs should be unique
        let mut unique_apos = apos.clone();
        unique_apos.sort();
        unique_apos.dedup();
        assert_eq!(unique_apos.len(), apos.len());

        // All holo IDs should be unique
        let mut unique_holos = holos.clone();
        unique_holos.sort();
        unique_holos.dedup();
        assert_eq!(unique_holos.len(), holos.len());
    }

    #[test]
    fn test_compute_rmsd_identical() {
        let conf = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!((compute_rmsd(&conf, &conf) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rmsd_different() {
        let conf1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let conf2 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let rmsd = compute_rmsd(&conf1, &conf2);
        // Expected: sqrt((0 + 1) / 2) = sqrt(0.5) ≈ 0.707
        assert!((rmsd - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_compute_centroid() {
        let conf = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let centroid = compute_centroid(&conf);
        assert!((centroid[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((centroid[1] - 1.0).abs() < 1e-6);
        assert!((centroid[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apo_holo_result_shows_improvement() {
        let result = ApoHoloResult {
            apo_pdb: "1AKE".to_string(),
            holo_pdb: "4AKE".to_string(),
            name: "Test".to_string(),
            motion_type: MotionType::DomainClosure,
            apo_holo_rmsd: 5.0,
            min_rmsd_to_holo: 3.0,
            best_sample_idx: 10,
            rmsd_improvement: 2.0,
            success: true,
            time_to_open: Some(5),
            n_residues: 100,
            rmsd_trajectory: vec![5.0, 4.5, 4.0, 3.5, 3.0],
        };

        assert!(result.shows_improvement());
        assert_eq!(result.fraction_improved(), 0.8); // 4 out of 5 improved
    }

    #[test]
    fn test_benchmark_summary_empty() {
        let summary = ApoHoloBenchmarkSummary {
            n_total: 0,
            n_success: 0,
            success_rate: 0.0,
            mean_rmsd_improvement: 0.0,
            mean_min_rmsd_to_holo: 0.0,
            results: Vec::new(),
            by_motion_type: MotionTypeStats::default(),
        };

        let md = summary.to_markdown();
        assert!(md.contains("Success Rate"));
        assert!(md.contains("0/0"));

        let latex = summary.to_latex();
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("\\end{table}"));
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = ApoHoloBenchmarkConfig::default();
        assert_eq!(config.data_dir, "data/benchmarks/apo_holo");
        assert!(config.store_trajectories);
        assert!(matches!(config.routing_strategy, RoutingStrategy::Auto));
    }

    #[test]
    fn test_benchmark_new() {
        let benchmark = ApoHoloBenchmark::new("custom/path");
        assert_eq!(benchmark.config.data_dir, "custom/path");
        assert!(benchmark.results.is_empty());
    }

    #[test]
    fn test_motion_stats_default() {
        let stats = MotionStats::default();
        assert_eq!(stats.n_total, 0);
        assert_eq!(stats.n_success, 0);
        assert_eq!(stats.mean_improvement, 0.0);
    }
}
