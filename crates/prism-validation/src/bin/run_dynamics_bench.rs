//! PRISM Dynamics Benchmark Runner - TRUE Experimental RMSF
//!
//! Uses NMR ensemble coordinate variance for REAL experimental flexibility,
//! NOT B-factor proxies which are corrupted by:
//! - Crystal packing artifacts
//! - Static disorder
//! - Resolution dependence
//!
//! # Data Sources
//!
//! 1. **NMR Ensembles**: Multi-model PDB files where RMSF = sqrt(variance across models)
//! 2. **MD Trajectory Databases**: Pre-computed true RMSF from validated simulations
//!
//! # Usage
//!
//! ```bash
//! # Download curated NMR ensembles first:
//! cargo run --release -p prism-validation --bin run-dynamics-bench -- --download-nmr
//!
//! # Run benchmark with true RMSF:
//! cargo run --release -p prism-validation --bin run-dynamics-bench -- \
//!     --mode enhanced-gnm --data-dir data/nmr_benchmark
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::{info, warn, error};
use serde::Serialize;

use prism_physics::{
    DynamicsEngine, DynamicsConfig, DynamicsMode, StructureInput,
    NmrEnsemble, load_nmr_ensemble, CURATED_NMR_PDBS,
};

/// Dynamics mode for CLI
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDynamicsMode {
    /// Enhanced GNM with structural weighting (fast, accurate)
    EnhancedGnm,
    /// All-atom AMBER ff14SB with HMC (detailed dynamics)
    AllAtom,
    /// Coarse-grained ANM with HMC (quick sampling)
    CoarseGrained,
    /// Transfer Entropy + GNM fusion (allosteric)
    TransferEntropy,
    /// ML-corrected GNM (maximum accuracy)
    MlCorrected,
    /// Run all modes and compare
    All,
}

impl From<CliDynamicsMode> for Option<DynamicsMode> {
    fn from(mode: CliDynamicsMode) -> Self {
        match mode {
            CliDynamicsMode::EnhancedGnm => Some(DynamicsMode::EnhancedGnm),
            CliDynamicsMode::AllAtom => Some(DynamicsMode::AllAtomAmber),
            CliDynamicsMode::CoarseGrained => Some(DynamicsMode::CoarseGrainedAnm),
            CliDynamicsMode::TransferEntropy => Some(DynamicsMode::TransferEntropyGnm),
            CliDynamicsMode::MlCorrected => Some(DynamicsMode::MlCorrectedGnm),
            CliDynamicsMode::All => None,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "run-dynamics-bench")]
#[command(about = "PRISM Dynamics Benchmark - TRUE Experimental RMSF from NMR Ensembles")]
struct Args {
    /// Data directory with NMR ensemble PDB files
    #[arg(long, default_value = "data/nmr_benchmark")]
    data_dir: PathBuf,

    /// Dynamics mode to benchmark
    #[arg(long, default_value = "enhanced-gnm")]
    mode: CliDynamicsMode,

    /// Limit number of proteins
    #[arg(long)]
    limit: Option<usize>,

    /// GNM cutoff distance in Angstroms
    #[arg(long, default_value = "7.3")]
    gnm_cutoff: f64,

    /// Temperature in Kelvin (for MD modes)
    #[arg(long, default_value = "310.0")]
    temperature: f32,

    /// Number of simulation steps (for MD modes)
    #[arg(long, default_value = "1000")]
    n_steps: usize,

    /// Output directory
    #[arg(long, default_value = "dynamics_bench_results")]
    output: PathBuf,

    /// Download curated NMR ensembles from RCSB PDB
    #[arg(long)]
    download_nmr: bool,

    /// Minimum number of NMR models required (default: 5)
    #[arg(long, default_value = "5")]
    min_models: usize,

    /// Use Kabsch alignment before computing RMSF
    #[arg(long)]
    align_models: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Per-protein benchmark result
#[derive(Debug, Clone, Serialize)]
struct ProteinResult {
    name: String,
    pdb_id: String,
    n_residues: usize,
    n_nmr_models: usize,
    mode: String,
    pearson_correlation: f64,
    spearman_correlation: f64,
    mean_predicted_rmsf: f64,
    mean_experimental_rmsf: f64,
    max_experimental_rmsf: f64,
    computation_time_ms: u64,
    passed: bool,
}

/// Summary for a single mode
#[derive(Debug, Clone, Serialize)]
struct ModeSummary {
    mode: String,
    n_proteins: usize,
    mean_pearson: f64,
    mean_spearman: f64,
    std_pearson: f64,
    pass_rate: f64,
    total_time_ms: u64,
    proteins: Vec<ProteinResult>,
}

/// Overall benchmark summary
#[derive(Debug, Clone, Serialize)]
struct BenchmarkSummary {
    dataset: String,
    data_source: String,
    n_proteins_total: usize,
    modes: Vec<ModeSummary>,
    best_mode: String,
    best_correlation: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    info!("╔═══════════════════════════════════════════════════════════════════════════╗");
    info!("║   PRISM Dynamics Benchmark - TRUE Experimental RMSF (NMR Ensembles)       ║");
    info!("╚═══════════════════════════════════════════════════════════════════════════╝");
    info!("");
    info!("  📊 Data source: NMR ensemble coordinate variance (NO B-factors!)");
    info!("  📁 NMR directory: {:?}", args.data_dir);
    info!("");

    // Create output directory
    fs::create_dir_all(&args.output)?;
    fs::create_dir_all(&args.data_dir)?;

    // Download NMR ensembles if requested
    if args.download_nmr {
        download_curated_nmr_ensembles(&args.data_dir).await?;
        return Ok(());
    }

    // Load NMR ensembles
    let ensembles = load_all_nmr_ensembles(&args.data_dir, args.min_models)?;

    if ensembles.is_empty() {
        error!("No NMR ensembles found in {:?}", args.data_dir);
        error!("");
        error!("To download curated NMR ensembles, run:");
        error!("  cargo run --release -p prism-validation --bin run-dynamics-bench -- --download-nmr");
        error!("");
        error!("Or manually place multi-model PDB files in the data directory.");
        anyhow::bail!("No NMR ensembles found");
    }

    let n_targets = args.limit.unwrap_or(ensembles.len()).min(ensembles.len());
    info!("  📈 Loaded {} NMR ensembles with ≥{} models each", n_targets, args.min_models);
    info!("");

    // Determine modes to run
    let modes_to_run: Vec<DynamicsMode> = match args.mode {
        CliDynamicsMode::All => vec![
            DynamicsMode::EnhancedGnm,
            DynamicsMode::CoarseGrainedAnm,
            DynamicsMode::AllAtomAmber,
        ],
        _ => vec![Option::<DynamicsMode>::from(args.mode).unwrap()],
    };

    let mut all_summaries = Vec::new();

    for mode in &modes_to_run {
        info!("═══════════════════════════════════════════════════════════════════════════");
        info!("  Running: {}", mode.name());
        info!("═══════════════════════════════════════════════════════════════════════════");
        info!("");

        let summary = run_benchmark_for_mode(
            *mode,
            &ensembles[..n_targets],
            &args,
        )?;

        all_summaries.push(summary);
    }

    // Find best mode (clone values before moving all_summaries)
    let (best_mode_name, best_correlation) = all_summaries.iter()
        .max_by(|a, b| a.mean_pearson.partial_cmp(&b.mean_pearson).unwrap())
        .map(|s| (s.mode.clone(), s.mean_pearson))
        .unwrap_or(("None".to_string(), 0.0));

    // Print comparison
    info!("");
    info!("╔═══════════════════════════════════════════════════════════════════════════╗");
    info!("║                         MODE COMPARISON SUMMARY                            ║");
    info!("║          (Using TRUE experimental RMSF from NMR ensembles)                 ║");
    info!("╠═══════════════════════════════════════════════════════════════════════════╣");
    info!("║  Mode                      │ Mean ρ   │ Pass Rate │ Time/Protein          ║");
    info!("╠────────────────────────────┼──────────┼───────────┼───────────────────────╣");

    for summary in &all_summaries {
        let time_per_protein = if summary.n_proteins > 0 {
            summary.total_time_ms as f64 / summary.n_proteins as f64
        } else {
            0.0
        };

        let is_best = summary.mode == best_mode_name;
        let marker = if is_best { "★" } else { " " };

        info!("║ {} {:<24} │ {:>8.3} │ {:>8.1}% │ {:>10.1} ms          ║",
              marker,
              summary.mode,
              summary.mean_pearson,
              summary.pass_rate * 100.0,
              time_per_protein);
    }

    info!("╚═══════════════════════════════════════════════════════════════════════════╝");
    info!("");

    // Literature baselines (from NMR-validated studies, not B-factor studies!)
    info!("  📊 Literature Baselines (NMR-validated):");
    info!("     Standard GNM:      ρ ≈ 0.45-0.55 (NMR ensembles)");
    info!("     ANM:               ρ ≈ 0.50-0.60 (NMR ensembles)");
    info!("     MD simulations:    ρ ≈ 0.60-0.70 (NMR ensembles)");
    info!("");

    if best_correlation >= 0.50 {
        info!("  ✅ Best mode ({}) achieves good NMR correlation!", &best_mode_name);
    }
    if best_correlation >= 0.60 {
        info!("  🏆 Best mode ({}) achieves MD-quality correlation!", &best_mode_name);
    }

    // Save results
    let summary = BenchmarkSummary {
        dataset: "NMR Ensemble Benchmark".to_string(),
        data_source: "TRUE experimental RMSF from NMR coordinate variance".to_string(),
        n_proteins_total: n_targets,
        modes: all_summaries,
        best_mode: best_mode_name.clone(),
        best_correlation,
    };

    let results_path = args.output.join("nmr_benchmark_results.json");
    fs::write(&results_path, serde_json::to_string_pretty(&summary)?)?;
    info!("");
    info!("  📄 Results saved to: {:?}", results_path);

    Ok(())
}

/// Download curated NMR ensembles from RCSB PDB
async fn download_curated_nmr_ensembles(data_dir: &PathBuf) -> Result<()> {
    info!("  📥 Downloading curated NMR ensembles from RCSB PDB...");
    info!("");

    let client = reqwest::Client::new();
    let mut success_count = 0;

    for pdb_id in CURATED_NMR_PDBS {
        let output_path = data_dir.join(format!("{}.pdb", pdb_id.to_lowercase()));

        if output_path.exists() {
            info!("  ⏭️  {} already exists, skipping", pdb_id);
            continue;
        }

        let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id.to_uppercase());

        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let content = response.text().await?;

                    // Verify it's an NMR structure with multiple models
                    let model_count = content.lines()
                        .filter(|line| line.starts_with("MODEL"))
                        .count();

                    if model_count >= 2 {
                        fs::write(&output_path, content)?;
                        info!("  ✅ {} - {} models", pdb_id, model_count);
                        success_count += 1;
                    } else {
                        warn!("  ⚠️  {} - only {} model(s), skipping", pdb_id, model_count);
                    }
                } else {
                    warn!("  ❌ {} - HTTP {}", pdb_id, response.status());
                }
            }
            Err(e) => {
                warn!("  ❌ {} - {}", pdb_id, e);
            }
        }
    }

    info!("");
    info!("  📊 Downloaded {} NMR ensembles to {:?}", success_count, data_dir);
    info!("");
    info!("  Now run the benchmark:");
    info!("    cargo run --release -p prism-validation --bin run-dynamics-bench -- \\");
    info!("        --mode enhanced-gnm --data-dir {:?}", data_dir);

    Ok(())
}

/// Load all NMR ensembles from a directory
fn load_all_nmr_ensembles(
    data_dir: &PathBuf,
    min_models: usize,
) -> Result<Vec<NmrEnsemble>> {
    let mut ensembles = Vec::new();

    // Find all PDB files
    let pdb_pattern = data_dir.join("*.pdb");
    let pattern_str = pdb_pattern.to_string_lossy();

    for entry in glob::glob(&pattern_str)? {
        match entry {
            Ok(path) => {
                match load_nmr_ensemble(&path) {
                    Ok(ensemble) => {
                        if ensemble.n_models() >= min_models {
                            info!("    ✓ {} - {} residues, {} models",
                                  ensemble.name, ensemble.n_residues, ensemble.n_models());
                            ensembles.push(ensemble);
                        } else {
                            info!("    ⏭  {} - only {} models (need ≥{})",
                                  path.file_name().unwrap_or_default().to_string_lossy(),
                                  ensemble.n_models(), min_models);
                        }
                    }
                    Err(e) => {
                        warn!("    ✗ {} - {}", path.display(), e);
                    }
                }
            }
            Err(e) => {
                warn!("    Glob error: {}", e);
            }
        }
    }

    Ok(ensembles)
}

fn run_benchmark_for_mode(
    mode: DynamicsMode,
    ensembles: &[NmrEnsemble],
    args: &Args,
) -> Result<ModeSummary> {
    // Create dynamics engine with appropriate config
    let config = DynamicsConfig {
        mode,
        temperature: args.temperature,
        n_steps: args.n_steps,
        gnm_cutoff: args.gnm_cutoff,
        use_distance_weighting: true,
        use_multi_cutoff: true,
        use_secondary_structure: true,
        use_sidechain_factors: true,
        use_sasa_modulation: true,
        ..Default::default()
    };

    let engine = DynamicsEngine::new(config)?;

    let mut results = Vec::new();
    let mut total_time_ms = 0u64;

    info!("  ┌──────────┬───────┬────────┬──────────┬──────────┬──────────┬────────┐");
    info!("  │ PDB      │ Res   │ Models │ ρ(Pears) │ ρ(Spear) │ Time(ms) │ Status │");
    info!("  ├──────────┼───────┼────────┼──────────┼──────────┼──────────┼────────┤");

    for ensemble in ensembles {
        // Compute TRUE experimental RMSF from NMR ensemble
        let true_rmsf = if args.align_models {
            ensemble.compute_rmsf_aligned()
        } else {
            ensemble.compute_rmsf()
        };

        // Convert ensemble to structure input (using mean structure)
        let structure = nmr_ensemble_to_structure(ensemble);

        // Run prediction
        let start = Instant::now();
        let result = match engine.predict_flexibility(&structure) {
            Ok(r) => r,
            Err(e) => {
                error!("  │ {:<8} │ FAIL  │        │ {:27} │",
                       &ensemble.name[..ensemble.name.len().min(8)],
                       format!("{}", e));
                continue;
            }
        };
        let elapsed_ms = start.elapsed().as_millis() as u64;
        total_time_ms += elapsed_ms;

        // Compute correlations against TRUE experimental RMSF
        let min_len = result.rmsf.len().min(true_rmsf.rmsf.len());
        let pred_slice = &result.rmsf[..min_len];
        let exp_slice = &true_rmsf.rmsf[..min_len];

        let pearson = pearson_correlation(pred_slice, exp_slice);
        let spearman = spearman_correlation(pred_slice, exp_slice);

        let mean_pred = pred_slice.iter().sum::<f64>() / pred_slice.len() as f64;
        let mean_exp = exp_slice.iter().sum::<f64>() / exp_slice.len() as f64;
        let max_exp = exp_slice.iter().cloned().fold(0.0, f64::max);

        let passed = pearson > 0.3;  // Liberal threshold
        let status = if passed { "✅ PASS" } else { "❌ FAIL" };

        info!("  │ {:<8} │ {:>5} │ {:>6} │ {:>8.3} │ {:>8.3} │ {:>8} │ {} │",
              &ensemble.name[..ensemble.name.len().min(8)],
              ensemble.n_residues,
              ensemble.n_models(),
              pearson,
              spearman,
              elapsed_ms,
              status);

        results.push(ProteinResult {
            name: ensemble.name.clone(),
            pdb_id: ensemble.name.clone(),
            n_residues: ensemble.n_residues,
            n_nmr_models: ensemble.n_models(),
            mode: mode.name().to_string(),
            pearson_correlation: pearson,
            spearman_correlation: spearman,
            mean_predicted_rmsf: mean_pred,
            mean_experimental_rmsf: mean_exp,
            max_experimental_rmsf: max_exp,
            computation_time_ms: elapsed_ms,
            passed,
        });
    }

    info!("  └──────────┴───────┴────────┴──────────┴──────────┴──────────┴────────┘");

    // Compute summary statistics
    let n_proteins = results.len();
    let n_passed = results.iter().filter(|r| r.passed).count();

    let mean_pearson = if n_proteins > 0 {
        results.iter().map(|r| r.pearson_correlation).sum::<f64>() / n_proteins as f64
    } else { 0.0 };

    let mean_spearman = if n_proteins > 0 {
        results.iter().map(|r| r.spearman_correlation).sum::<f64>() / n_proteins as f64
    } else { 0.0 };

    let std_pearson = if n_proteins > 1 {
        let variance = results.iter()
            .map(|r| (r.pearson_correlation - mean_pearson).powi(2))
            .sum::<f64>() / n_proteins as f64;
        variance.sqrt()
    } else { 0.0 };

    let pass_rate = if n_proteins > 0 {
        n_passed as f64 / n_proteins as f64
    } else { 0.0 };

    info!("");
    info!("  {} Summary (TRUE NMR RMSF): ρ={:.3}±{:.3}, pass rate={:.1}%, time={:.1}s",
          mode.name(), mean_pearson, std_pearson, pass_rate * 100.0,
          total_time_ms as f64 / 1000.0);

    Ok(ModeSummary {
        mode: mode.name().to_string(),
        n_proteins,
        mean_pearson,
        mean_spearman,
        std_pearson,
        pass_rate,
        total_time_ms,
        proteins: results,
    })
}

/// Convert NMR ensemble to StructureInput (using mean structure)
fn nmr_ensemble_to_structure(ensemble: &NmrEnsemble) -> StructureInput {
    let mean_structure = ensemble.mean_structure();

    // Convert chain IDs from String to char (take first char of each)
    let chain_chars: Vec<char> = ensemble.chain_ids.iter()
        .map(|s| s.chars().next().unwrap_or('A'))
        .collect();

    StructureInput {
        name: ensemble.name.clone(),
        pdb_id: Some(ensemble.name.clone()),
        ca_positions: mean_structure,
        all_positions: None,
        residue_names: ensemble.residue_names.clone(),
        atom_names: None,
        atom_residue_indices: None,
        chain_ids: Some(chain_chars),
        residue_seqs: Some(ensemble.residue_numbers.clone()),  // Already Vec<i32>
        b_factors: None,  // NO B-FACTORS - we use TRUE NMR RMSF!
    }
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
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

/// Compute Spearman correlation coefficient
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; v.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = rank as f64 + 1.0;
        }
        ranks
    }

    let rank_x = ranks(x);
    let rank_y = ranks(y);
    pearson_correlation(&rank_x, &rank_y)
}
