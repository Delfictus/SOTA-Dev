use clap::Parser;
use std::path::{Path, PathBuf};

use prism_lbs::{
    validation::{BenchmarkCase, BenchmarkSummary},
    LbsConfig, PrismLbs, ProteinStructure,
};

/// Batch benchmark runner for PRISM-LBS
#[derive(Parser)]
#[command(name = "prism-lbs-benchmark")]
#[command(about = "Benchmark PRISM-LBS predictions against ligand coordinates", long_about = None)]
struct Cli {
    /// Directory containing PDB files
    #[arg(long)]
    pdb_dir: PathBuf,
    /// Directory containing ligand XYZ files (stem must match PDB stem)
    #[arg(long)]
    ligand_dir: PathBuf,
    /// Config TOML file
    #[arg(long)]
    config: PathBuf,
    /// Distance threshold for success/coverage (Å)
    #[arg(long, default_value_t = 4.0)]
    threshold: f64,
    /// Optional JSON output path for summary
    #[arg(long)]
    summary_out: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let cfg = LbsConfig::from_file(&cli.config)?;
    let predictor = PrismLbs::new(cfg.clone())?;

    let pdbs = collect_files(&cli.pdb_dir, "pdb")?;
    let cases = BenchmarkCase::load_dir(&cli.ligand_dir, cli.threshold)?;
    if cases.is_empty() {
        anyhow::bail!("No ligand XYZ files found in {}", cli.ligand_dir.display());
    }

    let mut all_metrics = Vec::new();
    for pdb in pdbs {
        if let Some(stem) = pdb.file_stem().and_then(|s| s.to_str()) {
            if let Some(case) = cases.iter().find(|c| c.name == stem) {
                let structure = ProteinStructure::from_pdb_file(&pdb)?;
                let pockets = predictor.predict(&structure)?;
                let metrics = prism_lbs::validation::ValidationMetrics::compute_batch(
                    &pockets,
                    &case.ligand_coords,
                    case.threshold,
                );
                all_metrics.push(metrics.clone());
                log::info!(
                    "PDB {}: center_dist={:.2}Å coverage={:.2} precision={:.2}",
                    stem,
                    metrics.center_distance(),
                    metrics.ligand_coverage,
                    metrics.pocket_precision
                );
            }
        }
    }

    // Compute summary statistics
    let total_cases = all_metrics.len();
    let dcc_successes = all_metrics.iter().filter(|m| m.dcc_success).count();
    let mean_dcc = mean(&all_metrics, |m| m.dcc);
    let mean_coverage = mean(&all_metrics, |m| m.ligand_coverage);
    let mean_precision = mean(&all_metrics, |m| m.pocket_precision);

    let summary = BenchmarkSummary {
        total_cases,
        dcc_successes,
        dca_successes: all_metrics.iter().filter(|m| m.dca_success).count(),
        dcc_success_rate: dcc_successes as f64 / total_cases.max(1) as f64,
        dca_success_rate: all_metrics.iter().filter(|m| m.dca_success).count() as f64 / total_cases.max(1) as f64,
        top1_dcc_rate: 0.0,
        top3_dcc_rate: 0.0,
        top5_dcc_rate: 0.0,
        top1_dca_rate: 0.0,
        top3_dca_rate: 0.0,
        top5_dca_rate: 0.0,
        mean_dcc,
        mean_dca: mean(&all_metrics, |m| m.dca),
        std_dcc: 0.0,
        std_dca: 0.0,
        case_results: Vec::new(),
    };

    println!(
        "Cases: {} success: {:.2} mean_dist: {:.2}Å coverage: {:.2} precision: {:.2}",
        total_cases,
        summary.dcc_success_rate,
        mean_dcc,
        mean_coverage,
        mean_precision
    );

    if let Some(out) = cli.summary_out {
        let mut path = out;
        if path.extension().is_none() {
            path.set_extension("json");
        }
        std::fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
        log::info!("Wrote summary to {}", path.display());
    }

    Ok(())
}

fn collect_files(dir: &Path, ext: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|s| s.to_str())
                .map_or(false, |e| e.eq_ignore_ascii_case(ext))
            {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn mean<T, F>(items: &[T], f: F) -> f64
where
    F: Fn(&T) -> f64,
{
    if items.is_empty() {
        return 0.0;
    }
    let sum: f64 = items.iter().map(f).sum();
    sum / items.len() as f64
}
