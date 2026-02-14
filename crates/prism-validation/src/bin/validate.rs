//! PRISM-4D Validation Runner
//!
//! Runs the full validation suite against curated targets.
//!
//! ## Usage
//!
//! ```bash
//! prism-validate --config validation_config.json
//! ```

use anyhow::Result;
use prism_validation::{ValidationConfig, ValidationRunner};
use std::path::PathBuf;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          PRISM-4D NOVA Validation Framework                      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Multi-tier validation for dynamics and cryptic pocket discovery ║");
    println!("║  Demonstrating capabilities beyond AlphaFold3                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse arguments
    let config_path = std::env::args()
        .skip_while(|arg| arg != "--config")
        .nth(1)
        .map(PathBuf::from);

    let config = if let Some(path) = config_path {
        log::info!("Loading config from {:?}", path);
        let content = std::fs::read_to_string(&path)?;
        serde_json::from_str(&content)?
    } else {
        log::info!("Using default configuration");
        ValidationConfig::default()
    };

    log::info!("Configuration:");
    log::info!("  Data dir: {:?}", config.data_dir);
    log::info!("  Output dir: {:?}", config.output_dir);
    log::info!("  Steps per target: {}", config.steps_per_target);
    log::info!("  Temperature: {} K", config.temperature);
    log::info!("  Benchmarks: {:?}", config.benchmarks);

    // Create runner
    let runner = ValidationRunner::new(config)?;

    // Run all benchmarks
    log::info!("Starting validation run...");
    let summary = runner.run_all()?;

    // Save results
    let results_path = runner.save_results(&summary)?;

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    VALIDATION COMPLETE                           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Overall pass rate: {:.1}%", summary.overall_pass_rate * 100.0);
    println!("║  Overall score:     {:.1}/100", summary.overall_score);
    println!("║  Results saved to:  {:?}", results_path);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    for bench_summary in &summary.benchmark_summaries {
        println!("\n  {} Benchmark:", bench_summary.benchmark);
        println!("    Pass rate: {:.1}%", bench_summary.pass_rate * 100.0);
        println!("    Mean score: {:.1} ± {:.1}", bench_summary.mean_score, bench_summary.std_score);
        println!("    Best: {}", bench_summary.best_target);
        println!("    Worst: {}", bench_summary.worst_target);
    }

    Ok(())
}
