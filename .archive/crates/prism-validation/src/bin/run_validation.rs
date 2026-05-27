//! PRISM-4D End-to-End Validation Harness
//!
//! Runs the complete validation pipeline against curated targets:
//! 1. Loads curated manifest with provenance verification
//! 2. Extracts simulation-ready coordinates
//! 3. Runs all 4 benchmark tiers
//! 4. Computes metrics and generates reports
//!
//! ## Usage
//!
//! ```bash
//! prism-run-validation --manifest data/validation/curated/curation_manifest.json
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use prism_validation::pipeline::ValidationPipeline;
use prism_validation::reports::ValidationReport;
use prism_validation::ValidationConfig;
use std::path::PathBuf;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       PRISM-4D NOVA End-to-End Validation Harness                ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Benchmarks:                                                     ║");
    println!("║  • ATLAS Ensemble Recovery (dynamics)                            ║");
    println!("║  • Apo-Holo Transition (cryptic pockets)                         ║");
    println!("║  • Retrospective Blind (drug discovery)                          ║");
    println!("║  • Novel Cryptic Benchmark (PRISM-defined)                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse arguments
    let manifest_path = std::env::args()
        .skip_while(|arg| arg != "--manifest")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/validation/curated/curation_manifest.json"));

    let output_dir = std::env::args()
        .skip_while(|arg| arg != "--output")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("validation_results"));

    let steps: usize = std::env::args()
        .skip_while(|arg| arg != "--steps")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    log::info!("Manifest: {:?}", manifest_path);
    log::info!("Output: {:?}", output_dir);
    log::info!("Steps per target: {}", steps);

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Configure validation
    let config = ValidationConfig {
        data_dir: manifest_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf(),
        output_dir: output_dir.clone(),
        steps_per_target: steps,
        temperature: 310.0, // 37°C physiological
        compare_af3: true,
        benchmarks: vec![
            "atlas".to_string(),
            "apo_holo".to_string(),
            "retrospective".to_string(),
            "novel".to_string(),
        ],
        gpu_device: 0,
    };

    // Load pipeline
    log::info!("Loading validation pipeline...");
    let pipeline = ValidationPipeline::from_manifest(&manifest_path, config)
        .context("Failed to load validation pipeline")?;

    // Print status
    pipeline.print_status();

    // Compute initial metrics
    println!("\n📏 Initial Apo-Holo Distances:");
    println!("┌────────────────┬────────────────┬────────────────┐");
    println!("│ Target         │ CA RMSD (Å)    │ Pocket RMSD(Å) │");
    println!("├────────────────┼────────────────┼────────────────┤");

    for target_name in pipeline.target_names() {
        let apo = pipeline.get_apo_structure(&target_name);
        let holo = pipeline.get_holo_structure(&target_name);

        if let (Some(a), Some(h)) = (apo, holo) {
            let ca_rmsd = ValidationPipeline::compute_ca_rmsd(a, h)
                .map(|r| format!("{:.2}", r))
                .unwrap_or_else(|| "N/A".to_string());

            let pocket_rmsd = pipeline
                .compute_pocket_rmsd(&target_name)
                .map(|r| format!("{:.2}", r))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "│ {:<14} │ {:>14} │ {:>14} │",
                &target_name[..target_name.len().min(14)],
                ca_rmsd,
                pocket_rmsd
            );
        }
    }
    println!("└────────────────┴────────────────┴────────────────┘");

    // Run validation
    println!("\n🚀 Running validation benchmarks...\n");

    let summary = pipeline.run().context("Validation run failed")?;

    // Print results
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    VALIDATION RESULTS                            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Duration: {:.1}s",
        (summary.finished - summary.started).num_seconds()
    );
    println!(
        "║  Overall Pass Rate: {:.1}%",
        summary.overall_pass_rate * 100.0
    );
    println!("║  Overall Score: {:.1}/100", summary.overall_score);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n📊 Benchmark Results:");
    println!("┌──────────────────┬─────────┬───────────┬────────────┬────────────┐");
    println!("│ Benchmark        │ Targets │ Pass Rate │ Mean Score │ Std Score  │");
    println!("├──────────────────┼─────────┼───────────┼────────────┼────────────┤");

    for bench in &summary.benchmark_summaries {
        println!(
            "│ {:<16} │ {:>7} │ {:>8.1}% │ {:>10.1} │ {:>10.1} │",
            &bench.benchmark[..bench.benchmark.len().min(16)],
            bench.targets_run,
            bench.pass_rate * 100.0,
            bench.mean_score,
            bench.std_score
        );
    }
    println!("└──────────────────┴─────────┴───────────┴────────────┴────────────┘");

    // Best and worst performers
    println!("\n🏆 Top Performers by Benchmark:");
    for bench in &summary.benchmark_summaries {
        if !bench.best_target.is_empty() {
            println!(
                "  • {}: Best = {}, Challenging = {}",
                bench.benchmark, bench.best_target, bench.worst_target
            );
        }
    }

    // Generate report
    log::info!("Generating validation report...");
    let report = ValidationReport::from_summary(&summary);

    let report_md = output_dir.join(format!(
        "validation_report_{}.md",
        Utc::now().format("%Y%m%d_%H%M%S")
    ));
    let report_json = output_dir.join(format!(
        "validation_report_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    ));

    report.save_markdown(&report_md)?;
    report.save_json(&report_json)?;

    // Save summary JSON
    let summary_path = output_dir.join(format!(
        "validation_summary_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    ));
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, summary_json)?;

    println!("\n📁 Output Files:");
    println!("  • Report (MD):   {:?}", report_md);
    println!("  • Report (JSON): {:?}", report_json);
    println!("  • Summary:       {:?}", summary_path);

    // Final verdict
    let grade = match summary.overall_score {
        s if s >= 90.0 => ("A", "🌟 Excellent - Ready for publication"),
        s if s >= 80.0 => ("B", "✓ Good - Minor improvements possible"),
        s if s >= 70.0 => ("C", "⚠ Acceptable - Room for improvement"),
        s if s >= 60.0 => ("D", "⚠ Below expectations - Review needed"),
        _ => ("F", "✗ Needs significant work"),
    };

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  FINAL GRADE: {} - {}                          ",
        grade.0, grade.1
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    if summary.overall_pass_rate >= 0.8 {
        println!("\n✅ PRISM-4D NOVA demonstrates strong performance in dynamics-based");
        println!("   drug discovery, where AlphaFold3 cannot compete.");
    }

    Ok(())
}
