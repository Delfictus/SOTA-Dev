//! PRISM-NOVA Physics-Based Validation Runner
//!
//! End-to-end validation using real GPU-accelerated molecular dynamics.
//! This binary runs all 4 benchmark tiers against curated targets using
//! the PRISM-NOVA physics engine.
//!
//! Usage:
//!   cargo run --bin prism-validate-physics --features simulation -- [OPTIONS]
//!
//! Options:
//!   --manifest PATH    Path to curated manifest (required)
//!   --output PATH      Output directory (default: validation_results)
//!   --steps N          Simulation steps per target (default: 1000)
//!   --temp K           Temperature in Kelvin (default: 310)
//!   --gpu N            GPU device index (default: 0)
//!   --benchmark NAME   Run specific benchmark only

use anyhow::{Context, Result};
use chrono::Utc;
use env_logger;
use log;
use prism_validation::{
    benchmark_integration::SimulationBenchmarkRunner,
    data_curation::CurationManifest,
    pipeline::SimulationStructure,
    BenchmarkSummary, ValidationConfig, ValidationSummary,
};
use serde_json;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("═══════════════════════════════════════════════════════════════");
    log::info!("  PRISM-NOVA Physics-Based Validation");
    log::info!("  Dynamics-Based Drug Discovery Beyond AlphaFold3");
    log::info!("═══════════════════════════════════════════════════════════════");

    // Parse command line arguments
    let args = parse_args()?;

    log::info!("Configuration:");
    log::info!("  Manifest: {:?}", args.manifest_path);
    log::info!("  Output: {:?}", args.output_dir);
    log::info!("  Steps/target: {}", args.steps);
    log::info!("  Temperature: {} K", args.temperature);
    log::info!("  GPU device: {}", args.gpu_device);
    if let Some(ref bench) = args.benchmark_filter {
        log::info!("  Benchmark filter: {}", bench);
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Load curated manifest
    log::info!("\n📂 Loading curated manifest...");
    let manifest = load_manifest(&args.manifest_path)?;
    log::info!("  Loaded {} targets", manifest.targets.len());

    // Create validation config
    let config = ValidationConfig {
        data_dir: args.manifest_path.parent().unwrap_or(&PathBuf::from(".")).to_path_buf(),
        output_dir: args.output_dir.clone(),
        steps_per_target: args.steps,
        temperature: args.temperature,
        compare_af3: true,
        benchmarks: vec![
            "atlas".to_string(),
            "apo_holo".to_string(),
            "retrospective".to_string(),
            "novel".to_string(),
        ],
        gpu_device: args.gpu_device,
    };

    // Create benchmark runner
    log::info!("\n🚀 Initializing PRISM-NOVA benchmark runner...");
    let mut runner = SimulationBenchmarkRunner::new(&config)
        .context("Failed to create benchmark runner")?;

    log::info!("  Simulation enabled: {}", runner.simulation_enabled());

    // Load structures from manifest
    let (apo_structures, holo_structures) = load_structures(&manifest)?;

    // Run benchmarks
    let started = Utc::now();
    let mut all_results = Vec::new();
    let mut benchmark_summaries = Vec::new();

    // Define benchmarks to run
    let benchmarks_to_run: Vec<&str> = match &args.benchmark_filter {
        Some(name) => vec![name.as_str()],
        None => vec!["atlas", "apo_holo", "retrospective", "novel"],
    };

    for benchmark_name in benchmarks_to_run {
        log::info!("\n══════════════════════════════════════════════════════════════");
        log::info!("  Running {} benchmark", benchmark_name.to_uppercase());
        log::info!("══════════════════════════════════════════════════════════════");

        let mut results = Vec::new();
        let mut scores = Vec::new();
        let mut passed = 0;
        let mut best_score = 0.0f64;
        let mut worst_score = 100.0f64;
        let mut best_target = String::new();
        let mut worst_target = String::new();

        for target in &manifest.targets {
            if !target.valid_for_blind {
                log::warn!("  Skipping {} - not valid for blind validation", target.name);
                continue;
            }

            let apo = apo_structures.get(&target.name)
                .context(format!("Missing apo structure for {}", target.name))?;

            log::info!("\n  ▶ Target: {} (APO: {})", target.name, target.apo_provenance.pdb_id);

            let result = match benchmark_name {
                "atlas" => {
                    runner.run_atlas_benchmark(apo, None)?
                }
                "apo_holo" => {
                    let holo = holo_structures.get(&target.name)
                        .context(format!("Missing holo structure for {}", target.name))?;
                    runner.run_apo_holo_benchmark(apo, holo)?
                }
                "retrospective" => {
                    runner.run_retrospective_benchmark(apo, &target.pocket_residues)?
                }
                "novel" => {
                    runner.run_novel_cryptic_benchmark(apo)?
                }
                _ => {
                    log::warn!("Unknown benchmark: {}", benchmark_name);
                    continue;
                }
            };

            // Log result
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            log::info!("    {} - {}", status, result.reason);

            // Compute score (simple for now)
            let score = compute_benchmark_score(&result, benchmark_name);
            log::info!("    Score: {:.1}/100", score);

            if result.passed {
                passed += 1;
            }

            if score > best_score {
                best_score = score;
                best_target = target.name.clone();
            }
            if score < worst_score {
                worst_score = score;
                worst_target = target.name.clone();
            }

            scores.push(score);
            results.push(result);
        }

        // Compute summary statistics
        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        let std_score = if scores.len() > 1 {
            let variance: f64 = scores.iter()
                .map(|s| (s - mean_score).powi(2))
                .sum::<f64>() / (scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        log::info!("\n  {} Summary:", benchmark_name.to_uppercase());
        log::info!("    Targets: {}/{} passed ({:.0}%)",
            passed, results.len(), 100.0 * passed as f64 / results.len().max(1) as f64);
        log::info!("    Score: {:.1} ± {:.1}", mean_score, std_score);
        log::info!("    Best: {} ({:.1})", best_target, best_score);
        log::info!("    Worst: {} ({:.1})", worst_target, worst_score);

        benchmark_summaries.push(BenchmarkSummary {
            benchmark: benchmark_name.to_string(),
            targets_run: results.len(),
            targets_passed: passed,
            pass_rate: if results.is_empty() { 0.0 } else { passed as f64 / results.len() as f64 },
            mean_score,
            std_score,
            best_target,
            worst_target,
        });

        all_results.extend(results);
    }

    let finished = Utc::now();

    // Compute overall statistics
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

    // Print final summary
    log::info!("\n═══════════════════════════════════════════════════════════════");
    log::info!("  VALIDATION COMPLETE");
    log::info!("═══════════════════════════════════════════════════════════════");
    log::info!("");
    log::info!("  Overall Pass Rate: {:.0}%", overall_pass_rate * 100.0);
    log::info!("  Overall Score: {:.1}/100", overall_score);
    log::info!("  Duration: {:.1}s", (finished - started).num_milliseconds() as f64 / 1000.0);
    log::info!("");

    // Create validation summary
    let summary = ValidationSummary {
        started,
        finished,
        config,
        benchmark_summaries,
        overall_pass_rate,
        overall_score,
        af3_summary: None,
    };

    // Save results
    let timestamp = started.format("%Y%m%d_%H%M%S");
    let summary_path = args.output_dir.join(format!("physics_validation_{}.json", timestamp));
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, &summary_json)?;
    log::info!("📄 Results saved to: {:?}", summary_path);

    // Save detailed results
    let results_path = args.output_dir.join(format!("physics_results_{}.json", timestamp));
    let results_json = serde_json::to_string_pretty(&all_results)?;
    std::fs::write(&results_path, &results_json)?;
    log::info!("📄 Detailed results saved to: {:?}", results_path);

    // Grade the validation
    let grade = match overall_score {
        s if s >= 90.0 => "A (Excellent)",
        s if s >= 80.0 => "B (Good)",
        s if s >= 70.0 => "C (Satisfactory)",
        s if s >= 60.0 => "D (Needs Improvement)",
        _ => "F (Below Threshold)",
    };

    log::info!("");
    log::info!("  🎯 Grade: {}", grade);
    log::info!("");

    if overall_pass_rate >= 0.7 {
        log::info!("  ✅ Validation SUCCESSFUL - PRISM-NOVA demonstrates dynamics capability");
    } else {
        log::warn!("  ⚠️  Validation shows room for improvement");
    }

    Ok(())
}

/// Command line arguments
struct Args {
    manifest_path: PathBuf,
    output_dir: PathBuf,
    steps: usize,
    temperature: f32,
    gpu_device: usize,
    benchmark_filter: Option<String>,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut manifest_path = None;
    let mut output_dir = PathBuf::from("validation_results");
    let mut steps = 1000;
    let mut temperature = 310.0;
    let mut gpu_device = 0;
    let mut benchmark_filter = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest_path = Some(PathBuf::from(&args[i]));
            }
            "--output" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse()?;
            }
            "--temp" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--gpu" => {
                i += 1;
                gpu_device = args[i].parse()?;
            }
            "--benchmark" => {
                i += 1;
                benchmark_filter = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                log::warn!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    let manifest_path = manifest_path.context(
        "Missing --manifest argument. Use --help for usage."
    )?;

    Ok(Args {
        manifest_path,
        output_dir,
        steps,
        temperature,
        gpu_device,
        benchmark_filter,
    })
}

fn print_usage() {
    println!("PRISM-NOVA Physics-Based Validation");
    println!("");
    println!("Usage: prism-validate-physics --manifest PATH [OPTIONS]");
    println!("");
    println!("Options:");
    println!("  --manifest PATH    Path to curated manifest (required)");
    println!("  --output PATH      Output directory (default: validation_results)");
    println!("  --steps N          Simulation steps per target (default: 1000)");
    println!("  --temp K           Temperature in Kelvin (default: 310)");
    println!("  --gpu N            GPU device index (default: 0)");
    println!("  --benchmark NAME   Run specific benchmark only (atlas, apo_holo, retrospective, novel)");
    println!("  --help             Show this message");
}

fn load_manifest(path: &PathBuf) -> Result<CurationManifest> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read manifest file")?;
    let manifest: CurationManifest = serde_json::from_str(&content)
        .context("Failed to parse manifest JSON")?;
    Ok(manifest)
}

fn load_structures(
    manifest: &CurationManifest,
) -> Result<(HashMap<String, SimulationStructure>, HashMap<String, SimulationStructure>)> {
    let mut apo_structures = HashMap::new();
    let mut holo_structures = HashMap::new();

    for target in &manifest.targets {
        // Load APO structure
        let apo = SimulationStructure::from_metadata(
            &target.apo_metadata,
            Some(target.pocket_residues.clone()),
        );
        apo_structures.insert(target.name.clone(), apo);

        // Load HOLO structure
        let holo = SimulationStructure::from_metadata(
            &target.holo_metadata,
            Some(target.pocket_residues.clone()),
        );
        holo_structures.insert(target.name.clone(), holo);
    }

    Ok((apo_structures, holo_structures))
}

fn compute_benchmark_score(result: &prism_validation::BenchmarkResult, benchmark: &str) -> f64 {
    match benchmark {
        "atlas" => {
            let rmsf_corr = result.metrics.rmsf_correlation.unwrap_or(0.0) as f64;
            let acceptance = result.metrics.acceptance_rate.unwrap_or(0.0) as f64;
            (rmsf_corr * 0.7 + acceptance * 0.3) * 100.0
        }
        "apo_holo" => {
            let pocket_rmsd = result.metrics.pocket_rmsd.unwrap_or(10.0) as f64;
            let betti_2 = result.metrics.betti_2.unwrap_or(0.0) as f64;
            let rmsd_score = ((5.0 - pocket_rmsd) / 5.0 * 100.0).max(0.0).min(100.0);
            let betti_score = if betti_2 >= 1.0 { 100.0 } else { betti_2 * 80.0 };
            rmsd_score * 0.6 + betti_score * 0.4
        }
        "retrospective" => {
            let site_rank = result.metrics.custom.get("site_rank").copied().unwrap_or(10.0);
            let site_overlap = result.metrics.custom.get("site_overlap").copied().unwrap_or(0.0);
            let rank_score = ((4.0 - site_rank) / 3.0 * 100.0).max(0.0).min(100.0);
            let overlap_score = site_overlap * 100.0;
            rank_score * 0.5 + overlap_score * 0.5
        }
        "novel" => {
            let pocket_sig = result.metrics.pocket_signature.unwrap_or(0.0) as f64;
            let stability = result.metrics.pocket_stability.unwrap_or(0.0) as f64;
            (pocket_sig * 0.6 + stability * 0.4) * 100.0
        }
        _ => 50.0,
    }
}
