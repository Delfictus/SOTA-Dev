//! Apo-Holo Benchmark Binary - Run benchmark on 15 classic apo-holo pairs
//!
//! Usage: cargo run --release -p prism-validation --features cryptic-gpu --bin apo_holo_benchmark

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "apo_holo_benchmark")]
#[command(about = "Run apo-holo benchmark on 15 classic protein pairs")]
struct Args {
    /// Data directory containing PDB files
    #[arg(long, default_value = "data/benchmarks/apo_holo")]
    data_dir: String,

    /// Number of samples per structure
    #[arg(long, default_value = "100")]
    samples: usize,

    /// Run in mock mode (no GPU required, for testing)
    #[arg(long)]
    mock: bool,

    /// Output file for results
    #[arg(long, default_value = "results/apo_holo_benchmark.json")]
    output: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== PRISM Apo-Holo Benchmark ===");
    println!("Data directory: {}", args.data_dir);
    println!("Samples per structure: {}", args.samples);
    println!("Mode: {}", if args.mock { "MOCK" } else { "GPU" });
    println!();

    use prism_validation::apo_holo_benchmark::{ApoHoloBenchmark, ApoHoloBenchmarkConfig};
    use prism_validation::sampling::result::SamplingConfig;
    use prism_validation::sampling::router::RoutingStrategy;

    // Configure sampling
    // NOTE: steps_per_sample = 500 gives 50 fs relaxation between velocity resets
    // This prevents "thermal overheating" where frequent energy injection
    // overpowers the protein's ability to maintain its fold.
    let sampling_config = SamplingConfig {
        n_samples: args.samples,
        steps_per_sample: 500,  // 500 steps × 0.1 fs = 50 fs relaxation (was 50 → 5 fs)
        temperature: 310.0,
        seed: 42,
        // 0.1 fs required for stability with explicit hydrogens (C-H period ~10 fs)
        timestep_fs: Some(0.1),
        leapfrog_steps: Some(5),  // NOTE: Currently unused by amber_path.rs
    };

    // Configure benchmark
    let config = ApoHoloBenchmarkConfig {
        data_dir: args.data_dir.clone(),
        sampling_config,
        routing_strategy: RoutingStrategy::Auto,
        store_trajectories: false,
    };

    let mut benchmark = ApoHoloBenchmark::with_config(config);

    // Run benchmark
    let summary = if args.mock {
        println!("Running in MOCK mode...");
        benchmark.run_all_mock()
            .context("Mock benchmark failed")?
    } else {
        #[cfg(feature = "cryptic-gpu")]
        {
            use cudarc::driver::CudaContext;

            println!("Initializing CUDA...");
            let context = CudaContext::new(0)
                .context("Failed to initialize CUDA device")?;
            println!("CUDA initialized");
            println!();

            println!("Running GPU benchmark on 15 apo-holo pairs...");
            println!("This may take 20-30 minutes...");
            println!();

            benchmark.run_all(context)
                .context("GPU benchmark failed")?
        }

        #[cfg(not(feature = "cryptic-gpu"))]
        {
            println!("ERROR: cryptic-gpu feature not enabled");
            println!("Either use --mock flag or recompile with --features cryptic-gpu");
            std::process::exit(1);
        }
    };

    // Print results
    println!();
    println!("=== BENCHMARK RESULTS ===");
    println!();
    println!("{}", summary.to_markdown());

    // Save results
    std::fs::create_dir_all(std::path::Path::new(&args.output).parent().unwrap_or(std::path::Path::new(".")))?;
    summary.save_json(&args.output)?;
    println!("Results saved to: {}", args.output);

    // Print summary statistics
    println!();
    println!("=== SUMMARY ===");
    println!("Total pairs: {}", summary.n_total);
    println!("Successful: {} ({:.1}%)",
        summary.n_success,
        100.0 * summary.n_success as f64 / summary.n_total.max(1) as f64
    );
    println!("Overall success rate: {:.1}%", summary.success_rate * 100.0);
    println!("Mean min RMSD to holo: {:.2} Å", summary.mean_min_rmsd_to_holo);
    println!("Mean improvement: {:.1}%", summary.mean_rmsd_improvement * 100.0);

    // Check against targets
    println!();
    println!("=== TARGET CHECK ===");
    let success_target = 0.60;
    let rmsd_target = 3.5;

    if summary.success_rate >= success_target {
        println!("✅ Success rate {:.1}% >= {:.0}% target", summary.success_rate * 100.0, success_target * 100.0);
    } else {
        println!("❌ Success rate {:.1}% < {:.0}% target", summary.success_rate * 100.0, success_target * 100.0);
    }

    if summary.mean_min_rmsd_to_holo <= rmsd_target {
        println!("✅ Mean min RMSD {:.2}Å <= {:.1}Å target", summary.mean_min_rmsd_to_holo, rmsd_target);
    } else {
        println!("❌ Mean min RMSD {:.2}Å > {:.1}Å target", summary.mean_min_rmsd_to_holo, rmsd_target);
    }

    Ok(())
}
