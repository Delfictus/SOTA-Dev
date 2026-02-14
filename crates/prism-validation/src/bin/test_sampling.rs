//! Test Sampling Binary - Quick GPU sampling test on a single structure
//!
//! Usage: cargo run --release -p prism-validation --bin test_sampling -- --pdb <path> --samples <n>

use anyhow::{Context, Result};
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "test_sampling")]
#[command(about = "Test GPU sampling on a single PDB structure")]
struct Args {
    /// Path to PDB file
    #[arg(long)]
    pdb: PathBuf,

    /// Number of samples to generate
    #[arg(long, default_value = "10")]
    samples: usize,

    /// Steps per sample for decorrelation
    #[arg(long, default_value = "50")]
    steps_per_sample: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== PRISM GPU Sampling Test ===");
    println!("PDB: {}", args.pdb.display());
    println!("Samples: {}", args.samples);
    println!("Steps per sample: {}", args.steps_per_sample);
    println!();

    // Read PDB file
    let pdb_content = std::fs::read_to_string(&args.pdb)
        .with_context(|| format!("Failed to read PDB file: {}", args.pdb.display()))?;

    // Sanitize structure
    use prism_validation::pdb_sanitizer::sanitize_pdb;
    let pdb_id = args.pdb.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let structure = sanitize_pdb(&pdb_content, pdb_id)
        .context("Failed to sanitize PDB")?;

    println!("Structure loaded:");
    println!("  Atoms: {}", structure.n_atoms());
    println!("  Residues: {}", structure.n_residues());
    println!("  Chains: {:?}", structure.chains);
    println!();

    // Check if we can use GPU
    #[cfg(feature = "cryptic-gpu")]
    {
        use prism_validation::sampling::result::SamplingConfig;
        use prism_validation::sampling::router::HybridSampler;
        use prism_validation::sampling::contract::SamplingBackend;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        println!("Initializing CUDA...");
        let context = CudaContext::new(0)
            .context("Failed to initialize CUDA device")?;
        println!("CUDA initialized successfully");
        println!();

        // Create sampler
        let mut sampler = HybridSampler::new(context.clone())
            .context("Failed to create HybridSampler")?;

        // Load structure
        println!("Loading structure into sampler...");
        sampler.load_structure(&structure)
            .context("Failed to load structure")?;
        println!("Structure loaded successfully");
        println!();

        // Configure sampling
        let config = SamplingConfig {
            n_samples: args.samples,
            steps_per_sample: args.steps_per_sample,
            temperature: 310.0,
            seed: 42,
            timestep_fs: Some(2.0),
            leapfrog_steps: Some(5),
        };

        // Run sampling
        println!("Running GPU sampling...");
        let start = std::time::Instant::now();
        let result = sampler.sample(&config)
            .context("Sampling failed")?;
        let elapsed = start.elapsed();

        println!();
        println!("=== Results ===");
        println!("Backend: {:?}", result.metadata.backend);
        println!("Samples collected: {}", result.conformations.len());
        println!("Time: {:.2}s ({:.1} samples/sec)",
            elapsed.as_secs_f64(),
            result.conformations.len() as f64 / elapsed.as_secs_f64()
        );
        println!();

        // Report TDA data if available
        if let Some(betti) = &result.betti {
            println!("TDA Data (Betti numbers):");
            for (i, b) in betti.iter().take(5).enumerate() {
                println!("  Sample {}: β₀={}, β₁={}, β₂={}", i, b[0], b[1], b[2]);
            }
            if betti.len() > 5 {
                println!("  ... ({} more samples)", betti.len() - 5);
            }
        } else {
            println!("TDA Data: Not available (AMBER backend doesn't compute Betti numbers)");
        }
        println!();

        // Report energy data
        println!("Energy samples (first 5):");
        for (i, e) in result.energies.iter().take(5).enumerate() {
            println!("  Sample {}: {:.2} kcal/mol", i, e);
        }
        println!();

        // Report acceptance rate if available
        if let Some(rate) = result.metadata.acceptance_rate {
            println!("Acceptance rate: {:.1}%", rate * 100.0);
        }

        println!("=== Test Complete ===");
    }

    #[cfg(not(feature = "cryptic-gpu"))]
    {
        println!("ERROR: cryptic-gpu feature not enabled");
        println!("Recompile with: cargo run --release -p prism-validation --features cryptic-gpu --bin test_sampling ...");
        std::process::exit(1);
    }

    Ok(())
}
