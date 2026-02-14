//! Ensemble Speedup Test - Measures True Batched Parallel Speedup
//!
//! This test clones a SINGLE structure N times to measure the true batched
//! parallel speedup, eliminating the overhead from different structure sizes.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda -p prism-validation --bin ensemble_speedup_test -- \
//!     --topology results/sota_validation_fresh/6M0J_topology.json \
//!     --clones 4 8 16 32 \
//!     --steps 2000
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "ensemble-speedup-test")]
#[command(about = "Tests true batched parallel speedup by cloning a single structure")]
struct Args {
    /// Path to topology JSON file from prism-prep
    #[arg(long)]
    topology: PathBuf,

    /// Number of clones to test (e.g., 4 8 16 32)
    #[arg(long, num_args = 1.., required = true)]
    clones: Vec<usize>,

    /// MD steps per structure
    #[arg(long, default_value = "2000")]
    steps: usize,

    /// Temperature in Kelvin
    #[arg(long, default_value = "310.0")]
    temperature: f32,

    /// Timestep in femtoseconds
    #[arg(long, default_value = "2.0")]
    dt: f32,

    /// Output directory for results
    #[arg(long, default_value = "results/ensemble_speedup")]
    output: PathBuf,
}

/// prism-prep topology JSON format
#[derive(Debug, Deserialize)]
struct PrismPrepTopology {
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    positions: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondDef>,
    angles: Vec<AngleDef>,
    dihedrals: Vec<DihedralDef>,
    exclusions: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct LjParam {
    sigma: f32,
    epsilon: f32,
}

#[derive(Debug, Deserialize)]
struct BondDef {
    i: usize,
    j: usize,
    r0: f32,
    k: f32,
}

#[derive(Debug, Deserialize)]
struct AngleDef {
    i: usize,
    j: usize,
    k_idx: usize,
    theta0: f32,
    force_k: f32,
}

#[derive(Debug, Deserialize)]
struct DihedralDef {
    i: usize,
    j: usize,
    k_idx: usize,
    l: usize,
    periodicity: u32,
    phase: f32,
    force_k: f32,
}

/// Speedup test result
#[derive(Debug, Serialize)]
struct SpeedupResult {
    n_clones: usize,
    sequential_ms: f64,
    batched_ms: f64,
    speedup: f64,
    effective_ms_per_structure: f64,
    throughput_structures_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct TestReport {
    timestamp: String,
    structure: String,
    n_atoms: usize,
    steps: usize,
    results: Vec<SpeedupResult>,
    best_speedup: f64,
    best_n_clones: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ENSEMBLE SPEEDUP TEST - Clone Single Structure N Times   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    println!("\n📊 Configuration:");
    println!("   Topology: {:?}", args.topology);
    println!("   Clone counts: {:?}", args.clones);
    println!("   MD Steps: {}", args.steps);
    println!("   Temperature: {} K", args.temperature);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};

        // Initialize CUDA
        println!("\n🚀 Initializing CUDA...");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        println!("   GPU: Device {}", context.cu_device());

        // Load topology
        println!("\n📂 Loading topology...");
        let file = File::open(&args.topology)?;
        let reader = BufReader::new(file);
        let prism_topo: PrismPrepTopology = serde_json::from_reader(reader)?;

        let structure_topo = convert_prism_prep_to_structure_topology(&prism_topo);
        let n_atoms = prism_topo.n_atoms;

        println!("   ✓ {} atoms, {} bonds, {} angles, {} dihedrals",
                 n_atoms, prism_topo.bonds.len(),
                 prism_topo.angles.len(), prism_topo.dihedrals.len());

        let mut results: Vec<SpeedupResult> = Vec::new();

        for &n_clones in &args.clones {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Testing {} clones ({} total atoms)", n_clones, n_atoms * n_clones);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Sequential baseline - run each clone one at a time
            println!("\n   ⏱️  Sequential (1 at a time)...");
            let seq_start = Instant::now();

            for _ in 0..n_clones {
                let mut batch = AmberSimdBatch::new_with_config(
                    context.clone(),
                    n_atoms + 100,
                    1,
                    OptimizationConfig::default(), // Same config as batched for fair comparison
                )?;
                batch.add_structure(&structure_topo)?;
                batch.finalize_batch()?;
                batch.initialize_velocities(args.temperature)?;
                batch.equilibrate(100, args.dt, args.temperature)?;
                batch.run(args.steps, args.dt, args.temperature, 0.01)?;
            }

            let sequential_ms = seq_start.elapsed().as_secs_f64() * 1000.0;
            println!("      Sequential: {:.1}ms ({:.1}ms per structure)",
                     sequential_ms, sequential_ms / n_clones as f64);

            // Batched parallel - all clones in single kernel
            println!("\n   🚀 Batched parallel (all {} at once)...", n_clones);
            let batch_start = Instant::now();

            let mut batch = AmberSimdBatch::new_with_config(
                context.clone(),
                n_atoms + 100,
                n_clones,
                OptimizationConfig::default(), // Full SOTA
            )?;

            for _ in 0..n_clones {
                batch.add_structure(&structure_topo)?;
            }

            batch.finalize_batch()?;
            batch.initialize_velocities(args.temperature)?;
            batch.equilibrate(100, args.dt, args.temperature)?;
            batch.run(args.steps, args.dt, args.temperature, 0.01)?;

            let batched_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
            let speedup = sequential_ms / batched_ms;
            let effective_per_struct = batched_ms / n_clones as f64;
            let throughput = n_clones as f64 / (batched_ms / 1000.0);

            println!("      Batched: {:.1}ms ({:.1}ms effective per structure)",
                     batched_ms, effective_per_struct);
            println!("      ✨ SPEEDUP: {:.2}×", speedup);
            println!("      Throughput: {:.1} structures/sec", throughput);

            results.push(SpeedupResult {
                n_clones,
                sequential_ms,
                batched_ms,
                speedup,
                effective_ms_per_structure: effective_per_struct,
                throughput_structures_per_sec: throughput,
            });
        }

        // Find best speedup
        let (best_speedup, best_n_clones) = results.iter()
            .map(|r| (r.speedup, r.n_clones))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap_or((0.0, 0));

        // Print summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    SPEEDUP SUMMARY                            ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║   Clones │ Sequential │  Batched  │ Speedup │ Throughput     ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        for r in &results {
            println!("║   {:>5} │ {:>9.1}ms │ {:>8.1}ms │ {:>6.2}× │ {:>6.1} structs/s ║",
                     r.n_clones, r.sequential_ms, r.batched_ms,
                     r.speedup, r.throughput_structures_per_sec);
        }

        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║   BEST: {:.2}× speedup with {} clones                       ║",
                 best_speedup, best_n_clones);
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Save report
        let structure_name = args.topology.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let report = TestReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            structure: structure_name.clone(),
            n_atoms,
            steps: args.steps,
            results,
            best_speedup,
            best_n_clones,
        };

        let report_path = args.output.join(format!("{}_speedup.json", structure_name));
        let report_file = File::create(&report_path)?;
        serde_json::to_writer_pretty(report_file, &report)?;
        println!("\n📁 Report saved to: {:?}", report_path);
    }

    #[cfg(not(feature = "cuda"))]
    {
        bail!("CUDA feature not enabled. Rebuild with: cargo run --release --features cuda ...");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn convert_prism_prep_to_structure_topology(
    prism_topo: &PrismPrepTopology,
) -> prism_gpu::StructureTopology {
    let positions = prism_topo.positions.clone();
    let masses = prism_topo.masses.clone();
    let charges = prism_topo.charges.clone();

    let sigmas: Vec<f32> = prism_topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = prism_topo.lj_params.iter().map(|p| p.epsilon).collect();

    let bonds: Vec<(usize, usize, f32, f32)> = prism_topo.bonds.iter()
        .map(|b| (b.i, b.j, b.k, b.r0))
        .collect();

    let angles: Vec<(usize, usize, usize, f32, f32)> = prism_topo.angles.iter()
        .map(|a| (a.i, a.j, a.k_idx, a.force_k, a.theta0))
        .collect();

    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = prism_topo.dihedrals.iter()
        .map(|d| (d.i, d.j, d.k_idx, d.l, d.force_k, d.periodicity as f32, d.phase))
        .collect();

    let exclusions: Vec<HashSet<usize>> = prism_topo.exclusions.iter()
        .map(|e| e.iter().cloned().collect())
        .collect();

    prism_gpu::StructureTopology {
        positions,
        masses,
        charges,
        sigmas,
        epsilons,
        bonds,
        angles,
        dihedrals,
        exclusions,
    }
}
