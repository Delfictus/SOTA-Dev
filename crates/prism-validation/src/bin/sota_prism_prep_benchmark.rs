//! SOTA Benchmark using Official Prism-Prep Topologies
//!
//! This benchmark uses pre-prepared topology files from prism-prep v1.2.0
//! instead of inline sanitization. This ensures production-quality topologies
//! with proper AMBER ff14SB parameterization.
//!
//! ## Usage
//!
//! ```bash
//! # First, prepare structures with prism-prep:
//! scripts/prism-prep data/raw/6M0J.pdb results/prep/6M0J.json --use-amber
//!
//! # Then run benchmark:
//! cargo run --release --features cuda -p prism-validation --bin sota-prism-prep-benchmark -- \
//!     --topo-dir results/sota_prep_test \
//!     --topos 6M0J_topology.json 1HXY_topology.json 5IRE_topology.json \
//!     --steps 10000 \
//!     --output results/sota_benchmark
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
#[command(name = "sota-prism-prep-benchmark")]
#[command(about = "SOTA benchmark using official prism-prep topology files")]
struct Args {
    /// Directory containing prepared topology JSON files
    #[arg(long, default_value = "results/sota_prep_test")]
    topo_dir: PathBuf,

    /// Topology JSON files to process (from prism-prep)
    #[arg(long, num_args = 1.., required = true)]
    topos: Vec<String>,

    /// MD steps per structure
    #[arg(long, default_value = "10000")]
    steps: usize,

    /// Temperature in Kelvin
    #[arg(long, default_value = "310.0")]
    temperature: f32,

    /// Timestep in femtoseconds
    #[arg(long, default_value = "2.0")]
    dt: f32,

    /// Output directory for results
    #[arg(long, default_value = "results/sota_benchmark")]
    output: PathBuf,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
}

/// Prism-prep topology JSON format
#[derive(Debug, Deserialize)]
struct PrismPrepTopology {
    source_pdb: String,
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    positions: Vec<f64>,
    masses: Vec<f64>,
    charges: Vec<f64>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondEntry>,
    angles: Vec<AngleEntry>,
    dihedrals: Vec<DihedralEntry>,
    exclusions: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct LjParam {
    sigma: f64,
    epsilon: f64,
}

#[derive(Debug, Deserialize)]
struct BondEntry {
    i: usize,
    j: usize,
    r0: f64,
    k: f64,
}

#[derive(Debug, Deserialize)]
struct AngleEntry {
    i: usize,
    j: usize,
    k_idx: usize,  // Third atom index (middle atom is j)
    theta0: f64,
    force_k: f64,
}

#[derive(Debug, Deserialize)]
struct DihedralEntry {
    i: usize,
    j: usize,
    k_idx: usize,  // Third atom index
    l: usize,
    periodicity: i32,
    phase: f64,
    force_k: f64,
}

/// Benchmark result for a single structure
#[derive(Debug, Serialize)]
struct StructureResult {
    name: String,
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    load_ms: f64,
    seq_md_ms: f64,
    final_pe: f64,
    final_temp: f64,
    success: bool,
    error: Option<String>,
}

/// Full benchmark report
#[derive(Debug, Serialize)]
struct BenchmarkReport {
    timestamp: String,
    config: BenchmarkConfig,
    structures: Vec<StructureResult>,
    batched_results: BatchedResults,
    summary: BenchmarkSummary,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
    steps: usize,
    temperature: f32,
    dt: f32,
    topo_source: String,
}

#[derive(Debug, Serialize)]
struct BatchedResults {
    n_structures: usize,
    total_atoms: usize,
    max_atoms_per_structure: usize,
    sequential_total_ms: f64,
    batched_total_ms: f64,
    speedup: f64,
    verlet_rebuilds: u32,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    total_structures: usize,
    successful: usize,
    failed: usize,
    total_atoms: usize,
    sequential_total_ms: f64,
    batched_total_ms: f64,
    speedup: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      PRISM-4D SOTA Benchmark (Prism-Prep Topologies)         ║");
    println!("║      Official Production-Quality AMBER ff14SB Topologies     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    println!("\n📊 Configuration:");
    println!("   Topology Directory: {:?}", args.topo_dir);
    println!("   Topologies: {:?}", args.topos);
    println!("   MD Steps: {}", args.steps);
    println!("   Temperature: {} K", args.temperature);
    println!("   Timestep: {} fs", args.dt);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};
        use std::sync::Arc;

        // Initialize CUDA
        println!("\n🚀 Initializing CUDA...");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        println!("   GPU: Device {}", context.cu_device());

        let mut results: Vec<StructureResult> = Vec::new();
        let mut loaded_topos: Vec<(String, StructureTopology)> = Vec::new();

        // Load all topologies
        println!("\n📦 Loading topologies from prism-prep...");
        for topo_name in &args.topos {
            let topo_path = args.topo_dir.join(topo_name);
            println!("\n   Loading: {}", topo_name);

            let load_start = Instant::now();

            // Load and parse topology JSON
            let file = match File::open(&topo_path) {
                Ok(f) => f,
                Err(e) => {
                    println!("   ✗ Failed to open: {}", e);
                    results.push(StructureResult {
                        name: topo_name.clone(),
                        n_atoms: 0,
                        n_residues: 0,
                        n_chains: 0,
                        n_bonds: 0,
                        n_angles: 0,
                        n_dihedrals: 0,
                        load_ms: 0.0,
                        seq_md_ms: 0.0,
                        final_pe: 0.0,
                        final_temp: 0.0,
                        success: false,
                        error: Some(format!("Failed to open file: {}", e)),
                    });
                    continue;
                }
            };

            let reader = BufReader::new(file);
            let prep_topo: PrismPrepTopology = match serde_json::from_reader(reader) {
                Ok(t) => t,
                Err(e) => {
                    println!("   ✗ Failed to parse: {}", e);
                    results.push(StructureResult {
                        name: topo_name.clone(),
                        n_atoms: 0,
                        n_residues: 0,
                        n_chains: 0,
                        n_bonds: 0,
                        n_angles: 0,
                        n_dihedrals: 0,
                        load_ms: 0.0,
                        seq_md_ms: 0.0,
                        final_pe: 0.0,
                        final_temp: 0.0,
                        success: false,
                        error: Some(format!("Failed to parse JSON: {}", e)),
                    });
                    continue;
                }
            };

            // Convert to StructureTopology
            let structure_topo = convert_prism_prep_to_structure_topology(&prep_topo);
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            println!("   ✓ {} atoms, {} residues, {} chains ({:.1}ms)",
                     prep_topo.n_atoms, prep_topo.n_residues, prep_topo.n_chains, load_ms);
            println!("     {} bonds, {} angles, {} dihedrals",
                     prep_topo.bonds.len(), prep_topo.angles.len(), prep_topo.dihedrals.len());

            results.push(StructureResult {
                name: topo_name.clone(),
                n_atoms: prep_topo.n_atoms,
                n_residues: prep_topo.n_residues,
                n_chains: prep_topo.n_chains,
                n_bonds: prep_topo.bonds.len(),
                n_angles: prep_topo.angles.len(),
                n_dihedrals: prep_topo.dihedrals.len(),
                load_ms,
                seq_md_ms: 0.0,
                final_pe: 0.0,
                final_temp: 0.0,
                success: true,
                error: None,
            });

            loaded_topos.push((topo_name.clone(), structure_topo));
        }

        let successful_count = loaded_topos.len();
        if successful_count == 0 {
            bail!("No topologies loaded successfully");
        }

        // ============================================================
        // SEQUENTIAL BASELINE (one structure at a time)
        // ============================================================
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║           SEQUENTIAL BASELINE (One-by-One Processing)        ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        let max_atoms = loaded_topos.iter()
            .map(|(_, t)| t.positions.len() / 3)
            .max()
            .unwrap_or(0);

        let seq_start = Instant::now();
        let mut seq_results_data: Vec<(f64, f64)> = Vec::new();

        for (name, topo) in &loaded_topos {
            let n_atoms = topo.positions.len() / 3;
            println!("\n   Processing: {} ({} atoms)", name, n_atoms);

            let struct_start = Instant::now();

            // Create fresh batch for each structure (simulates no batching)
            let mut batch = AmberSimdBatch::new_with_config(
                context.clone(),
                max_atoms + 100,
                1,
                OptimizationConfig::legacy(), // No SOTA optimizations for baseline
            )?;

            batch.add_structure(topo)?;
            batch.finalize_batch()?;
            batch.initialize_velocities(args.temperature)?;
            batch.equilibrate(500, args.dt, args.temperature)?;
            batch.run(args.steps, args.dt, args.temperature, 0.01)?;

            let struct_result = batch.get_all_results()?;
            let elapsed_ms = struct_start.elapsed().as_secs_f64() * 1000.0;

            if !struct_result.is_empty() {
                let pe = struct_result[0].potential_energy;
                let temp = struct_result[0].temperature;
                seq_results_data.push((pe, temp));
                println!("     ✓ Done in {:.1}ms | PE: {:.1} kcal/mol | T: {:.1} K",
                         elapsed_ms, pe, temp);
            }
        }

        let seq_total_ms = seq_start.elapsed().as_secs_f64() * 1000.0;
        println!("\n   SEQUENTIAL TOTAL: {:.1}ms ({:.1}ms per structure)",
                 seq_total_ms, seq_total_ms / successful_count as f64);

        // ============================================================
        // PARALLEL BATCHED (all structures in single kernel)
        // ============================================================
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║           PARALLEL BATCHED (All {} Structures Together)      ║", successful_count);
        println!("╚══════════════════════════════════════════════════════════════╝");

        let total_atoms: usize = loaded_topos.iter()
            .map(|(_, t)| t.positions.len() / 3)
            .sum();

        println!("\n   Batch Configuration:");
        println!("     Structures in batch: {}", successful_count);
        println!("     Max atoms/structure: {}", max_atoms);
        println!("     Total atoms: {}", total_atoms);

        let batch_start = Instant::now();

        let mut batch = AmberSimdBatch::new_with_config(
            context.clone(),
            max_atoms + 100,
            successful_count,
            OptimizationConfig::default(), // Full SOTA optimizations
        )?;

        for (_, topo) in &loaded_topos {
            batch.add_structure(topo)?;
        }

        batch.finalize_batch()?;
        batch.initialize_velocities(args.temperature)?;
        batch.equilibrate(500, args.dt, args.temperature)?;
        batch.run(args.steps, args.dt, args.temperature, 0.01)?;

        let batch_results_data = batch.get_all_results()?;
        let batch_stats = batch.sota_stats();
        let batch_total_ms = batch_start.elapsed().as_secs_f64() * 1000.0;

        println!("\n   Batched Results:");
        for (i, res) in batch_results_data.iter().enumerate() {
            let name = &loaded_topos[i].0;
            println!("     {} | PE: {:.1} kcal/mol | T: {:.1} K",
                     name, res.potential_energy, res.temperature);
        }

        println!("\n   BATCHED TOTAL: {:.1}ms ({:.1}ms effective per structure)",
                 batch_total_ms, batch_total_ms / successful_count as f64);
        println!("   Verlet rebuilds: {}", batch_stats.verlet_rebuild_count);

        // Calculate speedup
        let speedup = seq_total_ms / batch_total_ms;

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    TRUE BATCHED SPEEDUP                       ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║   Sequential: {:>10.1}ms                                   ║", seq_total_ms);
        println!("║   Batched:    {:>10.1}ms                                   ║", batch_total_ms);
        println!("║   SPEEDUP:    {:>10.2}× 🚀                                 ║", speedup);
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Update results with final values
        for (i, result) in results.iter_mut().enumerate() {
            if result.success && i < seq_results_data.len() {
                result.seq_md_ms = seq_total_ms / successful_count as f64;
                result.final_pe = seq_results_data[i].0;
                result.final_temp = seq_results_data[i].1;
            }
        }

        // Create report
        let report = BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: BenchmarkConfig {
                steps: args.steps,
                temperature: args.temperature,
                dt: args.dt,
                topo_source: "prism-prep v1.2.0".into(),
            },
            structures: results,
            batched_results: BatchedResults {
                n_structures: successful_count,
                total_atoms,
                max_atoms_per_structure: max_atoms,
                sequential_total_ms: seq_total_ms,
                batched_total_ms: batch_total_ms,
                speedup,
                verlet_rebuilds: batch_stats.verlet_rebuild_count,
            },
            summary: BenchmarkSummary {
                total_structures: args.topos.len(),
                successful: successful_count,
                failed: args.topos.len() - successful_count,
                total_atoms,
                sequential_total_ms: seq_total_ms,
                batched_total_ms: batch_total_ms,
                speedup,
            },
        };

        // Save report
        let report_path = args.output.join("benchmark_report.json");
        let report_file = File::create(&report_path)?;
        serde_json::to_writer_pretty(report_file, &report)?;

        println!("\n📁 Report saved to: {:?}", report_path);

        // Print final summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BENCHMARK COMPLETE                         ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\n📊 Summary:");
        println!("   Structures: {}/{} successful", successful_count, args.topos.len());
        println!("   Total atoms: {}", total_atoms);
        println!("   Sequential: {:.1}ms total", seq_total_ms);
        println!("   Batched: {:.1}ms total", batch_total_ms);
        println!("   Speedup: {:.2}×", speedup);
    }

    #[cfg(not(feature = "cuda"))]
    {
        bail!("CUDA feature not enabled. Rebuild with: cargo run --release --features cuda ...");
    }

    Ok(())
}

/// Convert prism-prep topology JSON to StructureTopology format
#[cfg(feature = "cuda")]
fn convert_prism_prep_to_structure_topology(
    prep: &PrismPrepTopology,
) -> prism_gpu::StructureTopology {
    // Positions: f64 -> f32
    let positions: Vec<f32> = prep.positions.iter().map(|&p| p as f32).collect();

    // Masses: f64 -> f32
    let masses: Vec<f32> = prep.masses.iter().map(|&m| m as f32).collect();

    // Charges: f64 -> f32
    let charges: Vec<f32> = prep.charges.iter().map(|&c| c as f32).collect();

    // LJ params: extract sigma and epsilon
    let sigmas: Vec<f32> = prep.lj_params.iter().map(|p| p.sigma as f32).collect();
    let epsilons: Vec<f32> = prep.lj_params.iter().map(|p| p.epsilon as f32).collect();

    // Bonds: (i, j, k, r0)
    let bonds: Vec<(usize, usize, f32, f32)> = prep.bonds.iter()
        .map(|b| (b.i, b.j, b.k as f32, b.r0 as f32))
        .collect();

    // Angles: (i, j, k, force_k, theta0)
    // Note: j is the middle atom (vertex), i and k are the end atoms
    let angles: Vec<(usize, usize, usize, f32, f32)> = prep.angles.iter()
        .map(|a| (a.i, a.j, a.k_idx, a.force_k as f32, a.theta0 as f32))
        .collect();

    // Dihedrals: (i, j, k, l, force_k, periodicity, phase)
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = prep.dihedrals.iter()
        .map(|d| (d.i, d.j, d.k_idx, d.l, d.force_k as f32, d.periodicity as f32, d.phase as f32))
        .collect();

    // Exclusions: Vec<Vec<usize>> -> Vec<HashSet<usize>>
    let exclusions: Vec<HashSet<usize>> = prep.exclusions.iter()
        .map(|exc| exc.iter().cloned().collect())
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
