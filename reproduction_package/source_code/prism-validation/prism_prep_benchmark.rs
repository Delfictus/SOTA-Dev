//! PRISM-PREP Benchmark - Tests SOTA MD Engine with Official Preprocessing
//!
//! This benchmark uses topology JSON files produced by the official prism-prep tool,
//! ensuring production-ready structure preparation.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda -p prism-validation --bin prism-prep-benchmark -- \
//!     --topology-dir results/sota_validation_fresh \
//!     --topos 6M0J 1HXY 5IRE \
//!     --steps 10000
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "prism-prep-benchmark")]
#[command(about = "SOTA MD benchmark using prism-prep topology JSON files")]
struct Args {
    /// Directory containing topology JSON files from prism-prep
    #[arg(long, default_value = "results/sota_validation_fresh")]
    topology_dir: PathBuf,

    /// Topology base names to process (without _topology.json)
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
    #[arg(long, default_value = "results/prism_prep_benchmark")]
    output: PathBuf,

    /// Skip sequential baseline (batched only)
    #[arg(long)]
    skip_sequential: bool,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
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
    #[serde(default)]
    h_clusters: Option<Vec<HCluster>>,
    source_pdb: Option<String>,
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

#[derive(Debug, Deserialize)]
struct HCluster {
    #[serde(rename = "type")]
    cluster_type: u32,
    central_atom: usize,
    hydrogen_atoms: Vec<i32>,  // -1 used as padding for smaller clusters
    bond_lengths: Vec<f32>,
    n_hydrogens: u32,
    inv_mass_central: f32,
    inv_mass_h: f32,
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
    sequential_ms: Option<f64>,
    batched_ms: Option<f64>,
    final_pe: Option<f64>,
    final_temp: Option<f64>,
    success: bool,
    error: Option<String>,
}

/// Full benchmark report
#[derive(Debug, Serialize)]
struct BenchmarkReport {
    timestamp: String,
    config: BenchmarkConfig,
    structures: Vec<StructureResult>,
    summary: BenchmarkSummary,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
    steps: usize,
    temperature: f32,
    dt: f32,
    n_structures: usize,
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
    println!("║       PRISM-PREP BENCHMARK - Official Topology MD Test       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    println!("\n📊 Configuration:");
    println!("   Topology Directory: {:?}", args.topology_dir);
    println!("   Structures: {:?}", args.topos);
    println!("   MD Steps: {}", args.steps);
    println!("   Temperature: {} K", args.temperature);
    println!("   Timestep: {} fs", args.dt);
    println!("   Sequential baseline: {}", !args.skip_sequential);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};
        use std::sync::Arc;

        // Initialize CUDA
        println!("\n🚀 Initializing CUDA...");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        println!("   GPU: Device {}", context.cu_device());

        // Load all topologies
        println!("\n📂 Loading topologies from prism-prep...");
        let mut topologies: Vec<(String, StructureTopology, PrismPrepTopology)> = Vec::new();
        let mut results: Vec<StructureResult> = Vec::new();

        for name in &args.topos {
            let topo_path = args.topology_dir.join(format!("{}_topology.json", name));

            println!("\n   Loading {}...", name);

            if !topo_path.exists() {
                let error_msg = format!("Topology file not found: {:?}", topo_path);
                println!("   ✗ {}", error_msg);
                results.push(StructureResult {
                    name: name.clone(),
                    n_atoms: 0,
                    n_residues: 0,
                    n_chains: 0,
                    n_bonds: 0,
                    n_angles: 0,
                    n_dihedrals: 0,
                    sequential_ms: None,
                    batched_ms: None,
                    final_pe: None,
                    final_temp: None,
                    success: false,
                    error: Some(error_msg),
                });
                continue;
            }

            let file = File::open(&topo_path)
                .with_context(|| format!("Failed to open {:?}", topo_path))?;
            let reader = BufReader::new(file);

            let prism_topo: PrismPrepTopology = serde_json::from_reader(reader)
                .with_context(|| format!("Failed to parse {:?}", topo_path))?;

            // Convert to StructureTopology format
            let structure_topo = convert_prism_prep_to_structure_topology(&prism_topo);

            println!("   ✓ {} atoms, {} bonds, {} angles, {} dihedrals",
                     prism_topo.n_atoms, prism_topo.bonds.len(),
                     prism_topo.angles.len(), prism_topo.dihedrals.len());

            topologies.push((name.clone(), structure_topo, prism_topo));
        }

        if topologies.is_empty() {
            bail!("No valid topologies loaded!");
        }

        let n_batch = topologies.len();
        let max_atoms = topologies.iter().map(|(_, t, _)| t.positions.len() / 3).max().unwrap_or(0);
        let total_atoms: usize = topologies.iter().map(|(_, t, _)| t.positions.len() / 3).sum();

        println!("\n📊 Batch Summary:");
        println!("   Structures: {}", n_batch);
        println!("   Max atoms per structure: {}", max_atoms);
        println!("   Total atoms: {}", total_atoms);

        // ============================================================
        // SEQUENTIAL BASELINE (one at a time)
        // ============================================================
        let mut sequential_total_ms = 0.0;

        if !args.skip_sequential {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  ⏱️  SEQUENTIAL BASELINE (one structure at a time)");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let seq_start = Instant::now();

            for (name, topo, prism_topo) in &topologies {
                let struct_start = Instant::now();

                let n_atoms = topo.positions.len() / 3;
                let mut batch = AmberSimdBatch::new_with_config(
                    context.clone(),
                    n_atoms + 100,
                    1,
                    OptimizationConfig::legacy(), // No SOTA opts for baseline
                )?;

                batch.add_structure(topo)?;
                batch.finalize_batch()?;
                batch.initialize_velocities(args.temperature)?;
                batch.equilibrate(500, args.dt, args.temperature)?;
                batch.run(args.steps, args.dt, args.temperature, 0.01)?;

                let struct_elapsed = struct_start.elapsed().as_secs_f64() * 1000.0;
                let batch_results = batch.get_all_results()?;

                println!("   {} ({} atoms): {:.1}ms, PE={:.1} kcal/mol, T={:.1}K",
                         name, n_atoms, struct_elapsed,
                         batch_results.get(0).map(|r| r.potential_energy).unwrap_or(0.0),
                         batch_results.get(0).map(|r| r.temperature).unwrap_or(0.0));

                // Store sequential result
                results.push(StructureResult {
                    name: name.clone(),
                    n_atoms: prism_topo.n_atoms,
                    n_residues: prism_topo.n_residues,
                    n_chains: prism_topo.n_chains,
                    n_bonds: prism_topo.bonds.len(),
                    n_angles: prism_topo.angles.len(),
                    n_dihedrals: prism_topo.dihedrals.len(),
                    sequential_ms: Some(struct_elapsed),
                    batched_ms: None,
                    final_pe: batch_results.get(0).map(|r| r.potential_energy),
                    final_temp: batch_results.get(0).map(|r| r.temperature),
                    success: true,
                    error: None,
                });
            }

            sequential_total_ms = seq_start.elapsed().as_secs_f64() * 1000.0;
            println!("\n   Sequential total: {:.1}ms ({:.1}ms per structure)",
                     sequential_total_ms, sequential_total_ms / n_batch as f64);
        }

        // ============================================================
        // BATCHED PARALLEL (all structures simultaneously)
        // ============================================================
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  🚀 BATCHED PARALLEL ({} structures in single kernel)", n_batch);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let batch_start = Instant::now();

        let mut batch = AmberSimdBatch::new_with_config(
            context.clone(),
            max_atoms + 100,
            n_batch,
            OptimizationConfig::default(), // Full SOTA optimizations
        )?;

        for (_, topo, _) in &topologies {
            batch.add_structure(topo)?;
        }

        batch.finalize_batch()?;
        batch.initialize_velocities(args.temperature)?;
        batch.equilibrate(500, args.dt, args.temperature)?;
        batch.run(args.steps, args.dt, args.temperature, 0.01)?;

        let batched_total_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        let batch_results = batch.get_all_results()?;
        let batch_stats = batch.sota_stats();

        println!("   Batched total: {:.1}ms ({:.1}ms effective per structure)",
                 batched_total_ms, batched_total_ms / n_batch as f64);
        println!("   Verlet rebuilds: {}", batch_stats.verlet_rebuild_count);

        // Update results with batched times
        for (i, (name, topo, _)) in topologies.iter().enumerate() {
            if args.skip_sequential {
                // Create new result entry if we skipped sequential
                let prism_topo = &topologies[i].2;
                results.push(StructureResult {
                    name: name.clone(),
                    n_atoms: prism_topo.n_atoms,
                    n_residues: prism_topo.n_residues,
                    n_chains: prism_topo.n_chains,
                    n_bonds: prism_topo.bonds.len(),
                    n_angles: prism_topo.angles.len(),
                    n_dihedrals: prism_topo.dihedrals.len(),
                    sequential_ms: None,
                    batched_ms: Some(batched_total_ms / n_batch as f64),
                    final_pe: batch_results.get(i).map(|r| r.potential_energy),
                    final_temp: batch_results.get(i).map(|r| r.temperature),
                    success: true,
                    error: None,
                });
            } else if i < results.len() {
                results[i].batched_ms = Some(batched_total_ms / n_batch as f64);
                results[i].final_pe = batch_results.get(i).map(|r| r.potential_energy);
                results[i].final_temp = batch_results.get(i).map(|r| r.temperature);
            }

            if let Some(br) = batch_results.get(i) {
                println!("   {} ({} atoms): PE={:.1} kcal/mol, T={:.1}K",
                         name, topo.positions.len() / 3, br.potential_energy, br.temperature);
            }
        }

        // Calculate speedup
        let speedup = if sequential_total_ms > 0.0 {
            sequential_total_ms / batched_total_ms
        } else {
            0.0
        };

        // Print final results
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BENCHMARK RESULTS                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        if !args.skip_sequential {
            println!("║   Sequential: {:>10.1}ms                                   ║", sequential_total_ms);
        }
        println!("║   Batched:    {:>10.1}ms                                   ║", batched_total_ms);
        if !args.skip_sequential {
            println!("║   SPEEDUP:    {:>10.2}× 🚀                                 ║", speedup);
        }
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Create summary
        let summary = BenchmarkSummary {
            total_structures: n_batch,
            successful: results.iter().filter(|r| r.success).count(),
            failed: results.iter().filter(|r| !r.success).count(),
            total_atoms,
            sequential_total_ms,
            batched_total_ms,
            speedup,
        };

        // Create report
        let report = BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: BenchmarkConfig {
                steps: args.steps,
                temperature: args.temperature,
                dt: args.dt,
                n_structures: n_batch,
            },
            structures: results,
            summary,
        };

        // Save report
        let report_path = args.output.join("benchmark_report.json");
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
    // Positions are already flat in prism-prep output
    let positions = prism_topo.positions.clone();
    let masses = prism_topo.masses.clone();
    let charges = prism_topo.charges.clone();

    // LJ parameters
    let sigmas: Vec<f32> = prism_topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = prism_topo.lj_params.iter().map(|p| p.epsilon).collect();

    // Bonds: (i, j, k, r0)
    let bonds: Vec<(usize, usize, f32, f32)> = prism_topo.bonds.iter()
        .map(|b| (b.i, b.j, b.k, b.r0))
        .collect();

    // Angles: (i, j, k, force_k, theta0)
    // Note: prism-prep uses k_idx for the middle atom
    let angles: Vec<(usize, usize, usize, f32, f32)> = prism_topo.angles.iter()
        .map(|a| (a.i, a.j, a.k_idx, a.force_k, a.theta0))
        .collect();

    // Dihedrals: (i, j, k, l, force_k, periodicity, phase)
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = prism_topo.dihedrals.iter()
        .map(|d| (d.i, d.j, d.k_idx, d.l, d.force_k, d.periodicity as f32, d.phase))
        .collect();

    // Exclusions: convert Vec<Vec<usize>> to Vec<HashSet<usize>>
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
