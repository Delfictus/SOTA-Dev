//! SOTA Validation Benchmark - Raw PDB → SIMD Batched MD
//!
//! Validates the SOTA optimizations by processing raw PDB files through
//! the complete pipeline:
//!
//! 1. PDB Sanitization (remove waters, HETATM, alternate conformations)
//! 2. AMBER Topology Generation (bonds, angles, dihedrals, LJ params)
//! 3. SIMD Batched MD with SOTA optimizations
//! 4. Performance comparison (Legacy vs SOTA)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda -p prism-validation --bin sota-validation-benchmark -- \
//!     --pdb-dir data/curated_14 \
//!     --pdbs 3SQQ 4QWO 6LU7 1AKE 6M0J 4J1G \
//!     --steps 10000 \
//!     --output results/sota_validation
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "sota-validation-benchmark")]
#[command(about = "SOTA validation: Raw PDB → Sanitize → Topology → SIMD Batch MD")]
struct Args {
    /// Directory containing raw PDB files
    #[arg(long, default_value = "data/curated_14")]
    pdb_dir: PathBuf,

    /// PDB IDs to process (without .pdb extension)
    #[arg(long, num_args = 1.., required = true)]
    pdbs: Vec<String>,

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
    #[arg(long, default_value = "results/sota_validation")]
    output: PathBuf,

    /// Skip legacy comparison (SOTA only)
    #[arg(long)]
    skip_legacy: bool,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
}

/// Validation result for a single structure
#[derive(Debug, Serialize)]
struct StructureResult {
    pdb_id: String,
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    sanitization_ms: f64,
    topology_ms: f64,
    legacy_md_ms: Option<f64>,
    sota_md_ms: Option<f64>,
    speedup: Option<f64>,
    final_pe_legacy: Option<f64>,
    final_pe_sota: Option<f64>,
    final_temp_legacy: Option<f64>,
    final_temp_sota: Option<f64>,
    verlet_rebuilds: Option<u32>,
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
    sota_optimizations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkSummary {
    total_structures: usize,
    successful: usize,
    failed: usize,
    total_atoms: usize,
    avg_speedup: f64,
    max_speedup: f64,
    min_speedup: f64,
    total_legacy_ms: f64,
    total_sota_ms: f64,
    overall_speedup: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       PRISM-4D SOTA Validation Benchmark                     ║");
    println!("║       Raw PDB → Sanitize → Topology → SIMD Batch MD          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    println!("\n📊 Configuration:");
    println!("   PDB Directory: {:?}", args.pdb_dir);
    println!("   Structures: {:?}", args.pdbs);
    println!("   MD Steps: {}", args.steps);
    println!("   Temperature: {} K", args.temperature);
    println!("   Timestep: {} fs", args.dt);
    println!("   Legacy comparison: {}", !args.skip_legacy);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig, BATCH_SPATIAL_OFFSET};
        use prism_physics::amber_ff14sb::{AmberTopology, PdbAtom};
        use prism_validation::pdb_sanitizer::{PdbSanitizer, SanitizedStructure};
        use std::sync::Arc;

        // Initialize CUDA
        println!("\n🚀 Initializing CUDA...");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;

        // Get GPU info
        println!("   GPU: Device {}", context.cu_device());

        let mut results: Vec<StructureResult> = Vec::new();
        let sanitizer = PdbSanitizer::new();

        for pdb_id in &args.pdbs {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Processing: {}", pdb_id);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let mut result = StructureResult {
                pdb_id: pdb_id.clone(),
                n_atoms: 0,
                n_residues: 0,
                n_chains: 0,
                n_bonds: 0,
                n_angles: 0,
                n_dihedrals: 0,
                sanitization_ms: 0.0,
                topology_ms: 0.0,
                legacy_md_ms: None,
                sota_md_ms: None,
                speedup: None,
                final_pe_legacy: None,
                final_pe_sota: None,
                final_temp_legacy: None,
                final_temp_sota: None,
                verlet_rebuilds: None,
                success: false,
                error: None,
            };

            // Find PDB file
            let pdb_path = args.pdb_dir.join(format!("{}.pdb", pdb_id));
            if !pdb_path.exists() {
                // Try lowercase
                let pdb_path_lower = args.pdb_dir.join(format!("{}.pdb", pdb_id.to_lowercase()));
                if !pdb_path_lower.exists() {
                    result.error = Some(format!("PDB file not found: {:?}", pdb_path));
                    results.push(result);
                    continue;
                }
            }

            let pdb_path = if pdb_path.exists() { pdb_path } else {
                args.pdb_dir.join(format!("{}.pdb", pdb_id.to_lowercase()))
            };

            // Step 1: Read and sanitize PDB
            let sanitize_start = Instant::now();
            let pdb_content = match fs::read_to_string(&pdb_path) {
                Ok(c) => c,
                Err(e) => {
                    result.error = Some(format!("Failed to read PDB: {}", e));
                    results.push(result);
                    continue;
                }
            };

            let sanitized = match sanitizer.sanitize(&pdb_content, pdb_id) {
                Ok(s) => s,
                Err(e) => {
                    result.error = Some(format!("Sanitization failed: {}", e));
                    results.push(result);
                    continue;
                }
            };
            result.sanitization_ms = sanitize_start.elapsed().as_secs_f64() * 1000.0;

            result.n_atoms = sanitized.atoms.len();
            result.n_residues = sanitized.ca_residues.len();
            result.n_chains = sanitized.chains.len();

            println!("   ✓ Sanitized: {} atoms, {} residues, {} chains",
                     result.n_atoms, result.n_residues, result.n_chains);
            println!("     Removed: {} HETATM, {} waters, {} alt conformations",
                     sanitized.stats.hetatm_removed,
                     sanitized.stats.waters_removed,
                     sanitized.stats.altloc_removed);

            // Step 2: Generate AMBER topology
            let topology_start = Instant::now();
            let pdb_atoms: Vec<PdbAtom> = sanitized.atoms.iter().map(|a| {
                PdbAtom {
                    index: a.index,
                    name: a.name.clone(),
                    residue_name: a.residue_name.clone(),
                    residue_id: a.original_res_seq,
                    chain_id: a.chain_id,
                    x: a.position[0],
                    y: a.position[1],
                    z: a.position[2],
                }
            }).collect();

            let amber_topo = AmberTopology::from_pdb_atoms(&pdb_atoms);
            result.topology_ms = topology_start.elapsed().as_secs_f64() * 1000.0;

            result.n_bonds = amber_topo.bonds.len();
            result.n_angles = amber_topo.angles.len();
            result.n_dihedrals = amber_topo.dihedrals.len();

            println!("   ✓ Topology: {} bonds, {} angles, {} dihedrals",
                     result.n_bonds, result.n_angles, result.n_dihedrals);

            // Convert to StructureTopology format
            let structure_topo = convert_to_structure_topology(&sanitized, &amber_topo);

            results.push(result);
        }

        // ============================================================
        // BATCHED MD BENCHMARK - This is where the real speedup happens!
        // ============================================================

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║           BATCHED MD BENCHMARK (All Structures in Parallel)  ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Collect all successful topologies
        let successful_topos: Vec<_> = results.iter()
            .filter(|r| r.error.is_none())
            .collect();

        if successful_topos.is_empty() {
            println!("   ✗ No structures available for batched benchmark");
        } else {
            // Re-process structures to get topologies for batching
            let mut all_topos: Vec<prism_gpu::StructureTopology> = Vec::new();
            let mut all_names: Vec<String> = Vec::new();

            for pdb_id in &args.pdbs {
                let pdb_path = if args.pdb_dir.join(format!("{}.pdb", pdb_id)).exists() {
                    args.pdb_dir.join(format!("{}.pdb", pdb_id))
                } else {
                    args.pdb_dir.join(format!("{}.pdb", pdb_id.to_lowercase()))
                };

                if let Ok(content) = fs::read_to_string(&pdb_path) {
                    if let Ok(sanitized) = sanitizer.sanitize(&content, pdb_id) {
                        let pdb_atoms: Vec<PdbAtom> = sanitized.atoms.iter().map(|a| {
                            PdbAtom {
                                index: a.index,
                                name: a.name.clone(),
                                residue_name: a.residue_name.clone(),
                                residue_id: a.original_res_seq,
                                chain_id: a.chain_id,
                                x: a.position[0],
                                y: a.position[1],
                                z: a.position[2],
                            }
                        }).collect();
                        let amber_topo = AmberTopology::from_pdb_atoms(&pdb_atoms);
                        let structure_topo = convert_to_structure_topology(&sanitized, &amber_topo);
                        all_topos.push(structure_topo);
                        all_names.push(pdb_id.clone());
                    }
                }
            }

            let n_batch = all_topos.len();
            let max_atoms = all_topos.iter().map(|t| t.positions.len() / 3).max().unwrap_or(0);

            println!("\n📊 Batch Configuration:");
            println!("   Structures in batch: {}", n_batch);
            println!("   Max atoms/structure: {}", max_atoms);
            println!("   Total atoms: {}", all_topos.iter().map(|t| t.positions.len() / 3).sum::<usize>());

            // ---- SEQUENTIAL BASELINE (one at a time) ----
            println!("\n⏱️  SEQUENTIAL BASELINE (processing one-by-one)...");
            let seq_start = Instant::now();

            for topo in &all_topos {
                let mut batch = AmberSimdBatch::new_with_config(
                    context.clone(),
                    max_atoms + 100,
                    1,
                    OptimizationConfig::legacy(),
                )?;
                batch.add_structure(topo)?;
                batch.finalize_batch()?;
                batch.initialize_velocities(args.temperature)?;
                batch.equilibrate(500, args.dt, args.temperature)?;
                batch.run(args.steps, args.dt, args.temperature, 0.01)?;
            }

            let seq_elapsed = seq_start.elapsed().as_secs_f64() * 1000.0;
            println!("   Sequential total: {:.1}ms ({:.1}ms per structure)",
                     seq_elapsed, seq_elapsed / n_batch as f64);

            // ---- PARALLEL BATCHED (all at once with SOTA) ----
            println!("\n🚀 PARALLEL BATCHED (all {} structures in single kernel)...", n_batch);
            let batch_start = Instant::now();

            let mut batch = AmberSimdBatch::new_with_config(
                context.clone(),
                max_atoms + 100,
                n_batch,
                OptimizationConfig::default(), // Full SOTA optimizations
            )?;

            for topo in &all_topos {
                batch.add_structure(topo)?;
            }
            batch.finalize_batch()?;
            batch.initialize_velocities(args.temperature)?;
            batch.equilibrate(500, args.dt, args.temperature)?;
            batch.run(args.steps, args.dt, args.temperature, 0.01)?;

            let batch_elapsed = batch_start.elapsed().as_secs_f64() * 1000.0;
            let batch_results = batch.get_all_results()?;
            let batch_stats = batch.sota_stats();

            println!("   Batched total: {:.1}ms ({:.1}ms effective per structure)",
                     batch_elapsed, batch_elapsed / n_batch as f64);
            println!("   Verlet rebuilds: {}", batch_stats.verlet_rebuild_count);

            // Calculate real speedup
            let real_speedup = seq_elapsed / batch_elapsed;
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                    TRUE BATCHED SPEEDUP                       ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║   Sequential: {:>10.1}ms                                   ║", seq_elapsed);
            println!("║   Batched:    {:>10.1}ms                                   ║", batch_elapsed);
            println!("║   SPEEDUP:    {:>10.2}× 🚀                                 ║", real_speedup);
            println!("╚══════════════════════════════════════════════════════════════╝");

            // Update results with batch info
            for (i, result) in results.iter_mut().enumerate() {
                if i < batch_results.len() {
                    result.final_pe_sota = Some(batch_results[i].potential_energy);
                    result.final_temp_sota = Some(batch_results[i].temperature);
                    result.sota_md_ms = Some(batch_elapsed / n_batch as f64);
                    result.legacy_md_ms = Some(seq_elapsed / n_batch as f64);
                    result.speedup = Some(real_speedup);
                    result.verlet_rebuilds = Some(batch_stats.verlet_rebuild_count);
                    result.success = true;
                }
            }
        }

        // Generate summary
        let successful: Vec<_> = results.iter().filter(|r| r.success).collect();
        let speedups: Vec<f64> = successful.iter()
            .filter_map(|r| r.speedup)
            .collect();

        let total_legacy: f64 = successful.iter()
            .filter_map(|r| r.legacy_md_ms)
            .sum();
        let total_sota: f64 = successful.iter()
            .filter_map(|r| r.sota_md_ms)
            .sum();

        let summary = BenchmarkSummary {
            total_structures: results.len(),
            successful: successful.len(),
            failed: results.len() - successful.len(),
            total_atoms: successful.iter().map(|r| r.n_atoms).sum(),
            avg_speedup: if speedups.is_empty() { 0.0 } else {
                speedups.iter().sum::<f64>() / speedups.len() as f64
            },
            max_speedup: speedups.iter().cloned().fold(0.0_f64, f64::max),
            min_speedup: speedups.iter().cloned().fold(f64::MAX, f64::min),
            total_legacy_ms: total_legacy,
            total_sota_ms: total_sota,
            overall_speedup: if total_sota > 0.0 { total_legacy / total_sota } else { 0.0 },
        };

        // Create report
        let report = BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: BenchmarkConfig {
                steps: args.steps,
                temperature: args.temperature,
                dt: args.dt,
                sota_optimizations: vec![
                    "Verlet neighbor lists".into(),
                    "Tensor Core WMMA".into(),
                    "FP16 mixed precision".into(),
                    "Async stream overlap".into(),
                    "True batched processing".into(),
                ],
            },
            structures: results,
            summary: summary.clone(),
        };

        // Save report
        let report_path = args.output.join("benchmark_report.json");
        let report_file = File::create(&report_path)?;
        serde_json::to_writer_pretty(report_file, &report)?;

        // Print summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BENCHMARK COMPLETE                         ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\n📊 Summary:");
        println!("   Structures: {}/{} successful", summary.successful, summary.total_structures);
        println!("   Total atoms processed: {}", summary.total_atoms);
        if !args.skip_legacy {
            println!("   Legacy total: {:.1}ms", summary.total_legacy_ms);
            println!("   SOTA total: {:.1}ms", summary.total_sota_ms);
            println!("   Overall speedup: {:.2}×", summary.overall_speedup);
            if !speedups.is_empty() {
                println!("   Per-structure speedup: {:.2}× avg, {:.2}× max, {:.2}× min",
                         summary.avg_speedup, summary.max_speedup, summary.min_speedup);
            }
        }
        println!("\n📁 Report saved to: {:?}", report_path);
    }

    #[cfg(not(feature = "cuda"))]
    {
        bail!("CUDA feature not enabled. Rebuild with: cargo run --release --features cuda ...");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn convert_to_structure_topology(
    sanitized: &prism_validation::pdb_sanitizer::SanitizedStructure,
    amber: &prism_physics::amber_ff14sb::AmberTopology,
) -> prism_gpu::StructureTopology {
    use std::collections::HashSet;

    // Positions from sanitized structure
    let positions: Vec<f32> = sanitized.atoms.iter()
        .flat_map(|a| vec![a.position[0], a.position[1], a.position[2]])
        .collect();

    // Masses, charges, LJ params from AMBER topology
    let masses: Vec<f32> = amber.masses.clone();
    let charges: Vec<f32> = amber.charges.clone();

    // LJ params: convert rmin_half to sigma (sigma = rmin_half * 2^(5/6))
    // Actually for AMBER, we use rmin_half directly as sigma input
    let sigmas: Vec<f32> = amber.lj_params.iter().map(|p| p.rmin_half * 2.0).collect();
    let epsilons: Vec<f32> = amber.lj_params.iter().map(|p| p.epsilon).collect();

    // Bonds: (i, j, k, r0) - connectivity and params stored separately
    let bonds: Vec<(usize, usize, f32, f32)> = amber.bonds.iter()
        .zip(amber.bond_params.iter())
        .map(|((i, j), param)| (*i as usize, *j as usize, param.k, param.r0))
        .collect();

    // Angles: (i, j, k, force_k, theta0) - connectivity and params stored separately
    let angles: Vec<(usize, usize, usize, f32, f32)> = amber.angles.iter()
        .zip(amber.angle_params.iter())
        .map(|((i, j, k), param)| (*i as usize, *j as usize, *k as usize, param.k, param.theta0))
        .collect();

    // Dihedrals: (i, j, k, l, force_k, periodicity, phase)
    // Each dihedral can have multiple terms (Fourier series), we flatten them
    let mut dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = Vec::new();
    for ((i, j, k, l), params) in amber.dihedrals.iter().zip(amber.dihedral_params.iter()) {
        for param in params {
            dihedrals.push((
                *i as usize, *j as usize, *k as usize, *l as usize,
                param.k, param.n as f32, param.phase
            ));
        }
    }

    // Build exclusion lists
    let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); amber.n_atoms];

    // Add 1-2 exclusions (bonded atoms)
    for (i, j) in &amber.bonds {
        exclusions[*i as usize].insert(*j as usize);
        exclusions[*j as usize].insert(*i as usize);
    }

    // Add 1-3 exclusions (atoms connected through an angle)
    for (i, _, k) in &amber.angles {
        exclusions[*i as usize].insert(*k as usize);
        exclusions[*k as usize].insert(*i as usize);
    }

    // Add 1-4 exclusions (atoms connected through a dihedral)
    for (i, _, _, l) in &amber.dihedrals {
        exclusions[*i as usize].insert(*l as usize);
        exclusions[*l as usize].insert(*i as usize);
    }

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

#[cfg(feature = "cuda")]
fn run_md_benchmark(
    context: std::sync::Arc<cudarc::driver::CudaContext>,
    topo: &prism_gpu::StructureTopology,
    steps: usize,
    dt: f32,
    temperature: f32,
    opt_config: prism_gpu::OptimizationConfig,
) -> Result<(f64, f64)> {
    use prism_gpu::AmberSimdBatch;

    let n_atoms = topo.positions.len() / 3;

    // Create batch engine with specified config
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        n_atoms + 100,
        1, // Single structure
        opt_config,
    )?;

    batch.add_structure(topo)?;
    batch.finalize_batch()?;

    // Initialize velocities
    batch.initialize_velocities(temperature)?;

    // Equilibration
    batch.equilibrate(1000, dt, temperature)?;

    // Production
    batch.run(steps, dt, temperature, 0.01)?;

    // Get results
    let results = batch.get_all_results()?;
    if results.is_empty() {
        bail!("No results returned from MD");
    }

    Ok((results[0].potential_energy, results[0].temperature))
}

#[cfg(feature = "cuda")]
fn run_md_benchmark_sota(
    context: std::sync::Arc<cudarc::driver::CudaContext>,
    topo: &prism_gpu::StructureTopology,
    steps: usize,
    dt: f32,
    temperature: f32,
) -> Result<(f64, f64, u32)> {
    use prism_gpu::{AmberSimdBatch, OptimizationConfig};

    let n_atoms = topo.positions.len() / 3;

    // Create batch engine with SOTA optimizations
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        n_atoms + 100,
        1,
        OptimizationConfig::default(), // All SOTA optimizations enabled
    )?;

    batch.add_structure(topo)?;
    batch.finalize_batch()?;

    // Initialize velocities
    batch.initialize_velocities(temperature)?;

    // Equilibration
    batch.equilibrate(1000, dt, temperature)?;

    // Production with SOTA
    batch.run(steps, dt, temperature, 0.01)?;

    // Get results and stats
    let results = batch.get_all_results()?;
    if results.is_empty() {
        bail!("No results returned from MD");
    }

    let stats = batch.sota_stats();
    let rebuilds = stats.verlet_rebuild_count;

    Ok((results[0].potential_energy, results[0].temperature, rebuilds))
}
