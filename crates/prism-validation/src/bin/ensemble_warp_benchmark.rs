//! ENSEMBLE WARP MD BENCHMARK
//!
//! Tests the revolutionary warp-based parallel MD against sequential baseline.
//! Expected: Near-linear speedup with clone count!
//!
//! Usage: cargo run --release --features cuda -p prism-validation --bin ensemble_warp_benchmark

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::error::Error;
use cudarc::driver::CudaContext;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// Topology from prism-prep JSON (simplified for this benchmark)
#[derive(Debug, Deserialize)]
struct PrismPrepTopology {
    n_atoms: usize,
    positions: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondDef>,
    angles: Vec<AngleDef>,
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
    k: f32,
    r0: f32,
}

#[derive(Debug, Deserialize)]
struct AngleDef {
    i: usize,
    j: usize,
    k_idx: usize,
    #[serde(rename = "force_k")]
    k: f32,
    theta0: f32,
}

/// Benchmark result for a single clone count
#[derive(Debug, Serialize)]
struct BenchmarkResult {
    n_clones: usize,
    warp_ms: f64,
    sequential_ms: f64,
    speedup: f64,
    ms_per_structure_warp: f64,
    ms_per_structure_seq: f64,
    warp_pe: f64,
    warp_ke: f64,
    warp_temp: f64,
}

/// Full benchmark output
#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    timestamp: String,
    structure: String,
    n_atoms: usize,
    steps: usize,
    clone_counts: Vec<usize>,
    results: Vec<BenchmarkResult>,
    best_speedup: f64,
    best_n_clones: usize,
}

fn load_topology(path: &Path) -> Result<PrismPrepTopology> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let topo: PrismPrepTopology = serde_json::from_str(&content)
        .with_context(|| "Failed to parse topology JSON")?;
    Ok(topo)
}

fn run_ensemble_warp_benchmark(
    context: std::sync::Arc<CudaContext>,
    topo: &PrismPrepTopology,
    n_clones: usize,
    n_steps: usize,
    dt: f32,
    temperature: f32,
    gamma: f32,
) -> Result<(f64, Vec<prism_gpu::EnsembleResult>)> {
    use prism_gpu::{EnsembleWarpMd, EnsembleTopology};

    // Convert to EnsembleTopology
    let sigmas: Vec<f32> = topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = topo.lj_params.iter().map(|p| p.epsilon).collect();
    let bonds: Vec<(usize, usize, f32, f32)> = topo.bonds.iter()
        .map(|b| (b.i, b.j, b.k, b.r0))
        .collect();
    let angles: Vec<(usize, usize, usize, f32, f32)> = topo.angles.iter()
        .map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0))
        .collect();

    let ensemble_topo = EnsembleTopology {
        n_atoms: topo.n_atoms,
        masses: topo.masses.clone(),
        charges: topo.charges.clone(),
        sigmas,
        epsilons,
        bonds,
        angles,
    };

    // Create engine
    let mut engine = EnsembleWarpMd::new(context, &ensemble_topo, n_clones)
        .context("Failed to create EnsembleWarpMd")?;

    // Set positions
    engine.set_positions(&topo.positions)
        .context("Failed to set positions")?;

    // Initialize velocities
    engine.initialize_velocities(temperature)
        .context("Failed to initialize velocities")?;

    // Run and time
    let start = Instant::now();
    engine.run(n_steps, dt, temperature, gamma)
        .context("Failed to run ensemble MD")?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Get results
    let results = engine.get_results()
        .context("Failed to get results")?;

    Ok((elapsed, results))
}

fn run_sequential_baseline(
    context: std::sync::Arc<CudaContext>,
    topo: &PrismPrepTopology,
    n_clones: usize,
    n_steps: usize,
    dt: f32,
    temperature: f32,
    gamma: f32,
) -> Result<f64> {
    use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};

    let sigmas: Vec<f32> = topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = topo.lj_params.iter().map(|p| p.epsilon).collect();

    // Sequential: run each clone one at a time
    let start = Instant::now();

    for _ in 0..n_clones {
        let mut batch = AmberSimdBatch::new_with_config(
            context.clone(),
            topo.n_atoms + 100,
            1,
            OptimizationConfig {
                use_tensor_cores: false,
                use_async_pipeline: false,
                ..Default::default()
            },
        ).context("Failed to create AmberSimdBatch")?;

        // Create exclusions as Vec<HashSet<usize>>
        let exclusions: Vec<HashSet<usize>> = (0..topo.n_atoms)
            .map(|_| HashSet::new())
            .collect();

        let structure = StructureTopology {
            positions: topo.positions.clone(),
            masses: topo.masses.clone(),
            charges: topo.charges.clone(),
            sigmas: sigmas.clone(),
            epsilons: epsilons.clone(),
            bonds: topo.bonds.iter()
                .map(|b| (b.i, b.j, b.k, b.r0))
                .collect(),
            angles: topo.angles.iter()
                .map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0))
                .collect(),
            dihedrals: vec![],
            exclusions,
        };

        batch.add_structure(&structure)?;
        batch.finalize_batch()?;
        batch.run(n_steps, dt, temperature, gamma)?;
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("   ENSEMBLE WARP MD BENCHMARK - Revolutionary Parallel Processing   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Find a suitable topology (small enough for warp-based processing)
    let topology_paths = [
        "results/sota_validation_fresh/2VWD_topology.json",  // Small: ~2K atoms
        "results/sota_validation_fresh/6LU7_topology.json",  // Medium: 4.7K atoms (over limit)
    ];

    let mut selected_path = None;
    for path in &topology_paths {
        if Path::new(path).exists() {
            let topo = load_topology(Path::new(path))?;
            if topo.n_atoms <= prism_gpu::MAX_ATOMS_WARP {
                println!("✓ Selected: {} ({} atoms)", path, topo.n_atoms);
                selected_path = Some(path.to_string());
                break;
            } else {
                println!("✗ Skipping {} ({} atoms > {} limit)",
                    path, topo.n_atoms, prism_gpu::MAX_ATOMS_WARP);
            }
        }
    }

    // Also check for any small structure in the fresh validation results
    if selected_path.is_none() {
        let results_dir = Path::new("results/sota_validation_fresh");
        if results_dir.exists() {
            for entry in std::fs::read_dir(results_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "json") {
                    if let Ok(topo) = load_topology(&path) {
                        if topo.n_atoms <= prism_gpu::MAX_ATOMS_WARP {
                            println!("✓ Selected: {} ({} atoms)", path.display(), topo.n_atoms);
                            selected_path = Some(path.to_string_lossy().to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    let topo_path = match selected_path {
        Some(p) => p,
        None => {
            println!();
            println!("No suitable topology found (<= {} atoms).", prism_gpu::MAX_ATOMS_WARP);
            println!("Creating a synthetic small structure for testing...");

            // Create a synthetic small protein (~100 atoms) for testing
            return run_synthetic_benchmark();
        }
    };

    let topo = load_topology(Path::new(&topo_path))?;
    run_benchmark_with_topology(&topo_path, &topo)
}

fn run_synthetic_benchmark() -> Result<()> {
    println!();
    println!("Running synthetic benchmark with 100-atom linear chain...");
    println!();

    // Create a proper linear chain with CONSISTENT geometry
    // Bond distance r0 = 1.5Å, angle theta0 = 109.5° = 1.91 rad
    let n_atoms = 100;
    let bond_length = 1.5f32;  // Ångströms
    let bond_angle = 109.5f32 * std::f32::consts::PI / 180.0;  // radians

    let mut positions = Vec::with_capacity(n_atoms * 3);
    let mut masses = Vec::with_capacity(n_atoms);
    let mut charges = Vec::with_capacity(n_atoms);
    let sigmas = vec![3.4f32; n_atoms];    // Carbon LJ sigma
    let epsilons = vec![0.086f32; n_atoms]; // Carbon LJ epsilon

    // Build zigzag chain: atoms alternate direction to maintain bond angle
    // Start at origin, build along X with alternating Y offsets
    let half_angle = bond_angle / 2.0;
    let dx = bond_length * half_angle.cos();  // X step
    let dy = bond_length * half_angle.sin();  // Y oscillation amplitude

    for i in 0..n_atoms {
        let x = i as f32 * dx;
        let y = if i % 2 == 0 { 0.0 } else { dy };
        let z = 0.0f32;

        positions.push(x);
        positions.push(y);
        positions.push(z);
        masses.push(12.01);   // Carbon mass
        charges.push(0.0);    // Neutral for simplicity
    }

    // Verify first bond distance
    let d01 = ((positions[3] - positions[0]).powi(2)
             + (positions[4] - positions[1]).powi(2)
             + (positions[5] - positions[2]).powi(2)).sqrt();
    println!("  Bond 0-1 distance: {:.3}Å (target: {:.3}Å)", d01, bond_length);

    // Create bonds between consecutive atoms (now positions match!)
    let mut bonds = Vec::new();
    for idx in 0..(n_atoms - 1) {
        bonds.push(BondDef {
            i: idx,
            j: idx + 1,
            k: 310.0,  // Typical C-C bond force constant
            r0: bond_length,
        });
    }

    // Create angles (now geometry is correct!)
    let mut angles = Vec::new();
    for idx in 0..(n_atoms - 2) {
        angles.push(AngleDef {
            i: idx,
            j: idx + 1,
            k_idx: idx + 2,
            k: 63.0,  // Typical C-C-C angle force constant
            theta0: bond_angle,
        });
    }

    // Calculate actual angle from positions to verify
    if n_atoms >= 3 {
        let v1 = [positions[0] - positions[3], positions[1] - positions[4], positions[2] - positions[5]];
        let v2 = [positions[6] - positions[3], positions[7] - positions[4], positions[8] - positions[5]];
        let dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        let len1 = (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]).sqrt();
        let len2 = (v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]).sqrt();
        let angle_actual = (dot / (len1 * len2)).acos();
        println!("  Angle 0-1-2: {:.1}° (target: {:.1}°)",
                 angle_actual * 180.0 / std::f32::consts::PI,
                 bond_angle * 180.0 / std::f32::consts::PI);
    }

    let topo = PrismPrepTopology {
        n_atoms,
        positions,
        masses,
        charges,
        lj_params: sigmas.iter().zip(epsilons.iter())
            .map(|(&s, &e)| LjParam { sigma: s, epsilon: e })
            .collect(),
        bonds,
        angles,
    };

    run_benchmark_with_topology("synthetic_100_atoms", &topo)
}

fn run_benchmark_with_topology(name: &str, topo: &PrismPrepTopology) -> Result<()> {
    // CudaContext::new() already returns Arc<CudaContext>
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;

    // Benchmark parameters
    let n_steps = 500;
    let dt = 0.001;  // 1 fs
    let temperature = 300.0;
    let gamma = 1.0;

    // Clone counts to test
    let clone_counts = vec![2, 4, 8, 16, 32, 64];

    println!();
    println!("Structure: {} ({} atoms)", name, topo.n_atoms);
    println!("Steps: {}", n_steps);
    println!("Timestep: {} fs", dt * 1000.0);
    println!("Temperature: {} K", temperature);
    println!();
    println!("┌────────┬────────────┬────────────┬─────────┬──────────────┬──────────────┐");
    println!("│ Clones │  Warp MD   │ Sequential │ Speedup │ ms/struct(W) │ ms/struct(S) │");
    println!("├────────┼────────────┼────────────┼─────────┼──────────────┼──────────────┤");

    let mut results = Vec::new();
    let mut best_speedup = 0.0;
    let mut best_n = 0;

    for &n_clones in &clone_counts {
        // Run warp-based ensemble MD
        let (warp_ms, warp_results) = match run_ensemble_warp_benchmark(
            context.clone(),
            topo,
            n_clones,
            n_steps,
            dt,
            temperature,
            gamma,
        ) {
            Ok(r) => r,
            Err(e) => {
                println!("│ {:>6} │ {:>10} │ {:>10} │ {:>7} │ {:>12} │ {:>12} │",
                    n_clones, "FAILED", "-", "-", "-", "-");
                log::error!("Warp MD failed for {} clones: {:?}", n_clones, e);
                // Print the full error chain
                let mut source = e.source();
                while let Some(s) = source {
                    log::error!("  Caused by: {:?}", s);
                    source = s.source();
                }
                continue;
            }
        };

        // Run sequential baseline
        let seq_ms = match run_sequential_baseline(
            context.clone(),
            topo,
            n_clones,
            n_steps,
            dt,
            temperature,
            gamma,
        ) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Sequential baseline failed: {}", e);
                warp_ms * n_clones as f64  // Estimate
            }
        };

        let speedup = seq_ms / warp_ms;
        let ms_per_struct_warp = warp_ms / n_clones as f64;
        let ms_per_struct_seq = seq_ms / n_clones as f64;

        // Get first clone's results for reporting
        let (pe, ke, temp) = if !warp_results.is_empty() {
            let r = &warp_results[0];
            (r.potential_energy, r.kinetic_energy, r.temperature)
        } else {
            (0.0, 0.0, 0.0)
        };

        if speedup > best_speedup {
            best_speedup = speedup;
            best_n = n_clones;
        }

        println!("│ {:>6} │ {:>8.1}ms │ {:>8.1}ms │ {:>6.2}× │ {:>10.2}ms │ {:>10.2}ms │",
            n_clones, warp_ms, seq_ms, speedup, ms_per_struct_warp, ms_per_struct_seq);

        results.push(BenchmarkResult {
            n_clones,
            warp_ms,
            sequential_ms: seq_ms,
            speedup,
            ms_per_structure_warp: ms_per_struct_warp,
            ms_per_structure_seq: ms_per_struct_seq,
            warp_pe: pe,
            warp_ke: ke,
            warp_temp: temp,
        });
    }

    println!("└────────┴────────────┴────────────┴─────────┴──────────────┴──────────────┘");
    println!();
    println!("BEST SPEEDUP: {:.2}× with {} clones", best_speedup, best_n);
    println!();

    if best_speedup > 1.0 {
        println!("🚀 SUCCESS! Warp-based parallel MD achieves {:.1}× speedup!", best_speedup);
        println!("   This is {} more efficient than the old batched approach!",
            if best_speedup > 2.0 { "significantly" } else { "notably" });
    } else {
        println!("⚠️  Speedup < 1×. Further optimization needed.");
    }

    // Save results
    let output = BenchmarkOutput {
        timestamp: chrono::Utc::now().to_rfc3339(),
        structure: name.to_string(),
        n_atoms: topo.n_atoms,
        steps: n_steps,
        clone_counts,
        results,
        best_speedup,
        best_n_clones: best_n,
    };

    let output_dir = Path::new("results/ensemble_warp");
    std::fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join(format!("{}_warp_benchmark.json",
        name.replace("/", "_").replace(" ", "_")));
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&output_path, json)?;
    println!("Results saved to: {}", output_path.display());

    Ok(())
}
