//! Quick test: Run MD on 1L2Y using prism-prep topology

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct LjParam {
    sigma: f32,
    epsilon: f32,
}

#[derive(Debug, Deserialize)]
struct BondDef {
    #[serde(alias = "atom_i")]
    i: usize,
    #[serde(alias = "atom_j")]
    j: usize,
    #[serde(alias = "force_constant")]
    k: f32,
    #[serde(alias = "equilibrium_distance")]
    r0: f32,
}

#[derive(Debug, Deserialize)]
struct AngleDef {
    #[serde(alias = "atom_i")]
    i: usize,
    #[serde(alias = "atom_j")]
    j: usize,
    #[serde(alias = "atom_k")]
    k_idx: usize,
    #[serde(alias = "force_constant", rename = "force_k")]
    k: f32,
    #[serde(alias = "equilibrium_angle")]
    theta0: f32,
}

#[derive(Debug, Deserialize)]
struct Topology {
    n_atoms: usize,
    positions: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondDef>,
    angles: Vec<AngleDef>,
    #[serde(default)]
    exclusions: Vec<Vec<usize>>,  // Load exclusions from prism-prep!
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("   1L2Y Trp-cage MD Test using prism-prep topology");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load topology
    let topo_path = "results/prism_prep_test/1L2Y_topology.json";
    let content = std::fs::read_to_string(topo_path)
        .context("Failed to read topology")?;
    let topo: Topology = serde_json::from_str(&content)
        .context("Failed to parse topology")?;

    println!("Loaded: {} atoms, {} bonds, {} angles",
        topo.n_atoms, topo.bonds.len(), topo.angles.len());

    // Setup CUDA
    let context = CudaContext::new(0).context("CUDA context")?;

    // Create MD engine
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        topo.n_atoms + 100,
        1,
        OptimizationConfig {
            use_tensor_cores: false,
            use_async_pipeline: false,
            ..Default::default()
        },
    ).context("AmberSimdBatch")?;

    // Convert to StructureTopology
    let sigmas: Vec<f32> = topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = topo.lj_params.iter().map(|p| p.epsilon).collect();

    // Load REAL exclusions from prism-prep topology!
    let exclusions: Vec<HashSet<usize>> = if !topo.exclusions.is_empty() {
        println!("  Using {} exclusion lists from prism-prep", topo.exclusions.len());
        topo.exclusions.iter()
            .map(|excl| excl.iter().cloned().collect())
            .collect()
    } else {
        println!("  WARNING: No exclusions in topology!");
        (0..topo.n_atoms).map(|_| HashSet::new()).collect()
    };

    let structure = StructureTopology {
        positions: topo.positions.clone(),
        masses: topo.masses.clone(),
        charges: topo.charges.clone(),
        sigmas,
        epsilons,
        bonds: topo.bonds.iter().map(|b| (b.i, b.j, b.k, b.r0)).collect(),
        angles: topo.angles.iter().map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0)).collect(),
        dihedrals: vec![],
        exclusions,
    };

    batch.add_structure(&structure)?;
    batch.finalize_batch()?;

    // Get initial positions (first 3 atoms)
    let init_pos = topo.positions[0..9].to_vec();
    println!("\nInitial positions (first 3 atoms):");
    for i in 0..3 {
        println!("  Atom {}: ({:.3}, {:.3}, {:.3})",
            i, init_pos[i*3], init_pos[i*3+1], init_pos[i*3+2]);
    }

    // Run MD
    let n_steps = 1000;
    let dt = 0.001;  // 1 fs
    let temperature = 300.0;
    let gamma = 1.0;

    println!("\nRunning {} steps of Langevin MD at {}K...", n_steps, temperature);

    let start = Instant::now();
    batch.run(n_steps, dt, temperature, gamma)?;
    let elapsed = start.elapsed();

    // Get final positions
    let results = batch.get_all_results()?;
    if !results.is_empty() {
        let final_pos = &results[0].positions;
        println!("\nFinal positions (first 3 atoms):");
        for i in 0..3 {
            println!("  Atom {}: ({:.3}, {:.3}, {:.3})",
                i, final_pos[i*3], final_pos[i*3+1], final_pos[i*3+2]);
        }

        // Calculate RMSD of movement
        let mut sum_sq = 0.0f32;
        for i in 0..topo.n_atoms * 3 {
            let diff = final_pos[i] - topo.positions[i];
            sum_sq += diff * diff;
        }
        let rmsd = (sum_sq / topo.n_atoms as f32).sqrt();
        println!("\n  RMSD from start: {:.3} Å", rmsd);

        // Show energies if available
        println!("  Final PE: {:.1} kcal/mol", results[0].potential_energy);
        println!("  Final KE: {:.1} kcal/mol", results[0].kinetic_energy);
        println!("  Final Temp: {:.1} K", results[0].temperature);
    }

    println!("\n✓ MD completed in {:.2?}", elapsed);
    println!("  Throughput: {:.1} steps/sec", n_steps as f64 / elapsed.as_secs_f64());
    println!("  Time/step: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / n_steps as f64);

    Ok(())
}
