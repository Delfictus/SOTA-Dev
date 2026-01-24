//! Batch MD test - run MD on multiple prism-prep topologies

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct LjParam { sigma: f32, epsilon: f32 }

#[derive(Debug, Deserialize)]
struct BondDef {
    #[serde(alias = "atom_i")] i: usize,
    #[serde(alias = "atom_j")] j: usize,
    #[serde(alias = "force_constant")] k: f32,
    #[serde(alias = "equilibrium_distance")] r0: f32,
}

#[derive(Debug, Deserialize)]
struct AngleDef {
    #[serde(alias = "atom_i")] i: usize,
    #[serde(alias = "atom_j")] j: usize,
    #[serde(alias = "atom_k")] k_idx: usize,
    #[serde(alias = "force_constant", rename = "force_k")] k: f32,
    #[serde(alias = "equilibrium_angle")] theta0: f32,
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
    exclusions: Vec<Vec<usize>>,
}

fn run_md_test(name: &str, topo_path: &str, context: std::sync::Arc<CudaContext>) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let content = std::fs::read_to_string(topo_path)
        .with_context(|| format!("Failed to read {}", topo_path))?;
    let topo: Topology = serde_json::from_str(&content)
        .context("Failed to parse topology")?;
    
    println!("  Atoms: {}, Bonds: {}, Angles: {}", 
        topo.n_atoms, topo.bonds.len(), topo.angles.len());
    
    // Create MD engine with appropriate buffer size (need extra for large structures)
    let max_atoms = (topo.n_atoms as f32 * 1.2) as usize + 1000;
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        max_atoms,
        1,
        OptimizationConfig::default(),
    ).context("AmberSimdBatch creation")?;
    
    // Convert topology
    let sigmas: Vec<f32> = topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = topo.lj_params.iter().map(|p| p.epsilon).collect();
    let exclusions: Vec<HashSet<usize>> = if !topo.exclusions.is_empty() {
        topo.exclusions.iter().map(|e| e.iter().cloned().collect()).collect()
    } else {
        (0..topo.n_atoms).map(|_| HashSet::new()).collect()
    };
    
    let structure = StructureTopology {
        positions: topo.positions.clone(),
        masses: topo.masses,
        charges: topo.charges,
        sigmas, epsilons,
        bonds: topo.bonds.iter().map(|b| (b.i, b.j, b.k, b.r0)).collect(),
        angles: topo.angles.iter().map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0)).collect(),
        dihedrals: vec![],
        exclusions,
    };
    
    batch.add_structure(&structure)?;
    batch.finalize_batch()?;
    
    // Run 500 steps
    let n_steps = 500;
    let start = Instant::now();
    batch.run(n_steps, 0.001, 300.0, 1.0)?;
    let elapsed = start.elapsed();
    
    // Get results
    let results = batch.get_all_results()?;
    if let Some(r) = results.first() {
        let mut rmsd_sq = 0.0f32;
        for i in 0..topo.n_atoms * 3 {
            let diff = r.positions[i] - topo.positions[i];
            rmsd_sq += diff * diff;
        }
        let rmsd = (rmsd_sq / topo.n_atoms as f32).sqrt();
        
        println!("  RMSD: {:.4} Å", rmsd);
        println!("  PE: {:.1} kcal/mol", r.potential_energy);
        println!("  KE: {:.1} kcal/mol", r.kinetic_energy);
        println!("  Temp: {:.1} K", r.temperature);
    }
    
    let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
    let ns_per_day = steps_per_sec * 0.001 * 86400.0 / 1000.0;  // fs/step * sec/day / 1000
    
    println!("  Time: {:.2?} ({:.0} steps/sec, {:.2} ns/day)", elapsed, steps_per_sec, ns_per_day);
    println!();
    
    Ok(())
}

fn main() -> Result<()> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           BATCH MD TEST - prism-prep → AmberSimdBatch Pipeline              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    
    let context = CudaContext::new(0).context("CUDA context")?;
    
    let structures = [
        ("1L2Y (Trp-cage, 20 res)", "results/prism_prep_test/1L2Y_topology.json"),
        ("1HXY (Hemoglobin, 585 res)", "results/prism_prep_test/1HXY_apo_topology.json"),
        ("2VWD (Nipah G, 823 res)", "results/prism_prep_test/2VWD_apo_topology.json"),
        ("6M0J (SARS-CoV-2 RBD, 791 res)", "results/prism_prep_test/6M0J_apo_topology.json"),
        ("4B7Q (Influenza HA, 1548 res)", "results/prism_prep_test/4B7Q_apo_topology.json"),
    ];
    
    for (name, path) in &structures {
        if std::path::Path::new(path).exists() {
            if let Err(e) = run_md_test(name, path, context.clone()) {
                println!("  ✗ FAILED: {}", e);
                println!();
            }
        } else {
            println!("  ✗ Topology not found: {}", path);
            println!();
        }
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BATCH TEST COMPLETE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    Ok(())
}
