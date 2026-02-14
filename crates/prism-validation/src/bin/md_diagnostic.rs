//! MD Diagnostic - investigate energy and temperature differences

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
    n_residues: Option<usize>,
    positions: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondDef>,
    angles: Vec<AngleDef>,
    #[serde(default)]
    exclusions: Vec<Vec<usize>>,
}

fn analyze_structure(name: &str, topo: &Topology) {
    println!("\n=== {} Structure Analysis ===", name);
    println!("  Atoms: {}", topo.n_atoms);
    println!("  Residues: {}", topo.n_residues.unwrap_or(0));
    println!("  Bonds: {}", topo.bonds.len());
    println!("  Angles: {}", topo.angles.len());
    println!("  Exclusions: {} lists", topo.exclusions.len());
    
    // Check for issues
    let total_charge: f32 = topo.charges.iter().sum();
    println!("  Net charge: {:.1} e", total_charge);
    
    // Check mass distribution
    let avg_mass: f32 = topo.masses.iter().sum::<f32>() / topo.n_atoms as f32;
    let min_mass = topo.masses.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_mass = topo.masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Mass range: {:.2} - {:.2} (avg {:.2})", min_mass, max_mass, avg_mass);
    
    // Check LJ params
    let avg_sigma: f32 = topo.lj_params.iter().map(|p| p.sigma).sum::<f32>() / topo.n_atoms as f32;
    let avg_eps: f32 = topo.lj_params.iter().map(|p| p.epsilon).sum::<f32>() / topo.n_atoms as f32;
    println!("  Avg LJ sigma: {:.3} Å, epsilon: {:.4} kcal/mol", avg_sigma, avg_eps);
    
    // Check for close contacts in initial structure
    let mut close_contacts = 0;
    let mut very_close = 0;
    for i in 0..topo.n_atoms.min(1000) {  // Sample first 1000 atoms
        for j in (i+1)..topo.n_atoms.min(1000) {
            let dx = topo.positions[j*3] - topo.positions[i*3];
            let dy = topo.positions[j*3+1] - topo.positions[i*3+1];
            let dz = topo.positions[j*3+2] - topo.positions[i*3+2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt();
            if r < 1.5 { very_close += 1; }
            else if r < 2.0 { close_contacts += 1; }
        }
    }
    println!("  Close contacts (<2Å): {}, Very close (<1.5Å): {}", close_contacts, very_close);
}

fn run_diagnostic(name: &str, topo_path: &str, context: std::sync::Arc<CudaContext>) -> Result<()> {
    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("  DIAGNOSTIC: {}", name);
    println!("{}", sep);
    
    let content = std::fs::read_to_string(topo_path)
        .with_context(|| format!("Failed to read {}", topo_path))?;
    let topo: Topology = serde_json::from_str(&content)
        .context("Failed to parse topology")?;
    
    analyze_structure(name, &topo);
    
    // Create MD engine
    let max_atoms = (topo.n_atoms as f32 * 1.5) as usize + 1000;
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
        println!("  WARNING: No exclusions!");
        (0..topo.n_atoms).map(|_| HashSet::new()).collect()
    };
    
    let structure = StructureTopology {
        positions: topo.positions.clone(),
        masses: topo.masses.clone(),
        charges: topo.charges.clone(),
        sigmas, epsilons,
        bonds: topo.bonds.iter().map(|b| (b.i, b.j, b.k, b.r0)).collect(),
        angles: topo.angles.iter().map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0)).collect(),
        dihedrals: vec![],
        exclusions,
    };
    
    batch.add_structure(&structure)?;
    batch.finalize_batch()?;
    
    // Run multiple phases and track energy/temperature
    println!("\n  === MD Trajectory ===");
    println!("  {:>6} {:>12} {:>12} {:>10} {:>10}", "Step", "PE", "KE", "Temp", "RMSD");
    println!("  {:->6} {:->12} {:->12} {:->10} {:->10}", "", "", "", "", "");
    
    let dt = 0.001;  // 1 fs
    let temp_target = 300.0;
    let gamma = 1.0;  // Strong coupling for equilibration
    
    for phase in 0..5 {
        let steps = 200;
        batch.run(steps, dt, temp_target, gamma)?;
        
        let results = batch.get_all_results()?;
        if let Some(r) = results.first() {
            let mut rmsd_sq = 0.0f32;
            for i in 0..topo.n_atoms * 3 {
                let diff = r.positions[i] - topo.positions[i];
                rmsd_sq += diff * diff;
            }
            let rmsd = (rmsd_sq / topo.n_atoms as f32).sqrt();
            
            println!("  {:>6} {:>12.1} {:>12.1} {:>10.1} {:>10.4}",
                (phase + 1) * steps,
                r.potential_energy,
                r.kinetic_energy,
                r.temperature,
                rmsd);
        }
    }
    
    // Final analysis
    let results = batch.get_all_results()?;
    if let Some(r) = results.first() {
        println!("\n  === Final State ===");
        println!("  PE: {:.1} kcal/mol", r.potential_energy);
        println!("  KE: {:.1} kcal/mol", r.kinetic_energy);
        println!("  Total E: {:.1} kcal/mol", r.potential_energy + r.kinetic_energy);
        println!("  Temperature: {:.1} K (target: {})", r.temperature, temp_target);
        
        // Expected KE at 300K: KE = (3N - 6) * kB * T / 2
        let dof = 3 * topo.n_atoms - 6;
        let expected_ke = 0.5 * dof as f64 * 0.001987 * 300.0;  // kB in kcal/mol/K
        println!("  Expected KE at 300K: {:.1} kcal/mol", expected_ke);
        println!("  KE ratio: {:.2}%", 100.0 * r.kinetic_energy / expected_ke);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    MD DIAGNOSTIC - Energy & Temperature Analysis            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    
    let context = CudaContext::new(0).context("CUDA context")?;
    
    // Test structures that had different behaviors
    let structures = [
        ("1L2Y", "results/prism_prep_test/1L2Y_topology.json"),
        ("1HXY", "results/prism_prep_test/1HXY_apo_topology.json"),
        ("2VWD", "results/prism_prep_test/2VWD_apo_topology.json"),  // This had positive PE
        ("6M0J", "results/prism_prep_test/6M0J_apo_topology.json"),
    ];
    
    for (name, path) in &structures {
        if std::path::Path::new(path).exists() {
            if let Err(e) = run_diagnostic(name, path, context.clone()) {
                println!("  ✗ FAILED: {}", e);
            }
        } else {
            println!("  ✗ Not found: {}", path);
        }
    }
    
    let sep = "=".repeat(70);
    println!("\n{}", sep);
    println!("  DIAGNOSTIC COMPLETE");
    println!("{}", sep);
    
    Ok(())
}
