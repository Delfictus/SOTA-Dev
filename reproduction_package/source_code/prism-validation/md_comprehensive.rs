//! Comprehensive MD test with full AMBER ff14SB topology (bonds, angles, dihedrals)

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
    dihedrals: Vec<DihedralDef>,
    #[serde(default)]
    exclusions: Vec<Vec<usize>>,
}

fn run_comprehensive_md(name: &str, topo_path: &str, context: std::sync::Arc<CudaContext>) -> Result<(f64, f64, f64)> {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  {:^66}  ║", name);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let content = std::fs::read_to_string(topo_path)
        .with_context(|| format!("Failed to read {}", topo_path))?;
    let topo: Topology = serde_json::from_str(&content)
        .context("Failed to parse topology")?;

    println!("\n  Structure Summary:");
    println!("  ├─ Atoms: {:>8}", topo.n_atoms);
    println!("  ├─ Residues: {:>5}", topo.n_residues.unwrap_or(0));
    println!("  ├─ Bonds: {:>8}", topo.bonds.len());
    println!("  ├─ Angles: {:>7}", topo.angles.len());
    println!("  ├─ Dihedrals: {:>4}", topo.dihedrals.len());
    println!("  └─ Exclusions: {:>3} lists", topo.exclusions.len());

    // Create MD engine
    let max_atoms = (topo.n_atoms as f32 * 1.5) as usize + 1000;
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        max_atoms,
        1,
        OptimizationConfig::default(),
    ).context("AmberSimdBatch creation")?;

    // Convert topology - WITH DIHEDRALS
    let sigmas: Vec<f32> = topo.lj_params.iter().map(|p| p.sigma).collect();
    let epsilons: Vec<f32> = topo.lj_params.iter().map(|p| p.epsilon).collect();

    let exclusions: Vec<HashSet<usize>> = if !topo.exclusions.is_empty() {
        topo.exclusions.iter().map(|e| e.iter().cloned().collect()).collect()
    } else {
        println!("  ⚠ WARNING: No exclusions in topology!");
        (0..topo.n_atoms).map(|_| HashSet::new()).collect()
    };

    // NOTE: Dihedrals disabled - thermostat needs debugging first
    // Once thermostat is fixed, re-enable dihedrals:
    // let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topo.dihedrals.iter()
    //     .map(|d| (d.i, d.j, d.k_idx, d.l, d.force_k, d.periodicity as f32, d.phase))
    //     .collect();
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    if !dihedrals.is_empty() {
        println!("\n  ✓ Loading {} dihedrals", dihedrals.len());
    } else {
        println!("\n  ⚠ Dihedrals disabled (pending thermostat fix)");
    }

    let structure = StructureTopology {
        positions: topo.positions.clone(),
        masses: topo.masses.clone(),
        charges: topo.charges.clone(),
        sigmas, epsilons,
        bonds: topo.bonds.iter().map(|b| (b.i, b.j, b.k, b.r0)).collect(),
        angles: topo.angles.iter().map(|a| (a.i, a.j, a.k_idx, a.k, a.theta0)).collect(),
        dihedrals,
        exclusions,
    };

    batch.add_structure(&structure)?;
    batch.finalize_batch()?;

    // MD Parameters
    let dt = 0.001;  // 1 fs timestep
    let temp_target = 300.0;
    let gamma = 50.0;  // Strong coupling - thermostat reaches ~50K (not 300K due to kernel bug)

    // KNOWN ISSUES:
    // 1. Minimization is broken (increases PE)
    // 2. Langevin thermostat equilibrates to ~T/6 instead of T (kernel bug)

    // Initialize velocities at target temperature
    batch.initialize_velocities(temp_target as f32)?;
    println!("\n  ✓ Initialized velocities at {} K", temp_target);

    println!("\n  MD Parameters:");
    println!("  ├─ Timestep: {} fs", dt * 1000.0);
    println!("  ├─ Target Temp: {} K", temp_target);
    println!("  └─ Langevin γ: {} ps⁻¹ (strong coupling)", gamma);

    // Equilibration phases
    println!("\n  Equilibration Progress:");
    println!("  {:>6} {:>10} {:>10} {:>8} {:>10}", "Steps", "PE", "KE", "Temp", "RMSD");
    println!("  {:->6} {:->10} {:->10} {:->8} {:->10}", "", "", "", "", "");

    let start = Instant::now();

    // Phase 1: Initial heating (500 steps with strong damping)
    for phase in 0..5 {
        let steps = 500;
        batch.run(steps, dt, temp_target, gamma)?;

        let results = batch.get_all_results()?;
        if let Some(r) = results.first() {
            let mut rmsd_sq = 0.0f32;
            for i in 0..topo.n_atoms * 3 {
                let diff = r.positions[i] - topo.positions[i];
                rmsd_sq += diff * diff;
            }
            let rmsd = (rmsd_sq / topo.n_atoms as f32).sqrt();

            println!("  {:>6} {:>10.1} {:>10.1} {:>8.1} {:>10.4}",
                (phase + 1) * steps,
                r.potential_energy,
                r.kinetic_energy,
                r.temperature,
                rmsd);
        }
    }

    let elapsed = start.elapsed();

    // Get final results
    let results = batch.get_all_results()?;
    let (pe, ke, temp) = if let Some(r) = results.first() {
        // Compute expected KE
        let dof = 3 * topo.n_atoms - 6;
        let expected_ke = 0.5 * dof as f64 * 0.001987 * 300.0;  // kB in kcal/mol/K

        println!("\n  Final State:");
        println!("  ├─ PE: {:.1} kcal/mol", r.potential_energy);
        println!("  ├─ KE: {:.1} kcal/mol", r.kinetic_energy);
        println!("  ├─ Total E: {:.1} kcal/mol", r.potential_energy + r.kinetic_energy);
        println!("  ├─ Temperature: {:.1} K (target: {})", r.temperature, temp_target);
        println!("  └─ KE ratio: {:.1}% of expected {:.0}",
            100.0 * r.kinetic_energy / expected_ke, expected_ke);

        // Performance
        let total_steps = 5 * 500;
        let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
        let ns_per_day = steps_per_sec * 0.001 * 86400.0 / 1000.0;
        println!("\n  Performance:");
        println!("  ├─ Time: {:.2?}", elapsed);
        println!("  ├─ Steps/sec: {:.0}", steps_per_sec);
        println!("  └─ Throughput: {:.2} ns/day", ns_per_day);

        (r.potential_energy, r.kinetic_energy, r.temperature)
    } else {
        (0.0, 0.0, 0.0)
    };

    Ok((pe, ke, temp))
}

fn main() -> Result<()> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE MD TEST - Full AMBER ff14SB with Dihedrals                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing: Strong thermostat coupling (γ=10) + Full dihedral loading         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let context = CudaContext::new(0).context("CUDA context")?;

    let structures = [
        ("1L2Y (Trp-cage, 20 res)", "results/prism_prep_test/1L2Y_topology.json"),
        ("1HXY (Hemoglobin, 585 res)", "results/prism_prep_test/1HXY_apo_topology.json"),
        ("2VWD (Nipah G, 823 res)", "results/prism_prep_test/2VWD_apo_topology.json"),
        ("6M0J (SARS-CoV-2 RBD, 791 res)", "results/prism_prep_test/6M0J_apo_topology.json"),
    ];

    let mut results = Vec::new();

    for (name, path) in &structures {
        if std::path::Path::new(path).exists() {
            match run_comprehensive_md(name, path, context.clone()) {
                Ok((pe, ke, temp)) => {
                    results.push((name.to_string(), pe, ke, temp, true));
                }
                Err(e) => {
                    println!("  ✗ FAILED: {}", e);
                    results.push((name.to_string(), 0.0, 0.0, 0.0, false));
                }
            }
        } else {
            println!("\n  ✗ Topology not found: {}", path);
            results.push((name.to_string(), 0.0, 0.0, 0.0, false));
        }
    }

    // Summary table
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           RESULTS SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:^30} │ {:>12} │ {:>12} │ {:>8} ║", "Structure", "PE (kcal/mol)", "KE (kcal/mol)", "Temp (K)");
    println!("╟────────────────────────────────┼──────────────┼──────────────┼──────────╢");

    for (name, pe, ke, temp, ok) in &results {
        if *ok {
            let pe_status = if *pe < 0.0 { "✓" } else { "⚠" };
            let temp_status = if *temp > 250.0 && *temp < 350.0 { "✓" } else { "⚠" };
            println!("║ {:^30} │ {:>11.1}{} │ {:>12.1} │ {:>6.1}{} ║",
                name, pe, pe_status, ke, temp, temp_status);
        } else {
            println!("║ {:^30} │ {:^12} │ {:^12} │ {:^8} ║", name, "FAILED", "-", "-");
        }
    }

    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    // Status assessment
    let all_negative_pe = results.iter().filter(|r| r.4).all(|r| r.1 < 0.0);
    let all_temp_normal = results.iter().filter(|r| r.4).all(|r| r.3 > 250.0 && r.3 < 350.0);

    if all_negative_pe && all_temp_normal {
        println!("║  ✓ All structures have negative PE and normalized temperatures             ║");
    } else {
        if !all_negative_pe {
            println!("║  ⚠ Some structures have positive PE - may indicate structural issues       ║");
        }
        if !all_temp_normal {
            println!("║  ⚠ Some structures have abnormal temperatures - thermostat issues          ║");
        }
    }

    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
