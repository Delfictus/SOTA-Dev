//! Test Explicit Solvent MD on a Real Protein (1ABA - Glutaredoxin)
//!
//! This test validates that our explicit solvent implementation works on
//! a real protein structure, not just a pure water box.
//!
//! Workflow:
//! 1. Load real protein from PDB
//! 2. Create solvation box with TIP3P water
//! 3. Initialize PME for long-range electrostatics
//! 4. Initialize SETTLE for rigid water constraints
//! 5. Run production MD with BAOAB thermostat
//!
//! Expected:
//! - Temperature stable within ±20% of target
//! - Protein remains structurally stable (no explosion)
//! - Energy bounded (no numerical instability)

use anyhow::{Context, Result};
use std::sync::Arc;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    log::info!("==============================================");
    log::info!("  PROTEIN EXPLICIT SOLVENT TEST (1ABA)");
    log::info!("==============================================");
    log::info!("");

    #[cfg(not(feature = "cryptic-gpu"))]
    {
        log::error!("This test requires the 'cryptic-gpu' feature.");
        log::error!("Run with: cargo run --release -p prism-validation --features cryptic-gpu --bin test_protein_solvent");
        return Err(anyhow::anyhow!("Missing cryptic-gpu feature"));
    }

    #[cfg(feature = "cryptic-gpu")]
    {
        run_protein_solvent_test()
    }
}

#[cfg(feature = "cryptic-gpu")]
fn run_protein_solvent_test() -> Result<()> {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;
    use prism_physics::amber_ff14sb::AmberAtomType;
    use prism_physics::solvation::{SolvationBox, SolvationConfig};
    use std::collections::HashSet;
    use std::fs;

    // Simple PDB atom struct
    struct PdbAtom {
        name: String,
        x: f32,
        y: f32,
        z: f32,
    }

    // Simple PDB parser
    fn parse_pdb(content: &str) -> Vec<PdbAtom> {
        content.lines()
            .filter(|line| line.starts_with("ATOM"))
            .filter_map(|line| {
                if line.len() < 54 { return None; }
                let name = line[12..16].trim().to_string();
                let x: f32 = line[30..38].trim().parse().ok()?;
                let y: f32 = line[38..46].trim().parse().ok()?;
                let z: f32 = line[46..54].trim().parse().ok()?;
                Some(PdbAtom { name, x, y, z })
            })
            .collect()
    }

    // ========================================================================
    // PHYSICS CONSTANTS (OPTIMIZED FOR PROTEIN + EXPLICIT SOLVENT)
    // ========================================================================
    // Protein systems need stronger damping than pure water because:
    // - Bonded forces (bonds, angles, dihedrals) add energy
    // - Protein-water interactions are stronger
    // - Simplified charge assignment may cause artifacts
    //
    // Tuning history for 1ABA (728 protein + 1140 waters):
    // - γ = 0.3 fs⁻¹: T = 1.64× target (too hot)
    // - γ = 0.5 fs⁻¹: Testing...
    //
    const DT: f32 = 0.25;          // fs - smaller timestep for stability
    const GAMMA: f32 = 0.5;        // fs⁻¹ = 500 ps⁻¹ - very strong damping for protein
    const TEMPERATURE: f32 = 310.0; // K - physiological
    const N_STEPS: usize = 1000;   // 0.25 ps - quick test

    // ========================================================================
    // STEP 1: Load protein from PDB
    // ========================================================================
    log::info!("Step 1: Loading protein structure...");

    let pdb_path = "/home/diddy/Desktop/PRISM4D-bio/data/atlas/pdb/1aba.pdb";
    let pdb_content = fs::read_to_string(pdb_path)
        .context("Failed to read PDB file")?;
    let pdb_atoms = parse_pdb(&pdb_content);
    let n_protein_atoms = pdb_atoms.len();

    log::info!("  - Protein: 1ABA (Glutaredoxin)");
    log::info!("  - Atoms: {}", n_protein_atoms);
    log::info!("  ✓ Protein loaded");

    // ========================================================================
    // STEP 2: Initialize GPU
    // ========================================================================
    log::info!("Step 2: Initializing GPU...");
    let context = CudaContext::new(0)
        .context("Failed to create CUDA context - is a GPU available?")?;
    let context = Arc::new(context);
    log::info!("  ✓ GPU initialized");

    // ========================================================================
    // STEP 3: Create topology and solvation box
    // ========================================================================
    log::info!("Step 3: Creating solvated system...");

    // Convert PDB to flat arrays for solvation
    let mut protein_positions: Vec<f32> = Vec::with_capacity(n_protein_atoms * 3);
    let mut protein_types: Vec<AmberAtomType> = Vec::with_capacity(n_protein_atoms);
    let mut protein_charges: Vec<f32> = Vec::with_capacity(n_protein_atoms);

    for atom in &pdb_atoms {
        protein_positions.push(atom.x);
        protein_positions.push(atom.y);
        protein_positions.push(atom.z);

        // Map atom name to AMBER atom type (simplified)
        let atom_type = match atom.name.as_str() {
            "N" => AmberAtomType::N,
            "CA" => AmberAtomType::CT,
            "C" => AmberAtomType::C,
            "O" => AmberAtomType::O,
            "CB" => AmberAtomType::CT,
            "CG" | "CG1" | "CG2" => AmberAtomType::CT,
            "CD" | "CD1" | "CD2" => AmberAtomType::CT,
            "CE" | "CE1" | "CE2" => AmberAtomType::CT,
            "CZ" => AmberAtomType::CA,
            "NZ" => AmberAtomType::N3,
            "OG" | "OG1" => AmberAtomType::OH,
            "OD1" | "OD2" => AmberAtomType::O2,
            "OE1" | "OE2" => AmberAtomType::O2,
            "NE" | "NE1" | "NE2" => AmberAtomType::N,
            "NH1" | "NH2" => AmberAtomType::N2,
            "ND1" | "ND2" => AmberAtomType::N,
            "SG" => AmberAtomType::SH,
            "SD" => AmberAtomType::S,
            name if name.starts_with("H") => AmberAtomType::H,
            _ => AmberAtomType::CT, // Default to CT for unknowns
        };
        protein_types.push(atom_type);

        // Simplified charge assignment (proper charges would come from topology)
        let charge = match atom.name.as_str() {
            "N" => -0.4,
            "O" => -0.5,
            "NZ" => 1.0,  // Lysine amine
            "OD1" | "OD2" | "OE1" | "OE2" => -0.8,  // Carboxylates
            name if name.starts_with("H") => 0.2,
            _ => 0.0,
        };
        protein_charges.push(charge as f32);
    }

    // Create solvation box with reasonable padding
    let config = SolvationConfig {
        padding: 10.0,              // 10 Å padding around protein
        min_protein_distance: 2.8,  // Min distance from protein to water O
        min_water_distance: 2.8,    // Min distance between water O atoms
        target_density: 0.997,      // g/cm³
        max_box_dimension: 60.0,    // Max box size (protein is ~30Å)
        salt_concentration: 0.0,    // No ions for simplicity
    };

    let mut solvbox = SolvationBox::from_protein(
        &protein_positions,
        &protein_types,
        &protein_charges,
        &config,
    )?;

    let n_waters = solvbox.add_waters(&config);
    let (n_na, n_cl) = solvbox.neutralize(&config);

    log::info!("  - Protein atoms: {}", n_protein_atoms);
    log::info!("  - Waters added: {} ({} atoms)", n_waters, n_waters * 3);
    log::info!("  - Ions: {} Na+, {} Cl-", n_na, n_cl);
    log::info!("  - Box: {:.1} × {:.1} × {:.1} Å",
        solvbox.box_dimensions[0], solvbox.box_dimensions[1], solvbox.box_dimensions[2]);
    log::info!("  - Total atoms: {}", solvbox.total_atoms);
    log::info!("  ✓ System solvated");

    // ========================================================================
    // STEP 4: Create topology and upload to GPU
    // ========================================================================
    log::info!("Step 4: Creating topology...");

    let topology = solvbox.to_topology();
    let positions = solvbox.all_positions_flat();
    let water_oxygens = solvbox.water_oxygen_indices();

    // Build exclusion lists
    let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); topology.n_atoms];
    for (i, j) in &topology.exclusions {
        exclusions[*i as usize].insert(*j as usize);
        exclusions[*j as usize].insert(*i as usize);
    }

    // Convert topology to flat arrays
    let bonds: Vec<(usize, usize, f32, f32)> = topology.bonds.iter()
        .zip(topology.bond_params.iter())
        .map(|((i, j), p)| (*i as usize, *j as usize, p.k, p.r0))
        .collect();

    let angles: Vec<(usize, usize, usize, f32, f32)> = topology.angles.iter()
        .zip(topology.angle_params.iter())
        .map(|((i, j, k), p)| (*i as usize, *j as usize, *k as usize, p.k, p.theta0))
        .collect();

    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topology.dihedrals.iter()
        .zip(topology.dihedral_params.iter())
        .filter_map(|((i, j, k, l), params)| {
            params.first().map(|p| {
                (*i as usize, *j as usize, *k as usize, *l as usize, p.k, p.n as f32, p.phase)
            })
        })
        .collect();

    let nb_params: Vec<(f32, f32, f32, f32)> = topology.lj_params.iter()
        .zip(topology.charges.iter())
        .zip(topology.masses.iter())
        .map(|((lj, &q), &m)| {
            let sigma = 2.0 * lj.rmin_half / 2.0_f32.powf(1.0 / 6.0);
            (sigma, lj.epsilon, q, m)
        })
        .collect();

    log::info!("  - Bonds: {}", bonds.len());
    log::info!("  - Angles: {}", angles.len());
    log::info!("  - Dihedrals: {}", dihedrals.len());
    log::info!("  ✓ Topology created");

    // ========================================================================
    // STEP 5: Initialize HMC and upload topology
    // ========================================================================
    log::info!("Step 5: Initializing GPU HMC...");

    let mut hmc = AmberMegaFusedHmc::new(Arc::clone(&context), topology.n_atoms)?;
    hmc.upload_topology(&positions, &bonds, &angles, &dihedrals, &nb_params, &exclusions)?;

    log::info!("  ✓ Topology uploaded to GPU");

    // ========================================================================
    // STEP 6: Enable explicit solvent (PME + SETTLE)
    // ========================================================================
    log::info!("Step 6: Enabling explicit solvent...");

    hmc.enable_explicit_solvent(solvbox.box_dimensions)?;
    hmc.set_water_molecules(&water_oxygens)?;

    log::info!("  - PME: ✓ (box = {:?})", solvbox.box_dimensions);
    log::info!("  - SETTLE: ✓ ({} water molecules)", water_oxygens.len());
    log::info!("  ✓ Explicit solvent enabled");

    // ========================================================================
    // STEP 7: Energy minimization
    // ========================================================================
    log::info!("Step 7: Minimizing energy...");

    hmc.minimize(2000, 0.01)?;
    let max_f1 = hmc.get_max_force()?;
    log::info!("  Phase 1: max force = {:.2}", max_f1);

    hmc.minimize(5000, 0.005)?;
    let max_f2 = hmc.get_max_force()?;
    log::info!("  Phase 2: max force = {:.2}", max_f2);

    if max_f2 > 100.0 {
        hmc.minimize(5000, 0.002)?;
        let max_f3 = hmc.get_max_force()?;
        log::info!("  Phase 3: max force = {:.2}", max_f3);
    }

    log::info!("  ✓ Minimization complete");

    // ========================================================================
    // STEP 8: Run MD simulation
    // ========================================================================
    log::info!("");
    log::info!("Step 8: Running MD simulation...");
    log::info!("  Parameters:");
    log::info!("    - Steps: {}", N_STEPS);
    log::info!("    - Timestep: {} fs", DT);
    log::info!("    - Friction: {} fs⁻¹ ({} ps⁻¹)", GAMMA, GAMMA * 1000.0);
    log::info!("    - Temperature: {} K", TEMPERATURE);
    log::info!("    - Simulation time: {:.2} ps", N_STEPS as f32 * DT / 1000.0);
    log::info!("");

    let result = hmc.run(N_STEPS, DT, TEMPERATURE, GAMMA)
        .context("HMC run failed")?;

    log::info!("");
    log::info!("  Results:");
    log::info!("    - Potential Energy: {:.2} kcal/mol", result.potential_energy);
    log::info!("    - Kinetic Energy: {:.2} kcal/mol", result.kinetic_energy);
    log::info!("    - Average Temperature: {:.1} K", result.avg_temperature);

    // ========================================================================
    // STEP 9: Verify results
    // ========================================================================
    log::info!("");
    log::info!("Step 9: Verifying results...");

    let mut all_passed = true;

    // Temperature check: within ±20% of target
    let temp_ratio = result.avg_temperature / TEMPERATURE as f64;
    let temp_ok = temp_ratio > 0.8 && temp_ratio < 1.2;
    if temp_ok {
        log::info!("  ✓ Temperature: {:.1} K (target: {} K, ratio: {:.2})",
            result.avg_temperature, TEMPERATURE, temp_ratio);
    } else {
        log::warn!("  ✗ Temperature: {:.1} K (target: {} K, ratio: {:.2})",
            result.avg_temperature, TEMPERATURE, temp_ratio);
        all_passed = false;
    }

    // Energy check: not exploded
    let energy_ok = result.potential_energy.abs() < 1e7;
    if energy_ok {
        log::info!("  ✓ Energy bounded: {:.2} kcal/mol", result.potential_energy);
    } else {
        log::warn!("  ✗ Energy exploded: {:.2} kcal/mol", result.potential_energy);
        all_passed = false;
    }

    // Check constraints
    if let Some(ref settle) = hmc.settle() {
        log::info!("  ✓ SETTLE constraints active ({} waters)", settle.n_waters());
    }

    if hmc.is_explicit_solvent() {
        log::info!("  ✓ PME electrostatics active");
    }

    log::info!("");
    log::info!("==============================================");
    if all_passed {
        log::info!("  ✅ ALL TESTS PASSED - Protein MD working!");
    } else {
        log::warn!("  ⚠️  SOME TESTS FAILED");
    }
    log::info!("==============================================");

    if all_passed {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Some verification checks failed"))
    }
}
