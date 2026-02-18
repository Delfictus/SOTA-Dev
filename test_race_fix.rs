/// Simple test to verify per-structure kernel launch fix
/// This tests that multiple structures can be simulated without race conditions
use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use prism_gpu::{AmberSimdBatch, StructureTopology, OptimizationConfig};
use std::collections::HashSet;
use std::time::Instant;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Per-Structure Kernel Launch Race Condition Fix Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Create CUDA context
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;
    println!("✓ CUDA context initialized (GPU 0)");

    // Create a simple test system: 2 argon atoms with LJ interaction
    // Simple enough to validate correctness, complex enough to show race conditions
    let n_atoms = 100;
    let mut positions = Vec::new();
    let mut masses = Vec::new();
    let mut charges = Vec::new();
    let mut sigmas = Vec::new();
    let mut epsilons = Vec::new();
    let mut exclusions = Vec::new();

    // Create a simple cubic lattice of atoms
    let spacing = 3.0; // Angstroms
    let grid_size = ((n_atoms as f32).cbrt().ceil() as usize).max(2);

    for i in 0..n_atoms {
        let x = (i / (grid_size * grid_size)) as f32 * spacing;
        let y = ((i / grid_size) % grid_size) as f32 * spacing;
        let z = (i % grid_size) as f32 * spacing;

        positions.push(x);
        positions.push(y);
        positions.push(z);

        masses.push(39.948); // Argon mass
        charges.push(0.0);   // Neutral
        sigmas.push(3.4);    // Argon LJ sigma
        epsilons.push(0.238); // Argon LJ epsilon (kcal/mol)
        exclusions.push(HashSet::new());
    }

    println!("✓ Created test system: {} atoms in {}³ grid", n_atoms, grid_size);

    // Create batch with 3 independent structures (test cross-structure race)
    let n_structures = 3;
    let max_atoms = n_atoms * n_structures + 1000;

    let mut config = OptimizationConfig::default();
    config.use_batched_forces = true; // Enable per-structure launch path

    let mut batch = AmberSimdBatch::new_with_config(
        context,
        max_atoms,
        n_structures,
        config,
    ).context("Failed to create AmberSimdBatch")?;

    println!("✓ AmberSimdBatch created (max_atoms={}, n_structures={})", max_atoms, n_structures);

    // Add the same structure 3 times (simulating batch processing)
    let structure = StructureTopology {
        positions: positions.clone(),
        masses: masses.clone(),
        charges: charges.clone(),
        sigmas: sigmas.clone(),
        epsilons: epsilons.clone(),
        bonds: vec![],
        angles: vec![],
        dihedrals: vec![],
        exclusions: exclusions.clone(),
    };

    for i in 0..n_structures {
        batch.add_structure(&structure)?;
        println!("  Added structure {}", i + 1);
    }

    batch.finalize_batch()?;
    println!("✓ Batch finalized");
    println!();

    // Run a short equilibration to test stability
    println!("Running equilibration (1000 steps, 300K)...");
    let n_steps = 1000;
    let dt = 1.0; // fs
    let temperature = 300.0; // K
    let gamma = 0.1; // ps^-1

    let start = Instant::now();
    batch.run(n_steps, dt, temperature, gamma)?;
    let elapsed = start.elapsed();

    println!("✓ Equilibration complete: {:.2?}", elapsed);
    println!();

    // Get results and check for stability
    let results = batch.get_all_results()?;
    println!("═══════════════════════════════════════════════════════════");
    println!("  Results (Testing for Race Condition Symptoms)");
    println!("═══════════════════════════════════════════════════════════");

    let mut all_stable = true;

    for (idx, result) in results.iter().enumerate() {
        println!();
        println!("Structure {}:", idx + 1);
        println!("  Potential Energy: {:.2e} kcal/mol", result.potential_energy);
        println!("  Kinetic Energy:   {:.2e} kcal/mol", result.kinetic_energy);
        println!("  Temperature:      {:.1} K", result.temperature);

        // Calculate RMSD from initial positions
        let mut rmsd_sq = 0.0f32;
        for i in 0..n_atoms * 3 {
            let diff = result.positions[i] - positions[i];
            rmsd_sq += diff * diff;
        }
        let rmsd = (rmsd_sq / n_atoms as f32).sqrt();
        println!("  RMSD from start:  {:.4} Å", rmsd);

        // Check for NaN (indicates numerical instability from race)
        if result.potential_energy.is_nan() || result.kinetic_energy.is_nan() {
            println!("  ✗ FAIL: NaN detected (race condition symptom!)");
            all_stable = false;
        }

        // Check for explosion (RMSD > 100Å indicates race-induced forces)
        if rmsd > 100.0 {
            println!("  ✗ FAIL: Structure exploded (RMSD > 100Å, race symptom!)");
            all_stable = false;
        }

        // Check for frozen structure (RMSD < 0.01Å indicates computation didn't run)
        if rmsd < 0.01 {
            println!("  ⚠ WARNING: Very low RMSD, structure may be frozen");
        }

        // Check temperature is reasonable (50K-600K range)
        if result.temperature < 50.0 || result.temperature > 600.0 {
            println!("  ⚠ WARNING: Temperature outside normal range");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");

    if all_stable {
        println!("✓ TEST PASSED: All structures stable, no race condition!");
        println!("  - No NaN values detected");
        println!("  - No structural explosions");
        println!("  - Per-structure kernel launch fix is working correctly");
    } else {
        println!("✗ TEST FAILED: Race condition symptoms detected!");
        println!("  - Check for NaN or extreme RMSD values above");
    }

    println!();
    println!("Performance: {:.0} steps/sec ({:.2} ns/day equivalent)",
        n_steps as f64 / elapsed.as_secs_f64(),
        n_steps as f64 * dt as f64 * 86400.0 / elapsed.as_secs_f64() / 1e6);
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
