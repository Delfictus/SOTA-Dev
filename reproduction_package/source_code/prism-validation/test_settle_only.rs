//! Test SETTLE constraint solver in isolation
//!
//! This tests the SETTLE algorithm without running full MD,
//! to verify the constraint projection works correctly.

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TopologyJson {
    n_atoms: usize,
    positions: Vec<f32>,
    masses: Vec<f32>,
    water_oxygens: Vec<usize>,
}

fn main() -> Result<()> {
    env_logger::init();

    println!("SETTLE Constraint Solver - Isolated Test");
    println!("========================================");

    // Load solvated topology
    let topology_path = "data/prepared/prepared/1l2y_topology.json";
    println!("\nLoading topology: {}", topology_path);
    let topology_json = std::fs::read_to_string(topology_path)
        .context("Failed to read topology JSON")?;
    let topology: TopologyJson = serde_json::from_str(&topology_json)
        .context("Failed to parse topology JSON")?;

    println!("   Atoms: {}", topology.n_atoms);
    println!("   Water molecules: {}", topology.water_oxygens.len());

    // TIP3P targets
    let target_oh = 0.9572f32;
    let target_hh = 1.5136f32;

    // Check initial water geometry
    println!("\n=== Initial Water Geometry ===");
    let mut positions = topology.positions.clone();
    let (max_oh_err, max_hh_err) = check_water_geometry(&positions, &topology.water_oxygens, target_oh, target_hh);
    println!("   Max OH error: {:.4} Å (target: {} Å)", max_oh_err, target_oh);
    println!("   Max HH error: {:.4} Å (target: {} Å)", max_hh_err, target_hh);

    // Show actual geometry of first water
    let wo = topology.water_oxygens[0];
    let o = [positions[wo * 3], positions[wo * 3 + 1], positions[wo * 3 + 2]];
    let h1 = [positions[(wo + 1) * 3], positions[(wo + 1) * 3 + 1], positions[(wo + 1) * 3 + 2]];
    let h2 = [positions[(wo + 2) * 3], positions[(wo + 2) * 3 + 1], positions[(wo + 2) * 3 + 2]];
    let oh1 = dist(&o, &h1);
    let oh2 = dist(&o, &h2);
    let hh = dist(&h1, &h2);
    println!("   First water: OH1={:.4}, OH2={:.4}, HH={:.4}", oh1, oh2, hh);

    #[cfg(feature = "cuda")]
    {
        use prism_gpu::settle::Settle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        println!("\n=== Initializing CUDA ===");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        let stream = context.default_stream();

        // Create SETTLE solver
        println!("\n=== Creating SETTLE solver ===");
        let mut settle = Settle::new(context, &topology.water_oxygens, topology.n_atoms)?;
        println!("   Initialized for {} waters", topology.water_oxygens.len());

        // Upload positions to GPU
        println!("\n=== Uploading positions to GPU ===");
        let mut d_positions = stream.alloc_zeros::<f32>(topology.n_atoms * 3)?;
        stream.memcpy_htod(&positions, &mut d_positions)?;

        // Save old positions (for SETTLE algorithm)
        settle.save_positions(&d_positions)?;

        // Apply SETTLE constraint
        println!("\n=== Applying SETTLE ===");
        settle.apply(&mut d_positions, 0.001)?; // dt doesn't matter for position constraint

        // Download and check
        let mut new_positions = vec![0.0f32; topology.n_atoms * 3];
        stream.memcpy_dtoh(&d_positions, &mut new_positions)?;

        let (new_max_oh_err, new_max_hh_err) = check_water_geometry(&new_positions, &topology.water_oxygens, target_oh, target_hh);
        println!("\n=== After SETTLE ===");
        println!("   Max OH error: {:.4} Å -> {:.4} Å", max_oh_err, new_max_oh_err);
        println!("   Max HH error: {:.4} Å -> {:.4} Å", max_hh_err, new_max_hh_err);

        // Check first water after SETTLE
        let o_new = [new_positions[wo * 3], new_positions[wo * 3 + 1], new_positions[wo * 3 + 2]];
        let h1_new = [new_positions[(wo + 1) * 3], new_positions[(wo + 1) * 3 + 1], new_positions[(wo + 1) * 3 + 2]];
        let h2_new = [new_positions[(wo + 2) * 3], new_positions[(wo + 2) * 3 + 1], new_positions[(wo + 2) * 3 + 2]];
        let oh1_new = dist(&o_new, &h1_new);
        let oh2_new = dist(&o_new, &h2_new);
        let hh_new = dist(&h1_new, &h2_new);
        println!("   First water after: OH1={:.4}, OH2={:.4}, HH={:.4}", oh1_new, oh2_new, hh_new);

        // Apply multiple times to check convergence
        println!("\n=== Iterative SETTLE (5 iterations) ===");
        for i in 1..=5 {
            settle.save_positions(&d_positions)?;
            settle.apply(&mut d_positions, 0.001)?;
            stream.memcpy_dtoh(&d_positions, &mut new_positions)?;

            let (oh_err, hh_err) = check_water_geometry(&new_positions, &topology.water_oxygens, target_oh, target_hh);
            println!("   Iter {}: OH err={:.6} Å, HH err={:.6} Å", i, oh_err, hh_err);
        }

        // Final result
        stream.memcpy_dtoh(&d_positions, &mut new_positions)?;
        let (final_oh_err, final_hh_err) = check_water_geometry(&new_positions, &topology.water_oxygens, target_oh, target_hh);

        println!("\n=== Final Result ===");
        println!("   Max OH error: {:.6} Å (threshold: 0.01 Å)", final_oh_err);
        println!("   Max HH error: {:.6} Å (threshold: 0.01 Å)", final_hh_err);

        if final_oh_err < 0.01 && final_hh_err < 0.01 {
            println!("\n[PASS] SETTLE converged!");
        } else {
            println!("\n[FAIL] SETTLE did not converge");
            println!("       This indicates a bug in the SETTLE algorithm");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\nCUDA feature not enabled. Cannot run GPU tests.");
    }

    Ok(())
}

fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn check_water_geometry(positions: &[f32], water_oxygens: &[usize], target_oh: f32, target_hh: f32) -> (f32, f32) {
    let mut max_oh_err = 0.0f32;
    let mut max_hh_err = 0.0f32;

    for &wo in water_oxygens {
        let o = [positions[wo * 3], positions[wo * 3 + 1], positions[wo * 3 + 2]];
        let h1 = [positions[(wo + 1) * 3], positions[(wo + 1) * 3 + 1], positions[(wo + 1) * 3 + 2]];
        let h2 = [positions[(wo + 2) * 3], positions[(wo + 2) * 3 + 1], positions[(wo + 2) * 3 + 2]];

        let oh1 = dist(&o, &h1);
        let oh2 = dist(&o, &h2);
        let hh = dist(&h1, &h2);

        max_oh_err = max_oh_err.max((oh1 - target_oh).abs()).max((oh2 - target_oh).abs());
        max_hh_err = max_hh_err.max((hh - target_hh).abs());
    }

    (max_oh_err, max_hh_err)
}
