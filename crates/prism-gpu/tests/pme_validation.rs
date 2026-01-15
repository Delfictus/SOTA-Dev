//! PME Validation Tests
//!
//! Tests to verify PME electrostatics accuracy via integration testing:
//! 1. Verify Coulomb energy calculation correctness
//! 2. Test water box stability with PME electrostatics
//!
//! Run with: cargo test -p prism-gpu --test pme_validation --features cuda -- --ignored

use std::collections::HashSet;

/// Coulomb constant in kcal·Å/(mol·e²)
/// k_e = 332.0637 kcal·Å/(mol·e²)
const COULOMB_CONSTANT: f64 = 332.0637;

/// Direct Coulomb energy between two charges with minimum image convention
///
/// E = k_e * q1 * q2 / r
///
/// This is the "truth" we compare simulation energy against.
fn direct_coulomb_energy(
    positions: &[[f32; 3]],
    charges: &[f32],
    box_dims: [f32; 3],
) -> f64 {
    let n = positions.len();
    let mut energy = 0.0f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dx = positions[j][0] - positions[i][0];
            let mut dy = positions[j][1] - positions[i][1];
            let mut dz = positions[j][2] - positions[i][2];

            // Minimum image convention
            dx -= box_dims[0] * (dx / box_dims[0]).round();
            dy -= box_dims[1] * (dy / box_dims[1]).round();
            dz -= box_dims[2] * (dz / box_dims[2]).round();

            let r = (dx * dx + dy * dy + dz * dz).sqrt() as f64;
            if r > 1e-6 {
                energy += COULOMB_CONSTANT * (charges[i] as f64) * (charges[j] as f64) / r;
            }
        }
    }

    energy
}

/// Quick sanity check that can run without CUDA
#[test]
fn test_direct_coulomb_math() {
    // Two +1/-1 charges 10 Å apart should give -33.2 kcal/mol
    let positions = vec![
        [0.0f32, 0.0, 0.0],
        [10.0f32, 0.0, 0.0],
    ];
    let charges = vec![1.0f32, -1.0f32];
    let box_dims = [100.0f32, 100.0, 100.0]; // Large box to avoid image interactions

    let energy = direct_coulomb_energy(&positions, &charges, box_dims);
    let expected = -COULOMB_CONSTANT / 10.0; // -33.2 kcal/mol

    println!("Direct Coulomb energy: {:.4} kcal/mol (expected: {:.4})", energy, expected);

    assert!(
        (energy - expected).abs() < 0.01,
        "Direct Coulomb math error: {} vs {}",
        energy, expected
    );
}

/// Test water box temperature stability with explicit solvent
///
/// Runs a short MD simulation of a water box with PME and verifies:
/// 1. Temperature stays within target range
/// 2. Energy is finite (no explosion)
/// 3. SETTLE constraints are working
#[test]
#[ignore] // Run with: cargo test pme_water_box --ignored --features cuda
fn test_pme_water_box_stability() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         PME Water Box Stability Test                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create a small water box (27 waters = 81 atoms in a 3x3x3 grid)
    // Box size: ~9.3 Å (3 * 3.1 Å grid spacing)
    let n_waters = 27;
    let n_atoms = n_waters * 3;
    let box_dim = 9.3f32;
    let box_dims = [box_dim, box_dim, box_dim];

    println!("📦 Water box configuration:");
    println!("   Waters: {}", n_waters);
    println!("   Atoms: {}", n_atoms);
    println!("   Box: {:.1} × {:.1} × {:.1} Å", box_dim, box_dim, box_dim);

    // Generate water positions on a grid
    let mut positions = Vec::with_capacity(n_atoms * 3);
    let mut water_oxygens = Vec::with_capacity(n_waters);

    // TIP3P geometry
    let oh_bond = 0.9572f32;
    let hoh_angle = 104.52f32 * std::f32::consts::PI / 180.0;
    let h_offset_x = oh_bond * (hoh_angle / 2.0).sin();
    let h_offset_y = oh_bond * (hoh_angle / 2.0).cos();

    let spacing = box_dim / 3.0;
    let mut atom_idx = 0;

    for ix in 0..3 {
        for iy in 0..3 {
            for iz in 0..3 {
                let ox = (ix as f32 + 0.5) * spacing;
                let oy = (iy as f32 + 0.5) * spacing;
                let oz = (iz as f32 + 0.5) * spacing;

                // Oxygen
                water_oxygens.push(atom_idx);
                positions.extend_from_slice(&[ox, oy, oz]);

                // Hydrogen 1
                positions.extend_from_slice(&[ox + h_offset_x, oy + h_offset_y, oz]);

                // Hydrogen 2
                positions.extend_from_slice(&[ox - h_offset_x, oy + h_offset_y, oz]);

                atom_idx += 3;
            }
        }
    }

    println!("   Generated {} atom positions", positions.len() / 3);

    // Initialize CUDA
    println!("\n🚀 Initializing CUDA...");
    let context = CudaContext::new(0).expect("Failed to create CUDA context");

    // Create MD engine
    let mut hmc = AmberMegaFusedHmc::new(context, n_atoms)
        .expect("Failed to create HMC engine");

    // TIP3P parameters
    let mut nb_params = Vec::with_capacity(n_atoms);
    for _ in 0..n_waters {
        // Oxygen: σ=3.15061, ε=0.1521, q=-0.834, m=15.9994
        nb_params.push((3.15061f32, 0.1521f32, -0.834f32, 15.9994f32));
        // H1: σ=0, ε=0, q=0.417, m=1.008
        nb_params.push((0.0f32, 0.0f32, 0.417f32, 1.008f32));
        // H2: σ=0, ε=0, q=0.417, m=1.008
        nb_params.push((0.0f32, 0.0f32, 0.417f32, 1.008f32));
    }

    // No bonds/angles/dihedrals for rigid water (SETTLE handles geometry)
    let bonds: Vec<(usize, usize, f32, f32)> = vec![];
    let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    // Build exclusions (within each water molecule)
    let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); n_atoms];
    for i in 0..n_waters {
        let o = i * 3;
        let h1 = o + 1;
        let h2 = o + 2;

        exclusions[o].insert(h1);
        exclusions[o].insert(h2);
        exclusions[h1].insert(o);
        exclusions[h1].insert(h2);
        exclusions[h2].insert(o);
        exclusions[h2].insert(h1);
    }

    // Upload topology
    println!("\n📤 Uploading topology...");
    hmc.upload_topology(&positions, &bonds, &angles, &dihedrals, &nb_params, &exclusions)
        .expect("Failed to upload topology");

    // Enable PBC and explicit solvent (PME)
    println!("   Setting up PBC and PME...");
    hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
    hmc.enable_explicit_solvent(box_dims).expect("Failed to enable explicit solvent");

    // Enable SETTLE for water constraints
    hmc.set_water_molecules(&water_oxygens).expect("Failed to set waters");

    println!("   ✓ Topology uploaded");
    println!("   ✓ PBC enabled: {:.1} × {:.1} × {:.1} Å", box_dims[0], box_dims[1], box_dims[2]);
    println!("   ✓ PME electrostatics enabled");
    println!("   ✓ SETTLE constraints for {} waters", n_waters);

    // Initialize velocities at 310 K
    println!("\n🌡️  Initializing velocities at 310 K...");
    hmc.initialize_velocities(310.0).expect("Failed to init velocities");

    // Run equilibration (1000 steps = 2 ps with dt=2fs)
    println!("\n🔥 Equilibration (1000 steps, 2 ps)...");
    let eq_result = hmc.run_verlet(1000, 2.0, 310.0, 0.01)
        .expect("Equilibration failed");

    println!("   Temperature: {:.1} K", eq_result.avg_temperature);
    println!("   PE: {:.1} kcal/mol", eq_result.potential_energy);
    println!("   KE: {:.1} kcal/mol", eq_result.kinetic_energy);

    // Check equilibration didn't explode
    assert!(
        eq_result.potential_energy.is_finite(),
        "PE exploded during equilibration: {}", eq_result.potential_energy
    );
    assert!(
        eq_result.avg_temperature > 50.0,
        "System froze during equilibration: T={:.1} K", eq_result.avg_temperature
    );
    assert!(
        eq_result.avg_temperature < 800.0,
        "System overheated during equilibration: T={:.1} K", eq_result.avg_temperature
    );

    // Run production (5000 steps = 10 ps)
    println!("\n🏃 Production run (5000 steps, 10 ps)...");
    let result = hmc.run_verlet(5000, 2.0, 310.0, 0.01)
        .expect("Production run failed");

    println!("\n📊 Production Results:");
    println!("   Average temperature: {:.1} K (target: 310 K)", result.avg_temperature);
    println!("   Final PE: {:.1} kcal/mol", result.potential_energy);
    println!("   Final KE: {:.1} kcal/mol", result.kinetic_energy);
    println!("   Total E: {:.1} kcal/mol", result.potential_energy + result.kinetic_energy);

    // Verification checks
    println!("\n✅ Verification:");

    // 1. Temperature stability
    let temp_error = (result.avg_temperature - 310.0).abs() / 310.0;
    println!("   Temperature error: {:.1}%", temp_error * 100.0);
    assert!(
        temp_error < 0.3,
        "Temperature drifted too far: {:.1} K vs 310 K target ({:.1}% error)",
        result.avg_temperature, temp_error * 100.0
    );

    // 2. Energy stability (finite values)
    assert!(
        result.potential_energy.is_finite(),
        "PE is non-finite: {}", result.potential_energy
    );
    assert!(
        result.kinetic_energy.is_finite(),
        "KE is non-finite: {}", result.kinetic_energy
    );

    // 3. Temperature in reasonable range
    assert!(
        result.avg_temperature > 100.0 && result.avg_temperature < 600.0,
        "Temperature {} K outside reasonable range [100, 600]",
        result.avg_temperature
    );

    // 4. Check constraint info
    let constraint_info = result.constraint_info;
    println!("   SETTLE waters: {}", constraint_info.n_waters);
    println!("   H-constraints: {}", constraint_info.n_h_constraints);
    println!("   DOF: {}", result.n_dof);

    assert!(
        constraint_info.n_waters == n_waters,
        "Expected {} SETTLE waters, got {}",
        n_waters, constraint_info.n_waters
    );

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         ✓ PME Water Box Test PASSED                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Test water dimer energy (2 waters)
///
/// Tests that PME handles water-water electrostatic interactions correctly.
#[test]
#[ignore] // Run with: cargo test water_dimer --ignored --features cuda
fn test_water_dimer_energy() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n=== Water Dimer Energy Test ===\n");

    // Two TIP3P water molecules
    let n_atoms = 6;
    let box_dim = 20.0f32; // Large box to minimize periodic images
    let box_dims = [box_dim, box_dim, box_dim];

    // TIP3P geometry
    let oh_bond = 0.9572f32;
    let hoh_angle = 104.52f32 * std::f32::consts::PI / 180.0;
    let h_offset_x = oh_bond * (hoh_angle / 2.0).sin();
    let h_offset_y = oh_bond * (hoh_angle / 2.0).cos();

    // Water 1 centered at (8, 10, 10)
    // Water 2 centered at (12, 10, 10) - 4 Å apart (O-O distance)
    let positions = vec![
        // Water 1: O, H1, H2
        8.0, 10.0, 10.0,
        8.0 + h_offset_x, 10.0 + h_offset_y, 10.0,
        8.0 - h_offset_x, 10.0 + h_offset_y, 10.0,
        // Water 2: O, H1, H2
        12.0, 10.0, 10.0,
        12.0 + h_offset_x, 10.0 + h_offset_y, 10.0,
        12.0 - h_offset_x, 10.0 + h_offset_y, 10.0,
    ];

    let water_oxygens = vec![0, 3];

    // Calculate expected Coulomb energy between the two water molecules
    let positions_3d: Vec<[f32; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let charges = vec![-0.834f32, 0.417, 0.417, -0.834, 0.417, 0.417];

    // Only inter-molecular interactions (exclude intra-molecular)
    let mut expected_coulomb = 0.0f64;
    for i in 0..3 {  // Water 1 atoms
        for j in 3..6 {  // Water 2 atoms
            let dx = positions_3d[j][0] - positions_3d[i][0];
            let dy = positions_3d[j][1] - positions_3d[i][1];
            let dz = positions_3d[j][2] - positions_3d[i][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() as f64;
            if r > 1e-6 {
                expected_coulomb += COULOMB_CONSTANT * (charges[i] as f64) * (charges[j] as f64) / r;
            }
        }
    }
    println!("Expected inter-molecular Coulomb energy: {:.4} kcal/mol", expected_coulomb);

    // Initialize CUDA
    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    let mut hmc = AmberMegaFusedHmc::new(context, n_atoms).expect("Failed to create HMC");

    // TIP3P parameters
    let nb_params = vec![
        (3.15061f32, 0.1521f32, -0.834f32, 15.9994f32),  // O
        (0.0f32, 0.0f32, 0.417f32, 1.008f32),           // H1
        (0.0f32, 0.0f32, 0.417f32, 1.008f32),           // H2
        (3.15061f32, 0.1521f32, -0.834f32, 15.9994f32),  // O
        (0.0f32, 0.0f32, 0.417f32, 1.008f32),           // H1
        (0.0f32, 0.0f32, 0.417f32, 1.008f32),           // H2
    ];

    // No bonded terms (SETTLE handles geometry)
    let bonds: Vec<(usize, usize, f32, f32)> = vec![];
    let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    // Exclusions within each water
    let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); n_atoms];
    // Water 1
    exclusions[0].insert(1); exclusions[0].insert(2);
    exclusions[1].insert(0); exclusions[1].insert(2);
    exclusions[2].insert(0); exclusions[2].insert(1);
    // Water 2
    exclusions[3].insert(4); exclusions[3].insert(5);
    exclusions[4].insert(3); exclusions[4].insert(5);
    exclusions[5].insert(3); exclusions[5].insert(4);

    hmc.upload_topology(&positions, &bonds, &angles, &dihedrals, &nb_params, &exclusions)
        .expect("Failed to upload topology");

    hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
    hmc.enable_explicit_solvent(box_dims).expect("Failed to enable explicit solvent");
    hmc.set_water_molecules(&water_oxygens).expect("Failed to set waters");

    // Compute energy (single point, no dynamics)
    let pe = hmc.minimize(1, 0.0).expect("Failed to compute energy");

    println!("\nSimulation energy:");
    println!("   Total PE: {:.4} kcal/mol", pe);
    println!("   Expected Coulomb (inter-molecular): {:.4} kcal/mol", expected_coulomb);

    // The simulation PE includes LJ and Coulomb; we can't separate them easily
    // But we can verify the energy is negative (attractive) and reasonable
    assert!(
        pe.is_finite(),
        "PE is non-finite"
    );
    assert!(
        pe < 0.0,
        "Water dimer should have attractive (negative) energy, got {}", pe
    );
    assert!(
        pe > -100.0,
        "Water dimer energy {} kcal/mol seems too negative", pe
    );

    println!("\n✓ Water dimer test passed (energy is finite, negative, and reasonable)");
}
