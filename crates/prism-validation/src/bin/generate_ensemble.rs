//! Generate conformational ensemble via MD simulation
//!
//! Runs GPU-accelerated MD with SETTLE and H-constraints enabled,
//! saving snapshots at regular intervals to generate a conformational ensemble
//! for cryptic pocket analysis.
//!
//! Usage:
//!   cargo run --release --features cuda -p prism-validation --bin generate_ensemble -- \
//!     --topology data/prepared/prepared/1l2y_protein_only.json \
//!     --steps 500000 --dt 2.0 --save-interval 500 \
//!     --output ensemble.pdb

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate_ensemble")]
#[command(about = "Generate conformational ensemble via GPU MD simulation")]
struct Args {
    /// Path to prepared topology JSON (from prepare_protein.py)
    #[arg(long)]
    topology: PathBuf,

    /// Total number of MD steps
    #[arg(long, default_value = "100000")]
    steps: u64,

    /// Timestep in femtoseconds
    #[arg(long, default_value = "2.0")]
    dt: f32,

    /// Target temperature in Kelvin
    #[arg(long, default_value = "310.0")]
    temperature: f32,

    /// Langevin friction coefficient in fs⁻¹
    #[arg(long, default_value = "0.01")]
    gamma: f32,

    /// Save snapshot every N steps
    #[arg(long, default_value = "500")]
    save_interval: u64,

    /// Equilibration steps before saving (not included in output)
    #[arg(long, default_value = "10000")]
    equilibration: u64,

    /// Output PDB file (multi-MODEL format)
    #[arg(long)]
    output: PathBuf,

    /// Optional: original PDB file for atom names (otherwise uses element-based naming)
    #[arg(long)]
    pdb: Option<PathBuf>,

    /// Position restraint force constant (kcal/mol/Å², 0 to disable)
    #[arg(long, default_value = "0.0")]
    restraint_k: f32,

    /// Output CSV for energy timeseries (time_ps,PE,KE,total_E)
    #[arg(long)]
    energy_log: Option<PathBuf>,

    /// Output CSV for temperature timeseries (time_ps,T_instantaneous,T_rolling_avg,n_dof,n_settle,n_h_constraints)
    #[arg(long)]
    temperature_log: Option<PathBuf>,
}

// Topology structs matching prepare_protein.py output
#[derive(Debug, Deserialize)]
struct TopologyJson {
    n_atoms: usize,
    positions: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    lj_params: Vec<LjParam>,
    bonds: Vec<BondJson>,
    angles: Vec<AngleJson>,
    dihedrals: Vec<DihedralJson>,
    water_oxygens: Vec<usize>,
    h_clusters: Vec<HClusterJson>,
    box_vectors: Option<Vec<f32>>,
    exclusions: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct LjParam {
    sigma: f32,
    epsilon: f32,
}

#[derive(Debug, Deserialize)]
struct BondJson {
    i: usize,
    j: usize,
    r0: f32,
    #[serde(rename = "k")]
    force_k: f32,
}

#[derive(Debug, Deserialize)]
struct AngleJson {
    i: usize,
    j: usize,
    k_idx: usize,
    theta0: f32,
    force_k: f32,
}

#[derive(Debug, Deserialize)]
struct DihedralJson {
    i: usize,
    j: usize,
    k_idx: usize,
    l: usize,
    periodicity: i32,
    phase: f32,
    force_k: f32,
}

#[derive(Debug, Deserialize)]
struct HClusterJson {
    #[serde(rename = "type")]
    cluster_type: i32,
    central_atom: i32,
    hydrogen_atoms: Vec<i32>,
    bond_lengths: Vec<f32>,
    n_hydrogens: i32,
    inv_mass_central: f32,
    inv_mass_h: f32,
}

/// Simple atom info for PDB output
#[derive(Clone)]
struct AtomInfo {
    name: String,
    element: String,
    residue_name: String,
    residue_id: i32,
    chain: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         PRISM-4D Conformational Ensemble Generator           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Load topology
    println!("\n📂 Loading topology: {:?}", args.topology);
    let topology_json = std::fs::read_to_string(&args.topology)
        .context("Failed to read topology file")?;
    let topology: TopologyJson =
        serde_json::from_str(&topology_json).context("Failed to parse topology JSON")?;

    let n_atoms = topology.n_atoms;
    let n_waters = topology.water_oxygens.len();
    let n_h_clusters = topology.h_clusters.len();

    println!("   Atoms: {}", n_atoms);
    println!("   Bonds: {}", topology.bonds.len());
    println!("   Angles: {}", topology.angles.len());
    println!("   Dihedrals: {}", topology.dihedrals.len());
    println!("   Waters: {}", n_waters);
    println!("   H-bond clusters: {}", n_h_clusters);

    // Count H-cluster types
    let n_single = topology.h_clusters.iter().filter(|c| c.cluster_type == 1).count();
    let n_ch2 = topology.h_clusters.iter().filter(|c| c.cluster_type == 2).count();
    let n_ch3 = topology.h_clusters.iter().filter(|c| c.cluster_type == 3).count();
    let n_nh2 = topology.h_clusters.iter().filter(|c| c.cluster_type == 4).count();
    let n_nh3 = topology.h_clusters.iter().filter(|c| c.cluster_type == 5).count();
    if n_h_clusters > 0 {
        println!("   H-cluster breakdown: {} single, {} CH2, {} CH3, {} NH2, {} NH3",
                 n_single, n_ch2, n_ch3, n_nh2, n_nh3);
    }

    // Generate atom info for PDB output
    let atom_info = if let Some(ref pdb_path) = args.pdb {
        println!("\n📄 Loading atom names from PDB: {:?}", pdb_path);
        load_atom_info_from_pdb(pdb_path)?
    } else {
        println!("\n📄 Generating atom names from masses (no PDB provided)");
        generate_atom_info_from_masses(&topology.masses)
    };

    if atom_info.len() != n_atoms {
        println!("   Warning: atom info count ({}) differs from topology ({})",
                 atom_info.len(), n_atoms);
    }

    // Calculate simulation parameters
    let total_time_fs = args.steps as f64 * args.dt as f64;
    let total_time_ps = total_time_fs / 1000.0;
    let total_time_ns = total_time_ps / 1000.0;
    let production_steps = args.steps.saturating_sub(args.equilibration);
    let n_snapshots = production_steps / args.save_interval;
    let snapshot_interval_ps = args.save_interval as f64 * args.dt as f64 / 1000.0;

    println!("\n⚙️  Simulation parameters:");
    println!("   Total steps: {} ({:.3} ns)", args.steps, total_time_ns);
    println!("   Timestep: {} fs", args.dt);
    println!("   Temperature: {} K", args.temperature);
    println!("   Friction (γ): {} fs⁻¹", args.gamma);
    println!("   Equilibration: {} steps ({:.3} ps)",
             args.equilibration, args.equilibration as f64 * args.dt as f64 / 1000.0);
    println!("   Production: {} steps", production_steps);
    println!("   Save interval: {} steps ({:.2} ps)", args.save_interval, snapshot_interval_ps);
    println!("   Expected snapshots: {}", n_snapshots);
    if args.restraint_k > 0.0 {
        println!("   Position restraints: k={} kcal/(mol·Å²)", args.restraint_k);
    }

    // Initialize CUDA and run simulation
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        use prism_gpu::AmberMegaFusedHmc;
        use prism_gpu::HConstraintCluster;

        println!("\n🚀 Initializing CUDA...");
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;

        // Create MD engine
        let mut hmc = AmberMegaFusedHmc::new(context, n_atoms)
            .context("Failed to create MD engine")?;

        // Convert topology data
        let bonds: Vec<(usize, usize, f32, f32)> = topology
            .bonds
            .iter()
            .map(|b| (b.i, b.j, b.force_k, b.r0))
            .collect();

        let angles: Vec<(usize, usize, usize, f32, f32)> = topology
            .angles
            .iter()
            .map(|a| (a.i, a.j, a.k_idx, a.force_k, a.theta0))
            .collect();

        let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topology
            .dihedrals
            .iter()
            .map(|d| {
                (
                    d.i,
                    d.j,
                    d.k_idx,
                    d.l,
                    d.force_k,
                    d.periodicity as f32,
                    d.phase,
                )
            })
            .collect();

        let nb_params: Vec<(f32, f32, f32, f32)> = (0..n_atoms)
            .map(|i| {
                let lj = &topology.lj_params[i];
                (lj.sigma, lj.epsilon, topology.charges[i], topology.masses[i])
            })
            .collect();

        let exclusions: Vec<HashSet<usize>> = topology
            .exclusions
            .iter()
            .map(|e| e.iter().copied().collect())
            .collect();

        // Upload topology
        println!("   Uploading topology...");
        hmc.upload_topology(
            &topology.positions,
            &bonds,
            &angles,
            &dihedrals,
            &nb_params,
            &exclusions,
        )?;

        // Enable PBC if box vectors provided
        if let Some(ref box_vecs) = topology.box_vectors {
            if box_vecs.len() >= 3 {
                println!("   Box: {:.2} x {:.2} x {:.2} Å", box_vecs[0], box_vecs[1], box_vecs[2]);
                hmc.set_pbc_box([box_vecs[0], box_vecs[1], box_vecs[2]])?;
            }
        }

        // Enable SETTLE for waters
        if !topology.water_oxygens.is_empty() {
            hmc.set_water_molecules(&topology.water_oxygens)?;
            println!("   ✓ SETTLE enabled for {} water molecules", n_waters);
        }

        // Enable H-constraints
        if !topology.h_clusters.is_empty() {
            let h_clusters: Vec<HConstraintCluster> = topology
                .h_clusters
                .iter()
                .map(|c| HConstraintCluster {
                    central_atom: c.central_atom,
                    hydrogen_atoms: [
                        c.hydrogen_atoms.get(0).copied().unwrap_or(-1),
                        c.hydrogen_atoms.get(1).copied().unwrap_or(-1),
                        c.hydrogen_atoms.get(2).copied().unwrap_or(-1),
                    ],
                    bond_lengths: [
                        c.bond_lengths.get(0).copied().unwrap_or(0.0),
                        c.bond_lengths.get(1).copied().unwrap_or(0.0),
                        c.bond_lengths.get(2).copied().unwrap_or(0.0),
                    ],
                    inv_mass_central: c.inv_mass_central,
                    inv_mass_h: c.inv_mass_h,
                    n_hydrogens: c.n_hydrogens,
                    cluster_type: c.cluster_type,
                })
                .collect();
            hmc.set_h_constraints(&h_clusters)?;
            println!("   ✓ H-constraints enabled for {} clusters", n_h_clusters);
        }

        // Enable position restraints if requested
        if args.restraint_k > 0.0 {
            let heavy_atoms: Vec<usize> = (0..n_atoms)
                .filter(|&i| topology.masses[i] > 2.0)
                .collect();
            hmc.set_position_restraints(&heavy_atoms, args.restraint_k)?;
            println!("   ✓ Position restraints on {} heavy atoms", heavy_atoms.len());
        }

        // Energy minimization
        println!("\n⚡ Energy minimization...");
        let pe_before = hmc.minimize(1, 0.0001)?;
        let pe_after = hmc.minimize(200, 0.0001)?;
        println!("   PE: {:.1} → {:.1} kcal/mol", pe_before, pe_after);

        // Initialize velocities
        hmc.initialize_velocities(args.temperature)?;
        println!("   ✓ Velocities initialized at {} K", args.temperature);

        // Equilibration
        if args.equilibration > 0 {
            println!("\n🔥 Equilibration ({} steps, {:.2} ps)...",
                     args.equilibration,
                     args.equilibration as f64 * args.dt as f64 / 1000.0);

            let eq_result = hmc.run_verlet(
                args.equilibration as usize,
                args.dt,
                args.temperature,
                args.gamma,
            )?;
            println!("   Final T: {:.1} K, PE: {:.1} kcal/mol",
                     eq_result.avg_temperature, eq_result.potential_energy);
        }

        // Create output file
        if let Some(parent) = args.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut output = BufWriter::new(
            File::create(&args.output).context("Failed to create output file")?,
        );

        // Production run with snapshot saving
        println!("\n🏃 Production run ({} steps, {:.3} ns)...",
                 production_steps,
                 production_steps as f64 * args.dt as f64 / 1_000_000.0);

        let mut model_number = 1u32;
        let mut steps_run = 0u64;
        let mut last_progress = 0u32;

        // Collect energy trajectory for CSV output
        use prism_gpu::{EnergyRecord, ConstraintInfo};
        let mut all_energy_records: Vec<EnergyRecord> = Vec::new();
        let constraint_info_cached: ConstraintInfo = hmc.get_constraint_info();

        // Save initial frame
        let positions = hmc.get_positions()?;
        write_pdb_model(&mut output, &atom_info, &positions, model_number)?;
        model_number += 1;

        while steps_run < production_steps {
            // Run for save_interval steps
            let steps_this_round = args.save_interval.min(production_steps - steps_run) as usize;
            let result = hmc.run_verlet(
                steps_this_round,
                args.dt,
                args.temperature,
                args.gamma,
            )?;

            // Collect energy records with proper time offset
            let base_step = args.equilibration + steps_run;
            for record in &result.energy_trajectory {
                let adjusted_record = EnergyRecord {
                    step: base_step + record.step,
                    time_ps: (base_step + record.step) as f64 * args.dt as f64 / 1000.0,
                    potential_energy: record.potential_energy,
                    kinetic_energy: record.kinetic_energy,
                    total_energy: record.total_energy,
                    temperature: record.temperature,
                };
                all_energy_records.push(adjusted_record);
            }

            steps_run += steps_this_round as u64;

            // Save snapshot
            let positions = hmc.get_positions()?;
            write_pdb_model(&mut output, &atom_info, &positions, model_number)?;
            model_number += 1;

            // Progress update every 10%
            let progress = ((steps_run * 100) / production_steps) as u32;
            if progress >= last_progress + 10 {
                let current_time_ns = (args.equilibration + steps_run) as f64
                    * args.dt as f64 / 1_000_000.0;
                println!(
                    "   {:>3}% ({:.3} ns, {} snapshots, T={:.0}K, PE={:.0})",
                    progress,
                    current_time_ns,
                    model_number - 1,
                    result.avg_temperature,
                    result.potential_energy
                );
                last_progress = progress;
            }

            // Early termination check
            if result.avg_temperature > 5000.0 {
                println!("   ⚠️  Temperature exploded ({:.0} K), stopping early",
                         result.avg_temperature);
                break;
            }
        }

        writeln!(output, "END")?;
        output.flush()?;

        let total_snapshots = model_number - 1;
        let actual_time_ns = (args.equilibration + steps_run) as f64 * args.dt as f64 / 1_000_000.0;

        println!("\n✅ Ensemble generation complete!");
        println!("   Output: {:?}", args.output);
        println!("   Snapshots: {}", total_snapshots);
        println!("   Simulation time: {:.3} ns", actual_time_ns);

        // File size info
        if let Ok(metadata) = std::fs::metadata(&args.output) {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            println!("   File size: {:.1} MB", size_mb);
        }

        // Write energy timeseries CSV if requested
        if let Some(ref energy_path) = args.energy_log {
            if let Some(parent) = energy_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut energy_file = BufWriter::new(
                File::create(energy_path).context("Failed to create energy log file")?,
            );
            writeln!(energy_file, "time_ps,PE_kcal_mol,KE_kcal_mol,total_E_kcal_mol")?;
            for record in &all_energy_records {
                writeln!(
                    energy_file,
                    "{:.4},{:.4},{:.4},{:.4}",
                    record.time_ps,
                    record.potential_energy,
                    record.kinetic_energy,
                    record.total_energy
                )?;
            }
            energy_file.flush()?;
            println!("   Energy log: {:?} ({} records)", energy_path, all_energy_records.len());
        }

        // Write temperature timeseries CSV if requested
        if let Some(ref temp_path) = args.temperature_log {
            if let Some(parent) = temp_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut temp_file = BufWriter::new(
                File::create(temp_path).context("Failed to create temperature log file")?,
            );
            // Header with constraint info columns per plan
            writeln!(temp_file, "time_ps,T_instantaneous_K,T_rolling_avg_K,n_dof,n_settle_constraints,n_h_constraints")?;

            // Get DOF info from the cached constraint info
            let n_dof = hmc.compute_n_dof(true);  // true = remove COM DOFs

            // Compute rolling average (window of 10 points)
            let window_size = 10;
            for (i, record) in all_energy_records.iter().enumerate() {
                // Rolling average: average of last `window_size` temperature values
                let start = if i >= window_size { i - window_size + 1 } else { 0 };
                let t_avg: f64 = all_energy_records[start..=i]
                    .iter()
                    .map(|r| r.temperature)
                    .sum::<f64>()
                    / (i - start + 1) as f64;

                writeln!(
                    temp_file,
                    "{:.4},{:.2},{:.2},{},{},{}",
                    record.time_ps,
                    record.temperature,
                    t_avg,
                    n_dof,
                    constraint_info_cached.n_settle_constraints,
                    constraint_info_cached.n_h_constraints
                )?;
            }
            temp_file.flush()?;
            println!("   Temperature log: {:?} ({} records)", temp_path, all_energy_records.len());
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n❌ ERROR: CUDA feature not enabled.");
        println!("   Rebuild with: cargo run --release --features cuda ...");
        std::process::exit(1);
    }

    Ok(())
}

/// Write a single MODEL to PDB file
fn write_pdb_model<W: Write>(
    writer: &mut W,
    atoms: &[AtomInfo],
    positions: &[f32],
    model_number: u32,
) -> Result<()> {
    writeln!(writer, "MODEL     {:>4}", model_number)?;

    for (i, atom) in atoms.iter().enumerate() {
        if i * 3 + 2 >= positions.len() {
            break;
        }
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        // PDB ATOM format (80 columns)
        // ATOM  serial name  res  chain resSeq    x       y       z     occ  temp   element
        writeln!(
            writer,
            "ATOM  {:>5} {:^4} {:>3} {:>1}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}",
            (i + 1) % 100000,  // Wrap at 99999
            atom.name,
            atom.residue_name,
            atom.chain,
            atom.residue_id % 10000,  // Wrap at 9999
            x,
            y,
            z,
            atom.element
        )?;
    }

    writeln!(writer, "ENDMDL")?;
    Ok(())
}

/// Generate atom info from masses (when no PDB available)
fn generate_atom_info_from_masses(masses: &[f32]) -> Vec<AtomInfo> {
    let mut atom_count: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut residue_id = 1;
    let mut atoms_in_residue = 0;

    masses.iter().enumerate().map(|(i, &mass)| {
        // Determine element from mass
        let (element, name_prefix) = if mass < 1.5 {
            ("H", "H")
        } else if mass < 13.0 {
            ("C", "C")
        } else if mass < 15.0 {
            ("N", "N")
        } else if mass < 17.0 {
            ("O", "O")
        } else if mass < 33.0 {
            ("S", "S")
        } else {
            ("X", "X")
        };

        // Count this element type
        *atom_count.entry(element).or_insert(0) += 1;
        let count = atom_count[element];

        // Create atom name (max 4 chars)
        let name = if count < 100 {
            format!("{}{}", name_prefix, count)
        } else {
            format!("{}{}", name_prefix, count % 100)
        };

        // Simple residue tracking (~20 atoms per residue for proteins)
        atoms_in_residue += 1;
        if atoms_in_residue > 20 && element != "H" {
            residue_id += 1;
            atoms_in_residue = 1;
        }

        AtomInfo {
            name,
            element: element.to_string(),
            residue_name: "UNK".to_string(),
            residue_id,
            chain: "A".to_string(),
        }
    }).collect()
}

/// Load atom info from original PDB file
fn load_atom_info_from_pdb(path: &PathBuf) -> Result<Vec<AtomInfo>> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read PDB file")?;

    let mut atoms = Vec::new();

    for line in content.lines() {
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            // Parse PDB ATOM record
            let name = line.get(12..16).unwrap_or("    ").trim().to_string();
            let residue_name = line.get(17..20).unwrap_or("UNK").trim().to_string();
            let chain = line.get(21..22).unwrap_or("A").to_string();
            let residue_id: i32 = line.get(22..26)
                .unwrap_or("1")
                .trim()
                .parse()
                .unwrap_or(1);
            let element = line.get(76..78)
                .unwrap_or(&name[0..1])
                .trim()
                .to_string();

            atoms.push(AtomInfo {
                name,
                element,
                residue_name,
                residue_id,
                chain,
            });
        }
    }

    println!("   Loaded {} atoms from PDB", atoms.len());
    Ok(atoms)
}
