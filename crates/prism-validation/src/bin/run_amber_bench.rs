//! AMBER All-Atom Dynamics Benchmark
//!
//! Tests the AMBER ff14SB dynamics on ATLAS AlphaFlow-82 benchmark.
//! Compares RMSF from MD trajectory against experimental B-factors.
//!
//! This is a NEW file for Phase 2 - does NOT modify any locked files.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "run-amber-bench")]
#[command(about = "Run AMBER ff14SB all-atom dynamics benchmark on ATLAS proteins")]
struct Args {
    /// Number of simulation steps per protein
    #[arg(long, default_value = "100")]
    n_steps: usize,

    /// Save trajectory every N steps
    #[arg(long, default_value = "10")]
    save_every: usize,

    /// Temperature in Kelvin
    #[arg(long, default_value = "300.0")]
    temperature: f64,

    /// Timestep in femtoseconds
    #[arg(long, default_value = "2.0")]
    timestep: f64,

    /// Use Langevin thermostat (recommended)
    #[arg(long, default_value = "true")]
    use_langevin: bool,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Maximum number of proteins to test (0 = all)
    #[arg(long, default_value = "0")]
    max_proteins: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct AtlasTarget {
    pdb_id: String,
    #[serde(default)]
    chain: String,
    md_rmsf: Vec<f64>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let data_dir = PathBuf::from("data/atlas_alphaflow");
    let targets_path = data_dir.join("atlas_targets.json");
    let content = fs::read_to_string(&targets_path)?;
    let mut targets: Vec<AtlasTarget> = serde_json::from_str(&content)?;
    let pdb_dir = data_dir.join("pdb");

    // Limit proteins if requested
    if args.max_proteins > 0 && targets.len() > args.max_proteins {
        targets.truncate(args.max_proteins);
    }

    println!("{}",
             "=".repeat(80));
    println!("     AMBER ff14SB ALL-ATOM DYNAMICS BENCHMARK");
    println!("{}", "=".repeat(80));
    println!();
    println!("  Configuration:");
    println!("    Steps/protein: {}", args.n_steps);
    println!("    Save every:    {} steps", args.save_every);
    println!("    Temperature:   {} K", args.temperature);
    println!("    Timestep:      {} fs", args.timestep);
    println!("    Integrator:    {}", if args.use_langevin { "Langevin (BAOAB)" } else { "HMC" });
    println!("    Proteins:      {}", targets.len());
    println!();

    // Import the AMBER dynamics module
    use prism_physics::amber_ff14sb::PdbAtom;
    use prism_physics::amber_dynamics::{AmberSimulator, AmberSimConfig};

    let config = AmberSimConfig {
        temperature: args.temperature,
        timestep: args.timestep,
        n_leapfrog_steps: 10,
        friction: 1.0,
        use_langevin: args.use_langevin,
        seed: args.seed,
        use_gpu: true, // Enable GPU acceleration when available
    };

    let mut correlations = Vec::new();
    let mut total_time = 0.0;
    let mut successful = 0;
    let mut failed = 0;

    let overall_start = Instant::now();

    for (idx, target) in targets.iter().enumerate() {
        let pdb_path_with_chain = pdb_dir.join(format!("{}_{}.pdb", target.pdb_id.to_lowercase(), target.chain));
        let pdb_path_no_chain = pdb_dir.join(format!("{}.pdb", target.pdb_id.to_lowercase()));

        let pdb_path = if pdb_path_with_chain.exists() {
            pdb_path_with_chain
        } else if pdb_path_no_chain.exists() {
            pdb_path_no_chain
        } else {
            eprintln!("  [{}/{}] {} - PDB not found, skipping", idx + 1, targets.len(), target.pdb_id);
            failed += 1;
            continue;
        };

        let target_chain = if target.chain.is_empty() { None } else { Some(target.chain.as_str()) };

        // Parse PDB to get atoms
        match parse_pdb_atoms(&pdb_path, target_chain) {
            Ok(atoms) => {
                if atoms.len() < 50 {
                    eprintln!("  [{}/{}] {} - Too few atoms ({}), skipping",
                              idx + 1, targets.len(), target.pdb_id, atoms.len());
                    failed += 1;
                    continue;
                }

                let start = Instant::now();

                // Create simulator and run
                match AmberSimulator::new(&atoms, config.clone()) {
                    Ok(mut sim) => {
                        match sim.run(args.n_steps, args.save_every) {
                            Ok(result) => {
                                let elapsed = start.elapsed().as_secs_f64();
                                total_time += elapsed;

                                // Get CA indices for comparison with experimental RMSF
                                let ca_indices: Vec<usize> = atoms.iter()
                                    .enumerate()
                                    .filter(|(_, a)| a.name.trim() == "CA")
                                    .map(|(i, _)| i)
                                    .collect();

                                // Extract RMSF for CA atoms only
                                let ca_rmsf: Vec<f64> = ca_indices.iter()
                                    .filter_map(|&i| result.rmsf.get(i).copied())
                                    .collect();

                                // Compute correlation with experimental RMSF
                                if ca_rmsf.len() == target.md_rmsf.len() && !ca_rmsf.is_empty() {
                                    let corr = pearson_correlation(&ca_rmsf, &target.md_rmsf);
                                    if corr.is_finite() {
                                        correlations.push(corr);
                                        successful += 1;

                                        let marker = if corr >= 0.5 { "+" } else if corr >= 0.3 { "~" } else { "-" };
                                        println!("  [{}/{}] {} {} {:>5} atoms, {:>4} CA  rho={:.3}  PE={:.1} kcal/mol  T={:.1}K  time={:.2}s",
                                                 idx + 1, targets.len(), marker, target.pdb_id,
                                                 atoms.len(), ca_rmsf.len(),
                                                 corr, result.avg_potential_energy, result.avg_temperature, elapsed);
                                    } else {
                                        eprintln!("  [{}/{}] {} - Invalid correlation, skipping", idx + 1, targets.len(), target.pdb_id);
                                        failed += 1;
                                    }
                                } else {
                                    eprintln!("  [{}/{}] {} - Length mismatch (CA={}, exp={}), skipping",
                                              idx + 1, targets.len(), target.pdb_id, ca_rmsf.len(), target.md_rmsf.len());
                                    failed += 1;
                                }
                            }
                            Err(e) => {
                                eprintln!("  [{}/{}] {} - Simulation failed: {}", idx + 1, targets.len(), target.pdb_id, e);
                                failed += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  [{}/{}] {} - Initialization failed: {}", idx + 1, targets.len(), target.pdb_id, e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("  [{}/{}] {} - Parse failed: {}", idx + 1, targets.len(), target.pdb_id, e);
                failed += 1;
            }
        }
    }

    let overall_elapsed = overall_start.elapsed().as_secs_f64();

    println!();
    println!("{}", "=".repeat(80));
    println!("  AMBER BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));

    if !correlations.is_empty() {
        let mean_rho = correlations.iter().sum::<f64>() / correlations.len() as f64;
        let above_05 = correlations.iter().filter(|&&c| c >= 0.5).count();
        let above_03 = correlations.iter().filter(|&&c| c >= 0.3).count();

        println!();
        println!("  Mean Pearson rho:  {:.4}", mean_rho);
        println!("  >= 0.5:            {}/{} ({:.1}%)", above_05, correlations.len(),
                 100.0 * above_05 as f64 / correlations.len() as f64);
        println!("  >= 0.3:            {}/{} ({:.1}%)", above_03, correlations.len(),
                 100.0 * above_03 as f64 / correlations.len() as f64);
        println!();
        println!("  Successful:        {}", successful);
        println!("  Failed:            {}", failed);
        println!("  Total time:        {:.2}s", overall_elapsed);
        println!("  Avg time/protein:  {:.2}s", total_time / successful as f64);
    } else {
        println!();
        println!("  No successful simulations!");
        println!("  Failed: {}", failed);
    }

    println!();

    Ok(())
}

/// Parse PDB file and extract all atoms
fn parse_pdb_atoms(path: &PathBuf, target_chain: Option<&str>) -> Result<Vec<prism_physics::amber_ff14sb::PdbAtom>> {
    use prism_physics::amber_ff14sb::PdbAtom;

    let content = fs::read_to_string(path)?;
    let mut atoms = Vec::new();
    let mut atom_index = 0usize;

    for line in content.lines() {
        if !line.starts_with("ATOM") { continue; }

        let chain_id = line.get(21..22).unwrap_or(" ").chars().next().unwrap_or(' ');
        if let Some(target) = target_chain {
            if chain_id.to_string() != target { continue; }
        }

        // Skip alternate conformations except A
        let alt_loc = line.get(16..17).unwrap_or(" ");
        if alt_loc != " " && alt_loc != "A" { continue; }

        let name = line.get(12..16).unwrap_or("    ").to_string();
        let residue_name = line.get(17..20).unwrap_or("UNK").trim().to_string();
        let residue_id: i32 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);
        let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);

        atoms.push(PdbAtom {
            index: atom_index,
            name,
            residue_name,
            residue_id,
            chain_id,
            x, y, z,
        });
        atom_index += 1;
    }

    Ok(atoms)
}

/// Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() { return 0.0; }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}
