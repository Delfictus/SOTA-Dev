//! MD-Based Cryptic Site Detection CLI
//!
//! Uses Langevin molecular dynamics for conformational sampling with
//! residue-based Jaccard matching for pocket tracking.
//!
//! # Scientific Integrity
//!
//! All thresholds are literature-derived and set BEFORE validation.
//! DO NOT adjust thresholds to match expected results.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p prism-validation --bin md-cryptic-pilot \
//!     --features cryptic-gpu -- \
//!     --topology data/prepared/1BTL_topology.json \
//!     --output-dir results/1btl_md/
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

#[cfg(feature = "cryptic-gpu")]
use prism_validation::cryptic_site_pilot::{
    MdCrypticConfig, MdCrypticPipeline,
};

#[derive(Parser, Debug)]
#[command(name = "md-cryptic-pilot")]
#[command(about = "MD-based cryptic binding site detection with Langevin dynamics")]
#[command(version)]
struct Args {
    /// Input topology JSON file (from prism-prep)
    #[arg(short, long)]
    topology: PathBuf,

    /// Output directory for results
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Number of production frames (default: 200 = 4 ns)
    #[arg(long, default_value = "200")]
    frames: usize,

    /// Temperature in Kelvin (default: 310 K physiological)
    #[arg(long, default_value = "310.0")]
    temperature: f32,

    /// Use quick test mode (50 frames, 1 ns)
    #[arg(long)]
    quick: bool,

    /// Use production mode (400 frames, 8 ns)
    #[arg(long)]
    production: bool,

    /// Use accelerated mode (4 replicas, 4fs timestep - requires HMR topology)
    #[arg(long)]
    accelerated: bool,

    /// Number of parallel replicas (overrides mode default)
    #[arg(long)]
    replicas: Option<usize>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "cryptic-gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    println!();
    println!("{}", "=".repeat(80));
    println!("  PRISM-4D MD-BASED CRYPTIC SITE DETECTION");
    println!("  Langevin Dynamics + Jaccard Pocket Tracking");
    println!("{}", "=".repeat(80));
    println!();

    // Validate input
    if !args.topology.exists() {
        anyhow::bail!("Topology file not found: {}", args.topology.display());
    }

    // Build configuration
    let mut config = if args.quick {
        println!("  Mode:         QUICK TEST (50 frames, ~1 ns)");
        MdCrypticConfig::quick_test()
    } else if args.production {
        println!("  Mode:         PRODUCTION (400 frames, ~8 ns)");
        MdCrypticConfig::production()
    } else if args.accelerated {
        println!("  Mode:         ACCELERATED (4 replicas, 4fs timestep)");
        println!("                ⚠️  Requires topology from prism-prep --hmr");
        MdCrypticConfig::accelerated()
    } else {
        println!("  Mode:         STANDARD ({} frames)", args.frames);
        let mut config = MdCrypticConfig::default();
        config.n_frames = args.frames;
        config.temperature_k = args.temperature;
        config
    };

    // Override replicas if specified
    if let Some(n) = args.replicas {
        config.n_replicas = n;
        println!("  Replicas:     {} (override)", n);
    }

    println!("  Input:        {}", args.topology.display());
    println!("  Output:       {}", args.output_dir.display());
    println!("  Frames:       {}", config.n_frames);
    println!("  Sim Time:     {:.1} ns", config.total_time_ps() / 1000.0);
    println!("  Temperature:  {:.1} K", config.temperature_k);
    println!("  Timestep:     {:.1} fs", config.dt_fs);
    println!("  Replicas:     {}", config.n_replicas);
    println!();
    println!("  CLASSIFICATION THRESHOLDS (Literature-Derived, Pre-Set):");
    println!("  - CV threshold:       {:.2} (CryptoSite, PocketMiner)", config.cv_threshold);
    println!("  - Open frequency:     {:.0}% - {:.0}%",
        config.min_open_frequency * 100.0, config.max_open_frequency * 100.0);
    println!("  - Jaccard threshold:  {:.0}%", config.jaccard_threshold * 100.0);
    println!("  - Min volume:         {:.0} Å³", config.min_pocket_volume);
    println!();
    println!("  ⚠️  DO NOT adjust thresholds post-hoc to match expected results.");
    println!("      If validation fails, investigate physics, not thresholds.");
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Initialize pipeline
    println!("{}", "-".repeat(80));
    println!("  INITIALIZING MD ENGINE");
    println!("{}", "-".repeat(80));
    println!();

    let pipeline = MdCrypticPipeline::new(config.clone())?;

    // Run analysis
    println!("{}", "-".repeat(80));
    println!("  RUNNING MD SIMULATION");
    println!("{}", "-".repeat(80));
    println!();

    let start = Instant::now();
    let result = pipeline.run(args.topology.to_str().unwrap())?;
    let elapsed = start.elapsed();

    // Print results
    println!();
    println!("{}", "-".repeat(80));
    println!("  ANALYSIS RESULTS");
    println!("{}", "-".repeat(80));
    println!();
    println!("  PDB ID:           {}", result.pdb_id);
    println!("  Simulation Time:  {:.1} ns", result.total_time_ps / 1000.0);
    println!("  Frames Analyzed:  {}", result.n_frames);
    println!("  Wall Time:        {:.1} s", elapsed.as_secs_f64());
    println!();
    println!("  POCKET STATISTICS:");
    println!("  - Total pockets tracked:  {}", result.all_pockets.len());
    println!("  - CV range:               {:.3} - {:.3}",
        result.diagnostics.cv_min, result.diagnostics.cv_max);
    println!("  - CV mean:                {:.3}", result.diagnostics.cv_mean);
    println!("  - Freq range:             {:.1}% - {:.1}%",
        result.diagnostics.freq_min * 100.0, result.diagnostics.freq_max * 100.0);
    println!();
    println!("  CRYPTIC SITES DETECTED:   {}", result.cryptic_sites.len());
    println!();

    if result.cryptic_sites.is_empty() {
        println!("  ⚠️  No cryptic sites detected.");
        println!();
        println!("  DIAGNOSTIC NOTES:");
        println!("  - Max CV observed: {:.3} (threshold: {:.3})",
            result.diagnostics.cv_max, config.cv_threshold);

        if result.diagnostics.cv_max > 0.0 && result.diagnostics.cv_max < config.cv_threshold {
            println!("  - Pockets show variance but below threshold.");
            println!("  - Consider: longer simulation or elevated temperature.");
        } else if result.diagnostics.cv_max == 0.0 {
            println!("  - No volume variance detected.");
            println!("  - Check: is MD sampling properly? RMSD of target region?");
        }

        println!();
        println!("  DO NOT adjust cv_threshold to make this pass.");
        println!("  Report actual values and investigate physics.");
    } else {
        println!("  {:<6} {:<12} {:<10} {:<10} {:<10} {:<12}",
            "Rank", "Site ID", "CV", "Open %", "Volume", "Residues");
        println!("  {}", "-".repeat(66));

        for site in &result.cryptic_sites {
            println!("  {:<6} {:<12} {:<10.3} {:<10.1} {:<10.0} {:<12}",
                site.rank,
                site.site_id,
                site.cv_volume,
                site.open_frequency * 100.0,
                site.mean_volume,
                site.residues.len());

            if args.verbose {
                println!("         Residues: {:?}", &site.residues[..site.residues.len().min(10)]);
                if let Some(ref drug) = site.druggability {
                    println!("         Druggability: {:.2} ({})",
                        drug.score, drug.classification.name());
                }
                println!();
            }
        }
    }

    // Write results to JSON
    let json_path = args.output_dir.join(format!("{}_md_cryptic_result.json", result.pdb_id));
    let json_content = serde_json::to_string_pretty(&result)?;
    std::fs::write(&json_path, json_content)?;
    println!();
    println!("  Results written to: {}", json_path.display());

    // Write summary CSV
    let csv_path = args.output_dir.join(format!("{}_md_cryptic_summary.csv", result.pdb_id));
    let mut csv_content = String::from("pocket_id,cv_volume,cv_sasa,open_frequency,mean_volume,mean_sasa,n_residues,is_cryptic\n");
    for pocket in &result.all_pockets {
        let is_cryptic = pocket.cv > config.cv_threshold
            && pocket.open_frequency >= config.min_open_frequency
            && pocket.open_frequency <= config.max_open_frequency;
        csv_content.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:.2},{:.2},{},{}\n",
            pocket.pocket_id,
            pocket.cv,
            pocket.cv_sasa,
            pocket.open_frequency,
            pocket.mean_volume,
            pocket.mean_sasa,
            pocket.defining_residues.len(),
            is_cryptic
        ));
    }
    std::fs::write(&csv_path, csv_content)?;
    println!("  Summary CSV: {}", csv_path.display());

    println!();
    println!("{}", "=".repeat(80));
    println!("  ANALYSIS COMPLETE");
    println!("{}", "=".repeat(80));
    println!();

    // Validation target hint
    let pdb_upper = result.pdb_id.to_uppercase();
    if pdb_upper.contains("1BTL") {
        println!("  ★ VALIDATION TARGET: TEM-1 β-lactamase (1BTL)");
        println!("    Expected cryptic site: Ω-loop region (residues 214-220, 244-250)");
        println!();
        if result.cryptic_sites.is_empty() {
            println!("    RESULT: NOT DETECTED");
            println!("    Action: Investigate MD sampling of Ω-loop region.");
        } else {
            // Check if any detected site contains omega loop residues
            let omega_residues: Vec<i32> = (214..=220).chain(244..=250).collect();
            for site in &result.cryptic_sites {
                let overlap: usize = site.residues.iter()
                    .filter(|r| omega_residues.contains(r))
                    .count();
                if overlap >= 3 {
                    println!("    RESULT: DETECTED (Site {} contains {} Ω-loop residues)",
                        site.site_id, overlap);
                }
            }
        }
        println!();
    }

    Ok(())
}

#[cfg(not(feature = "cryptic-gpu"))]
fn main() {
    eprintln!("Error: This binary requires the 'cryptic-gpu' feature.");
    eprintln!("Rebuild with: cargo build --features cryptic-gpu");
    std::process::exit(1);
}
