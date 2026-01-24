//! Cryptic Site Pilot CLI - Production-Ready Pharmaceutical Pipeline
//!
//! Full-featured cryptic binding site detection with executive reporting
//! and all deliverables required for pharmaceutical pilot studies.
//!
//! # Outputs
//!
//! - Multi-MODEL PDB trajectory (≥100 frames, Kabsch-aligned)
//! - Per-site open conformation PDBs (top 5 per site)
//! - RMSF CSV with flexibility classification
//! - Volume time series CSV
//! - Contact residue lists for docking
//! - Executive HTML report
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p prism-validation --bin cryptic-pilot -- \
//!     --pdb input.pdb --output-dir results/
//! ```
//!
//! # Validation Targets
//!
//! Known cryptic sites that must be detected (≥4/5):
//! - TEM-1 β-lactamase (1BTL) - Omega loop pocket
//! - p38 MAP kinase (1A9U) - DFG-out pocket
//! - IL-2 (1M47) - Composite groove
//! - BCL-xL (1MAZ) - BH3 groove extension
//! - PDK1 (1H1W) - PIF pocket

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use prism_validation::cryptic_site_pilot::{
    CrypticPilotConfig,
    CrypticPilotPipeline,
};

#[derive(Parser, Debug)]
#[command(name = "cryptic-pilot")]
#[command(about = "Production-ready cryptic binding site detection for pharmaceutical pilots")]
#[command(version)]
struct Args {
    /// Input topology JSON file (from prism-prep) - RECOMMENDED
    #[arg(short, long, conflicts_with = "pdb")]
    topology: Option<PathBuf>,

    /// Input raw PDB file (NOT RECOMMENDED - use prism-prep first)
    #[arg(short, long, conflicts_with = "topology")]
    pdb: Option<PathBuf>,

    /// Output directory for results
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Number of conformations to generate
    #[arg(long, default_value = "100")]
    frames: usize,

    /// Temperature in Kelvin
    #[arg(long, default_value = "310.0")]
    temperature: f64,

    /// Minimum pocket volume (Å³)
    #[arg(long, default_value = "100.0")]
    min_volume: f64,

    /// Maximum pocket volume (Å³)
    #[arg(long, default_value = "2000.0")]
    max_volume: f64,

    /// Volume CV threshold for cryptic classification (0.0-1.0)
    #[arg(long, default_value = "0.20")]
    cv_threshold: f64,

    /// Number of representative structures per site
    #[arg(long, default_value = "5")]
    n_representatives: usize,

    /// Use production preset (200 frames, strict thresholds)
    #[arg(long)]
    production: bool,

    /// Use quick preset for testing (50 frames)
    #[arg(long)]
    quick: bool,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!();
    println!("{}", "=".repeat(80));
    println!("  PRISM-4D CRYPTIC SITE PILOT");
    println!("  Production-Ready Pharmaceutical Analysis Pipeline");
    println!("{}", "=".repeat(80));
    println!();

    // Determine input type and validate
    let (input_path, use_topology) = if let Some(ref topology) = args.topology {
        if !topology.exists() {
            anyhow::bail!("Input topology file not found: {}", topology.display());
        }
        println!("  Input Type:   TOPOLOGY (prism-prep sanitized) ✓");
        (topology.clone(), true)
    } else if let Some(ref pdb) = args.pdb {
        if !pdb.exists() {
            anyhow::bail!("Input PDB file not found: {}", pdb.display());
        }
        println!("  Input Type:   RAW PDB (⚠ not recommended - use prism-prep first)");
        (pdb.clone(), false)
    } else {
        anyhow::bail!("Either --topology or --pdb must be specified");
    };

    // Build configuration
    let config = if args.production {
        println!("  Mode:         PRODUCTION (200 frames, strict)");
        CrypticPilotConfig::production()
    } else if args.quick {
        println!("  Mode:         QUICK (50 frames, testing)");
        CrypticPilotConfig::quick()
    } else {
        println!("  Mode:         CUSTOM");
        let mut config = CrypticPilotConfig::default();
        config.n_frames = args.frames.max(50); // Minimum 50 frames
        config.temperature_k = args.temperature as f32;
        config.min_pocket_volume = args.min_volume;
        config.max_pocket_volume = args.max_volume;
        config.cryptic_volume_cv_threshold = args.cv_threshold;
        config.n_representative_structures = args.n_representatives;
        config.seed = args.seed;
        config
    };

    println!("  Input:        {}", input_path.display());
    println!("  Output:       {}", args.output_dir.display());
    println!("  Frames:       {}", config.n_frames);
    println!("  Temperature:  {:.1} K", config.temperature_k);
    println!("  CV Threshold: {:.0}%", config.cryptic_volume_cv_threshold * 100.0);
    println!("  Volume:       {:.0} - {:.0} Å³", config.min_pocket_volume, config.max_pocket_volume);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Initialize pipeline
    let pipeline = CrypticPilotPipeline::new(config.clone())?;

    // Run analysis - use topology or raw PDB based on input type
    let start = Instant::now();
    let result = if use_topology {
        pipeline.analyze_topology(input_path.to_str().unwrap())?
    } else {
        pipeline.analyze_raw_pdb(input_path.to_str().unwrap())?
    };
    let elapsed = start.elapsed();

    // Print results
    println!("{}", "-".repeat(80));
    println!("  ANALYSIS RESULTS");
    println!("{}", "-".repeat(80));
    println!();
    println!("  PDB ID:           {}", result.pdb_id);
    println!("  Input Hash:       {}", result.input_hash);
    println!("  Residues:         {}", result.n_residues);
    println!("  Frames:           {}", result.n_frames);
    println!("  Mean RMSD:        {:.2} Å", result.mean_rmsd);
    println!("  RMSD Std:         {:.2} Å", result.rmsd_std);
    println!();
    println!("  Total Pockets:    {}", result.all_pockets.len());
    println!("  Cryptic Sites:    {}", result.cryptic_sites.len());
    println!("  Computation Time: {:.2} s", elapsed.as_secs_f64());
    println!();

    if result.cryptic_sites.is_empty() {
        println!("  ⚠ No cryptic binding sites detected.");
        println!();
        println!("  Possible reasons:");
        println!("  - Structure is rigid with no cryptic pockets");
        println!("  - Need more frames for statistical significance");
        println!("  - CV threshold may be too strict for this structure");
    } else {
        println!("  DETECTED CRYPTIC SITES:");
        println!();
        println!("  {:<6} {:<12} {:<10} {:<10} {:<10} {:<12}",
                 "Rank", "Site ID", "Volume", "Open %", "Drug Score", "Class");
        println!("  {}", "-".repeat(66));

        for site in &result.cryptic_sites {
            println!("  {:<6} {:<12} {:<10.0} {:<10.0} {:<10.2} {:<12}",
                     site.rank,
                     site.site_id,
                     site.volume_series.stats.mean_volume,
                     site.volume_series.stats.open_frequency * 100.0,
                     site.druggability.score,
                     site.druggability.classification.name());

            if args.verbose {
                println!("         Residues:  {} residues", site.residues.len());
                println!("         Centroid:  ({:.1}, {:.1}, {:.1}) Å",
                         site.centroid[0], site.centroid[1], site.centroid[2]);
                println!("         Breathing: {:.0} Å³ amplitude",
                         site.volume_series.stats.breathing_amplitude);
                println!("         Contacts:  {} residues for docking", site.contacts.len());
                println!("         Rep Frame: {}", site.representative_frame);
                println!("         Affinity:  {:.1} - {:.1} kcal/mol (estimated)",
                         site.druggability.estimated_affinity_range.0,
                         site.druggability.estimated_affinity_range.1);
                println!();
            }
        }
    }

    println!();

    // Write all outputs
    println!("{}", "-".repeat(80));
    println!("  GENERATING OUTPUTS");
    println!("{}", "-".repeat(80));
    println!();

    result.write_all_outputs(&args.output_dir)?;

    // List generated files
    println!();
    println!("  Generated files:");
    let files = std::fs::read_dir(&args.output_dir)?;
    for entry in files {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Ok(metadata) = entry.metadata() {
                let size_kb = metadata.len() as f64 / 1024.0;
                println!("    - {} ({:.1} KB)", path.file_name().unwrap().to_string_lossy(), size_kb);
            }
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("  ANALYSIS COMPLETE");
    println!("{}", "=".repeat(80));
    println!();
    println!("  Summary:");
    println!("  - {} cryptic binding sites detected", result.cryptic_sites.len());
    if !result.cryptic_sites.is_empty() {
        let top_druggable = &result.cryptic_sites[0];
        println!("  - Top druggable site: {} (score={:.2}, {})",
                 top_druggable.site_id,
                 top_druggable.druggability.score,
                 top_druggable.druggability.classification.name());
    }
    println!("  - Results in: {}", args.output_dir.display());
    println!();

    // Validation hint for known targets
    let pdb_id_upper = result.pdb_id.to_uppercase();
    if pdb_id_upper.contains("1BTL") || pdb_id_upper.contains("1A9U") ||
       pdb_id_upper.contains("1M47") || pdb_id_upper.contains("1MAZ") ||
       pdb_id_upper.contains("1H1W") {
        println!("  ★ VALIDATION TARGET DETECTED");
        println!("    This is one of the 5 known cryptic site validation targets.");
        println!("    Please verify the detected sites match published locations.");
        println!();
    }

    Ok(())
}
