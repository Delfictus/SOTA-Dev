//! PRISM-Cryptic: Official Cryptic Binding Site Detection Module
//!
//! The PRISM-Cryptic module uses GPU-accelerated Langevin molecular dynamics
//! with residue-based Jaccard pocket tracking to identify cryptic binding sites
//! in protein structures.
//!
//! # Scientific Background
//!
//! Cryptic binding sites are pockets that are not visible in static crystal structures
//! but become accessible through conformational dynamics. PRISM-Cryptic identifies
//! these sites by:
//!
//! 1. Running Langevin dynamics simulations with AMBER ff14SB force field
//! 2. Tracking pocket volumes and SASA across the trajectory
//! 3. Using Jaccard coefficient for pocket identity matching across frames
//! 4. Classifying pockets based on coefficient of variation (CV) and open frequency
//!
//! # Classification Criteria (Pre-Set, Literature-Derived)
//!
//! - CV(SASA) > 0.20 - Significant conformational variability
//! - Open frequency: 5% - 90% - Neither always open nor always closed
//! - Minimum volume: 100 Å³ - Large enough for drug binding
//!
//! # Usage
//!
//! ```bash
//! # Step 1: Preprocess PDB with prism-prep
//! prism-prep input.pdb topology.json --use-amber --mode cryptic --strict
//!
//! # Step 2: Run cryptic site detection
//! prism-cryptic detect --topology topology.json --output-dir results/
//!
//! # Accelerated mode (4 replicas, 4fs timestep - requires HMR topology)
//! prism-prep input.pdb topology.json --use-amber --mode cryptic --hmr --strict
//! prism-cryptic detect --topology topology.json --output-dir results/ --accelerated
//! ```
//!
//! # Validation Targets
//!
//! The module has been validated against known cryptic sites:
//! - TEM-1 β-lactamase (1BTL) - Ω-loop pocket
//! - p38 MAP kinase (1A9U) - DFG-out pocket
//! - IL-2 (1M47) - Composite groove
//! - BCL-xL (1MAZ) - BH3 groove extension
//! - PDK1 (1H1W) - PIF pocket

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[cfg(feature = "cryptic-gpu")]
use prism_validation::cryptic_site_pilot::{
    MdCrypticConfig, MdCrypticPipeline, MdCrypticResult,
};

const VERSION: &str = "1.0.0";
const BANNER: &str = r#"
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PRISM-CRYPTIC v1.0.0                                 ║
║           Official Cryptic Binding Site Detection Module                      ║
║                                                                              ║
║     GPU-Accelerated Langevin MD + Jaccard Pocket Tracking                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"#;

#[derive(Parser, Debug)]
#[command(name = "prism-cryptic")]
#[command(author = "PRISM4D Team")]
#[command(version = VERSION)]
#[command(about = "Cryptic binding site detection using GPU-accelerated molecular dynamics")]
#[command(long_about = r#"
PRISM-Cryptic: Official Cryptic Binding Site Detection Module

Identifies cryptic binding sites in protein structures using Langevin molecular
dynamics with the AMBER ff14SB force field. Pockets are tracked across the
trajectory using residue-based Jaccard matching and classified based on
conformational variability.

WORKFLOW:
  1. Preprocess PDB with prism-prep (required)
  2. Run cryptic detection with prism-cryptic detect
  3. Review results in output directory

PREREQUISITE:
  prism-prep input.pdb topology.json --use-amber --mode cryptic --strict
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Detect cryptic binding sites in a protein structure
    Detect(DetectArgs),

    /// Run batch detection on multiple structures
    Batch(BatchArgs),

    /// Validate a topology file for cryptic detection
    Validate(ValidateArgs),

    /// Show classification thresholds and methodology
    Info,

    /// Check system requirements and GPU status
    Check,
}

#[derive(Parser, Debug)]
struct DetectArgs {
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

    /// Quick test mode (50 frames, ~1 ns)
    #[arg(long)]
    quick: bool,

    /// Production mode (400 frames, ~8 ns)
    #[arg(long)]
    production: bool,

    /// Accelerated mode (4 replicas, 4fs timestep - requires HMR topology)
    #[arg(long)]
    accelerated: bool,

    /// Number of parallel replicas (overrides mode default)
    #[arg(long)]
    replicas: Option<usize>,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    quiet: bool,

    /// Verbose mode (detailed output)
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct BatchArgs {
    /// Manifest file with topology paths (one per line)
    #[arg(short, long)]
    manifest: PathBuf,

    /// Output directory for all results
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Accelerated mode for all structures
    #[arg(long)]
    accelerated: bool,

    /// Continue on error (don't stop on failed structures)
    #[arg(long)]
    continue_on_error: bool,

    /// Verbose mode
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct ValidateArgs {
    /// Topology JSON file to validate
    #[arg(short, long)]
    topology: PathBuf,

    /// Verbose validation output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "cryptic-gpu")]
fn cmd_detect(args: DetectArgs) -> Result<()> {
    if !args.quiet {
        println!("{}", BANNER);
    }

    // Validate input
    if !args.topology.exists() {
        anyhow::bail!("Topology file not found: {}\n\nHint: First run:\n  prism-prep input.pdb {} --use-amber --mode cryptic --strict",
            args.topology.display(),
            args.topology.display());
    }

    // Build configuration
    let mut config = if args.quick {
        if !args.quiet {
            println!("  Mode:         QUICK TEST (50 frames, ~1 ns)");
        }
        MdCrypticConfig::quick_test()
    } else if args.production {
        if !args.quiet {
            println!("  Mode:         PRODUCTION (400 frames, ~8 ns)");
        }
        MdCrypticConfig::production()
    } else if args.accelerated {
        if !args.quiet {
            println!("  Mode:         ACCELERATED (4 replicas, 4fs timestep)");
            println!("                Requires topology from: prism-prep --hmr");
        }
        MdCrypticConfig::accelerated()
    } else {
        if !args.quiet {
            println!("  Mode:         STANDARD ({} frames)", args.frames);
        }
        let mut config = MdCrypticConfig::default();
        config.n_frames = args.frames;
        config.temperature_k = args.temperature;
        config
    };

    // Override replicas if specified
    if let Some(n) = args.replicas {
        config.n_replicas = n;
    }

    if !args.quiet {
        println!("  Input:        {}", args.topology.display());
        println!("  Output:       {}", args.output_dir.display());
        println!("  Frames:       {}", config.n_frames);
        println!("  Sim Time:     {:.1} ns", config.total_time_ps() / 1000.0);
        println!("  Temperature:  {:.1} K", config.temperature_k);
        println!("  Timestep:     {:.1} fs", config.dt_fs);
        println!("  Replicas:     {}", config.n_replicas);
        println!();
        println!("  CLASSIFICATION THRESHOLDS (Literature-Derived, Pre-Set):");
        println!("    CV threshold:       {:.2} (CryptoSite, PocketMiner)", config.cv_threshold);
        println!("    Open frequency:     {:.0}% - {:.0}%",
            config.min_open_frequency * 100.0, config.max_open_frequency * 100.0);
        println!("    Jaccard threshold:  {:.0}%", config.jaccard_threshold * 100.0);
        println!("    Min volume:         {:.0} Å³", config.min_pocket_volume);
        println!();
    }

    // Create output directory
    fs::create_dir_all(&args.output_dir)?;

    // Initialize pipeline
    if !args.quiet {
        println!("{}", "─".repeat(80));
        println!("  INITIALIZING GPU MD ENGINE");
        println!("{}", "─".repeat(80));
        println!();
    }

    let pipeline = MdCrypticPipeline::new(config.clone())?;

    // Run analysis
    if !args.quiet {
        println!("{}", "─".repeat(80));
        println!("  RUNNING LANGEVIN DYNAMICS SIMULATION");
        println!("{}", "─".repeat(80));
        println!();
    }

    let start = Instant::now();
    let result = pipeline.run(args.topology.to_str().unwrap())?;
    let elapsed = start.elapsed();

    // Print results
    print_results(&result, &config, elapsed.as_secs_f64(), args.verbose, args.quiet)?;

    // Write outputs
    write_outputs(&result, &args.output_dir, &config)?;

    if !args.quiet {
        println!();
        println!("{}", "═".repeat(80));
        println!("  ANALYSIS COMPLETE");
        println!("{}", "═".repeat(80));
        println!();
    }

    Ok(())
}

#[cfg(feature = "cryptic-gpu")]
fn print_results(
    result: &MdCrypticResult,
    config: &MdCrypticConfig,
    elapsed_secs: f64,
    verbose: bool,
    quiet: bool,
) -> Result<()> {
    if quiet {
        // Minimal output for scripting
        println!("{},{},{}", result.pdb_id, result.cryptic_sites.len(), elapsed_secs);
        return Ok(());
    }

    println!();
    println!("{}", "─".repeat(80));
    println!("  ANALYSIS RESULTS");
    println!("{}", "─".repeat(80));
    println!();
    println!("  PDB ID:           {}", result.pdb_id);
    println!("  Simulation Time:  {:.1} ns", result.total_time_ps / 1000.0);
    println!("  Frames Analyzed:  {}", result.n_frames);
    println!("  Wall Time:        {:.1} s ({:.1} min)", elapsed_secs, elapsed_secs / 60.0);
    println!();
    println!("  POCKET STATISTICS:");
    println!("    Total pockets tracked:  {}", result.all_pockets.len());
    println!("    CV range:               {:.3} - {:.3}",
        result.diagnostics.cv_min, result.diagnostics.cv_max);
    println!("    CV mean:                {:.3}", result.diagnostics.cv_mean);
    println!("    Freq range:             {:.1}% - {:.1}%",
        result.diagnostics.freq_min * 100.0, result.diagnostics.freq_max * 100.0);
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────────┐");
    println!("  │  CRYPTIC SITES DETECTED: {:>3}                                    │",
        result.cryptic_sites.len());
    println!("  └─────────────────────────────────────────────────────────────────┘");
    println!();

    if result.cryptic_sites.is_empty() {
        println!("    No cryptic sites detected.");
        println!();
        println!("    DIAGNOSTIC NOTES:");
        println!("    Max CV observed: {:.3} (threshold: {:.3})",
            result.diagnostics.cv_max, config.cv_threshold);

        if result.diagnostics.cv_max > 0.0 && result.diagnostics.cv_max < config.cv_threshold {
            println!("    Pockets show variance but below threshold.");
            println!("    Consider: longer simulation or elevated temperature.");
        } else if result.diagnostics.cv_max == 0.0 {
            println!("    No volume variance detected.");
            println!("    Check: is MD sampling properly? RMSD of target region?");
        }
    } else {
        println!("  {:<6} {:<12} {:<10} {:<10} {:<10} {:<12}",
            "Rank", "Site ID", "CV", "Open %", "Volume", "Residues");
        println!("  {}", "─".repeat(66));

        for site in &result.cryptic_sites {
            println!("  {:<6} {:<12} {:<10.3} {:<10.1} {:<10.0} {:<12}",
                site.rank,
                site.site_id,
                site.cv_volume,
                site.open_frequency * 100.0,
                site.mean_volume,
                site.residues.len());

            if verbose {
                println!("         Residues: {:?}", &site.residues[..site.residues.len().min(10)]);
                if let Some(ref drug) = site.druggability {
                    println!("         Druggability: {:.2} ({})",
                        drug.score, drug.classification.name());
                }
                println!();
            }
        }
    }

    // Print validation target hints
    let pdb_upper = result.pdb_id.to_uppercase();
    if pdb_upper.contains("1BTL") {
        println!();
        println!("  VALIDATION TARGET: TEM-1 β-lactamase (1BTL)");
        println!("    Expected cryptic site: Ω-loop region (residues 214-220, 244-250)");

        let omega_residues: Vec<i32> = (214..=220).chain(244..=250).collect();
        for site in &result.cryptic_sites {
            let overlap: usize = site.residues.iter()
                .filter(|r| omega_residues.contains(r))
                .count();
            if overlap >= 3 {
                println!("    ✓ DETECTED (Site {} contains {} Ω-loop residues)", site.site_id, overlap);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cryptic-gpu")]
fn write_outputs(result: &MdCrypticResult, output_dir: &PathBuf, config: &MdCrypticConfig) -> Result<()> {
    // Write JSON result
    let json_path = output_dir.join(format!("{}_cryptic_result.json", result.pdb_id));
    let json_content = serde_json::to_string_pretty(&result)?;
    fs::write(&json_path, json_content)?;
    println!("\n  Output files:");
    println!("    Result JSON:  {}", json_path.display());

    // Write summary CSV
    let csv_path = output_dir.join(format!("{}_cryptic_summary.csv", result.pdb_id));
    let mut csv_content = String::from(
        "pocket_id,cv_volume,cv_sasa,open_frequency,mean_volume,mean_sasa,n_residues,is_cryptic\n"
    );

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
    fs::write(&csv_path, csv_content)?;
    println!("    Summary CSV:  {}", csv_path.display());

    // Write cryptic sites list
    if !result.cryptic_sites.is_empty() {
        let sites_path = output_dir.join(format!("{}_cryptic_sites.txt", result.pdb_id));
        let mut sites_content = String::new();
        sites_content.push_str(&format!("# Cryptic Sites for {}\n", result.pdb_id));
        sites_content.push_str(&format!("# Simulation: {:.1} ns, {} frames\n",
            result.total_time_ps / 1000.0, result.n_frames));
        sites_content.push_str("#\n");

        for site in &result.cryptic_sites {
            sites_content.push_str(&format!(
                "Site {}: CV={:.3}, Open={:.1}%, Residues={:?}\n",
                site.rank, site.cv_volume, site.open_frequency * 100.0, site.residues
            ));
        }
        fs::write(&sites_path, sites_content)?;
        println!("    Sites list:   {}", sites_path.display());
    }

    Ok(())
}

#[cfg(feature = "cryptic-gpu")]
fn cmd_batch(args: BatchArgs) -> Result<()> {
    println!("{}", BANNER);

    if !args.manifest.exists() {
        anyhow::bail!("Manifest file not found: {}", args.manifest.display());
    }

    // Read manifest
    let manifest_content = fs::read_to_string(&args.manifest)?;
    let topologies: Vec<PathBuf> = manifest_content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .map(|l| PathBuf::from(l.trim()))
        .collect();

    println!("  Batch Processing: {} structures", topologies.len());
    println!("  Output Directory: {}", args.output_dir.display());
    println!();

    fs::create_dir_all(&args.output_dir)?;

    let mut success = 0;
    let mut failed = 0;
    let mut failed_list = Vec::new();

    for (i, topology) in topologies.iter().enumerate() {
        println!("[{}/{}] Processing: {}", i + 1, topologies.len(), topology.display());

        let output_subdir = args.output_dir.join(
            topology.file_stem().unwrap_or_default().to_str().unwrap_or("unknown")
        );

        let detect_args = DetectArgs {
            topology: topology.clone(),
            output_dir: output_subdir,
            frames: 200,
            temperature: 310.0,
            quick: false,
            production: false,
            accelerated: args.accelerated,
            replicas: None,
            quiet: !args.verbose,
            verbose: args.verbose,
        };

        match cmd_detect(detect_args) {
            Ok(_) => {
                success += 1;
                println!("  ✓ Success");
            }
            Err(e) => {
                failed += 1;
                failed_list.push((topology.clone(), e.to_string()));
                println!("  ✗ Failed: {}", e);

                if !args.continue_on_error {
                    break;
                }
            }
        }
        println!();
    }

    // Print summary
    println!("{}", "═".repeat(80));
    println!("  BATCH SUMMARY");
    println!("{}", "═".repeat(80));
    println!("  Total:   {}", topologies.len());
    println!("  Success: {}", success);
    println!("  Failed:  {}", failed);

    if !failed_list.is_empty() {
        println!("\n  Failed structures:");
        for (path, err) in &failed_list {
            println!("    - {}: {}", path.display(), err);
        }
    }

    Ok(())
}

#[cfg(feature = "cryptic-gpu")]
fn cmd_validate(args: ValidateArgs) -> Result<()> {
    println!("{}", BANNER);
    println!("  Validating topology: {}", args.topology.display());
    println!();

    if !args.topology.exists() {
        anyhow::bail!("Topology file not found: {}", args.topology.display());
    }

    // Load and validate topology
    let content = fs::read_to_string(&args.topology)
        .context("Failed to read topology file")?;

    let topology: serde_json::Value = serde_json::from_str(&content)
        .context("Failed to parse topology JSON")?;

    let n_atoms = topology.get("n_atoms").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_residues = topology.get("n_residues").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_bonds = topology.get("bonds").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    let has_charges = topology.get("charges").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
    let has_masses = topology.get("masses").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
    let has_gb_radii = topology.get("gb_radii").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false);

    println!("  TOPOLOGY VALIDATION");
    println!("  {}", "─".repeat(60));
    println!("  Atoms:      {:>8}  {}", n_atoms, if n_atoms > 0 { "✓" } else { "✗" });
    println!("  Residues:   {:>8}  {}", n_residues, if n_residues > 0 { "✓" } else { "✗" });
    println!("  Bonds:      {:>8}  {}", n_bonds, if n_bonds > 0 { "✓" } else { "✗" });
    println!("  Charges:    {:>8}  {}", if has_charges { "present" } else { "missing" }, if has_charges { "✓" } else { "✗" });
    println!("  Masses:     {:>8}  {}", if has_masses { "present" } else { "missing" }, if has_masses { "✓" } else { "✗" });
    println!("  GB Radii:   {:>8}  {}", if has_gb_radii { "present" } else { "missing" }, if has_gb_radii { "✓ (implicit solvent)" } else { "✗" });
    println!();

    // Check for HMR (hydrogen mass repartitioning)
    if let Some(masses) = topology.get("masses").and_then(|v| v.as_array()) {
        let heavy_h = masses.iter()
            .filter_map(|m| m.as_f64())
            .filter(|&m| m > 1.5 && m < 4.0)
            .count();
        if heavy_h > 0 {
            println!("  HMR:        detected  ✓ (4 fs timestep compatible)");
        }
    }

    let valid = n_atoms > 0 && n_bonds > 0 && has_charges && has_masses && has_gb_radii;

    println!();
    if valid {
        println!("  ✅ TOPOLOGY VALID FOR CRYPTIC DETECTION");
    } else {
        println!("  ❌ TOPOLOGY INCOMPLETE");
        println!();
        println!("  Run prism-prep to generate a complete topology:");
        println!("    prism-prep input.pdb {} --use-amber --mode cryptic --strict",
            args.topology.display());
    }
    println!();

    Ok(())
}

fn cmd_info() {
    println!("{}", BANNER);
    println!(r#"
  CLASSIFICATION METHODOLOGY
  ══════════════════════════════════════════════════════════════════════════════

  PRISM-Cryptic identifies cryptic binding sites using three criteria:

  1. COEFFICIENT OF VARIATION (CV) > 0.20
     Measures pocket volume/SASA variability across the trajectory.
     High CV indicates significant conformational changes.
     Reference: CryptoSite (Cimermancic et al., 2016)

  2. OPEN FREQUENCY: 5% - 90%
     Fraction of frames where the pocket is "open" (volume > threshold).
     - < 5%:  Too rarely open → likely noise
     - > 90%: Too often open → constitutive, not cryptic
     Reference: PocketMiner (Meller et al., 2023)

  3. MINIMUM VOLUME: 100 Å³
     Ensures pocket is large enough for drug-like molecules.
     Reference: Druggability literature (Schmidtke & Barril, 2010)

  POCKET TRACKING
  ══════════════════════════════════════════════════════════════════════════════

  Pockets are tracked across frames using residue-based Jaccard matching:

    Jaccard(P₁, P₂) = |R₁ ∩ R₂| / |R₁ ∪ R₂|

  Where R₁, R₂ are the sets of residues defining each pocket.
  Threshold: 30% overlap for pocket identity.

  SIMULATION PARAMETERS
  ══════════════════════════════════════════════════════════════════════════════

  Integrator:     BAOAB Langevin (Leimkuhler & Matthews, 2013)
  Force Field:    AMBER ff14SB
  Solvent:        Generalized Born implicit (GBn2)
  Temperature:    310 K (physiological)
  Friction:       200 ps⁻¹ (aggressive thermostat)
  Timestep:       2 fs (standard) or 4 fs (with HMR)

  Standard Mode:    1 replica, 200 frames, ~4 ns
  Production Mode:  1 replica, 400 frames, ~8 ns
  Accelerated Mode: 4 replicas, 200 frames each, ~4 ns total, 4 fs timestep

  SCIENTIFIC INTEGRITY
  ══════════════════════════════════════════════════════════════════════════════

  All thresholds are FIXED based on literature and validation benchmarks.
  DO NOT adjust thresholds post-hoc to match expected results.
  If detection fails, investigate the physics, not the thresholds.

"#);
}

fn cmd_check() -> Result<()> {
    println!("{}", BANNER);
    println!("  SYSTEM CHECK");
    println!("  {}", "═".repeat(60));
    println!();

    // Check CUDA
    println!("  GPU/CUDA Status:");

    #[cfg(feature = "cryptic-gpu")]
    {
        use cudarc::driver::CudaContext;

        match CudaContext::new(0) {
            Ok(_device) => {
                println!("    CUDA Device:  ✓ Available");
                println!("    Device Name:  GPU 0 (primary)");
                println!("    Status:       Ready for computation");
            }
            Err(e) => {
                println!("    CUDA Device:  ✗ Not available");
                println!("    Error:        {}", e);
                println!();
                println!("    PRISM-Cryptic requires an NVIDIA GPU with CUDA support.");
                println!("    Please ensure CUDA drivers are installed and a GPU is available.");
            }
        }
    }

    #[cfg(not(feature = "cryptic-gpu"))]
    {
        println!("    CUDA Support: ✗ Not compiled with GPU support");
        println!();
        println!("    Rebuild with: cargo build --release --features cryptic-gpu");
    }

    println!();
    println!("  Dependencies:");
    println!("    prism-prep:   Check with: prism-prep --check-deps");
    println!();
    println!("  Memory Requirements:");
    println!("    Small (<500 atoms):   ~512 MB VRAM");
    println!("    Medium (<5000 atoms): ~2 GB VRAM");
    println!("    Large (<20000 atoms): ~8 GB VRAM");
    println!();

    Ok(())
}

#[cfg(feature = "cryptic-gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Detect(args) => cmd_detect(args),
        Commands::Batch(args) => cmd_batch(args),
        Commands::Validate(args) => cmd_validate(args),
        Commands::Info => { cmd_info(); Ok(()) },
        Commands::Check => cmd_check(),
    }
}

#[cfg(not(feature = "cryptic-gpu"))]
fn main() {
    eprintln!("Error: PRISM-Cryptic requires the 'cryptic-gpu' feature.");
    eprintln!();
    eprintln!("Rebuild with:");
    eprintln!("  cargo build --release -p prism-validation --features cryptic-gpu --bin prism-cryptic");
    eprintln!();
    std::process::exit(1);
}
