//! GNM-Enhanced Cryptic Site Detection CLI
//!
//! Detects hidden (cryptic) binding sites using physics-based GNM flexibility
//! prediction combined with multi-signal analysis.
//!
//! Usage:
//!   cargo run --release -p prism-validation --bin detect-cryptic-sites -- \
//!       --pdb input.pdb [OPTIONS]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use prism_validation::cryptic_sites::{
    CrypticSiteConfig, CrypticSiteDetector, CrypticConfidence,
    parse_pdb_simple,
};

#[derive(Parser, Debug)]
#[command(name = "detect-cryptic-sites")]
#[command(about = "Detect cryptic binding sites using GNM-enhanced analysis")]
struct Args {
    /// Input PDB file
    #[arg(short, long)]
    pdb: PathBuf,

    /// Output JSON file (optional, prints to stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// GNM cutoff distance (Ångströms)
    #[arg(long, default_value = "7.3")]
    gnm_cutoff: f64,

    /// Minimum flexibility z-score threshold
    #[arg(long, default_value = "1.0")]
    flexibility_threshold: f64,

    /// Minimum cluster size
    #[arg(long, default_value = "5")]
    min_cluster: usize,

    /// Maximum cluster size
    #[arg(long, default_value = "30")]
    max_cluster: usize,

    /// Minimum cryptic score to report
    #[arg(long, default_value = "0.3")]
    min_score: f64,

    /// Use experimental B-factors in addition to GNM
    #[arg(long, default_value = "true")]
    use_bfactors: bool,

    /// Chain to analyze (default: all chains)
    #[arg(short, long)]
    chain: Option<char>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Read PDB file
    let content = fs::read_to_string(&args.pdb)?;
    let pdb_name = args.pdb
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Parse atoms
    let mut atoms = parse_pdb_simple(&content);

    // Filter by chain if specified
    if let Some(chain) = args.chain {
        atoms.retain(|a| a.chain_id == chain);
    }

    println!("{}", "=".repeat(80));
    println!("  GNM-ENHANCED CRYPTIC SITE DETECTION");
    println!("{}", "=".repeat(80));
    println!();
    println!("  Input:          {}", args.pdb.display());
    println!("  Atoms:          {}", atoms.len());
    println!("  GNM cutoff:     {} Å", args.gnm_cutoff);
    println!("  Flex threshold: {} σ", args.flexibility_threshold);
    println!("  Use B-factors:  {}", args.use_bfactors);
    println!();

    // Configure detector
    let config = CrypticSiteConfig {
        gnm_cutoff: args.gnm_cutoff,
        flexibility_threshold: args.flexibility_threshold,
        min_cluster_size: args.min_cluster,
        max_cluster_size: args.max_cluster,
        min_score: args.min_score,
        use_bfactors: args.use_bfactors,
        ..Default::default()
    };

    let detector = CrypticSiteDetector::with_config(config);

    // Run detection
    let start = Instant::now();
    let result = detector.detect(&pdb_name, &atoms);
    let elapsed = start.elapsed();

    // Print results
    println!("{}", "-".repeat(80));
    println!("  RESULTS");
    println!("{}", "-".repeat(80));
    println!();
    println!("  Residues analyzed:  {}", result.n_residues);
    println!("  GNM used:           {}", result.gnm_used);
    println!("  B-factors used:     {}", result.bfactors_used);
    println!("  Candidates found:   {}", result.n_candidates);
    println!("  Time:               {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    if result.candidates.is_empty() {
        println!("  No cryptic sites detected above threshold ({:.2})", args.min_score);
    } else {
        println!("  CRYPTIC SITE CANDIDATES:");
        println!("  {:<5} {:<8} {:<10} {:<12} {:<8} {:<8} {:<8}",
                 "Rank", "Conf.", "Score", "Volume(Å³)", "Flex", "Pack", "Hydro");
        println!("  {}", "-".repeat(65));

        for candidate in &result.candidates {
            let conf_str = match candidate.confidence {
                CrypticConfidence::High => "HIGH",
                CrypticConfidence::Medium => "MEDIUM",
                CrypticConfidence::Low => "LOW",
            };

            println!("  {:<5} {:<8} {:<10.3} {:<12.0} {:<8.3} {:<8.3} {:<8.3}",
                     candidate.rank,
                     conf_str,
                     candidate.score,
                     candidate.volume,
                     candidate.flexibility_score,
                     candidate.packing_score,
                     candidate.hydrophobicity_score);

            if args.verbose {
                println!("        Residues: {:?}", candidate.residues);
                println!("        Rationale: {}", candidate.rationale);
                println!("        Centroid: ({:.1}, {:.1}, {:.1})",
                         candidate.centroid[0], candidate.centroid[1], candidate.centroid[2]);
                println!();
            }
        }
    }

    println!();

    // Show distribution of confidence levels
    let high_conf = result.candidates.iter()
        .filter(|c| c.confidence == CrypticConfidence::High)
        .count();
    let med_conf = result.candidates.iter()
        .filter(|c| c.confidence == CrypticConfidence::Medium)
        .count();
    let low_conf = result.candidates.iter()
        .filter(|c| c.confidence == CrypticConfidence::Low)
        .count();

    if result.n_candidates > 0 {
        println!("  Confidence Distribution:");
        println!("    HIGH:   {} ({:.0}%)", high_conf, 100.0 * high_conf as f64 / result.n_candidates as f64);
        println!("    MEDIUM: {} ({:.0}%)", med_conf, 100.0 * med_conf as f64 / result.n_candidates as f64);
        println!("    LOW:    {} ({:.0}%)", low_conf, 100.0 * low_conf as f64 / result.n_candidates as f64);
        println!();
    }

    // Output JSON if requested
    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&result.candidates)?;
        fs::write(&output_path, json)?;
        println!("  Results saved to: {}", output_path.display());
    }

    println!("{}", "=".repeat(80));

    // Print insight
    println!();
    println!("★ Insight ─────────────────────────────────────");
    println!("  GNM-enhanced detection uses physics-based flexibility prediction");
    println!("  to identify cryptic sites that may not be apparent from B-factors");
    println!("  alone. High-confidence sites (score ≥ 0.7) are strong candidates");
    println!("  for induced-fit binding that could reveal druggable pockets.");
    println!("─────────────────────────────────────────────────");
    println!();

    Ok(())
}
