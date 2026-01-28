//! GNM-Enhanced Cryptic Site Benchmark
//!
//! Runs cryptic site detection across the ATLAS benchmark proteins
//! and produces comprehensive statistics.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use prism_validation::cryptic_sites::{
    CrypticSiteConfig, CrypticSiteDetector, CrypticConfidence,
    CrypticSiteResult, parse_pdb_simple,
};

#[derive(Parser, Debug)]
#[command(name = "cryptic-bench")]
#[command(about = "Benchmark GNM-enhanced cryptic site detection on ATLAS proteins")]
struct Args {
    /// Directory containing PDB files
    #[arg(short, long, default_value = "data/atlas_alphaflow/pdb")]
    pdb_dir: PathBuf,

    /// Maximum number of proteins to analyze (0 = all)
    #[arg(short, long, default_value = "0")]
    max_proteins: usize,

    /// GNM cutoff distance (Ångströms)
    #[arg(long, default_value = "7.3")]
    gnm_cutoff: f64,

    /// Minimum cryptic score to report
    #[arg(long, default_value = "0.3")]
    min_score: f64,

    /// Output JSON file for results
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=".repeat(80));
    println!("  GNM-ENHANCED CRYPTIC SITE BENCHMARK");
    println!("{}", "=".repeat(80));
    println!();

    // Find all PDB files
    let mut pdb_files: Vec<PathBuf> = fs::read_dir(&args.pdb_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "pdb").unwrap_or(false))
        .collect();

    pdb_files.sort();

    if args.max_proteins > 0 && pdb_files.len() > args.max_proteins {
        pdb_files.truncate(args.max_proteins);
    }

    println!("  Configuration:");
    println!("    PDB directory:    {}", args.pdb_dir.display());
    println!("    Proteins:         {}", pdb_files.len());
    println!("    GNM cutoff:       {} Å", args.gnm_cutoff);
    println!("    Min score:        {}", args.min_score);
    println!();

    // Configure detector
    let config = CrypticSiteConfig {
        gnm_cutoff: args.gnm_cutoff,
        min_score: args.min_score,
        ..Default::default()
    };

    let detector = CrypticSiteDetector::with_config(config);

    // Run detection on all proteins
    let overall_start = Instant::now();
    let mut results: Vec<CrypticSiteResult> = Vec::new();
    let mut total_candidates = 0;
    let mut high_conf_candidates = 0;
    let mut medium_conf_candidates = 0;
    let mut proteins_with_sites = 0;

    println!("{}", "-".repeat(80));
    println!("  {:<20} {:>8} {:>10} {:>10} {:>10} {:>10}",
             "Protein", "Residues", "Candidates", "High", "Medium", "Time(ms)");
    println!("{}", "-".repeat(80));

    for pdb_path in &pdb_files {
        let pdb_name = pdb_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let content = match fs::read_to_string(pdb_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let atoms = parse_pdb_simple(&content);
        let start = Instant::now();
        let result = detector.detect(&pdb_name, &atoms);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let high = result.candidates.iter()
            .filter(|c| c.confidence == CrypticConfidence::High)
            .count();
        let medium = result.candidates.iter()
            .filter(|c| c.confidence == CrypticConfidence::Medium)
            .count();

        if result.n_candidates > 0 {
            proteins_with_sites += 1;
        }

        total_candidates += result.n_candidates;
        high_conf_candidates += high;
        medium_conf_candidates += medium;

        let marker = if high > 0 { "★" } else if medium > 0 { "●" } else { " " };

        println!("  {:<20} {:>8} {:>10} {:>10} {:>10} {:>10.1}",
                 format!("{} {}", marker, pdb_name),
                 result.n_residues,
                 result.n_candidates,
                 high,
                 medium,
                 elapsed_ms);

        results.push(result);
    }

    let overall_elapsed = overall_start.elapsed();

    println!();
    println!("{}", "=".repeat(80));
    println!("  BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));
    println!();
    println!("  Total proteins:        {}", pdb_files.len());
    println!("  With cryptic sites:    {} ({:.1}%)",
             proteins_with_sites,
             100.0 * proteins_with_sites as f64 / pdb_files.len() as f64);
    println!();
    println!("  Total candidates:      {}", total_candidates);
    println!("    HIGH confidence:     {} ({:.1}%)",
             high_conf_candidates,
             if total_candidates > 0 { 100.0 * high_conf_candidates as f64 / total_candidates as f64 } else { 0.0 });
    println!("    MEDIUM confidence:   {} ({:.1}%)",
             medium_conf_candidates,
             if total_candidates > 0 { 100.0 * medium_conf_candidates as f64 / total_candidates as f64 } else { 0.0 });
    println!();
    println!("  Avg candidates/protein: {:.2}",
             total_candidates as f64 / pdb_files.len() as f64);
    println!("  Total time:             {:.2}s", overall_elapsed.as_secs_f64());
    println!("  Avg time/protein:       {:.1}ms",
             overall_elapsed.as_secs_f64() * 1000.0 / pdb_files.len() as f64);
    println!();

    // Compute score distribution
    let all_scores: Vec<f64> = results.iter()
        .flat_map(|r| r.candidates.iter().map(|c| c.score))
        .collect();

    if !all_scores.is_empty() {
        let min_score = all_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = all_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_score = all_scores.iter().sum::<f64>() / all_scores.len() as f64;

        println!("  Score Distribution:");
        println!("    Min:  {:.3}", min_score);
        println!("    Max:  {:.3}", max_score);
        println!("    Mean: {:.3}", mean_score);
    }

    println!();
    println!("{}", "=".repeat(80));

    // Print insight
    println!();
    println!("★ Insight ─────────────────────────────────────");
    println!("  Cryptic site detection identifies \"hidden\" binding pockets");
    println!("  that are not visible in the static structure but can open");
    println!("  upon ligand binding (induced fit). These sites are valuable");
    println!("  drug discovery targets often missed by structure-based methods.");
    println!("─────────────────────────────────────────────────");
    println!();

    // Save results if output specified
    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&results.iter()
            .map(|r| &r.candidates)
            .collect::<Vec<_>>())?;
        fs::write(&output_path, json)?;
        println!("Results saved to: {}", output_path.display());
    }

    Ok(())
}
