//! PRISM-Delta Blind Validation Pipeline Runner
//!
//! Runs retrospective blind validation on control structures:
//! - 6VXX: SARS-CoV-2 Spike (Coronavirus)
//! - 2VWD: HIV-1 gp120 Core (Lentivirus)
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p prism-validation --bin run-blind-validation -- \
//!     --pdb data/raw/6VXX.pdb \
//!     --chain A \
//!     --output results/6vxx_blind_prediction.json
//! ```

use std::path::PathBuf;
use std::collections::HashMap;
use std::fs;
use anyhow::{Result, Context, anyhow};
use clap::Parser;
use chrono::Utc;

use prism_validation::blind_validation_pipeline::{
    BlindValidationPipeline, BlindValidationConfig, BlindPrediction,
    ValidationReport, StructureValidationResult, ValidationMetrics,
};

#[derive(Parser, Debug)]
#[command(name = "run-blind-validation")]
#[command(about = "Run PRISM-Delta blind validation on control structures")]
struct Args {
    /// Path to PDB file (6VXX.pdb or 2VWD.pdb)
    #[arg(short, long)]
    pdb: PathBuf,

    /// Chain ID to process (e.g., A, B, C)
    #[arg(short, long, default_value = "A")]
    chain: String,

    /// Output JSON file for predictions
    #[arg(short, long, default_value = "blind_prediction.json")]
    output: PathBuf,

    /// Output markdown report
    #[arg(short, long)]
    report: Option<PathBuf>,

    /// Number of conformations to generate
    #[arg(long, default_value = "100")]
    n_conformations: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Compare to ground truth after prediction
    #[arg(long)]
    validate: bool,

    /// Export PDB with B-factors colored by cryptic score for PyMOL
    #[arg(long)]
    pymol_pdb: Option<PathBuf>,
}

/// Parse PDB file to extract CA coordinates and sequence
fn parse_pdb_chain(pdb_path: &PathBuf, chain_id: &str) -> Result<(Vec<[f32; 3]>, String, Vec<i32>)> {
    let content = fs::read_to_string(pdb_path)
        .with_context(|| format!("Failed to read PDB file: {:?}", pdb_path))?;

    let mut ca_coords: Vec<[f32; 3]> = Vec::new();
    let mut residue_numbers: Vec<i32> = Vec::new();
    let mut residue_map: HashMap<i32, char> = HashMap::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        // Parse chain ID (column 22, 0-indexed = 21)
        let line_chain = line.chars().nth(21).unwrap_or(' ');
        if line_chain.to_string() != chain_id {
            continue;
        }

        // Parse atom name (columns 13-16)
        let atom_name: String = line[12..16].trim().to_string();
        if atom_name != "CA" {
            continue;
        }

        // Parse residue number (columns 23-26)
        let res_num: i32 = line[22..26].trim().parse().unwrap_or(0);

        // Skip if we already have this residue (alternate conformations)
        if residue_numbers.contains(&res_num) {
            continue;
        }

        // Parse coordinates (columns 31-38, 39-46, 47-54)
        let x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f32 = line[46..54].trim().parse().unwrap_or(0.0);

        // Parse residue name (columns 18-20)
        let res_name: &str = line[17..20].trim();
        let aa = residue_name_to_char(res_name);

        ca_coords.push([x, y, z]);
        residue_numbers.push(res_num);
        residue_map.insert(res_num, aa);
    }

    if ca_coords.is_empty() {
        return Err(anyhow!("No CA atoms found for chain {} in {:?}", chain_id, pdb_path));
    }

    // Build sequence from residue map
    let sequence: String = residue_numbers.iter()
        .map(|r| residue_map.get(r).copied().unwrap_or('X'))
        .collect();

    Ok((ca_coords, sequence, residue_numbers))
}

/// Convert 3-letter amino acid code to 1-letter code
fn residue_name_to_char(name: &str) -> char {
    match name {
        "ALA" => 'A', "CYS" => 'C', "ASP" => 'D', "GLU" => 'E',
        "PHE" => 'F', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
        "LYS" => 'K', "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
        "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R', "SER" => 'S',
        "THR" => 'T', "VAL" => 'V', "TRP" => 'W', "TYR" => 'Y',
        // Non-standard
        "MSE" => 'M', // Selenomethionine
        "SEC" => 'U', // Selenocysteine
        "PYL" => 'O', // Pyrrolysine
        _ => 'X',
    }
}

/// Export PDB with B-factors colored by cryptic score for PyMOL visualization
///
/// Creates a modified PDB where each atom's B-factor is replaced with the
/// residue's cryptic score (0-1). In PyMOL, use:
///   `spectrum b, blue_white_red, minimum=0, maximum=1`
///
/// Blue = low cryptic score (stable)
/// White = medium
/// Red = high cryptic score (cryptic binding potential)
fn export_pymol_pdb(
    pdb_path: &PathBuf,
    chain_id: &str,
    prediction: &prism_validation::blind_validation_pipeline::BlindPrediction,
    output_path: &PathBuf,
) -> Result<()> {
    let content = fs::read_to_string(pdb_path)?;

    // Build residue number -> cryptic score map
    let mut score_map: HashMap<i32, f64> = HashMap::new();
    for pred in &prediction.residue_predictions {
        score_map.insert(pred.residue_num, pred.cryptic_score);
    }

    let mut output_lines = Vec::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            output_lines.push(line.to_string());
            continue;
        }

        // Check chain
        let line_chain = line.chars().nth(21).unwrap_or(' ');
        if line_chain.to_string() != chain_id {
            output_lines.push(line.to_string());
            continue;
        }

        // Parse residue number
        let res_num: i32 = line[22..26].trim().parse().unwrap_or(0);

        // Get cryptic score for this residue (default 0 if not in map)
        let cryptic_score = score_map.get(&res_num).copied().unwrap_or(0.0);

        // Scale to 0-100 range for B-factor (PDB B-factor is typically 0-100)
        let b_factor = (cryptic_score * 100.0).min(99.99);

        // Replace B-factor (columns 61-66 in PDB format)
        if line.len() >= 66 {
            let mut new_line = line[..60].to_string();
            new_line.push_str(&format!("{:6.2}", b_factor));
            if line.len() > 66 {
                new_line.push_str(&line[66..]);
            }
            output_lines.push(new_line);
        } else {
            output_lines.push(line.to_string());
        }
    }

    // Add PyMOL script header as REMARK
    let mut final_output = Vec::new();
    final_output.push("REMARK PRISM-Delta Cryptic Site Prediction".to_string());
    final_output.push("REMARK B-factor = Cryptic Score (0-100 scale)".to_string());
    final_output.push("REMARK PyMOL: spectrum b, blue_white_red, minimum=0, maximum=100".to_string());
    final_output.push("REMARK        or: color red, b > 50".to_string());
    final_output.extend(output_lines);

    fs::write(output_path, final_output.join("\n"))?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("================================================================================");
    println!("         PRISM-DELTA BLIND VALIDATION PIPELINE");
    println!("         Control Structure Analysis");
    println!("================================================================================");
    println!();

    // Parse PDB file
    let pdb_filename = args.pdb.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    println!("Loading structure: {:?}", args.pdb);
    println!("Chain: {}", args.chain);

    let (ca_coords, sequence, residue_numbers) = parse_pdb_chain(&args.pdb, &args.chain)?;

    println!("Parsed {} residues from chain {}", ca_coords.len(), args.chain);
    println!("Sequence length: {}", sequence.len());
    println!();

    // Configure pipeline
    let config = BlindValidationConfig {
        n_ensemble_conformations: args.n_conformations,
        verbose: args.verbose,
        ..Default::default()
    };

    // Create pipeline
    let mut pipeline = BlindValidationPipeline::with_config(config);
    pipeline.load_ground_truth();

    // Read full PDB content for AMBER ff14SB physics (bonds, angles, dihedrals)
    let pdb_content = fs::read_to_string(&args.pdb)
        .with_context(|| format!("Failed to read PDB file for full-atom physics: {:?}", args.pdb))?;

    // Pass full-atom PDB to pipeline for proper AMBER force field
    // This enables bonds, angles, dihedrals instead of CA-only elastic network
    let chain_char = args.chain.chars().next();
    pipeline.set_full_atom_pdb(pdb_content, chain_char);

    println!("📊 Full-atom PDB loaded for AMBER ff14SB physics");
    println!("   - Bonds: E = k(r - r0)²");
    println!("   - Angles: E = k(θ - θ0)²");
    println!("   - Dihedrals: E = k(1 + cos(nφ - γ))");
    println!();

    println!("================================================================================");
    println!("  STAGE 1-5: BLIND PREDICTION (Ground truth NOT accessible)");
    println!("================================================================================");
    println!();

    // Run blind prediction
    let prediction = pipeline.run_blind(
        &ca_coords,
        &sequence,
        &pdb_filename,
        &args.chain,
        &residue_numbers,
        None, // No MSA entropy for now
    )?;

    // Print prediction summary
    println!("PREDICTION LOCKED");
    println!("-----------------");
    println!("Total residues analyzed: {}", prediction.summary.n_residues);
    println!("Cryptic residues predicted: {}", prediction.summary.n_cryptic_residues);
    println!("Predicted binding sites: {}", prediction.summary.n_predicted_sites);
    println!("Mean cryptic score: {:.4}", prediction.summary.mean_cryptic_score);
    println!("Max cryptic score: {:.4}", prediction.summary.max_cryptic_score);
    println!("Mean escape resistance: {:.4}", prediction.summary.mean_escape_resistance);
    println!("Mean priority score: {:.4}", prediction.summary.mean_priority_score);
    println!("Threshold used: {:.4}", prediction.summary.threshold_used);
    println!();

    // Print timing breakdown
    println!("Timing Breakdown:");
    println!("  Ensemble generation: {} ms", prediction.timing.ensemble_generation_ms);
    println!("  Kabsch alignment: {} ms", prediction.timing.alignment_ms);
    println!("  Feature extraction: {} ms", prediction.timing.feature_extraction_ms);
    println!("  Cryptic scoring: {} ms", prediction.timing.cryptic_scoring_ms);
    println!("  Clustering: {} ms", prediction.timing.clustering_ms);
    println!("  Total: {} ms", prediction.timing.total_ms);
    println!();

    // Print top predicted sites
    if !prediction.predicted_sites.is_empty() {
        println!("TOP PREDICTED CRYPTIC SITES:");
        println!("{:-<80}", "");
        for (i, site) in prediction.predicted_sites.iter().take(5).enumerate() {
            let residue_nums: Vec<i32> = site.residues.iter()
                .map(|r| r.residue_num)
                .collect();
            println!("  Site {}: {} residues, score={:.3}, escape_resistance={:.3}",
                     i + 1,
                     site.residues.len(),
                     site.mean_cryptic_score,
                     site.mean_escape_resistance);
            println!("    Residues: {:?}", &residue_nums[..residue_nums.len().min(10)]);
            if let Some(ref domain) = site.representative.domain {
                println!("    Domain: {}", domain);
            }
        }
        println!();
    }

    // Save prediction to JSON
    let prediction_json = serde_json::to_string_pretty(&prediction)?;
    fs::write(&args.output, &prediction_json)?;
    println!("Prediction saved to: {:?}", args.output);

    // Export PyMOL-compatible PDB if requested
    if let Some(pymol_path) = &args.pymol_pdb {
        export_pymol_pdb(&args.pdb, &args.chain, &prediction, pymol_path)?;
        println!("PyMOL PDB exported to: {:?}", pymol_path);
        println!("  Open in PyMOL and use: spectrum b, blue_white_red, minimum=0, maximum=1");
    }

    // Validate against ground truth if requested
    if args.validate {
        println!();
        println!("================================================================================");
        println!("  STAGE 6: VALIDATION (Comparing to hidden ground truth)");
        println!("================================================================================");
        println!();

        let report = pipeline.validate(&[prediction.clone()])?;

        // Print validation results
        if let Some(result) = report.structure_results.first() {
            println!("VALIDATION METRICS:");
            println!("{:-<80}", "");
            println!("  ROC AUC: {:.4}", result.metrics.cryptic_roc_auc);
            println!("  PR AUC: {:.4}", result.metrics.cryptic_pr_auc);
            println!("  Recall: {:.4}", result.metrics.cryptic_recall);
            println!("  Precision: {:.4}", result.metrics.cryptic_precision);
            println!("  F1 Score: {:.4}", result.metrics.cryptic_f1);
            println!("  Success: {}", if result.metrics.success { "YES" } else { "NO" });
            println!("  Epitope overlap: {:.4}", result.metrics.epitope_overlap);
            println!();

            println!("Ground Truth:");
            println!("  Cryptic residues: {:?}", &result.ground_truth.cryptic_residues);
            println!("  Source: {}", result.ground_truth.source);
        }

        // Save report if requested
        if let Some(report_path) = &args.report {
            let report_json = serde_json::to_string_pretty(&report)?;
            fs::write(report_path, &report_json)?;
            println!();
            println!("Validation report saved to: {:?}", report_path);
        }
    }

    println!();
    println!("================================================================================");
    println!("  PIPELINE COMPLETE");
    println!("================================================================================");

    Ok(())
}
