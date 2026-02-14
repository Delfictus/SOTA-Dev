//! PRISM4D Stage 4: Ensemble Analysis
//!
//! Native Rust implementation for analyzing MD ensemble trajectories.
//! Performs Kabsch alignment and computes RMSD/RMSF statistics.
//!
//! Usage:
//!   cargo run --release -p prism-validation --bin analyze-ensemble -- \
//!     --ensemble ensemble.pdb --output analysis.json
//!
//! This binary completes the native pipeline:
//!   Stage 1-2: Python (PDBFixer + OpenMM) - topology preparation
//!   Stage 3:   Rust/CUDA (generate-ensemble) - MD simulation
//!   Stage 4:   Rust (analyze-ensemble) - analysis ← THIS BINARY

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

// Import our native Kabsch alignment module
use prism_validation::kabsch_alignment::{
    align_and_compute_displacement, compute_rmsf, compute_rmsd,
};

#[derive(Parser, Debug)]
#[command(name = "analyze-ensemble")]
#[command(about = "Analyze MD ensemble with Kabsch alignment (Native Rust)")]
struct Args {
    /// Input ensemble PDB file (multi-MODEL format)
    #[arg(long)]
    ensemble: PathBuf,

    /// Output JSON file for analysis results
    #[arg(long)]
    output: PathBuf,

    /// Use only CA atoms for alignment and RMSD (default: all atoms)
    #[arg(long, default_value = "false")]
    ca_only: bool,

    /// Reference frame index (0-based, default: 0 = first frame)
    #[arg(long, default_value = "0")]
    reference_frame: usize,

    /// Suppress output
    #[arg(long, short)]
    quiet: bool,
}

/// Analysis results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisResults {
    /// Input file path
    pub input_file: String,
    /// Number of frames in ensemble
    pub n_frames: usize,
    /// Number of atoms per frame
    pub n_atoms: usize,
    /// Reference frame index used
    pub reference_frame: usize,
    /// Whether CA-only analysis was used
    pub ca_only: bool,
    /// Per-frame RMSD values (Angstroms)
    pub frame_rmsds: Vec<f64>,
    /// Mean RMSD across all frames
    pub mean_rmsd: f64,
    /// Standard deviation of RMSD
    pub std_rmsd: f64,
    /// Min RMSD
    pub min_rmsd: f64,
    /// Max RMSD
    pub max_rmsd: f64,
    /// Per-atom RMSF values (Angstroms)
    pub atom_rmsf: Vec<f64>,
    /// Mean RMSF across all atoms
    pub mean_rmsf: f64,
    /// Standard deviation of RMSF
    pub std_rmsf: f64,
    /// Number of high-flexibility atoms (RMSF > 1.0 Å)
    pub high_flex_count: usize,
    /// Residue-level RMSF (if available)
    pub residue_rmsf: Option<Vec<ResidueRmsf>>,
}

/// Per-residue RMSF data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResidueRmsf {
    pub residue_id: usize,
    pub residue_name: String,
    pub ca_rmsf: f64,
}

/// Parsed atom from PDB
#[derive(Debug, Clone)]
struct PdbAtom {
    pub name: String,
    pub residue_name: String,
    pub residue_id: i32,
    pub chain_id: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Parse multi-MODEL PDB file into frames
fn parse_ensemble_pdb(content: &str) -> Result<(Vec<Vec<[f32; 3]>>, Vec<PdbAtom>)> {
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut current_frame: Vec<[f32; 3]> = Vec::new();
    let mut atom_info: Vec<PdbAtom> = Vec::new();
    let mut first_frame = true;

    for line in content.lines() {
        if line.starts_with("MODEL") {
            current_frame = Vec::new();
        } else if line.starts_with("ENDMDL") {
            if !current_frame.is_empty() {
                frames.push(current_frame.clone());
                first_frame = false;
            }
            current_frame = Vec::new();
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            if line.len() < 54 {
                continue;
            }

            let x: f32 = line
                .get(30..38)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let y: f32 = line
                .get(38..46)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let z: f32 = line
                .get(46..54)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);

            current_frame.push([x, y, z]);

            // Store atom info from first frame only
            if first_frame {
                let atom_name = line.get(12..16).unwrap_or("    ").trim().to_string();
                let residue_name = line.get(17..20).unwrap_or("UNK").trim().to_string();
                let chain_id = line.get(21..22).unwrap_or("A").chars().next().unwrap_or('A');
                let residue_id: i32 = line
                    .get(22..26)
                    .unwrap_or("0")
                    .trim()
                    .parse()
                    .unwrap_or(0);

                atom_info.push(PdbAtom {
                    name: atom_name,
                    residue_name,
                    residue_id,
                    chain_id,
                    x,
                    y,
                    z,
                });
            }
        }
    }

    // Handle case where last frame doesn't have ENDMDL
    if !current_frame.is_empty() {
        frames.push(current_frame);
    }

    // If no MODEL/ENDMDL records, treat entire file as single frame
    if frames.is_empty() && !atom_info.is_empty() {
        let coords: Vec<[f32; 3]> = atom_info.iter().map(|a| [a.x, a.y, a.z]).collect();
        frames.push(coords);
    }

    Ok((frames, atom_info))
}

/// Extract CA atom indices
fn get_ca_indices(atoms: &[PdbAtom]) -> Vec<usize> {
    atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.name == "CA")
        .map(|(i, _)| i)
        .collect()
}

/// Extract subset of coordinates by indices
fn extract_coords(coords: &[[f32; 3]], indices: &[usize]) -> Vec<[f32; 3]> {
    indices.iter().map(|&i| coords[i]).collect()
}

/// Compute mean and standard deviation
fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.quiet {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║         PRISM-4D Ensemble Analysis (Native Rust)             ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    }

    // Load ensemble PDB
    if !args.quiet {
        println!("\n📂 Loading ensemble: {:?}", args.ensemble);
    }

    let content = std::fs::read_to_string(&args.ensemble)
        .context("Failed to read ensemble PDB file")?;

    let (frames, atom_info) = parse_ensemble_pdb(&content)?;

    if frames.is_empty() {
        anyhow::bail!("No frames found in ensemble PDB");
    }

    let n_frames = frames.len();
    let n_atoms = frames[0].len();

    if !args.quiet {
        println!("   Frames: {}", n_frames);
        println!("   Atoms per frame: {}", n_atoms);
    }

    // Get CA indices if needed
    let ca_indices = get_ca_indices(&atom_info);
    if !args.quiet {
        println!("   CA atoms: {}", ca_indices.len());
    }

    // Select coordinates for analysis
    let analysis_frames: Vec<Vec<[f32; 3]>> = if args.ca_only {
        if !args.quiet {
            println!("\n⚙️  Using CA-only alignment");
        }
        frames
            .iter()
            .map(|f| extract_coords(f, &ca_indices))
            .collect()
    } else {
        if !args.quiet {
            println!("\n⚙️  Using all-atom alignment");
        }
        frames.clone()
    };

    let n_analysis_atoms = analysis_frames[0].len();

    // Validate reference frame
    if args.reference_frame >= n_frames {
        anyhow::bail!(
            "Reference frame {} out of range (0..{})",
            args.reference_frame,
            n_frames
        );
    }

    let reference = &analysis_frames[args.reference_frame];

    if !args.quiet {
        println!("   Reference frame: {}", args.reference_frame);
        println!("   Analysis atoms: {}", n_analysis_atoms);
    }

    // Perform Kabsch alignment and compute displacements
    if !args.quiet {
        println!("\n🔄 Performing Kabsch alignment...");
    }

    let mut all_displacements: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n_frames);
    let mut frame_rmsds: Vec<f64> = Vec::with_capacity(n_frames);

    for (i, frame) in analysis_frames.iter().enumerate() {
        let (aligned, displacement) = align_and_compute_displacement(reference, frame);

        // Compute RMSD for this frame
        let rmsd = compute_rmsd(reference, &aligned);
        frame_rmsds.push(rmsd);

        all_displacements.push(displacement);

        // Progress update
        if !args.quiet && (i + 1) % 100 == 0 {
            println!("   Processed {}/{} frames", i + 1, n_frames);
        }
    }

    if !args.quiet {
        println!("   ✓ Aligned {} frames", n_frames);
    }

    // Compute RMSF
    if !args.quiet {
        println!("\n📊 Computing RMSF...");
    }

    let atom_rmsf = compute_rmsf(&all_displacements);

    // Compute statistics
    let (mean_rmsd, std_rmsd) = mean_std(&frame_rmsds);
    let min_rmsd = frame_rmsds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rmsd = frame_rmsds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let (mean_rmsf, std_rmsf) = mean_std(&atom_rmsf);
    let high_flex_count = atom_rmsf.iter().filter(|&&r| r > 1.0).count();

    // Compute residue-level RMSF (CA atoms only)
    let residue_rmsf: Option<Vec<ResidueRmsf>> = if args.ca_only {
        // In CA-only mode, each atom is a residue
        Some(
            atom_rmsf
                .iter()
                .enumerate()
                .map(|(i, &rmsf)| {
                    let atom = &atom_info[ca_indices[i]];
                    ResidueRmsf {
                        residue_id: atom.residue_id as usize,
                        residue_name: atom.residue_name.clone(),
                        ca_rmsf: rmsf,
                    }
                })
                .collect(),
        )
    } else if !ca_indices.is_empty() {
        // In all-atom mode, extract CA RMSF
        Some(
            ca_indices
                .iter()
                .map(|&i| {
                    let atom = &atom_info[i];
                    ResidueRmsf {
                        residue_id: atom.residue_id as usize,
                        residue_name: atom.residue_name.clone(),
                        ca_rmsf: atom_rmsf[i],
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    // Build results
    let results = AnalysisResults {
        input_file: args.ensemble.display().to_string(),
        n_frames,
        n_atoms: n_analysis_atoms,
        reference_frame: args.reference_frame,
        ca_only: args.ca_only,
        frame_rmsds,
        mean_rmsd,
        std_rmsd,
        min_rmsd,
        max_rmsd,
        atom_rmsf,
        mean_rmsf,
        std_rmsf,
        high_flex_count,
        residue_rmsf,
    };

    // Print summary
    if !args.quiet {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    Analysis Results                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Frames analyzed:    {:>6}                                  ║", n_frames);
        println!("║  Atoms per frame:    {:>6}                                  ║", n_analysis_atoms);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  RMSD Statistics (Å):                                        ║");
        println!("║    Mean ± Std:       {:>6.3} ± {:<6.3}                       ║", mean_rmsd, std_rmsd);
        println!("║    Min / Max:        {:>6.3} / {:<6.3}                       ║", min_rmsd, max_rmsd);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  RMSF Statistics (Å):                                        ║");
        println!("║    Mean ± Std:       {:>6.3} ± {:<6.3}                       ║", mean_rmsf, std_rmsf);
        println!("║    High-flex atoms:  {:>6} (RMSF > 1.0 Å)                   ║", high_flex_count);
        println!("╚══════════════════════════════════════════════════════════════╝");
    }

    // Write output JSON
    if !args.quiet {
        println!("\n💾 Writing results to {:?}", args.output);
    }

    let output_file = File::create(&args.output)
        .context("Failed to create output file")?;
    let mut writer = BufWriter::new(output_file);

    serde_json::to_writer_pretty(&mut writer, &results)
        .context("Failed to write JSON")?;
    writer.flush()?;

    if !args.quiet {
        println!("✅ Analysis complete!");
    }

    Ok(())
}
