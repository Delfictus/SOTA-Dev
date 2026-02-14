//! PRISM4D Stage 4: Batched Ensemble Analysis
//!
//! Processes multiple ensemble PDB files in parallel, computing RMSD/RMSF
//! statistics for each using native Rust Kabsch alignment.
//!
//! This is the batched/optimized version of analyze_ensemble for high throughput.
//!
//! Usage:
//!   cargo run --release -p prism-validation --bin analyze_ensemble_batch -- \
//!     --ensembles dir/*.pdb --output-dir results/

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

// Import native Kabsch alignment
use prism_validation::kabsch_alignment::{
    align_and_compute_displacement, compute_rmsf, compute_rmsd,
};

#[derive(Parser, Debug)]
#[command(name = "analyze_ensemble_batch")]
#[command(about = "Analyze multiple MD ensembles in parallel (Native Rust)")]
struct Args {
    /// Input ensemble PDB files (multi-MODEL format)
    #[arg(long, num_args = 1.., required = true)]
    ensembles: Vec<PathBuf>,

    /// Output directory for analysis JSON files
    #[arg(long)]
    output_dir: PathBuf,

    /// Use only CA atoms for alignment and RMSD (default: all atoms)
    #[arg(long, default_value = "false")]
    ca_only: bool,

    /// Reference frame index (0-based, default: 0 = first frame)
    #[arg(long, default_value = "0")]
    reference_frame: usize,

    /// Number of parallel threads (default: all available CPUs)
    #[arg(long)]
    threads: Option<usize>,

    /// Quiet mode (minimal output)
    #[arg(long, short)]
    quiet: bool,
}

/// Analysis results structure (same as single version)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisResults {
    pub input_file: String,
    pub n_frames: usize,
    pub n_atoms: usize,
    pub reference_frame: usize,
    pub ca_only: bool,
    pub frame_rmsds: Vec<f64>,
    pub mean_rmsd: f64,
    pub std_rmsd: f64,
    pub min_rmsd: f64,
    pub max_rmsd: f64,
    pub atom_rmsf: Vec<f64>,
    pub mean_rmsf: f64,
    pub std_rmsf: f64,
    pub high_flex_count: usize,
    pub residue_rmsf: Option<Vec<ResidueRmsf>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResidueRmsf {
    pub residue_id: usize,
    pub residue_name: String,
    pub ca_rmsf: f64,
}

/// Batch summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchSummary {
    pub total_files: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_frames: usize,
    pub mean_rmsd_across_all: f64,
    pub std_rmsd_across_all: f64,
    pub elapsed_secs: f64,
    pub throughput_files_per_sec: f64,
}

/// Parsed atom from PDB
#[derive(Debug, Clone)]
struct PdbAtom {
    pub name: String,
    pub residue_name: String,
    pub residue_id: i32,
    #[allow(dead_code)]
    pub chain_id: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Result of processing a single file
struct FileResult {
    name: String,
    result: Option<AnalysisResults>,
    error: Option<String>,
    elapsed_secs: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.quiet {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║     PRISM-4D Batched Ensemble Analysis (Native Rust)         ║");
        println!("║              Multi-File Parallel Processing                  ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    }

    let n_files = args.ensembles.len();
    if n_files == 0 {
        anyhow::bail!("No ensemble files provided");
    }

    // Configure thread pool
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let n_threads = rayon::current_num_threads();

    if !args.quiet {
        println!("\n📊 Batch configuration:");
        println!("   Files to process: {}", n_files);
        println!("   Parallel threads: {}", n_threads);
        println!("   CA-only mode: {}", args.ca_only);
        println!("   Reference frame: {}", args.reference_frame);
        println!("   Output directory: {:?}", args.output_dir);
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .context("Failed to create output directory")?;

    let start_time = Instant::now();

    // Process all files in parallel using rayon
    if !args.quiet {
        println!("\n🔄 Processing {} files in parallel...", n_files);
    }

    let results: Vec<FileResult> = args.ensembles
        .par_iter()
        .map(|path| {
            process_single_file(
                path,
                &args.output_dir,
                args.ca_only,
                args.reference_frame,
            )
        })
        .collect();

    let total_elapsed = start_time.elapsed().as_secs_f64();

    // Compute summary statistics
    let successful: Vec<_> = results.iter().filter(|r| r.result.is_some()).collect();
    let failed: Vec<_> = results.iter().filter(|r| r.result.is_none()).collect();

    let total_frames: usize = successful
        .iter()
        .filter_map(|r| r.result.as_ref())
        .map(|r| r.n_frames)
        .sum();

    let all_mean_rmsds: Vec<f64> = successful
        .iter()
        .filter_map(|r| r.result.as_ref())
        .map(|r| r.mean_rmsd)
        .collect();

    let (mean_rmsd_all, std_rmsd_all) = if !all_mean_rmsds.is_empty() {
        mean_std(&all_mean_rmsds)
    } else {
        (0.0, 0.0)
    };

    let summary = BatchSummary {
        total_files: n_files,
        successful: successful.len(),
        failed: failed.len(),
        total_frames,
        mean_rmsd_across_all: mean_rmsd_all,
        std_rmsd_across_all: std_rmsd_all,
        elapsed_secs: total_elapsed,
        throughput_files_per_sec: n_files as f64 / total_elapsed,
    };

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                  BATCH ANALYSIS COMPLETE                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\n📊 Summary:");
    println!("   Files processed: {}/{}", summary.successful, summary.total_files);
    println!("   Total frames: {}", summary.total_frames);
    println!("   Mean RMSD (all files): {:.3} ± {:.3} Å",
             summary.mean_rmsd_across_all, summary.std_rmsd_across_all);
    println!("   Total time: {:.2}s", summary.elapsed_secs);
    println!("   Throughput: {:.2} files/sec", summary.throughput_files_per_sec);

    // List failures
    if !failed.is_empty() {
        println!("\n⚠️  Failures ({}):", failed.len());
        for f in &failed {
            println!("   - {}: {}", f.name, f.error.as_ref().unwrap_or(&"Unknown".to_string()));
        }
    }

    // Write batch summary
    let summary_path = args.output_dir.join("batch_summary.json");
    let summary_file = File::create(&summary_path)?;
    serde_json::to_writer_pretty(BufWriter::new(summary_file), &summary)?;

    if !args.quiet {
        println!("\n📁 Output directory: {:?}", args.output_dir);
        println!("   - batch_summary.json (overall statistics)");
        for result in &results {
            if result.result.is_some() {
                println!("   - {}_analysis.json", result.name);
            }
        }
    }

    println!("\n✅ Batch analysis complete!");

    Ok(())
}

fn process_single_file(
    path: &PathBuf,
    output_dir: &PathBuf,
    ca_only: bool,
    reference_frame: usize,
) -> FileResult {
    let start = Instant::now();

    let file_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Load ensemble PDB
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return FileResult {
            name: file_name,
            result: None,
            error: Some(format!("Failed to read file: {}", e)),
            elapsed_secs: start.elapsed().as_secs_f64(),
        },
    };

    let (frames, atom_info) = match parse_ensemble_pdb(&content) {
        Ok(r) => r,
        Err(e) => return FileResult {
            name: file_name,
            result: None,
            error: Some(format!("Failed to parse PDB: {}", e)),
            elapsed_secs: start.elapsed().as_secs_f64(),
        },
    };

    if frames.is_empty() {
        return FileResult {
            name: file_name,
            result: None,
            error: Some("No frames found".to_string()),
            elapsed_secs: start.elapsed().as_secs_f64(),
        };
    }

    let n_frames = frames.len();
    let n_atoms = frames[0].len();

    // Get CA indices
    let ca_indices = get_ca_indices(&atom_info);

    // Select coordinates
    let analysis_frames: Vec<Vec<[f32; 3]>> = if ca_only {
        frames.iter().map(|f| extract_coords(f, &ca_indices)).collect()
    } else {
        frames.clone()
    };

    let n_analysis_atoms = analysis_frames[0].len();

    // Validate reference frame
    if reference_frame >= n_frames {
        return FileResult {
            name: file_name,
            result: None,
            error: Some(format!("Reference frame {} out of range", reference_frame)),
            elapsed_secs: start.elapsed().as_secs_f64(),
        };
    }

    let reference = &analysis_frames[reference_frame];

    // Kabsch alignment
    let mut all_displacements: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n_frames);
    let mut frame_rmsds: Vec<f64> = Vec::with_capacity(n_frames);

    for frame in &analysis_frames {
        let (aligned, displacement) = align_and_compute_displacement(reference, frame);
        let rmsd = compute_rmsd(reference, &aligned);
        frame_rmsds.push(rmsd);
        all_displacements.push(displacement);
    }

    // Compute RMSF
    let atom_rmsf = compute_rmsf(&all_displacements);

    // Statistics
    let (mean_rmsd, std_rmsd) = mean_std(&frame_rmsds);
    let min_rmsd = frame_rmsds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rmsd = frame_rmsds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let (mean_rmsf, std_rmsf) = mean_std(&atom_rmsf);
    let high_flex_count = atom_rmsf.iter().filter(|&&r| r > 1.0).count();

    // Residue-level RMSF
    let residue_rmsf: Option<Vec<ResidueRmsf>> = if ca_only {
        Some(atom_rmsf.iter().enumerate().map(|(i, &rmsf)| {
            let atom = &atom_info[ca_indices[i]];
            ResidueRmsf {
                residue_id: atom.residue_id as usize,
                residue_name: atom.residue_name.clone(),
                ca_rmsf: rmsf,
            }
        }).collect())
    } else if !ca_indices.is_empty() {
        Some(ca_indices.iter().map(|&i| {
            let atom = &atom_info[i];
            ResidueRmsf {
                residue_id: atom.residue_id as usize,
                residue_name: atom.residue_name.clone(),
                ca_rmsf: atom_rmsf[i],
            }
        }).collect())
    } else {
        None
    };

    let results = AnalysisResults {
        input_file: path.display().to_string(),
        n_frames,
        n_atoms: n_analysis_atoms,
        reference_frame,
        ca_only,
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

    // Write output
    let output_path = output_dir.join(format!("{}_analysis.json", file_name));
    if let Ok(file) = File::create(&output_path) {
        let _ = serde_json::to_writer_pretty(BufWriter::new(file), &results);
    }

    FileResult {
        name: file_name,
        result: Some(results),
        error: None,
        elapsed_secs: start.elapsed().as_secs_f64(),
    }
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

            let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
            let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
            let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);

            current_frame.push([x, y, z]);

            if first_frame {
                let atom_name = line.get(12..16).unwrap_or("    ").trim().to_string();
                let residue_name = line.get(17..20).unwrap_or("UNK").trim().to_string();
                let chain_id = line.get(21..22).unwrap_or("A").chars().next().unwrap_or('A');
                let residue_id: i32 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);

                atom_info.push(PdbAtom {
                    name: atom_name,
                    residue_name,
                    residue_id,
                    chain_id,
                    x, y, z,
                });
            }
        }
    }

    if !current_frame.is_empty() {
        frames.push(current_frame);
    }

    if frames.is_empty() && !atom_info.is_empty() {
        let coords: Vec<[f32; 3]> = atom_info.iter().map(|a| [a.x, a.y, a.z]).collect();
        frames.push(coords);
    }

    Ok((frames, atom_info))
}

fn get_ca_indices(atoms: &[PdbAtom]) -> Vec<usize> {
    atoms.iter().enumerate()
        .filter(|(_, a)| a.name == "CA")
        .map(|(i, _)| i)
        .collect()
}

fn extract_coords(coords: &[[f32; 3]], indices: &[usize]) -> Vec<[f32; 3]> {
    indices.iter().map(|&i| coords[i]).collect()
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}
