//! NHS Cryptic Site Detection CLI
//!
//! GPU-accelerated cryptic binding site detection using the holographic negative principle.
//!
//! ## Usage
//!
//! ```bash
//! # Run detection on PRISM-PREP topology
//! nhs-detect input_topology.json --output results/
//!
//! # With custom parameters
//! nhs-detect input_topology.json --output results/ --frames 500 --spacing 0.5
//! ```
//!
//! ## Input Requirements
//!
//! **CRITICAL**: Input MUST be a PRISM-PREP sanitized topology JSON.
//! Raw PDB files must go through `prism-prep` first:
//!
//! ```bash
//! prism-prep input.pdb output_topology.json --use-amber --strict
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use prism_nhs::{
    avalanche::CrypticSiteEvent,
    config::NhsConfig,
    input::NhsPreparedInput,
};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::NhsGpuEngine;

/// NHS Cryptic Site Detection - GPU-accelerated holographic water inference
#[derive(Parser, Debug)]
#[command(name = "nhs-detect")]
#[command(author = "PRISM-4D Team")]
#[command(version)]
#[command(about = "Detect cryptic binding sites using neuromorphic holographic streaming")]
struct Args {
    /// Input PRISM-PREP topology JSON file
    #[arg(required = true)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "nhs_results")]
    output: PathBuf,

    /// Number of frames to simulate
    #[arg(short, long, default_value = "200")]
    frames: usize,

    /// Grid spacing in Angstroms
    #[arg(short, long, default_value = "0.5")]
    spacing: f32,

    /// Grid padding around protein (Angstroms)
    #[arg(short, long, default_value = "5.0")]
    padding: f32,

    /// Enable UV bias perturbation for causal validation
    #[arg(long, default_value = "true")]
    uv_bias: bool,

    /// LIF membrane time constant (frames)
    #[arg(long, default_value = "10.0")]
    tau_mem: f32,

    /// Dewetting sensitivity multiplier
    #[arg(long, default_value = "1.0")]
    sensitivity: f32,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: i32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-NHS: Neuromorphic Holographic Stream Detection       ║");
    println!("║              Cryptic Binding Site Discovery                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Verify input exists
    if !args.input.exists() {
        anyhow::bail!(
            "Input file not found: {}\n\
             Note: Input must be a PRISM-PREP topology JSON.\n\
             Run: prism-prep your_structure.pdb output.json --use-amber --strict",
            args.input.display()
        );
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {}", args.output.display()))?;

    log::info!("Loading PRISM-PREP topology: {}", args.input.display());
    let prepared = NhsPreparedInput::load(&args.input, args.spacing, args.padding)?;

    log::info!(
        "Structure: {} atoms, {} residues",
        prepared.topology.n_atoms,
        prepared.topology.n_residues
    );

    // Print atom type statistics
    let stats = prepared.topology.atom_type_stats();
    println!("\nAtom Classification:");
    println!("  Hydrophobic:      {:>5}", stats.get(&prism_nhs::NhsAtomType::Hydrophobic).unwrap_or(&0));
    println!("  Polar:            {:>5}", stats.get(&prism_nhs::NhsAtomType::Polar).unwrap_or(&0));
    println!("  Charged (+):      {:>5}", stats.get(&prism_nhs::NhsAtomType::ChargedPositive).unwrap_or(&0));
    println!("  Charged (-):      {:>5}", stats.get(&prism_nhs::NhsAtomType::ChargedNegative).unwrap_or(&0));
    println!("  Aromatic (UV):    {:>5}", stats.get(&prism_nhs::NhsAtomType::Aromatic).unwrap_or(&0));
    println!("  Backbone:         {:>5}", stats.get(&prism_nhs::NhsAtomType::Backbone).unwrap_or(&0));
    println!();

    // List aromatic residues (UV bias targets)
    let aromatic = prepared.topology.aromatic_residues();
    if !aromatic.is_empty() {
        println!("UV Bias Targets ({} aromatic residues):", aromatic.len());
        for (i, res_id) in aromatic.iter().take(10).enumerate() {
            // Find first atom with this residue ID to get the residue name
            let res_name = prepared.topology.residue_ids.iter()
                .position(|&r| r == *res_id)
                .map(|atom_idx| prepared.topology.residue_names[atom_idx].as_str())
                .unwrap_or("???");
            println!("  {} - Residue {} ({})", i + 1, res_id, res_name);
        }
        if aromatic.len() > 10 {
            println!("  ... and {} more", aromatic.len() - 10);
        }
        println!();
    }

    println!("Grid Configuration:");
    println!("  Dimension:  {}³ voxels", prepared.grid_dim);
    println!("  Spacing:    {:.2} Å", args.spacing);
    println!("  Origin:     [{:.1}, {:.1}, {:.1}]",
             prepared.grid_origin[0], prepared.grid_origin[1], prepared.grid_origin[2]);
    println!("  Frames:     {}", args.frames);
    println!();

    // Run GPU detection
    #[cfg(feature = "gpu")]
    {
        run_gpu_detection(&args, &prepared)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        log::error!("NHS requires GPU support. Compile with --features gpu");
        anyhow::bail!("GPU feature not enabled");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_detection(args: &Args, prepared: &NhsPreparedInput) -> Result<()> {
    println!("Initializing CUDA device {}...", args.device);

    let context = CudaContext::new(args.device as usize)
        .context("Failed to initialize CUDA. Ensure GPU is available.")?;

    log::info!("Creating NHS GPU Engine...");
    let mut engine = NhsGpuEngine::new(
        context,
        prepared.grid_dim,
        prepared.topology.n_atoms,
    )?;

    engine.set_lif_params(args.tau_mem, args.sensitivity);
    engine.initialize(prepared.grid_origin)?;

    println!("Running NHS detection ({} frames)...\n", args.frames);

    let mut total_spikes = 0;
    let mut cryptic_events: Vec<CrypticSiteEvent> = Vec::new();
    let start_time = Instant::now();

    // For now, we run multiple "frames" with the same positions
    // In real use, this would process trajectory frames
    for frame in 0..args.frames {
        let frame_result = engine.process_frame(
            &prepared.positions,
            &prepared.types,
            &prepared.charges,
            &prepared.residues,
        )?;

        total_spikes += frame_result.spike_count;

        if frame_result.spike_count > 0 && args.verbose {
            log::info!(
                "Frame {}: {} spikes detected",
                frame,
                frame_result.spike_count
            );
        }

        // Progress indicator
        if (frame + 1) % 50 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = (frame + 1) as f64 / elapsed;
            println!(
                "  Frame {}/{} ({:.1} fps, {} total spikes)",
                frame + 1,
                args.frames,
                fps,
                total_spikes
            );
        }
    }

    let elapsed = start_time.elapsed();
    let fps = args.frames as f64 / elapsed.as_secs_f64();
    let ms_per_frame = elapsed.as_secs_f64() * 1000.0 / args.frames as f64;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                 ");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Performance:");
    println!("  Total time:       {:.2}s", elapsed.as_secs_f64());
    println!("  Frames/second:    {:.1}", fps);
    println!("  ms/frame:         {:.2}", ms_per_frame);
    println!();

    println!("Detection Summary:");
    println!("  Total spikes:     {}", total_spikes);
    println!("  Avg spikes/frame: {:.1}", total_spikes as f64 / args.frames as f64);
    println!();

    // Get final pocket probability
    let pocket_prob = engine.finalize_pocket_probability(0.05)?;
    let high_prob_voxels: usize = pocket_prob.iter().filter(|&&p| p > 0.5).count();
    let cryptic_voxels: usize = pocket_prob.iter().filter(|&&p| p > 0.7).count();

    println!("Pocket Analysis:");
    println!("  High probability voxels (>0.5): {}", high_prob_voxels);
    println!("  Cryptic site voxels (>0.7):     {}", cryptic_voxels);
    println!();

    // Performance target check
    if ms_per_frame < 2.0 {
        println!("✓ Performance target MET: {:.2} ms/frame < 2.0 ms", ms_per_frame);
    } else {
        println!("✗ Performance target MISSED: {:.2} ms/frame > 2.0 ms", ms_per_frame);
    }

    // Save results
    let output_json = args.output.join("nhs_results.json");
    let results = serde_json::json!({
        "input": args.input.to_string_lossy(),
        "n_atoms": prepared.topology.n_atoms,
        "n_residues": prepared.topology.n_residues,
        "n_frames": args.frames,
        "grid_dim": prepared.grid_dim,
        "grid_spacing": args.spacing,
        "total_spikes": total_spikes,
        "avg_spikes_per_frame": total_spikes as f64 / args.frames as f64,
        "high_prob_voxels": high_prob_voxels,
        "cryptic_voxels": cryptic_voxels,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "ms_per_frame": ms_per_frame,
        "performance_target_met": ms_per_frame < 2.0,
    });

    std::fs::write(&output_json, serde_json::to_string_pretty(&results)?)?;
    println!("\nResults saved to: {}", output_json.display());

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_gpu_detection(_args: &Args, _prepared: &NhsPreparedInput) -> Result<()> {
    anyhow::bail!("GPU feature not enabled. Compile with --features gpu")
}
