//! NHS RT-Full: Complete E2E Pipeline with All RT Capabilities
//!
//! This binary runs the full neuromorphic holographic binding site detection
//! pipeline with all RT-core accelerated features enabled:
//!
//! - RT-accelerated spike clustering (OptiX BVH)
//! - Aromatic proximity analysis
//! - Site persistence tracking
//! - Visualization output (PDB, PyMOL, ChimeraX)
//!
//! Usage:
//!   Single structure:
//!     nhs-rt-full -t topology.json -o output_dir --steps 500000
//!
//!   From Stage 1B manifest (batch mode):
//!     nhs-rt-full --manifest batch_manifest.json -o output_dir --fast

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{
    PersistentBatchConfig, PersistentNhsEngine, CryoUvProtocol,
    ClusteredBindingSite,
    enhance_sites_with_aromatics,
    write_binding_site_visualizations,
    PrismPrepTopology,
    ParallelReplicaEngine,
    sdst_bridge::{SdstBridge, PrismThermAnalysis, ThermClass},
    sdst_report,
    spike_thermodynamic_integration::{compute_binding_free_energy, StiConfig, BindingFreeEnergy},
};

#[derive(Parser, Clone)]
#[command(name = "nhs-rt-full")]
#[command(about = "Full NHS pipeline with RT-core acceleration")]
struct Args {
    /// Topology JSON file (for single structure mode)
    #[arg(short, long, required_unless_present = "manifest")]
    topology: Option<PathBuf>,

    /// Batch manifest from Stage 1B (for batch mode)
    #[arg(long, conflicts_with = "topology")]
    manifest: Option<PathBuf>,

    /// Output directory
    #[arg(short, long, default_value = "rt_full_output")]
    output: PathBuf,

    /// Total simulation steps
    #[arg(long, default_value = "500000")]
    steps: i32,

    /// Target temperature (K)
    #[arg(long, default_value = "300.0")]
    temperature: f32,

    /// Cryo temperature (K)
    #[arg(long, default_value = "50.0")]
    cryo_temp: f32,

    /// Enable RT clustering
    #[arg(long, default_value = "true")]
    rt_clustering: bool,

    /// Cluster matching threshold for persistence tracking (Å)
    #[arg(long, default_value = "5.0")]
    cluster_threshold: f32,

    /// Enable UltimateEngine for 2-4x faster MD (requires SM86+)
    #[arg(long, default_value = "true")]
    ultimate_mode: bool,

    /// Lining residue cutoff distance (Å) - use 8+ to capture catalytic residues
    #[arg(long, default_value = "8.0")]
    lining_cutoff: f32,

    /// Number of replicas to run in parallel (improves sampling accuracy)
    #[arg(long, default_value = "1")]
    replicas: usize,

    /// Base random seed for replica initialization (each replica uses seed + replica_id)
    #[arg(long, default_value = "42")]
    replica_seed: u64,

    /// Enable multi-scale clustering for structure-agnostic detection
    /// Runs clustering at multiple epsilon values and finds persistent sites
    #[arg(long, default_value = "false")]
    multi_scale: bool,

    /// Fast 35K protocol - high-energy UV (42 kcal/mol, +40%) for faster detection
    /// 14K cold + 6K ramp + 15K warm = 35K total, UV burst every 250 steps
    #[arg(long, default_value = "false")]
    fast: bool,

    /// Ultra-fast 25K protocol: 8K cold + 4K ramp + 8K warm = 20K + 5K hysteresis
    /// 55% faster than --fast (35K). For rapid screening of large datasets.
    #[arg(long, default_value = "false")]
    fast_25k: bool,

    /// Enable true parallel replica execution via AmberSimdBatch
    /// All replicas run simultaneously on GPU (vs sequential when disabled)
    #[arg(long, default_value = "false")]
    parallel: bool,

    /// True multi-stream concurrency: N independent CUDA streams each running
    /// the FULL cryo-UV-BNZ-RT pipeline. Creates N PersistentNhsEngine instances
    /// on separate streams for maximum GPU utilization. Results are aggregated
    /// via consensus clustering across all streams.
    #[arg(long, default_value = "0")]
    multi_stream: usize,

    /// Enable adaptive epsilon selection from k-NN distribution
    /// Automatically determines optimal clustering scales per structure
    #[arg(long, default_value = "true")]
    adaptive_epsilon: bool,

    /// Enable CCNS hysteresis protocol: full thermal cycle (cold→hot→cold)
    /// Runs 5-phase protocol for Conformational Crackling Noise Spectroscopy
    #[arg(long, default_value = "false")]
    hysteresis: bool,

    /// Spike intensity percentile filter (0-100). Only keep spikes above this
    /// percentile of intensity. Higher = stricter = fewer spikes.
    /// Default 70 = keep top 30%. Use 90+ to suppress thermal noise.
    #[arg(long, default_value = "70")]
    spike_percentile: u32,

    /// Enable PRISM-Therm: feed spike events into SDST and run hysteresis + CCNS
    /// thermodynamic analysis after the simulation completes.
    /// Produces a "prism_therm" section in the output JSON with per-site
    /// asymmetry scores (heating vs cooling), CCNS tau exponents, and
    /// independently-detected hysteretic pockets.
    /// Best used with --hysteresis (5-phase thermal cycle required for meaningful
    /// asymmetry data, though the flag works without it).
    #[arg(long, default_value = "false")]
    prism_therm: bool,

    /// Enable Hydrogen Mass Repartitioning (HMR).
    /// Redistributes mass from central atoms to bound hydrogens (3x factor),
    /// enabling dt=4fs instead of 2fs for a straight 2x speedup.
    /// SHAKE constraints maintain bond-length accuracy.
    #[arg(long, default_value = "false")]
    hmr: bool,

    /// Fused multi-step: run N AMBER integration steps per 1 multi-LIF
    /// observation step. Since multi-LIF is 99% of GPU time, this gives
    /// ~Nx wall-clock speedup. Use 4 for ~4x, 8 for ~8x.
    /// Default 6 (validated safe for Nyquist sampling of the RAF oscillator).
    #[arg(long, default_value = "6")]
    fused_steps: u32,

    /// Adaptive timestep: use 1.5x dt during hold phases (constant T) where
    /// forces change slowly. Reverts to base dt during ramp phases.
    /// Stacks with HMR: base 4fs → 6fs during holds.
    #[arg(long, default_value = "false")]
    adaptive_dt: bool,

    /// Emit per-site spike_events.json files (large, slow to write).
    /// Off by default to save ~55s per run. Use --emit-spike-json to opt in.
    #[arg(long, default_value = "false")]
    emit_spike_json: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifest Data Structures (for Stage 1B integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Structure entry from manifest
#[derive(Debug, Clone, Deserialize)]
struct ManifestStructure {
    name: String,
    topology_path: String,
    atoms: usize,
    #[allow(dead_code)]
    residues: usize,
    #[allow(dead_code)]
    chains: Vec<String>,
    #[allow(dead_code)]
    memory_tier: String,
    #[allow(dead_code)]
    estimated_gpu_mb: usize,
}

/// Batch entry from manifest
#[derive(Debug, Clone, Deserialize)]
struct ManifestBatch {
    batch_id: usize,
    structures: Vec<ManifestStructure>,
    concurrency: usize,
    memory_tier: String,
    #[allow(dead_code)]
    estimated_total_gpu_mb: usize,
    /// GPU-informed replica count for this batch (defaults to 1 for backward compatibility)
    #[serde(default = "default_replicas")]
    replicas_per_structure: usize,
}

fn default_replicas() -> usize {
    1
}

/// Complete batch manifest from Stage 1B
#[derive(Debug, Clone, Deserialize)]
struct BatchManifest {
    #[allow(dead_code)]
    generated_at: String,
    #[allow(dead_code)]
    gpu_memory_mb: usize,
    replicas: usize,
    total_structures: usize,
    total_batches: usize,
    batches: Vec<ManifestBatch>,
    #[allow(dead_code)]
    execution_order: Vec<String>,
}

/// Result summary for manifest mode
#[derive(Debug, Clone, Serialize)]
struct ManifestRunSummary {
    manifest_path: String,
    total_structures: usize,
    successful: usize,
    failed: usize,
    total_elapsed_seconds: f64,
    results: Vec<StructureRunResult>,
}

#[derive(Debug, Clone, Serialize)]
struct StructureRunResult {
    name: String,
    success: bool,
    error: Option<String>,
    elapsed_seconds: f64,
    sites_found: Option<usize>,
    druggable_sites: Option<usize>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    {
        if let Some(manifest_path) = &args.manifest {
            run_from_manifest(&args, manifest_path)?;
        } else if let Some(topology_path) = &args.topology {
            run_single_structure(&args, topology_path)?;
        } else {
            anyhow::bail!("Either --topology or --manifest must be provided");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        anyhow::bail!("GPU feature required for nhs-rt-full");
    }

    Ok(())
}

/// Run from Stage 1B manifest (batch mode)
/// ONE CudaContext created once. Structures sorted by atom count into size tiers.
/// Each tier gets a right-sized AmberSimdBatch (no padding waste). Tiers run
/// SEQUENTIALLY on the same context. Batch dropped between tiers to free GPU memory.
/// ZERO threads.
#[cfg(feature = "gpu")]
fn run_from_manifest(args: &Args, manifest_path: &PathBuf) -> Result<()> {
    use cudarc::driver::CudaContext;

    let total_start = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     PRISM4D NHS RT-FULL - BATCH MODE (from manifest)          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Load manifest
    log::info!("Loading manifest: {}", manifest_path.display());
    let manifest_content = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("Failed to read manifest: {}", manifest_path.display()))?;
    let manifest: BatchManifest = serde_json::from_str(&manifest_content)
        .context("Failed to parse manifest JSON")?;

    log::info!("Manifest loaded: {} structures in {} batches",
        manifest.total_structures, manifest.total_batches);

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Collect ALL structures from ALL batches into ONE flat list
    let mut all_structures: Vec<ManifestStructure> = Vec::new();
    let mut max_batch_replicas: usize = 0;

    for batch in &manifest.batches {
        let batch_replicas = batch.replicas_per_structure;
        max_batch_replicas = max_batch_replicas.max(batch_replicas);
        log::info!("  Collecting batch {}: {} structures, {} replicas (tier: {})",
            batch.batch_id, batch.structures.len(), batch_replicas, batch.memory_tier);
        all_structures.extend(batch.structures.clone());
    }

    // Determine replica count: CLI override > manifest-level > max per-batch > 1
    let replicas = if args.replicas > 1 {
        args.replicas
    } else if manifest.replicas > 1 {
        manifest.replicas
    } else {
        max_batch_replicas.max(1)
    };

    // Sort ALL structures by atom count for size-tier grouping
    all_structures.sort_by_key(|s| s.atoms);

    // Group into size tiers: small (≤5000), medium (≤20000), large (>20000)
    // Each tier gets a right-sized AmberSimdBatch — no padding waste
    let mut tier_small: Vec<ManifestStructure> = Vec::new();
    let mut tier_medium: Vec<ManifestStructure> = Vec::new();
    let mut tier_large: Vec<ManifestStructure> = Vec::new();

    for structure in &all_structures {
        match structure.atoms {
            0..=5000 => tier_small.push(structure.clone()),
            5001..=20000 => tier_medium.push(structure.clone()),
            _ => tier_large.push(structure.clone()),
        }
    }

    let tiers: Vec<(&str, Vec<ManifestStructure>)> = vec![
        ("small (≤5K atoms)", tier_small),
        ("medium (5K-20K atoms)", tier_medium),
        ("large (>20K atoms)", tier_large),
    ].into_iter().filter(|(_, v)| !v.is_empty()).collect();

    log::info!("SIZE-TIER SEQUENTIAL BATCHING: {} structures across {} tiers",
        all_structures.len(), tiers.len());
    for (name, tier) in &tiers {
        let min_atoms = tier.iter().map(|s| s.atoms).min().unwrap_or(0);
        let max_atoms = tier.iter().map(|s| s.atoms).max().unwrap_or(0);
        log::info!("  Tier {}: {} structures ({}-{} atoms), {} entries with {} replicas",
            name, tier.len(), min_atoms, max_atoms, tier.len() * replicas, replicas);
    }

    // Create ONE CudaContext for the entire run
    log::info!("Creating ONE CudaContext (device 0)...");
    let context = CudaContext::new(0)?;
    log::info!("ONE CudaContext. ZERO threads. {} tiers sequentially.", tiers.len());

    // Run each tier SEQUENTIALLY on the same context
    // Each tier creates a right-sized AmberSimdBatch, runs MD, drops batch to free GPU memory
    let mut all_results: Vec<StructureRunResult> = Vec::new();

    for (tier_idx, (tier_name, tier_structures)) in tiers.iter().enumerate() {
        let tier_start = Instant::now();
        let max_atoms = tier_structures.iter().map(|s| s.atoms).max().unwrap_or(0);

        log::info!("═══ Tier {}/{}: {} ({} structures, max {} atoms) ═══",
            tier_idx + 1, tiers.len(), tier_name, tier_structures.len(), max_atoms);

        // Create right-sized batch for this tier (Arc::clone keeps context alive)
        match run_batch_gpu_concurrent(tier_structures, args, replicas, context.clone()) {
            Ok(tier_results) => {
                let tier_success = tier_results.iter().filter(|r| r.success).count();
                log::info!("  Tier {} complete: {}/{} successful in {:.1}s",
                    tier_name, tier_success, tier_structures.len(),
                    tier_start.elapsed().as_secs_f64());
                all_results.extend(tier_results);
            }
            Err(e) => {
                log::error!("  Tier {} failed: {}", tier_name, e);
                for s in tier_structures {
                    all_results.push(StructureRunResult {
                        name: s.name.clone(),
                        success: false,
                        error: Some(format!("Tier GPU execution failed: {}", e)),
                        elapsed_seconds: 0.0,
                        sites_found: None,
                        druggable_sites: None,
                    });
                }
            }
        }
        // AmberSimdBatch is dropped here — GPU memory freed before next tier
        log::info!("  Tier {} batch dropped, GPU memory freed.", tier_name);
    }

    let successful = all_results.iter().filter(|r| r.success).count();
    let failed = all_results.iter().filter(|r| !r.success).count();

    // Write summary
    let total_elapsed = total_start.elapsed().as_secs_f64();
    let summary = ManifestRunSummary {
        manifest_path: manifest_path.to_string_lossy().to_string(),
        total_structures: manifest.total_structures,
        successful,
        failed,
        total_elapsed_seconds: total_elapsed,
        results: all_results,
    };

    let summary_path = args.output.join("batch_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, &summary_json)?;

    // Print final summary
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    BATCH RUN COMPLETE                         ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Total structures: {:>4}                                       ║", manifest.total_structures);
    println!("║ Successful:       {:>4}                                       ║", successful);
    println!("║ Failed:           {:>4}                                       ║", failed);
    println!("║ Size tiers:       {:>4}                                       ║", tiers.len());
    println!("║ Total time:       {:>6.1}s                                    ║", total_elapsed);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Summary written to: {}", summary_path.display());

    Ok(())
}

/// Run single structure (original behavior)
#[cfg(feature = "gpu")]
fn run_single_structure(args: &Args, topology_path: &PathBuf) -> Result<()> {
    if args.multi_stream > 1 {
        return run_multi_stream_pipeline(args, topology_path, args.multi_stream);
    }
    run_single_structure_internal(topology_path, &args.output, args, args.replicas)?;
    Ok(())
}

/// Internal implementation for running a single structure
#[cfg(feature = "gpu")]
fn run_single_structure_internal(
    topology_path: &PathBuf,
    output_dir: &PathBuf,
    args: &Args,
    replicas: usize,
) -> Result<(usize, usize)> {
    run_full_pipeline_internal(topology_path, output_dir, args, replicas)
}


/// Main pipeline implementation (extracted from original run_full_pipeline)
#[cfg(feature = "gpu")]
fn run_full_pipeline_internal(
    topology_path: &PathBuf,
    output_dir: &PathBuf,
    args: &Args,
    n_replicas: usize,
) -> Result<(usize, usize)> {
    let start_time = Instant::now();

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║     PRISM4D NHS RT-FULL PIPELINE                              ║");
    log::info!("║     RT-Core Accelerated Binding Site Detection                ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // Load topology
    log::info!("\n[1/6] Loading topology: {}", topology_path.display());
    let mut topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

    let structure_name = topology_path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "structure".to_string());

    log::info!("  Atoms: {}", topology.n_atoms);
    log::info!("  Residues: {}", topology.residue_ids.iter().max().unwrap_or(&0) + 1);

    // Apply HMR if requested (must be before engine creation)
    if args.hmr {
        log::info!("  HMR: Applying 3x hydrogen mass repartitioning (dt=4fs)");
        topology.apply_hmr(3.0);
    }

    // Extract aromatic positions for later analysis
    let aromatic_positions = extract_aromatic_positions(&topology);
    log::info!("  Aromatics: {} (TRP/TYR/PHE)", aromatic_positions.len());

    // Minimum atom guard: GPU buffers require >= 500 atoms
    if topology.n_atoms < 500 {
        log::warn!("Protein too small for GPU analysis (minimum 500 atoms, got {})", topology.n_atoms);
        let output_base = output_dir.join(&structure_name);
        let json_path = output_base.with_extension("binding_sites.json");
        let json_output = serde_json::json!({
            "structure": structure_name,
            "total_steps": 0,
            "simulation_time_sec": 0.0,
            "spike_count": 0,
            "binding_sites": 0,
            "druggable_sites": 0,
            "skipped": true,
            "skip_reason": format!("Protein too small for GPU analysis ({} atoms, minimum 500)", topology.n_atoms),
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("Empty result written to {}", json_path.display());
        let total_time = start_time.elapsed();
        log::info!("\n╔═══════════════════════════════════════════════════════════════╗");
        log::info!("║  PIPELINE COMPLETE                                            ║");
        log::info!("╠═══════════════════════════════════════════════════════════════╣");
        log::info!("║  Structure: {:<48} ║", structure_name);
        log::info!("║  Total time: {:<46.1}s ║", total_time.as_secs_f64());
        log::info!("║  SKIPPED: Too few atoms ({:<4} < 500)                         ║", topology.n_atoms);
        log::info!("║  Binding sites: {:<43} ║", 0);
        log::info!("║  Druggable sites: {:<41} ║", 0);
        log::info!("╚═══════════════════════════════════════════════════════════════╝");
        return Ok((0, 0));
    }

    // Initialize engine
    log::info!("\n[2/6] Initializing GPU engine...");
    // Adaptive grid: scale to protein bounding box
    // Small proteins (<500 atoms): 64³
    // Medium (500-2000): 96³
    // Large (2000-5000): 128³
    // Very large (>5000): 160³
    let adaptive_grid_dim = if topology.n_atoms < 500 {
        64
    } else if topology.n_atoms < 2000 {
        96
    } else if topology.n_atoms < 5000 {
        128
    } else {
        160
    };
    log::info!("  Adaptive grid: {}³ for {} atoms", adaptive_grid_dim, topology.n_atoms);
    let config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
        grid_dim: adaptive_grid_dim,
        ..Default::default()
    };

    let mut engine = PersistentNhsEngine::new(&config)?;
    engine.load_topology(&topology)?;

    // Apply HMR timestep if enabled
    if args.hmr {
        engine.set_dt(0.004)?;  // 4fs with HMR masses
    }

    // Apply fused multi-step if requested
    if args.fused_steps > 1 {
        engine.set_fused_inner_steps(args.fused_steps)?;
    }

    // Apply adaptive dt if requested
    if args.adaptive_dt {
        engine.set_adaptive_dt(true)?;
    }

    // Check RT clustering availability
    let has_rt = engine.has_rt_clustering();
    log::info!("  RT-core clustering: {}", if has_rt { "✓ Available" } else { "✗ Fallback mode" });

    // Configure cryo-UV protocol
    let protocol = if args.fast_25k {
        log::info!("  Protocol: Fast 25K (high-energy UV, 42 kcal/mol, burst every 250 steps)");
        CryoUvProtocol::fast_25k()
    } else if args.fast {
        log::info!("  Protocol: Fast 35K (high-energy UV, 42 kcal/mol, burst every 250 steps)");
        CryoUvProtocol::fast_35k()
    } else {
        // Standard protocol with user-configurable temperatures
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: config.cryo_hold,
            ramp_steps: config.convergence_steps / 2,
            warm_hold_steps: config.convergence_steps / 2,
            current_step: 0,
            uv_burst_energy: 50.0,
            uv_burst_interval: 100,
            uv_burst_duration: 50,
            // Full aromatic coverage: TRP, TYR, PHE, HIS (all protonation states)
            scan_wavelengths: vec![280.0, 274.0, 258.0, 254.0, 211.0],
            wavelength_dwell_steps: 500,
            ramp_down_steps: 0,
            cold_return_steps: 0,
        }
    };
    // Apply hysteresis if requested (adds cooling ramp + cold return)
    let protocol = if args.hysteresis {
        log::info!("  CCNS Hysteresis: ENABLED (full thermal cycle {}K → {}K → {}K)",
            protocol.start_temp, protocol.end_temp, protocol.start_temp);
        protocol.with_hysteresis()
    } else {
        protocol
    };
    let target_end_temp = protocol.end_temp;
    engine.set_cryo_uv_protocol(protocol.clone())?;

    // Enable spike accumulation for analysis
    engine.set_spike_accumulation(true);

    // Enable UltimateEngine for faster MD (2-4x speedup on SM86+)
    if args.ultimate_mode {
        match engine.enable_ultimate_mode(&topology) {
            Ok(()) => log::info!("  UltimateEngine: ✓ Enabled (2-4x faster MD)"),
            Err(e) => log::warn!("  UltimateEngine: ✗ Failed to enable: {}", e),
        }
    }

    // Run simulation with replicas for improved sampling
    let n_replicas = n_replicas.max(1);

    // --fast/--fast-25k uses full protocol length (respects hysteresis), otherwise user --steps
    let steps_per_replica = if args.fast || args.fast_25k {
        protocol.total_steps()
    } else {
        args.steps
    };

    log::info!("\n[3/6] Running MD simulation ({} steps x {} replicas)...",
        steps_per_replica, n_replicas);

    let sim_start = Instant::now();
    let mut all_spikes = Vec::new();
    let mut final_temperature = 0.0f32;
    let mut total_snapshots = 0usize;
    let mut all_snapshots: Vec<prism_nhs::fused_engine::EnsembleSnapshot> = Vec::new();

    // Choose parallel or sequential execution
    if args.parallel && n_replicas > 1 {
        // Parallel replica execution via AmberSimdBatch
        log::info!("  Mode: PARALLEL (AmberSimdBatch, {} replicas simultaneous)", n_replicas);

        let mut parallel_engine = ParallelReplicaEngine::new(
            n_replicas,
            &topology,
            protocol.clone(),
        )?;

        let frame_interval = 500; // Extract frames every 500 steps for spike detection
        let result = parallel_engine.run(steps_per_replica as usize, frame_interval)?;

        // Convert parallel spikes to GpuSpikeEvent format
        for spike in result.spikes {
            all_spikes.push(prism_nhs::fused_engine::GpuSpikeEvent {
                timestep: spike.timestep as i32,
                voxel_idx: 0,
                position: spike.position,
                intensity: spike.intensity,
                nearby_residues: [0; 8],
                n_residues: 0,
                spike_source: 0,
                wavelength_nm: 0.0,
                aromatic_type: -1,
                aromatic_residue_id: -1,
                water_density: 0.0,
                vibrational_energy: 0.0,
                n_nearby_excited: 0,
                wd_change: 0.0,
            });
        }

        final_temperature = target_end_temp;
        total_snapshots = 0; // Not tracked in parallel mode

        log::info!("  ✓ Parallel complete: {:.1}s ({:.0} steps/sec aggregate)",
            result.elapsed_seconds, result.throughput);
    } else {
        // Sequential replica execution (original behavior)
        if n_replicas > 1 {
            log::info!("  Mode: SEQUENTIAL ({} replicas one at a time)", n_replicas);
        }

        for replica_id in 0..n_replicas {
            let replica_seed = args.replica_seed + replica_id as u64;

            if n_replicas > 1 {
                log::info!("  Replica {}/{} (seed: {})...", replica_id + 1, n_replicas, replica_seed);

                // Reset engine state for each replica (re-initialize with different seed)
                engine.reset_for_replica(replica_seed)?;
            }

            let summary = engine.run(steps_per_replica)?;

            // Collect spikes from this replica
            let replica_spikes = engine.get_accumulated_spikes();
            let spike_count = replica_spikes.len();
            all_spikes.extend(replica_spikes);

            final_temperature = summary.end_temperature;
            let snapshots = engine.get_snapshots();
            total_snapshots += snapshots.len();
            all_snapshots.extend(snapshots);

            if n_replicas > 1 {
                log::info!("    Replica {} complete: {} spikes, T={:.1}K",
                    replica_id + 1, spike_count, summary.end_temperature);
            }
        }

        let sim_time_seq = sim_start.elapsed();
        let total_steps_seq = steps_per_replica as usize * n_replicas;
        log::info!("  ✓ Completed in {:.1}s ({:.0} steps/sec)",
            sim_time_seq.as_secs_f64(),
            total_steps_seq as f64 / sim_time_seq.as_secs_f64());
    }

    let sim_time = sim_start.elapsed();
    let _total_steps = steps_per_replica as usize * n_replicas;

    log::info!("  Raw spikes collected: {} (from {} replicas)", all_spikes.len(), n_replicas);
    log::info!("  Snapshots: {}", total_snapshots);
    log::info!("  Final temperature: {:.1}K", final_temperature);

    // Intensity pre-filtering: keep only spikes above --spike-percentile
    let pct = (args.spike_percentile.min(99) as f32) / 100.0;
    let accumulated_spikes = if all_spikes.len() > 1000 {
        let mut intensities: Vec<f32> = all_spikes.iter()
            .map(|s| s.intensity)
            .collect();
        intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (intensities.len() as f32 * pct) as usize;
        let intensity_threshold = intensities.get(threshold_idx).copied().unwrap_or(0.0);

        let filtered: Vec<_> = all_spikes.into_iter()
            .filter(|s| s.intensity >= intensity_threshold)
            .collect();
        log::info!("  Intensity filter: kept {} spikes (top {}%, threshold={:.2})",
            filtered.len(), 100 - args.spike_percentile, intensity_threshold);
        filtered
    } else {
        all_spikes
    };

    // RT-accelerated spike clustering
    let cluster_mode = if args.multi_scale { "multi-scale" } else { "single-scale" };
    log::info!("\n[4/6] RT-accelerated spike clustering ({})...", cluster_mode);

    // Track epsilon info for JSON output (outside block for scope)
    let mut epsilon_info: Option<(Vec<f32>, bool, Option<usize>, Option<usize>)> = None;

    let mut clustered_sites = if !accumulated_spikes.is_empty() && args.rt_clustering {
        // Copy positions from packed struct to avoid alignment issues
        let positions: Vec<f32> = accumulated_spikes.iter()
            .flat_map(|s| {
                let pos = s.position;  // Copy packed field
                [pos[0], pos[1], pos[2]].into_iter()
            })
            .collect();

        let cluster_start = Instant::now();

        if args.multi_scale {
            // Multi-scale clustering for structure-agnostic detection
            // Use adaptive epsilon (k-NN based) or fixed values
            let custom_epsilon = if args.adaptive_epsilon {
                None // Let the engine compute from k-NN distribution
            } else {
                Some(vec![1.5f32, 2.5, 3.5, 5.0]) // Tighter for high-density runs
            };
            match engine.multi_scale_cluster_spikes_with_epsilon(&positions, custom_epsilon) {
                Ok(ms_result) => {
                    log::info!("  ✓ Multi-scale clustering complete: {} persistent clusters",
                        ms_result.num_clusters());

                    // Capture epsilon info for JSON output
                    epsilon_info = Some((
                        ms_result.epsilon_values.clone(),
                        ms_result.adaptive_epsilon,
                        ms_result.knn_k,
                        ms_result.num_spikes_sampled,
                    ));

                    // Convert multi-scale result to cluster IDs for site building
                    let cluster_ids = ms_result.to_cluster_ids(accumulated_spikes.len());
                    let fake_result = prism_nhs::rt_clustering::RtClusteringResult {
                        cluster_ids,
                        num_clusters: ms_result.num_clusters(),
                        total_neighbors: 0, // Not tracked in multi-scale
                        gpu_time_ms: cluster_start.elapsed().as_secs_f64() * 1000.0,
                    };

                    // Build clustered binding sites
                    let all_sites = build_sites_from_clustering(&accumulated_spikes, &fake_result);

                    // Post-filter: keep only significant clusters (min 2% of total spikes)
                    // Adaptive min-spikes: scale with aromatic count so that
                    // per-aromatic clusters survive filtering in large proteins.
                    // min = max(50, 0.3 * spikes_per_aromatic)
                    let n_arom = aromatic_positions.len().max(1);
                    let spikes_per_arom = accumulated_spikes.len() as f64 / n_arom as f64;
                    let min_spikes = (spikes_per_arom * 0.3).ceil().max(50.0) as usize;
                    let sites: Vec<_> = all_sites.into_iter()
                        .filter(|s| s.spike_count >= min_spikes)
                        .collect();
                    log::info!("  Binding sites: {} (filtered, min {} spikes = 2%)",
                        sites.len(), min_spikes);
                    sites
                }
                Err(e) => {
                    log::warn!("  ⚠ Multi-scale clustering failed: {}", e);
                    Vec::new()
                }
            }
        } else {
            // Single-scale clustering (original behavior)
            match engine.cluster_spikes(&positions) {
                Ok(mut result) => {
                    log::info!("  ✓ Clustering complete: {} clusters, {} neighbor pairs, {:.2}ms",
                        result.num_clusters, result.total_neighbors, result.gpu_time_ms);

                    // ── Mega-cluster subdivision via voxel density peaks ──
                    // When a single DBSCAN cluster absorbs >50% of all spikes,
                    // the protein is compact enough that all spike clouds form one
                    // density-connected component (percolation).  Instead of
                    // re-running DBSCAN at tighter epsilon (which causes a phase
                    // transition from one mega-cluster to dust), we find density
                    // peaks on a 3D voxel grid and partition spikes around them.
                    let total_spikes = accumulated_spikes.len();
                    let mega_threshold = (total_spikes as f64 * 0.50) as usize;
                    {
                        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
                        for &cid in &result.cluster_ids {
                            if cid >= 0 {
                                *counts.entry(cid).or_insert(0) += 1;
                            }
                        }
                        let mega = counts.iter()
                            .max_by_key(|(_, &c)| c)
                            .map(|(&id, &c)| (id, c));

                        if let Some((mega_id, mega_count)) = mega {
                            if mega_count > mega_threshold {
                                log::info!("  Mega-cluster {} detected: {} spikes ({:.0}% of total)",
                                    mega_id, mega_count,
                                    mega_count as f64 / total_spikes as f64 * 100.0);
                                log::info!("  Applying voxel density peak subdivision...");

                                // Extract mega-cluster spike indices
                                let mega_indices: Vec<usize> = result.cluster_ids.iter()
                                    .enumerate()
                                    .filter(|(_, &cid)| cid == mega_id)
                                    .map(|(i, _)| i)
                                    .collect();

                                // Compute bounding box
                                let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
                                let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
                                for &i in &mega_indices {
                                    let (x, y, z) = (positions[i*3], positions[i*3+1], positions[i*3+2]);
                                    min_x = min_x.min(x); min_y = min_y.min(y); min_z = min_z.min(z);
                                    max_x = max_x.max(x); max_y = max_y.max(y); max_z = max_z.max(z);
                                }

                                // Voxel grid: 3Å cells (roughly 1 aromatic spike cloud diameter)
                                let cell = 3.0_f32;
                                let nx = ((max_x - min_x) / cell).ceil() as usize + 1;
                                let ny = ((max_y - min_y) / cell).ceil() as usize + 1;
                                let nz = ((max_z - min_z) / cell).ceil() as usize + 1;

                                // Count spikes per voxel
                                let mut grid = vec![0u32; nx * ny * nz];
                                let voxel_idx = |x: f32, y: f32, z: f32| -> usize {
                                    let ix = ((x - min_x) / cell) as usize;
                                    let iy = ((y - min_y) / cell) as usize;
                                    let iz = ((z - min_z) / cell) as usize;
                                    ix.min(nx-1) + iy.min(ny-1) * nx + iz.min(nz-1) * nx * ny
                                };

                                for &i in &mega_indices {
                                    let vi = voxel_idx(positions[i*3], positions[i*3+1], positions[i*3+2]);
                                    grid[vi] += 1;
                                }

                                // Find density peaks: voxels that are local maxima among 26 neighbors
                                let mut peaks: Vec<(usize, u32)> = Vec::new(); // (voxel_idx, count)
                                for iz in 0..nz {
                                    for iy in 0..ny {
                                        for ix in 0..nx {
                                            let vi = ix + iy * nx + iz * nx * ny;
                                            let c = grid[vi];
                                            if c == 0 { continue; }
                                            let mut is_peak = true;
                                            for dz in -1i32..=1 {
                                                for dy in -1i32..=1 {
                                                    for dx in -1i32..=1 {
                                                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                                                        let (jx, jy, jz) = (ix as i32 + dx, iy as i32 + dy, iz as i32 + dz);
                                                        if jx >= 0 && jx < nx as i32 && jy >= 0 && jy < ny as i32 && jz >= 0 && jz < nz as i32 {
                                                            let ji = jx as usize + jy as usize * nx + jz as usize * nx * ny;
                                                            if grid[ji] > c {
                                                                is_peak = false;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            if is_peak {
                                                peaks.push((vi, c));
                                            }
                                        }
                                    }
                                }

                                // Sort peaks by density (descending) and filter weak ones
                                peaks.sort_by(|a, b| b.1.cmp(&a.1));
                                let peak_threshold = if let Some(top) = peaks.first() {
                                    (top.1 as f32 * 0.05) as u32  // keep peaks with >5% of max density
                                } else {
                                    0
                                };
                                let peaks: Vec<_> = peaks.into_iter().filter(|(_, c)| *c >= peak_threshold.max(10)).collect();

                                if peaks.len() >= 2 {
                                    // Compute peak centers in Angstrom coordinates
                                    let peak_centers: Vec<[f32; 3]> = peaks.iter().map(|&(vi, _)| {
                                        let iz = vi / (nx * ny);
                                        let iy = (vi % (nx * ny)) / nx;
                                        let ix = vi % nx;
                                        [
                                            min_x + (ix as f32 + 0.5) * cell,
                                            min_y + (iy as f32 + 0.5) * cell,
                                            min_z + (iz as f32 + 0.5) * cell,
                                        ]
                                    }).collect();

                                    // Assign each mega-cluster spike to nearest peak
                                    let max_existing = result.cluster_ids.iter().max().copied().unwrap_or(0);
                                    for &i in &mega_indices {
                                        let (x, y, z) = (positions[i*3], positions[i*3+1], positions[i*3+2]);
                                        let mut best_peak = 0usize;
                                        let mut best_d2 = f32::MAX;
                                        for (pi, pc) in peak_centers.iter().enumerate() {
                                            let d2 = (x - pc[0]).powi(2) + (y - pc[1]).powi(2) + (z - pc[2]).powi(2);
                                            if d2 < best_d2 {
                                                best_d2 = d2;
                                                best_peak = pi;
                                            }
                                        }
                                        result.cluster_ids[i] = max_existing + 1 + best_peak as i32;
                                    }
                                    let new_max = result.cluster_ids.iter().max().copied().unwrap_or(0);
                                    result.num_clusters = (new_max + 1) as usize;

                                    log::info!("  Voxel grid: {}x{}x{} (cell={:.1}Å), {} density peaks found",
                                        nx, ny, nz, cell, peak_centers.len());
                                    for (pi, (pc, &(_, count))) in peak_centers.iter().zip(peaks.iter()).enumerate().take(10) {
                                        log::info!("    Peak {}: ({:.1}, {:.1}, {:.1}) density={}",
                                            pi, pc[0], pc[1], pc[2], count);
                                    }
                                } else {
                                    log::info!("  Only {} density peak(s) found; keeping original mega-cluster", peaks.len());
                                }
                            }
                        }
                    }

                    // Build clustered binding sites
                    let all_sites = build_sites_from_clustering(&accumulated_spikes, &result);

                    // Post-filter: keep only significant clusters (min 2% of total spikes)
                    // Adaptive min-spikes: scale with aromatic count so that
                    // per-aromatic clusters survive filtering in large proteins.
                    // min = max(50, 0.3 * spikes_per_aromatic)
                    let n_arom = aromatic_positions.len().max(1);
                    let spikes_per_arom = accumulated_spikes.len() as f64 / n_arom as f64;
                    let min_spikes = (spikes_per_arom * 0.3).ceil().max(50.0) as usize;
                    let sites: Vec<_> = all_sites.into_iter()
                        .filter(|s| s.spike_count >= min_spikes)
                        .collect();
                    log::info!("  Binding sites: {} (filtered from {} clusters, min {} spikes = 2%)",
                        sites.len(), result.num_clusters, min_spikes);
                    sites
                }
                Err(e) => {
                    log::warn!("  ⚠ Clustering failed: {}", e);
                    Vec::new()
                }
            }
        }
    } else {
        log::info!("  Skipped (no spikes or RT disabled)");
        Vec::new()
    };

    // Dimer-aware merge for single-stream path
    merge_symmetric_sites(&mut clustered_sites, &accumulated_spikes, &topology, 12.0);

    // Aromatic proximity analysis
    log::info!("\n[5/6] Aromatic proximity analysis...");
    if !clustered_sites.is_empty() && !aromatic_positions.is_empty() {
        enhance_sites_with_aromatics(&mut clustered_sites, &aromatic_positions);

        let druggable_count = clustered_sites.iter().filter(|s| s.druggability.is_druggable).count();
        log::info!("  ✓ Analyzed {} sites, {} druggable", clustered_sites.len(), druggable_count);

        // Create mapping from internal index to PDB ID
        let mut pdb_id_map = Vec::new();
        if !topology.residues.is_empty() {
            let max_idx = topology.residues.iter().map(|r| r.residue_idx).max().unwrap_or(0);
            pdb_id_map.resize(max_idx + 1, 0);
            for r in &topology.residues {
                if r.residue_idx < pdb_id_map.len() {
                    pdb_id_map[r.residue_idx] = r.residue_id;
                }
            }
        } else {
            // Fallback if metadata is missing
            let max_id = *topology.residue_ids.iter().max().unwrap_or(&0);
            pdb_id_map = (0..=max_id).map(|i| i as i32).collect();
        }
        // Compute lining residues for top sites (limit to top 100 for performance)
        let lining_cutoff = args.lining_cutoff;
        for site in clustered_sites.iter_mut().take(100) {
            site.compute_lining_residues(
                &topology.positions,
                &topology.residue_ids,
                &topology.residue_names,
                &topology.chain_ids,
                &pdb_id_map,
                lining_cutoff,
            );
        }

        // Log top sites with residue info (highlighting catalytic residues)
        let catalytic_residues = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
        for (i, site) in clustered_sites.iter().take(5).enumerate() {
            let res_str = if site.lining_residues.is_empty() {
                "no residues".to_string()
            } else {
                site.lining_residues_str()
            };
            // Count catalytic residues
            let catalytic_count = site.lining_residues.iter()
                .filter(|r| catalytic_residues.contains(&r.resname.as_str()))
                .count();
            log::info!("    #{}: {:?} at ({:.1}, {:.1}, {:.1}), quality={:.2}, druggable={}",
                i + 1,
                site.classification,
                site.centroid[0], site.centroid[1], site.centroid[2],
                site.quality_score,
                site.druggability.is_druggable);
            log::info!("        Residues ({}, {} catalytic): {}",
                site.lining_residues.len(),
                catalytic_count,
                if res_str.len() > 70 { format!("{}...", &res_str[..67]) } else { res_str });
            // Log catalytic residues specifically if any
            if catalytic_count > 0 {
                let cat_list: Vec<_> = site.lining_residues.iter()
                    .filter(|r| catalytic_residues.contains(&r.resname.as_str()))
                    .map(|r| format!("{}:{}{} ({:.1}Å)", r.chain, r.resname, r.resid, r.min_distance))
                    .collect();
                log::info!("        Catalytic: {}", cat_list.join(", "));
            }
        }
    } else {
        log::info!("  Skipped (no sites or aromatics)");
    }

    // Generate visualization output
    log::info!("\n[6/6] Generating visualization output...");
    let output_base = output_dir.join(&structure_name);

    if !clustered_sites.is_empty() {
        write_binding_site_visualizations(&clustered_sites, &output_base, &structure_name)?;

        // Also write JSON summary (with lining residues for top 100 sites)
        let json_path = output_base.with_extension("binding_sites.json");
        let catalytic_residues = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];

        // Build adaptive epsilon info for JSON
        let epsilon_json = if let Some((values, is_adaptive, knn_k, num_sampled)) = &epsilon_info {
            serde_json::json!({
                "computed_values": values,
                "source": if *is_adaptive { "knn_adaptive" } else { "fixed" },
                "knn_k": knn_k,
                "num_spikes_sampled": num_sampled,
            })
        } else {
            serde_json::json!({
                "computed_values": null,
                "source": "single_scale",
                "knn_k": null,
                "num_spikes_sampled": null,
            })
        };

        // Compute per-site CCNS time-series data from spike events
        // FRAME ALIGNMENT FIX: Use cluster-assigned spikes from spike_indices,
        // not radius queries. Single-replica path always has spike_indices populated.
        let site_radius = args.lining_cutoff + 2.0;
        let frame_window = 1000i32;
        let mut all_pockets_json = Vec::new();
        let mut cryptic_sites_json = Vec::new();

        for site in clustered_sites.iter().take(100) {
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];

            // Use cluster-assigned spikes (frame-aligned with sites[].spike_count)
            let site_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> = if !site.spike_indices.is_empty() {
                site.spike_indices.iter()
                    .filter_map(|&idx| accumulated_spikes.get(idx))
                    .collect()
            } else {
                // Fallback: assign to nearest centroid (shouldn't happen in single-replica)
                accumulated_spikes.iter()
                    .filter(|s| {
                        let dx = s.position[0] - cx;
                        let dy = s.position[1] - cy;
                        let dz = s.position[2] - cz;
                        (dx*dx + dy*dy + dz*dz).sqrt() <= site_radius
                    })
                    .collect()
            };

            let max_ts = site_spikes.iter().map(|s| s.timestep).max().unwrap_or(0);
            let n_frames = (max_ts / frame_window + 1) as usize;
            let mut frame_spike_counts = vec![0usize; n_frames];
            let mut frame_intensity_sums = vec![0.0f32; n_frames];

            for s in &site_spikes {
                let frame = (s.timestep / frame_window) as usize;
                if frame < n_frames {
                    frame_spike_counts[frame] += 1;
                    frame_intensity_sums[frame] += s.intensity;
                }
            }

            let voxel_vol = 27.0f32;
            let volumes: Vec<f64> = frame_spike_counts.iter()
                .map(|&c| (c as f32 * voxel_vol) as f64)
                .collect();
            let mean_volume: f64 = if !volumes.is_empty() {
                volumes.iter().sum::<f64>() / volumes.len() as f64
            } else { 0.0 };
            let cv_volume = if mean_volume > 0.0 {
                let variance = volumes.iter().map(|v| (v - mean_volume).powi(2)).sum::<f64>()
                    / volumes.len() as f64;
                variance.sqrt() / mean_volume
            } else { 0.0 };

            all_pockets_json.push(serde_json::json!({
                "site_id": site.cluster_id,
                "centroid": site.centroid,
                "mean_volume": mean_volume,
                "cv_volume": cv_volume,
                "n_frames": n_frames,
                "volumes": volumes,
            }));

            let spike_frames: Vec<usize> = frame_spike_counts.iter().enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(i, _)| i)
                .collect();
            let spike_amplitudes: Vec<f32> = spike_frames.iter()
                .map(|&f| {
                    if frame_spike_counts[f] > 0 {
                        frame_intensity_sums[f] / frame_spike_counts[f] as f32
                    } else { 0.0 }
                })
                .collect();
            let inter_spike_intervals: Vec<f32> = if spike_frames.len() >= 2 {
                spike_frames.windows(2).map(|w| (w[1] - w[0]) as f32).collect()
            } else {
                Vec::new()
            };

            cryptic_sites_json.push(serde_json::json!({
                "site_id": site.cluster_id,
                "centroid": site.centroid,
                "spike_count": site_spikes.len(),
                "consensus_spike_count": site.spike_count,
                "spike_source": if !site.spike_indices.is_empty() { "cluster_assigned" } else { "radius_fallback" },
                "spike_frames": spike_frames,
                "spike_amplitudes": spike_amplitudes,
                "inter_spike_intervals": inter_spike_intervals,
                "volume": site.estimated_volume,
                "druggability": site.druggability.overall,
                "classification": format!("{:?}", site.classification),
            }));
        }

        // ── PRISM-Therm: SDST hysteresis + CCNS analysis ──
        let prism_therm_result: Option<PrismThermAnalysis> = if args.prism_therm {
            log::info!("\n[PRISM-Therm] Initializing SDST thermodynamic analysis...");
            match SdstBridge::new(&topology, &protocol, accumulated_spikes.len()) {
                Err(e) => {
                    log::warn!("  PRISM-Therm: SDST init failed ({}), skipping", e);
                    None
                }
                Ok(bridge) => {
                    match bridge.ingest_all_spikes(&accumulated_spikes) {
                        Err(e) => {
                            log::warn!("  PRISM-Therm: spike ingestion failed ({}), skipping", e);
                            None
                        }
                        Ok(event_count) => {
                            log::info!("  PRISM-Therm: {} events ingested into SDST", event_count);
                            match bridge.analyze(&clustered_sites) {
                                Err(e) => {
                                    log::warn!("  PRISM-Therm: analysis failed ({})", e);
                                    None
                                }
                                Ok(analysis) => {
                                    log::info!("  PRISM-Therm: {} hysteretic / {} NHS sites | {} SDST global pockets",
                                        analysis.hysteretic_site_count,
                                        clustered_sites.len(),
                                        analysis.global_pockets.len());
                                    Some(analysis)
                                }
                            }
                        }
                    }
                }
            }
        } else {
            None
        };

        // Build per-site JSON, merging PRISM-Therm therm_class when available
        let mut sites_json: Vec<serde_json::Value> = clustered_sites.iter().take(100).map(|s| {
            let catalytic_count = s.lining_residues.iter()
                .filter(|r| catalytic_residues.contains(&r.resname.as_str()))
                .count();
            serde_json::json!({
                "id": s.cluster_id,
                "centroid": s.centroid,
                "volume": s.estimated_volume,
                "spike_count": s.spike_count,
                "quality_score": s.quality_score,
                "druggability": s.druggability.overall,
                "is_druggable": s.druggability.is_druggable,
                "classification": format!("{:?}", s.classification),
                "aromatic_score": s.aromatic_proximity.as_ref().map(|p| p.aromatic_score),
                "catalytic_residue_count": catalytic_count,
                "lining_residues": s.lining_residues.iter().map(|r| {
                    let is_catalytic = catalytic_residues.contains(&r.resname.as_str());
                    serde_json::json!({
                        "chain": r.chain,
                        "resid": r.resid,
                        "resname": r.resname,
                        "min_distance": r.min_distance,
                        "n_atoms": r.n_atoms_in_pocket,
                        "is_catalytic": is_catalytic,
                    })
                }).collect::<Vec<_>>(),
                "residue_ids": s.lining_residue_ids(),
            })
        }).collect();

        // Inject PRISM-Therm classification into each site (authoritative physics-based)
        if let Some(ref analysis) = prism_therm_result {
            for site_json in sites_json.iter_mut() {
                let site_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
                if let Some(therm_site) = analysis.sites.iter().find(|s| s.site_id == site_id) {
                    site_json["therm_class"] = serde_json::Value::String(
                        therm_site.therm_class.to_string()
                    );
                    site_json["hysteresis_asymmetry"] = serde_json::json!(therm_site.asymmetry_score);
                    site_json["relative_asymmetry"] = serde_json::json!(therm_site.relative_asymmetry);
                    site_json["ccns_tau"] = serde_json::json!(therm_site.tau);
                    // Override heuristic classification if PRISM-Therm says CRYPTIC
                    if therm_site.therm_class.to_string() == "CRYPTIC" {
                        site_json["classification"] = serde_json::Value::String("Cryptic".to_string());
                    }
                }
            }
        }

        let json_output = serde_json::json!({
            "structure": structure_name,
            "total_steps": steps_per_replica,
            "simulation_time_sec": sim_time.as_secs_f64(),
            "spike_count": accumulated_spikes.len(),
            "binding_sites": clustered_sites.len(),
            "druggable_sites": clustered_sites.iter().filter(|s| s.druggability.is_druggable).count(),
            "lining_residue_cutoff_angstroms": args.lining_cutoff,
            "adaptive_epsilon": epsilon_json,
            "sites": sites_json,
            "all_pockets": all_pockets_json,
            "cryptic_sites": cryptic_sites_json,
            "prism_therm": prism_therm_result,
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON summary: {}", json_path.display());

        // ── PRISM-Therm standalone report ──
        if let Some(ref analysis) = prism_therm_result {
            let site_centroids: Vec<([f32; 3], i32)> = clustered_sites.iter()
                .map(|s| (s.centroid, s.cluster_id))
                .collect();
            let report = sdst_report::build_report(analysis, &topology, &structure_name, &site_centroids);
            sdst_report::print_summary_table(&report);
            if let Err(e) = sdst_report::write_json(&report, output_dir, &structure_name) {
                log::warn!("  PRISM-Therm JSON write failed: {}", e);
            }
            if let Err(e) = sdst_report::write_druggability_pdb(&report, &topology, output_dir, &structure_name) {
                log::warn!("  PRISM-Therm druggability PDB failed: {}", e);
            }
        }

        // Write ensemble trajectory PDB
        write_ensemble_trajectory(&all_snapshots, &topology, &output_base)?;
    }

    // Final summary
    let total_time = start_time.elapsed();
    log::info!("\n╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  PIPELINE COMPLETE                                            ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    log::info!("║  Structure: {:<48} ║", structure_name);
    log::info!("║  Total time: {:<46.1}s ║", total_time.as_secs_f64());
    log::info!("║  Spikes detected: {:<41} ║", accumulated_spikes.len());
    log::info!("║  Binding sites: {:<43} ║", clustered_sites.len());
    log::info!("║  Druggable sites: {:<41} ║",
        clustered_sites.iter().filter(|s| s.druggability.is_druggable).count());
    log::info!("║  RT cores used: {:<43} ║", if has_rt { "Yes" } else { "No" });
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // Return counts for manifest mode
    let total_sites = clustered_sites.len();
    let druggable_sites = clustered_sites.iter().filter(|s| s.druggability.is_druggable).count();

    Ok((total_sites, druggable_sites))
}

/// Run batch of structures on GPU using AmberSimdBatch with an externally-provided CudaContext.
/// The batch is right-sized to max_atoms of the provided structures (no padding waste).
/// Caller creates ONE CudaContext and passes it to each tier.
#[cfg(feature = "gpu")]
fn run_batch_gpu_concurrent(
    structures: &[ManifestStructure],
    args: &Args,
    replicas: usize,
    context: std::sync::Arc<cudarc::driver::CudaContext>,
) -> Result<Vec<StructureRunResult>> {
    use prism_gpu::{AmberSimdBatch, OptimizationConfig};
    use prism_nhs::fused_engine::GpuSpikeEvent;

    let batch_start = Instant::now();

    // Find max atoms for batch sizing — right-sized to this tier
    let max_atoms = structures.iter().map(|s| s.atoms).max().unwrap_or(0);
    let n_structures = structures.len();
    let total_entries = n_structures * replicas;

    log::info!("    Creating AmberSimdBatch: {} structures × {} replicas = {} total entries, max {} atoms",
        n_structures, replicas, total_entries, max_atoms);

    // Use MAXIMUM config: Verlet + Tensor Cores + FP16 + Async pipeline
    // RTX 5080 Blackwell has 5th gen Tensor Cores - use them
    let opt_config = OptimizationConfig::maximum();
    let mut batch = AmberSimdBatch::new_with_config(
        context,
        max_atoms,
        total_entries,
        opt_config,
    )?;

    // Load each structure topology and extract aromatic positions
    // We track (structure_idx, replica_idx) for each batch entry
    let mut entry_mapping = Vec::new(); // Vec<(structure_idx, replica_idx)>
    let mut max_atoms_seen: usize = 0;
    let mut structure_ids = Vec::new();
    let mut topologies = Vec::new();
    let mut aromatic_positions_per_structure = Vec::new();
    let mut aromatic_indices_per_structure = Vec::new();

    for (struct_idx, structure) in structures.iter().enumerate() {
        let topology_path = PathBuf::from(&structure.topology_path);
        let topology = PrismPrepTopology::load(&topology_path)
            .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

        // Extract aromatic positions for UV burst targeting
        max_atoms_seen = max_atoms_seen.max(topology.n_atoms);
        let aromatic_positions = extract_aromatic_positions(&topology);

        // Extract aromatic atom indices for spike detection
        let aromatic_residue_ids = topology.aromatic_residues();
        let aromatic_residues: std::collections::HashSet<usize> = aromatic_residue_ids.into_iter().collect();
        let aromatic_indices: Vec<usize> = topology.residue_ids
            .iter()
            .enumerate()
            .filter(|(_, &res_id)| aromatic_residues.contains(&res_id))
            .map(|(atom_idx, _)| atom_idx)
            .collect();

        // Convert to StructureTopology format
        let struct_topo = prism_nhs::simd_batch_integration::convert_to_structure_topology(&topology)?;

        // Add N replicas of this structure
        for replica_idx in 0..replicas {
            let id = batch.add_structure(&struct_topo)?;
            structure_ids.push(id);
            entry_mapping.push((struct_idx, replica_idx));

            if replica_idx == 0 {
                log::info!("      Loaded: {} ({} atoms, {} aromatics) → {} replicas starting at ID {}",
                    structure.name, structure.atoms, aromatic_indices.len(), replicas, id);
            }
        }

        topologies.push(topology);
        aromatic_positions_per_structure.push(aromatic_positions);
        aromatic_indices_per_structure.push(aromatic_indices);
    }

    // Finalize batch
    log::info!("    Finalizing batch for GPU upload...");
    batch.finalize_batch()?;
    log::info!("      ✓ Batch ready: {} total entries ({} structures × {} replicas) on GPU",
        total_entries, n_structures, replicas);

    // Configure protocol (steps determined after hysteresis decision)
    let protocol = if args.fast_25k {
        let base = CryoUvProtocol::fast_25k();
        let extra_warm = ((max_atoms_seen.saturating_sub(5000) / 1000) * 2000) as i32;
        let protocol_sized = CryoUvProtocol { warm_hold_steps: base.warm_hold_steps + extra_warm, ..base };
        log::info!("  Adaptive warm_hold: {} steps ({} atoms, +{} extra)",
            protocol_sized.warm_hold_steps, max_atoms_seen, extra_warm);
        protocol_sized
    } else if args.fast {
        let base = CryoUvProtocol::fast_35k();
        let extra_warm = ((max_atoms_seen.saturating_sub(5000) / 1000) * 2000) as i32;
        let protocol_sized = CryoUvProtocol { warm_hold_steps: base.warm_hold_steps + extra_warm, ..base };
        log::info!("  Adaptive warm_hold: {} steps ({} atoms, +{} extra)",
            protocol_sized.warm_hold_steps, max_atoms_seen, extra_warm);
        protocol_sized
    } else {
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: 50000,
            ramp_steps: args.steps / 4,
            warm_hold_steps: args.steps / 4,
            current_step: 0,
            uv_burst_energy: 50.0,
            uv_burst_interval: 100,
            uv_burst_duration: 50,
            scan_wavelengths: vec![280.0, 274.0, 258.0, 254.0, 211.0],
            wavelength_dwell_steps: 500,
            ramp_down_steps: 0,
            cold_return_steps: 0,
        }
    };
    let protocol = if args.hysteresis {
        protocol.with_hysteresis()
    } else {
        protocol
    };

    // Determine steps: --fast/--fast-25k uses full protocol length (respects hysteresis)
    let steps_per_structure = if args.fast || args.fast_25k {
        protocol.total_steps() as usize
    } else {
        args.steps as usize
    };

    // Compute simulation phases
    let total_protocol_steps = protocol.total_steps() as usize;

    let scale = if steps_per_structure < total_protocol_steps {
        steps_per_structure as f64 / total_protocol_steps as f64
    } else {
        1.0
    };

    let cold_steps = ((protocol.cold_hold_steps as f64 * scale) as usize).max(100);
    let ramp_steps = ((protocol.ramp_steps as f64 * scale) as usize).max(100);
    let warm_steps = steps_per_structure.saturating_sub(cold_steps + ramp_steps);

    log::info!("    Running {} steps per replica (batch executes in lockstep)...", steps_per_structure);
    log::info!("      Protocol phases: cold={}, ramp={}, warm={}", cold_steps, ramp_steps, warm_steps);

    // Initialize spike storage per entry (structure × replica)
    let frame_interval = 500;
    let mut all_entry_spikes: Vec<Vec<GpuSpikeEvent>> = vec![Vec::new(); total_entries];
    let mut previous_positions: Vec<Vec<f32>> = vec![Vec::new(); total_entries];

    // Track timestep
    let mut current_step = 0usize;
    let dt = 0.002f32;
    let gamma = 1.0f32;

    // Phase 1: Cold hold
    log::info!("    [1/3] Cold hold at {:.0}K ({} steps)...", protocol.start_temp, cold_steps);
    run_batch_phase(
        &mut batch,
        &structure_ids,
        &topologies,
        &aromatic_indices_per_structure,
        &entry_mapping,
        replicas,
        cold_steps,
        frame_interval,
        protocol.start_temp,
        dt,
        gamma,
        protocol.uv_burst_interval as usize,
        protocol.uv_burst_energy,
        &mut all_entry_spikes,
        &mut previous_positions,
        &mut current_step,
    )?;

    // Phase 2: Temperature ramp
    log::info!("    [2/3] Ramping {:.0}K → {:.0}K ({} steps)...",
        protocol.start_temp, protocol.end_temp, ramp_steps);
    run_batch_ramp_phase(
        &mut batch,
        &structure_ids,
        &topologies,
        &aromatic_indices_per_structure,
        &entry_mapping,
        replicas,
        ramp_steps,
        frame_interval,
        protocol.start_temp,
        protocol.end_temp,
        dt,
        gamma,
        protocol.uv_burst_interval as usize,
        protocol.uv_burst_energy,
        &mut all_entry_spikes,
        &mut previous_positions,
        &mut current_step,
    )?;

    // Phase 3: Warm hold
    if warm_steps > 0 {
        log::info!("    [3/3] Warm hold at {:.0}K ({} steps)...",
            protocol.end_temp, warm_steps);
        run_batch_phase(
            &mut batch,
            &structure_ids,
            &topologies,
            &aromatic_indices_per_structure,
            &entry_mapping,
            replicas,
            warm_steps,
            frame_interval,
            protocol.end_temp,
            dt,
            gamma,
            protocol.uv_burst_interval as usize,
            protocol.uv_burst_energy,
            &mut all_entry_spikes,
            &mut previous_positions,
            &mut current_step,
        )?;
    }

    let md_elapsed = batch_start.elapsed().as_secs_f64();
    log::info!("    ✓ Batch MD complete in {:.1}s", md_elapsed);

    // Process results per structure (aggregating across replicas)
    let mut results = Vec::new();

    // Create RT clustering engine once for all structures
    // Adaptive grid dim based on largest structure in batch
    let batch_grid_dim = if max_atoms < 500 {
        64
    } else if max_atoms < 2000 {
        96
    } else if max_atoms < 5000 {
        128
    } else {
        160
    };
    log::info!("  Adaptive grid: {}³ for batch (max {} atoms)", batch_grid_dim, max_atoms);
    let config = PersistentBatchConfig {
        max_atoms: max_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
        grid_dim: batch_grid_dim,
        ..Default::default()
    };
    let mut engine = PersistentNhsEngine::new(&config)?;
    let _has_rt = engine.has_rt_clustering();

    for (struct_idx, (structure, topology)) in structures.iter().zip(topologies.iter()).enumerate() {
        let structure_start = Instant::now();
        let structure_output = args.output.join(&structure.name);
        std::fs::create_dir_all(&structure_output)?;

        // Aggregate spikes across all replicas for this structure
        let mut per_replica_spikes: Vec<Vec<GpuSpikeEvent>> = vec![Vec::new(); replicas];
        for (entry_idx, &(s_idx, r_idx)) in entry_mapping.iter().enumerate() {
            if s_idx == struct_idx {
                per_replica_spikes[r_idx].extend(all_entry_spikes[entry_idx].clone());
            }
        }

        let total_raw_spikes: usize = per_replica_spikes.iter().map(|s| s.len()).sum();
        log::info!("    Processing {} ({} replicas): {} total raw spikes",
            structure.name, replicas, total_raw_spikes);
        for (r_idx, replica_spikes) in per_replica_spikes.iter().enumerate() {
            log::info!("      Replica {}: {} spikes", r_idx, replica_spikes.len());
        }

        // Process per-replica clustering for consensus analysis
        let mut per_replica_sites: Vec<Vec<ClusteredBindingSite>> = Vec::new();

        for (replica_idx, replica_spikes) in per_replica_spikes.iter().enumerate() {
            // Apply intensity filtering per replica (top 2%)
            let filtered_spikes = if replica_spikes.len() > 1000 {
                let mut intensities: Vec<f32> = replica_spikes.iter()
                    .map(|s| s.intensity)
                    .collect();
                intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let threshold_idx = (intensities.len() as f32 * 0.98) as usize;
                let intensity_threshold = intensities.get(threshold_idx).copied().unwrap_or(0.0);

                replica_spikes.iter()
                    .filter(|s| s.intensity >= intensity_threshold)
                    .cloned()
                    .collect()
            } else {
                replica_spikes.clone()
            };

            // Cluster this replica's spikes
            let replica_sites = if !filtered_spikes.is_empty() && args.rt_clustering {
                let positions: Vec<f32> = filtered_spikes.iter()
                    .flat_map(|s| {
                        let pos = s.position;
                        [pos[0], pos[1], pos[2]].into_iter()
                    })
                    .collect();

                if args.multi_scale {
                    let custom_epsilon = if args.adaptive_epsilon {
                        None
                    } else {
                        Some(vec![2.5f32, 3.5, 5.0, 7.0])
                    };

                    match engine.multi_scale_cluster_spikes_with_epsilon(&positions, custom_epsilon) {
                        Ok(ms_result) => {
                            let cluster_ids = ms_result.to_cluster_ids(filtered_spikes.len());
                            let fake_result = prism_nhs::rt_clustering::RtClusteringResult {
                                cluster_ids,
                                num_clusters: ms_result.num_clusters(),
                                total_neighbors: 0,
                                gpu_time_ms: 0.0,
                            };

                            let all_sites = build_sites_from_clustering(&filtered_spikes, &fake_result);
                            let min_spikes = (filtered_spikes.len() as f64 * 0.02).ceil() as usize;
                            all_sites.into_iter()
                                .filter(|s| s.spike_count >= min_spikes)
                                .collect()
                        }
                        Err(_) => Vec::new()
                    }
                } else {
                    match engine.cluster_spikes(&positions) {
                        Ok(result) => {
                            let all_sites = build_sites_from_clustering(&filtered_spikes, &result);
                            let min_spikes = (filtered_spikes.len() as f64 * 0.02).ceil() as usize;
                            all_sites.into_iter()
                                .filter(|s| s.spike_count >= min_spikes)
                                .collect()
                        }
                        Err(_) => Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            log::info!("      Replica {}: {} filtered spikes → {} sites",
                replica_idx, filtered_spikes.len(), replica_sites.len());
            per_replica_sites.push(replica_sites);
        }

        // Perform consensus analysis: site must appear in N out of M replicas
        let consensus_threshold = if replicas >= 3 {
            (replicas as f32 * 0.67).ceil() as usize  // 2+ out of 3, 3+ out of 4, etc.
        } else {
            1  // For 1-2 replicas, any detection counts
        };

        log::info!("      Consensus analysis: site must appear in {}/{} replicas", consensus_threshold, replicas);

        // Build consensus sites by finding spatially overlapping sites across replicas
        // Note: batch/manifest mode doesn't have global spike offsets; consensus
        // spike_indices will be empty and the nearest-centroid fallback is used.
        let empty_offsets: Vec<usize> = Vec::new();
        let clustered_sites = build_consensus_sites(&per_replica_sites, consensus_threshold, 5.0, &empty_offsets);
        log::info!("      Consensus sites: {}", clustered_sites.len());

        // Prepare per-replica stats BEFORE moving per_replica_sites
        let per_replica_stats: Vec<_> = per_replica_spikes.iter()
            .enumerate()
            .map(|(r_idx, spikes)| {
                let sites_found = if r_idx < per_replica_sites.len() {
                    per_replica_sites[r_idx].len()
                } else {
                    0
                };
                let druggable_sites = if r_idx < per_replica_sites.len() {
                    per_replica_sites[r_idx].iter().filter(|s| s.druggability.is_druggable).count()
                } else {
                    0
                };
                serde_json::json!({
                    "replica_id": r_idx,
                    "raw_spikes": spikes.len(),
                    "sites_found": sites_found,
                    "druggable_sites": druggable_sites,
                })
            })
            .collect();

        // Use consensus sites (or single replica if replicas=1)
        let mut clustered_sites = if replicas == 1 && !per_replica_sites.is_empty() {
            per_replica_sites.into_iter().next().unwrap_or_default()
        } else {
            clustered_sites
        };

        log::info!("      Final binding sites: {}", clustered_sites.len());

        // Aromatic proximity analysis
        let aromatic_positions = &aromatic_positions_per_structure[struct_idx];
        if !clustered_sites.is_empty() && !aromatic_positions.is_empty() {
            enhance_sites_with_aromatics(&mut clustered_sites, aromatic_positions);
            let druggable_count = clustered_sites.iter()
                .filter(|s| s.druggability.is_druggable)
                .count();
            log::info!("      Aromatic analysis: {} druggable sites", druggable_count);
        }

        // Create mapping from internal index to PDB ID
        let mut pdb_id_map = Vec::new();
        if !topology.residues.is_empty() {
            let max_idx = topology.residues.iter().map(|r| r.residue_idx).max().unwrap_or(0);
            pdb_id_map.resize(max_idx + 1, 0);
            for r in &topology.residues {
                if r.residue_idx < pdb_id_map.len() {
                    pdb_id_map[r.residue_idx] = r.residue_id;
                }
            }
        } else {
            // Fallback if metadata is missing
            let max_id = *topology.residue_ids.iter().max().unwrap_or(&0);
            pdb_id_map = (0..=max_id).map(|i| i as i32).collect();
        }
        // Compute lining residues
        let lining_cutoff = args.lining_cutoff;
        for site in clustered_sites.iter_mut().take(100) {
            site.compute_lining_residues(
                &topology.positions,
                &topology.residue_ids,
                &topology.residue_names,
                &topology.chain_ids,
                &pdb_id_map,
                lining_cutoff,
            );
        }

        // Write visualization outputs
        let structure_name = structure.name.clone();
        let output_base = structure_output.join(&structure_name);

        if !clustered_sites.is_empty() {
            write_binding_site_visualizations(&clustered_sites, &output_base, &structure_name)?;

            // Write JSON summary
            let json_path = output_base.with_extension("binding_sites.json");
            let catalytic_residues = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];

            let json_output = serde_json::json!({
                "structure": structure_name,
                "total_steps": steps_per_structure,
                "simulation_time_sec": md_elapsed / n_structures as f64,
                "replicas": replicas,
                "consensus_threshold": consensus_threshold,
                "spike_count": total_raw_spikes,
                "per_replica_stats": per_replica_stats,
                "binding_sites": clustered_sites.len(),
                "druggable_sites": clustered_sites.iter().filter(|s| s.druggability.is_druggable).count(),
                "lining_residue_cutoff_angstroms": lining_cutoff,
                "sites": clustered_sites.iter().take(100).map(|s| {
                    let catalytic_count = s.lining_residues.iter()
                        .filter(|r| catalytic_residues.contains(&r.resname.as_str()))
                        .count();
                    serde_json::json!({
                        "id": s.cluster_id,
                        "centroid": s.centroid,
                        "volume": s.estimated_volume,
                        "spike_count": s.spike_count,
                        "quality_score": s.quality_score,
                        "druggability": s.druggability.overall,
                        "is_druggable": s.druggability.is_druggable,
                        "classification": format!("{:?}", s.classification),
                        "aromatic_score": s.aromatic_proximity.as_ref().map(|p| p.aromatic_score),
                        "catalytic_residue_count": catalytic_count,
                        "lining_residues": s.lining_residues.iter().map(|r| {
                            let is_catalytic = catalytic_residues.contains(&r.resname.as_str());
                            serde_json::json!({
                                "chain": r.chain,
                                "resid": r.resid,
                                "resname": r.resname,
                                "min_distance": r.min_distance,
                                "n_atoms": r.n_atoms_in_pocket,
                                "is_catalytic": is_catalytic,
                            })
                        }).collect::<Vec<_>>(),
                        "residue_ids": s.lining_residue_ids(),
                    })
                }).collect::<Vec<_>>(),
            });
            std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        }

        let total_sites = clustered_sites.len();
        let druggable_sites = clustered_sites.iter()
            .filter(|s| s.druggability.is_druggable)
            .count();

        let elapsed = structure_start.elapsed().as_secs_f64();
        log::info!("      ✓ Complete: {} sites ({} druggable) in {:.1}s",
            total_sites, druggable_sites, elapsed);

        results.push(StructureRunResult {
            name: structure.name.clone(),
            success: true,
            error: None,
            elapsed_seconds: md_elapsed / n_structures as f64 + elapsed,
            sites_found: Some(total_sites),
            druggable_sites: Some(druggable_sites),
        });
    }

    Ok(results)
}

/// Run a simulation phase at constant temperature for batch
#[cfg(feature = "gpu")]
fn run_batch_phase(
    batch: &mut prism_gpu::AmberSimdBatch,
    _structure_ids: &[usize],
    topologies: &[PrismPrepTopology],
    aromatic_indices_per_structure: &[Vec<usize>],
    entry_mapping: &[(usize, usize)],
    _replicas: usize,
    steps: usize,
    frame_interval: usize,
    temperature: f32,
    dt: f32,
    gamma: f32,
    uv_interval: usize,
    uv_energy: f32,
    all_entry_spikes: &mut [Vec<prism_nhs::fused_engine::GpuSpikeEvent>],
    previous_positions: &mut [Vec<f32>],
    current_step: &mut usize,
) -> Result<()> {
    let n_chunks = steps / frame_interval;

    for _chunk in 0..n_chunks {
        // Run MD chunk
        batch.run(frame_interval, dt, temperature, gamma)?;
        *current_step += frame_interval;

        // Apply UV burst if at interval
        if *current_step % uv_interval < frame_interval {
            apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy, *current_step)?;
        }

        // Extract positions and detect spikes per entry (structure × replica)
        let all_positions = batch.get_positions()?;
        for (entry_idx, &(struct_idx, _replica_idx)) in entry_mapping.iter().enumerate() {
            let topology = &topologies[struct_idx];
            let n_atoms = topology.n_atoms;
            let aromatic_indices = &aromatic_indices_per_structure[struct_idx];

            // Extract per-entry positions from batch
            let start_idx = entry_idx * batch.max_atoms_per_struct() * 3;
            let end_idx = start_idx + n_atoms * 3;
            let positions = &all_positions[start_idx..end_idx];

            let spikes = detect_spikes_from_positions(
                positions,
                &previous_positions[entry_idx],
                topology,
                aromatic_indices,
                *current_step,
            );
            all_entry_spikes[entry_idx].extend(spikes);

            previous_positions[entry_idx] = positions.to_vec();
        }
    }

    // Run remaining steps
    let remaining = steps % frame_interval;
    if remaining > 0 {
        batch.run(remaining, dt, temperature, gamma)?;
        *current_step += remaining;
    }

    Ok(())
}

/// Run temperature ramp phase for batch
#[cfg(feature = "gpu")]
fn run_batch_ramp_phase(
    batch: &mut prism_gpu::AmberSimdBatch,
    _structure_ids: &[usize],
    topologies: &[PrismPrepTopology],
    aromatic_indices_per_structure: &[Vec<usize>],
    entry_mapping: &[(usize, usize)],
    _replicas: usize,
    steps: usize,
    frame_interval: usize,
    start_temp: f32,
    end_temp: f32,
    dt: f32,
    gamma: f32,
    uv_interval: usize,
    uv_energy: f32,
    all_entry_spikes: &mut [Vec<prism_nhs::fused_engine::GpuSpikeEvent>],
    previous_positions: &mut [Vec<f32>],
    current_step: &mut usize,
) -> Result<()> {
    let n_chunks = steps / frame_interval;

    for chunk in 0..n_chunks {
        // Linear temperature interpolation
        let progress = chunk as f32 / n_chunks as f32;
        let temp = start_temp + progress * (end_temp - start_temp);

        // Run MD chunk at current temperature
        batch.run(frame_interval, dt, temp, gamma)?;
        *current_step += frame_interval;

        // Apply UV burst if at interval
        if *current_step % uv_interval < frame_interval {
            apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy, *current_step)?;
        }

        // Extract positions and detect spikes per entry (structure × replica)
        let all_positions = batch.get_positions()?;
        for (entry_idx, &(struct_idx, _replica_idx)) in entry_mapping.iter().enumerate() {
            let topology = &topologies[struct_idx];
            let n_atoms = topology.n_atoms;
            let aromatic_indices = &aromatic_indices_per_structure[struct_idx];

            // Extract per-entry positions from batch
            let start_idx = entry_idx * batch.max_atoms_per_struct() * 3;
            let end_idx = start_idx + n_atoms * 3;
            let positions = &all_positions[start_idx..end_idx];

            let spikes = detect_spikes_from_positions(
                positions,
                &previous_positions[entry_idx],
                topology,
                aromatic_indices,
                *current_step,
            );
            all_entry_spikes[entry_idx].extend(spikes);

            previous_positions[entry_idx] = positions.to_vec();
        }
    }

    Ok(())
}

/// Apply UV burst to aromatic atoms in batch
#[cfg(feature = "gpu")]
fn apply_batch_uv_burst(
    batch: &mut prism_gpu::AmberSimdBatch,
    aromatic_indices_per_structure: &[Vec<usize>],
    topologies: &[PrismPrepTopology],
    _energy: f32,
    current_step: usize,
) -> Result<()> {
    use prism_nhs::config::{
        extinction_to_cross_section, wavelength_to_ev,
        CALIBRATED_PHOTON_FLUENCE,
        HEAT_YIELD_TRP, HEAT_YIELD_TYR, HEAT_YIELD_PHE,
        KB_EV_K, NEFF_TRP, NEFF_TYR, NEFF_PHE,
    };

    // Wavelength cycling: rotate through chromophore-specific wavelengths
    // 280nm=TRP, 274nm=TYR, 258nm=PHE, 211nm=HIS
    let wavelengths = [280.0f32, 274.0, 258.0, 211.0];
    let wavelength = wavelengths[current_step / 250 % wavelengths.len()];

    let mut velocities = batch.get_velocities()?;
    let max_stride = batch.max_atoms_per_struct() * 3;

    // Seed RNG from current_step for reproducible but varying directions
    let mut rng_state: u64 = current_step as u64 * 6364136223846793005 + 1442695040888963407;

    for (struct_idx, aromatic_indices) in aromatic_indices_per_structure.iter().enumerate() {
        if aromatic_indices.is_empty() {
            continue;
        }

        let topology = &topologies[struct_idx];
        let offset = struct_idx * max_stride;

        for &atom_idx in aromatic_indices {
            if atom_idx >= topology.residue_names.len() {
                continue;
            }

            // Classify chromophore from residue name
            let res_name = &topology.residue_names[atom_idx];
            let (chromophore_type, heat_yield, n_eff) = match res_name.as_str() {
                "TRP" => (0i32, HEAT_YIELD_TRP, NEFF_TRP),
                "TYR" => (1, HEAT_YIELD_TYR, NEFF_TYR),
                "PHE" => (2, HEAT_YIELD_PHE, NEFF_PHE),
                "HIS" | "HID" | "HIE" | "HIP" => (3, 0.95f32, 6.0f32),
                _ => continue,
            };

            // Wavelength-dependent extinction (Gaussian band model, FWHM ~15nm)
            let (peak_wavelength, peak_extinction) = match chromophore_type {
                0 => (280.0f32, 5500.0f32),  // TRP: indole
                1 => (274.0, 1490.0),          // TYR: phenol
                2 => (258.0, 200.0),           // PHE: benzene
                3 => (211.0, 5700.0),          // HIS: imidazole
                _ => continue,
            };
            let sigma_nm = 7.5f32; // Gaussian width
            let delta = wavelength - peak_wavelength;
            let extinction = peak_extinction * (-0.5 * (delta / sigma_nm).powi(2)).exp();

            // Skip if negligible absorption at this wavelength
            if extinction < 10.0 {
                continue;
            }

            // Physics: absorption cross-section and probability
            let cross_section = extinction_to_cross_section(extinction);
            let p_absorb = cross_section * CALIBRATED_PHOTON_FLUENCE;

            // Stochastic absorption check (PCG-style fast hash)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(
                (struct_idx as u64) << 32 | atom_idx as u64
            );
            let rand_val = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            if rand_val > p_absorb {
                continue; // Photon not absorbed by this chromophore
            }

            // Energy deposited: E_photon * heat_yield
            let e_photon = wavelength_to_ev(wavelength);
            let e_dep = e_photon * heat_yield;

            // Local heating: delta_T = E_dep / (1.5 * k_B * N_eff)
            let delta_t_kelvin = e_dep / (1.5 * KB_EV_K * n_eff);

            // Use real atomic mass from topology
            let mass_amu = if atom_idx < topology.masses.len() {
                topology.masses[atom_idx].max(1.0)
            } else {
                12.0
            };

            // KE -> velocity: v = sqrt(2 * KE_eV * 96.485 / mass_amu) in Å/ps
            let ke_ev = 1.5 * KB_EV_K * delta_t_kelvin;
            let velocity_boost = (2.0 * ke_ev * 96.485 / mass_amu).sqrt().max(0.0);

            // Proper uniform random direction on unit sphere
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng_state >> 33) as f32) / (u32::MAX as f32);
            let cos_theta = 2.0 * u1 - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * std::f32::consts::PI * u2;

            let base = offset + atom_idx * 3;
            if base + 2 < velocities.len() {
                velocities[base]     += velocity_boost * sin_theta * phi.cos();
                velocities[base + 1] += velocity_boost * sin_theta * phi.sin();
                velocities[base + 2] += velocity_boost * cos_theta;
            }
        }
    }

    batch.set_velocities(&velocities)?;
    Ok(())
}

/// Detect spikes from position changes (simplified aromatic proximity method)
#[cfg(feature = "gpu")]
fn detect_spikes_from_positions(
    positions: &[f32],
    previous_positions: &[f32],
    topology: &PrismPrepTopology,
    aromatic_indices: &[usize],
    timestep: usize,
) -> Vec<prism_nhs::fused_engine::GpuSpikeEvent> {
    use prism_nhs::fused_engine::GpuSpikeEvent;

    let mut spikes = Vec::new();

    // If no previous positions, initialize and return
    if previous_positions.is_empty() {
        return spikes;
    }

    // Simple spike detection: large displacement of aromatic atoms
    // This is a proxy for dewetting events
    let displacement_threshold = 0.5; // Angstroms per frame
    let proximity_threshold = 6.0; // Angstroms

    for &atom_idx in aromatic_indices {
        let idx = atom_idx * 3;
        if idx + 2 >= positions.len() || idx + 2 >= previous_positions.len() {
            continue;
        }

        // Compute displacement
        let dx = positions[idx] - previous_positions[idx];
        let dy = positions[idx + 1] - previous_positions[idx + 1];
        let dz = positions[idx + 2] - previous_positions[idx + 2];
        let displacement = (dx * dx + dy * dy + dz * dz).sqrt();

        // If significant displacement, check for nearby atoms (potential binding pocket)
        if displacement > displacement_threshold {
            let pos = [positions[idx], positions[idx + 1], positions[idx + 2]];

            // Count nearby heavy atoms (potential pocket)
            let mut nearby_count = 0;
            for i in 0..topology.n_atoms {
                if i == atom_idx {
                    continue;
                }
                let i_idx = i * 3;
                if i_idx + 2 >= positions.len() {
                    continue;
                }

                let dx = positions[i_idx] - pos[0];
                let dy = positions[i_idx + 1] - pos[1];
                let dz = positions[i_idx + 2] - pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < proximity_threshold {
                    nearby_count += 1;
                }
            }

            // If sufficiently isolated (potential pocket), register spike
            if nearby_count < 20 {
                let intensity = displacement * (20.0 - nearby_count as f32) / 20.0;
                spikes.push(GpuSpikeEvent {
                    timestep: timestep as i32,
                    voxel_idx: 0,
                    position: pos,
                    intensity,
                    nearby_residues: [0; 8],
                    n_residues: 0,
                    spike_source: 0,
                    wavelength_nm: 0.0,
                    aromatic_type: -1,
                    aromatic_residue_id: -1,
                    water_density: 0.0,
                    vibrational_energy: 0.0,
                    n_nearby_excited: 0,
                    wd_change: 0.0,
                });
            }
        }
    }

    spikes
}

/// True multi-stream pipeline: N independent CUDA streams, each running
/// the full cryo-UV-BNZ-RT stack. One CudaContext, one PTX module, N streams.
/// Results aggregated via consensus clustering.
#[cfg(feature = "gpu")]
fn run_multi_stream_pipeline(
    args: &Args,
    topology_path: &PathBuf,
    n_streams: usize,
) -> Result<()> {
    use cudarc::driver::CudaContext;
    use cudarc::nvrtc::Ptx;
    use std::path::Path;

    let total_start = Instant::now();

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  TRUE MULTI-STREAM PIPELINE ({} concurrent streams)           ║", n_streams);
    log::info!("║  Full cryo-UV-BNZ-RT on each independent CUDA stream          ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // ── ONE context, ONE module ──
    let context = CudaContext::new(0).context("CUDA context")?;

    let ptx_candidates = [
        "../prism-gpu/src/kernels/nhs_amber_fused.ptx",
        "crates/prism-gpu/src/kernels/nhs_amber_fused.ptx",
        "target/ptx/nhs_amber_fused.ptx",
    ];
    let ptx_path = ptx_candidates.iter()
        .find(|p| Path::new(p).exists())
        .ok_or_else(|| anyhow::anyhow!("nhs_amber_fused.ptx not found"))?;
    let module = context
        .load_module(Ptx::from_file(ptx_path))
        .context("Failed to load PTX")?;

    // ── N independent streams ──
    let streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>> = (0..n_streams)
        .map(|_| context.new_stream())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Failed to create CUDA streams")?;
    log::info!("  ✓ {} CUDA streams created on shared context", n_streams);

    // ── Load topology ONCE ──
    let mut topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

    let structure_name = topology_path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "structure".to_string());

    // Apply HMR if requested
    if args.hmr {
        log::info!("  HMR: Applying 3x hydrogen mass repartitioning (dt=4fs)");
        topology.apply_hmr(3.0);
    }

    let aromatic_positions = extract_aromatic_positions(&topology);
    log::info!("  Structure: {} ({} atoms, {} aromatics)",
        structure_name, topology.n_atoms, aromatic_positions.len());

    if topology.n_atoms < 500 {
        anyhow::bail!("Protein too small for GPU analysis ({} atoms, min 500)", topology.n_atoms);
    }

    // Adaptive grid: scale to protein bounding box
    let ms_grid_dim = if topology.n_atoms < 500 {
        64
    } else if topology.n_atoms < 2000 {
        96
    } else if topology.n_atoms < 5000 {
        128
    } else {
        160
    };
    log::info!("  Adaptive grid: {}³ for {} atoms", ms_grid_dim, topology.n_atoms);
    let config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
        grid_dim: ms_grid_dim,
        ..Default::default()
    };

    let protocol = if args.fast_25k {
        let base = CryoUvProtocol::fast_25k();
        let extra_warm = ((topology.n_atoms.saturating_sub(5000) / 1000) * 2000) as i32;
        let protocol_sized = CryoUvProtocol { warm_hold_steps: base.warm_hold_steps + extra_warm, ..base };
        log::info!("  Adaptive warm_hold: {} steps ({} atoms, +{} extra)",
            protocol_sized.warm_hold_steps, topology.n_atoms, extra_warm);
        protocol_sized
    } else if args.fast {
        let base = CryoUvProtocol::fast_35k();
        let extra_warm = ((topology.n_atoms.saturating_sub(5000) / 1000) * 2000) as i32;
        let protocol_sized = CryoUvProtocol { warm_hold_steps: base.warm_hold_steps + extra_warm, ..base };
        log::info!("  Adaptive warm_hold: {} steps ({} atoms, +{} extra)",
            protocol_sized.warm_hold_steps, topology.n_atoms, extra_warm);
        protocol_sized
    } else {
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: config.cryo_hold,
            ramp_steps: config.convergence_steps / 2,
            warm_hold_steps: config.convergence_steps / 2,
            current_step: 0,
            uv_burst_energy: 50.0,
            uv_burst_interval: 100,
            uv_burst_duration: 50,
            scan_wavelengths: vec![280.0, 274.0, 258.0, 254.0, 211.0],
            wavelength_dwell_steps: 500,
            ramp_down_steps: 0,
            cold_return_steps: 0,
        }
    };
    let protocol = if args.hysteresis {
        protocol.with_hysteresis()
    } else {
        protocol
    };
    let _target_end_temp = protocol.end_temp;

    // Steps per stream: use full protocol length for --fast/--fast-25k (respects hysteresis),
    // otherwise use user-specified --steps
    let steps_per_stream = if args.fast || args.fast_25k {
        protocol.total_steps()
    } else {
        args.steps
    };

    // ── Run N engines on N threads (scoped for safe borrowing) ──
    log::info!("\n  🚀 Launching {} independent trajectories...", n_streams);
    let sim_start = Instant::now();

    let stream_results: Vec<Result<(Vec<prism_nhs::fused_engine::GpuSpikeEvent>, Vec<prism_nhs::fused_engine::EnsembleSnapshot>)>> =
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_streams).map(|i| {
                let ctx = context.clone();
                let mod_ = module.clone();
                let stream_i = streams[i].clone();
                let topo_ref = &topology;
                let config_ref = &config;
                let prot = protocol.clone();
                let seed = args.replica_seed + i as u64 * 12345;
                let ultimate = args.ultimate_mode;
                let steps = steps_per_stream;
                let hmr_enabled = args.hmr;
                let fused_steps = args.fused_steps;
                let adaptive_dt = args.adaptive_dt;

                s.spawn(move || -> Result<(Vec<prism_nhs::fused_engine::GpuSpikeEvent>, Vec<prism_nhs::fused_engine::EnsembleSnapshot>)> {
                    log::info!("    [stream {}] Starting (seed: {})...", i, seed);

                    let mut engine = PersistentNhsEngine::new_on_stream(
                        config_ref, ctx, mod_, stream_i,
                    )?;
                    engine.load_topology(topo_ref)?;
                    if hmr_enabled {
                        engine.set_dt(0.004)?;  // 4fs with HMR masses
                    }
                    if fused_steps > 1 {
                        engine.set_fused_inner_steps(fused_steps)?;
                    }
                    if adaptive_dt {
                        engine.set_adaptive_dt(true)?;
                    }
                    engine.set_cryo_uv_protocol(prot)?;
                    engine.set_spike_accumulation(true);

                    if ultimate {
                        match engine.enable_ultimate_mode(topo_ref) {
                            Ok(()) => log::info!("    [stream {}] UltimateEngine: ✓", i),
                            Err(e) => log::warn!("    [stream {}] UltimateEngine: ✗ {}", i, e),
                        }
                    }

                    engine.reset_for_replica(seed)?;
                    let summary = engine.run(steps)?;
                    let spikes = engine.get_accumulated_spikes();
                    let snapshots = engine.get_snapshots();

                    log::info!("    [stream {}] Complete: {} spikes, {} snapshots, T={:.1}K",
                        i, spikes.len(), snapshots.len(), summary.end_temperature);
                    Ok((spikes, snapshots))
                })
            }).collect();

            handles.into_iter()
                .map(|h| h.join().expect("stream thread panicked"))
                .collect()
        });

    let sim_elapsed = sim_start.elapsed();
    log::info!("  ✓ All {} streams complete in {:.1}s", n_streams, sim_elapsed.as_secs_f64());

    // ── Aggregate: per-stream filtering + clustering → consensus ──
    log::info!("\n  Aggregating results across {} streams...", n_streams);

    let mut cluster_engine = PersistentNhsEngine::new(&config)?;
    cluster_engine.load_topology(&topology)?;

    let mut per_stream_sites: Vec<Vec<ClusteredBindingSite>> = Vec::new();
    let mut per_stream_stats: Vec<serde_json::Value> = Vec::new();
    let mut all_stream_snapshots: Vec<Vec<prism_nhs::fused_engine::EnsembleSnapshot>> = Vec::new();
    let mut all_stream_spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent> = Vec::new();
    let mut stream_spike_offsets: Vec<usize> = Vec::new(); // offset into all_stream_spikes for each stream

    for (i, result) in stream_results.into_iter().enumerate() {
        let (raw_spikes, stream_snapshots) = match result {
            Ok((spikes, snaps)) => (spikes, snaps),
            Err(e) => {
                log::error!("    Stream {} failed: {}", i, e);
                per_stream_stats.push(serde_json::json!({
                    "stream_id": i, "error": e.to_string(),
                }));
                per_stream_sites.push(Vec::new());
                all_stream_snapshots.push(Vec::new());
                stream_spike_offsets.push(all_stream_spikes.len());
                continue;
            }
        };

        let pct_f = (args.spike_percentile.min(99) as f32) / 100.0;
        let filtered = if raw_spikes.len() > 1000 {
            let mut intensities: Vec<f32> = raw_spikes.iter().map(|s| s.intensity).collect();
            intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (intensities.len() as f32 * pct_f) as usize;
            let threshold = intensities.get(idx).copied().unwrap_or(0.0);
            raw_spikes.into_iter().filter(|s| s.intensity >= threshold).collect::<Vec<_>>()
        } else {
            raw_spikes
        };

        // Track offset so per-stream spike_indices can be remapped to all_stream_spikes
        stream_spike_offsets.push(all_stream_spikes.len());
        all_stream_spikes.extend(filtered.iter().cloned());
        let sites = if !filtered.is_empty() && args.rt_clustering {
            let positions: Vec<f32> = filtered.iter()
                .flat_map(|s| { let p = s.position; [p[0], p[1], p[2]].into_iter() })
                .collect();

            if args.multi_scale {
                let eps = if args.adaptive_epsilon { None } else { Some(vec![2.5f32, 3.5, 5.0, 7.0]) };
                match cluster_engine.multi_scale_cluster_spikes_with_epsilon(&positions, eps) {
                    Ok(ms) => {
                        let ids = ms.to_cluster_ids(filtered.len());
                        let fake = prism_nhs::rt_clustering::RtClusteringResult {
                            cluster_ids: ids, num_clusters: ms.num_clusters(),
                            total_neighbors: 0, gpu_time_ms: 0.0,
                        };
                        let all = build_sites_from_clustering(&filtered, &fake);
                        let min_s = (filtered.len() as f64 * 0.02).ceil() as usize;
                        all.into_iter().filter(|s| s.spike_count >= min_s).collect()
                    }
                    Err(_) => Vec::new()
                }
            } else {
                match cluster_engine.cluster_spikes(&positions) {
                    Ok(r) => {
                        let all = build_sites_from_clustering(&filtered, &r);
                        let min_s = (filtered.len() as f64 * 0.02).ceil() as usize;
                        all.into_iter().filter(|s| s.spike_count >= min_s).collect()
                    }
                    Err(_) => Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        log::info!("    Stream {}: {} filtered spikes → {} sites", i, filtered.len(), sites.len());
        per_stream_stats.push(serde_json::json!({
            "stream_id": i,
            "raw_spikes": filtered.len(),
            "sites_found": sites.len(),
            "druggable_sites": sites.iter().filter(|s| s.druggability.is_druggable).count(),
        }));
        per_stream_sites.push(sites);
        all_stream_snapshots.push(stream_snapshots);
    }

    let consensus_threshold = if n_streams >= 3 {
        (n_streams as f32 * 0.5).ceil() as usize
    } else {
        1
    };
    log::info!("  Consensus threshold: {}/{} streams", consensus_threshold, n_streams);

    // DEBUG: Log per-stream site centroids
    for (i, sites) in per_stream_sites.iter().enumerate() {
        for (j, site) in sites.iter().enumerate() {
            log::info!("    Stream {} site {}: centroid=[{:.1}, {:.1}, {:.1}], spikes={}, intensity={:.3}",
                i, j, site.centroid[0], site.centroid[1], site.centroid[2],
                site.spike_count, site.avg_intensity);
        }
    }

    let mut clustered_sites = if n_streams == 1 && !per_stream_sites.is_empty() {
        per_stream_sites.into_iter().next().unwrap_or_default()
    } else {
        build_consensus_sites(&per_stream_sites, consensus_threshold, 10.0, &stream_spike_offsets)
    };
    log::info!("  Consensus binding sites: {}", clustered_sites.len());

    // Dimer-aware merge: detect homodimer symmetry and merge symmetric site pairs
    merge_symmetric_sites(&mut clustered_sites, &all_stream_spikes, &topology, 12.0);

    // ========== SNDC: Spike-Native Density Clustering (PRIMARY) ==========
    // SNDC (OptiX RT clustering) — deprecated on SM120+ (RTX 5080).
    // OptiX fails with OPTIX_ERROR_PIPELINE_LINK_ERROR on every run;
    // LIGSITE + spike density + SDST gives identical results.
    // The OptiX code is preserved in git history for future re-enablement.
    if !all_stream_spikes.is_empty() && args.rt_clustering {
        log::warn!("  SNDC (OptiX RT clustering) is deprecated on SM120+. \
            Using LIGSITE + spike density overlay (identical results, ~2s faster).");
    }

    // Dynamic LIGSITE: Geometry proposes pockets, physics scores them.
    // Runs unconditionally — geometry finds pockets independent of spike-cluster sites.
    log::info!("  Running Dynamic LIGSITE pocket detection...");
    recalculate_enclosure_volume(&mut clustered_sites, &all_stream_spikes, &topology.positions);

    // ========== Cubical PH: density-peak centroid refinement ==========
    // Build a spike density grid and run 0-dim persistent homology to find
    // density peaks. For each LIGSITE site, if a PH peak is within 5A,
    // refine the centroid to the density maximum (birth voxel).
    if !all_stream_spikes.is_empty() && !clustered_sites.is_empty() {
        let ph_start = std::time::Instant::now();
        let (grid_origin, grid_dims, grid_spacing) =
            prism_nhs::cubical_ph::compute_density_grid_bounds(
                &topology.positions, 10.0, 1.0,
            );

        let grid_n = grid_dims[0] * grid_dims[1] * grid_dims[2];
        if grid_n > 0 && grid_n < 8_000_000 {
            // Gaussian splatting of spike density onto the grid
            let mut density = vec![0.0f32; grid_n];
            let sigma = 2.0f32;
            let cutoff = (3.0 * sigma / grid_spacing) as i32 + 1;
            let inv_2sig2 = 1.0 / (2.0 * sigma * sigma);

            for spike in &all_stream_spikes {
                let ix = ((spike.position[0] - grid_origin[0]) / grid_spacing) as i32;
                let iy = ((spike.position[1] - grid_origin[1]) / grid_spacing) as i32;
                let iz = ((spike.position[2] - grid_origin[2]) / grid_spacing) as i32;
                let w = spike.intensity * spike.intensity; // intensity^2 weighting

                for dz in -cutoff..=cutoff {
                    for dy in -cutoff..=cutoff {
                        for dx in -cutoff..=cutoff {
                            let gx = ix + dx;
                            let gy = iy + dy;
                            let gz = iz + dz;
                            if gx < 0 || gy < 0 || gz < 0 { continue; }
                            let (gx, gy, gz) = (gx as usize, gy as usize, gz as usize);
                            if gx >= grid_dims[0] || gy >= grid_dims[1] || gz >= grid_dims[2] {
                                continue;
                            }
                            let r2 = (dx * dx + dy * dy + dz * dz) as f32
                                * grid_spacing * grid_spacing;
                            let val = w * (-r2 * inv_2sig2).exp();
                            density[(gz * grid_dims[1] + gy) * grid_dims[0] + gx] += val;
                        }
                    }
                }
            }

            // Run cubical PH
            let ph_pockets = prism_nhs::cubical_ph::compute_cubical_ph_cpu(
                &density,
                grid_dims,
                grid_origin,
                grid_spacing,
                0.0,  // min_persistence — filter below
                10,   // min_component_size (voxels)
            );

            // Adaptive persistence threshold: 10th percentile
            let persistence_threshold = if ph_pockets.len() > 3 {
                let mut persis: Vec<f32> = ph_pockets.iter().map(|p| p.persistence).collect();
                persis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                persis[persis.len() / 10]
            } else {
                0.0
            };

            let significant: Vec<_> = ph_pockets
                .iter()
                .filter(|p| p.persistence >= persistence_threshold)
                .collect();

            let ph_elapsed = ph_start.elapsed();
            log::info!(
                "  Cubical PH: {} total pairs, {} significant (persistence > {:.2}), {:.1}ms",
                ph_pockets.len(),
                significant.len(),
                persistence_threshold,
                ph_elapsed.as_secs_f64() * 1000.0
            );

            // Refine LIGSITE centroids: for each site, find nearest PH peak within 5A
            let ph_refine_radius = 5.0f32;
            let mut refined_count = 0usize;
            for site in clustered_sites.iter_mut() {
                let mut best_dist = f32::MAX;
                let mut best_centroid: Option<[f32; 3]> = None;
                let mut best_persistence = 0.0f32;
                for ph in &significant {
                    let dx = ph.centroid[0] - site.centroid[0];
                    let dy = ph.centroid[1] - site.centroid[1];
                    let dz = ph.centroid[2] - site.centroid[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < ph_refine_radius && dist < best_dist {
                        best_dist = dist;
                        best_centroid = Some(ph.centroid);
                        best_persistence = ph.persistence;
                    }
                }
                if let Some(new_c) = best_centroid {
                    log::info!(
                        "  PH refine site {}: shift {:.1}A -> ({:.1},{:.1},{:.1}), persistence={:.2}",
                        site.cluster_id,
                        best_dist,
                        new_c[0], new_c[1], new_c[2],
                        best_persistence
                    );
                    site.centroid = new_c;
                    refined_count += 1;
                }
            }
            log::info!(
                "  Cubical PH: refined {}/{} site centroids to density peaks",
                refined_count,
                clustered_sites.len()
            );
        } else if grid_n >= 8_000_000 {
            log::warn!(
                "  Cubical PH: grid too large ({} voxels > 8M), skipping",
                grid_n
            );
        }
    }

    if !clustered_sites.is_empty() && !aromatic_positions.is_empty() {
        enhance_sites_with_aromatics(&mut clustered_sites, &aromatic_positions);
    }

    let mut pdb_id_map = Vec::new();
    if !topology.residues.is_empty() {
        let max_idx = topology.residues.iter().map(|r| r.residue_idx).max().unwrap_or(0);
        pdb_id_map.resize(max_idx + 1, 0);
        for r in &topology.residues {
            if r.residue_idx < pdb_id_map.len() {
                pdb_id_map[r.residue_idx] = r.residue_id;
            }
        }
    } else {
        let max_id = *topology.residue_ids.iter().max().unwrap_or(&0);
        pdb_id_map = (0..=max_id).map(|i| i as i32).collect();
    }

    let catalytic_residues = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
    for site in clustered_sites.iter_mut().take(100) {
        site.compute_lining_residues(
            &topology.positions, &topology.residue_ids,
            &topology.residue_names, &topology.chain_ids,
            &pdb_id_map, args.lining_cutoff,
        );
    }

    std::fs::create_dir_all(&args.output)?;
    let output_base = args.output.join(&structure_name);

    if !clustered_sites.is_empty() {
        write_binding_site_visualizations(&clustered_sites, &output_base, &structure_name)?;

        // Compute per-site CCNS time-series data from spike events
        let lining_cutoff = args.lining_cutoff;
        let site_radius = lining_cutoff + 2.0;
        let frame_window = 1000; // timesteps per frame for binning

        // Build all_pockets (per-frame volumes) and cryptic_sites (spike time-series)
        // FRAME ALIGNMENT FIX: Use cluster-assigned spikes, not radius queries.
        // Previously, cryptic_sites/all_pockets used all spikes within a 10Å radius
        // (site_radius = lining_cutoff + 2.0), which included spikes from neighboring
        // clusters and noise — producing spike counts 5-13x larger than the consensus
        // cluster assignment in sites[]. This caused frame alignment mismatches in
        // downstream analysis (STI, pharmacophore, dG calculations).
        //
        // Fix: For sites with spike_indices (per-stream path), use those directly.
        // For consensus sites (spike_indices empty), assign each spike to its nearest
        // consensus centroid within the cluster's bounding radius.

        // Pre-assign spikes to consensus sites if spike_indices are empty
        let mut site_spike_assignments: Vec<Vec<usize>> = vec![Vec::new(); clustered_sites.len().min(100)];
        {
            let n_sites = clustered_sites.len().min(100);
            let any_have_indices = clustered_sites.iter().take(n_sites)
                .any(|s| !s.spike_indices.is_empty());

            if any_have_indices {
                // Per-stream or single-stream path: spike_indices are valid indices into
                // the spike array used for clustering. Use them directly.
                for (si, site) in clustered_sites.iter().take(n_sites).enumerate() {
                    let total: Vec<usize> = site.spike_indices.iter()
                        .copied()
                        .collect();
                    let (valid, invalid): (Vec<usize>, Vec<usize>) = total.into_iter()
                        .partition(|&idx| idx < all_stream_spikes.len());
                    if !invalid.is_empty() {
                        log::warn!("Site {} has {} OOB spike indices (max={})",
                            site.cluster_id, invalid.len(), all_stream_spikes.len());
                    }
                    site_spike_assignments[si] = valid;
                }
            } else {
                log::info!("Using nearest-centroid fallback for {} consensus sites ({} total spikes)", n_sites, all_stream_spikes.len());
                // Consensus path: spike_indices are empty. Assign each spike to its
                // nearest consensus centroid within the cluster's bounding radius.
                // Use max(bounding_box_half_diagonal, site_radius) as assignment radius
                // to capture all cluster-relevant spikes without over-counting.
                let mut site_radii: Vec<f32> = Vec::with_capacity(n_sites);
                for site in clustered_sites.iter().take(n_sites) {
                    let bb = site.bounding_box;
                    let half_diag = (bb[0]*bb[0] + bb[1]*bb[1] + bb[2]*bb[2]).sqrt() / 2.0;
                    // Use bounding box half-diagonal + 2Å margin, clamped to [3, site_radius]
                    site_radii.push((half_diag + 2.0).clamp(3.0, site_radius));
                }

                for (spike_idx, spike) in all_stream_spikes.iter().enumerate() {
                    let mut best_site: Option<usize> = None;
                    let mut best_dist = f32::MAX;
                    for (si, site) in clustered_sites.iter().take(n_sites).enumerate() {
                        let dx = spike.position[0] - site.centroid[0];
                        let dy = spike.position[1] - site.centroid[1];
                        let dz = spike.position[2] - site.centroid[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                        if dist <= site_radii[si] && dist < best_dist {
                            best_dist = dist;
                            best_site = Some(si);
                        }
                    }
                    if let Some(si) = best_site {
                        site_spike_assignments[si].push(spike_idx);
                    }
                }
            }
        }

        // ─── Per-spike binding-site-ness scoring ───
        // Score each spike individually based on local environment features,
        // then aggregate per-site. This is the PRISM4D analog of P2Rank's
        // per-surface-point scoring — instead of ranking by aggregate pocket
        // properties (which lose spatial info), we rank by the quality of
        // individual spike signals.
        //
        // Per-spike features:
        //   1. Residue crowding: n_residues (0-8) — buried spikes see more residues
        //   2. Multi-aromatic: n_nearby_excited — π-stacking environment
        //   3. Water displacement: wd_change — high = dewetting event = pocket signal
        //   4. Intensity: high = strong pocket signal
        //   5. Source diversity: UV + LIF convergence = strongest signal
        let mut per_spike_scores: Vec<f32> = vec![0.0; all_stream_spikes.len()];
        for (idx, spike) in all_stream_spikes.iter().enumerate() {
            // Burial: spikes surrounded by many residues are in enclosed pockets
            let burial_q = (spike.n_residues as f32 / 6.0).clamp(0.0, 1.0);

            // Aromatic environment: π-stacking = functional pocket
            let arom_q = (spike.n_nearby_excited as f32 / 3.0).clamp(0.0, 1.0);

            // Water displacement: large wd_change = dewetting event
            let wd_q = (spike.wd_change * 20.0).clamp(0.0, 1.0);

            // Intensity: high-intensity spikes are stronger signals
            let int_q = (spike.intensity / 30.0).clamp(0.0, 1.0);

            // Composite per-spike score: burial is most important
            per_spike_scores[idx] =
                0.40 * burial_q +    // pocket burial (dominant)
                0.20 * arom_q +      // aromatic environment
                0.20 * wd_q +        // water displacement signal
                0.20 * int_q;        // signal intensity
        }

        let mut all_pockets_json = Vec::new();
        let mut cryptic_sites_json = Vec::new();
        let mut physics_signals: std::collections::HashMap<i32, (f32, f32, f32, f32)> = std::collections::HashMap::new();

        // Per-site STI free energy and UV enrichment (Task 1 + Task 7)
        let mut sti_results: std::collections::HashMap<i32, BindingFreeEnergy> = std::collections::HashMap::new();
        let mut uv_enrichment_scores: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();

        // Build StiConfig from protocol parameters
        let sti_config = StiConfig {
            temperature: protocol.end_temp as f64,
            protocol_start_temp: protocol.start_temp as f64,
            protocol_end_temp: protocol.end_temp as f64,
            ramp_steps: protocol.ramp_steps,
            cold_hold_steps: protocol.cold_hold_steps,
            warm_hold_steps: protocol.warm_hold_steps,
        };

        let mut spatial_signals: std::collections::HashMap<i32, (f32, f32, f32)> = std::collections::HashMap::new();

        for (site_idx, site) in clustered_sites.iter_mut().take(100).enumerate() {
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];

            // Collect cluster-assigned spikes for this site (frame-aligned)
            let site_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> =
                site_spike_assignments[site_idx].iter()
                    .filter_map(|&idx| all_stream_spikes.get(idx))
                    .collect();

            // Aggregate per-spike scores for this site
            let site_spike_quality: f32 = if !site_spike_assignments[site_idx].is_empty() {
                let sum: f32 = site_spike_assignments[site_idx].iter()
                    .filter_map(|&idx| per_spike_scores.get(idx))
                    .sum();
                sum / site_spike_assignments[site_idx].len() as f32
            } else {
                0.0
            };

            // ─── Temporal onset: early activation = low barrier = real pocket ───
            let onset_score = if !site_spikes.is_empty() {
                let mut timesteps: Vec<i32> = site_spikes.iter().map(|s| s.timestep).collect();
                timesteps.sort();
                let median_ts = timesteps[timesteps.len() / 2];
                let total_steps = timesteps.last().copied().unwrap_or(1).max(1);
                // Protocol phases: cold_hold=14000, ramp=6000, warm_hold=15000
                // Sites that spike during cold_hold or early ramp have lowest activation barrier
                let onset_fraction = median_ts as f32 / total_steps as f32;
                (1.0 - onset_fraction).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // ─── Source diversity: UV+LIF+EFP convergence = real pocket ───
            let (source_diversity, source_entropy) = if !site_spikes.is_empty() {
                let mut source_counts = [0u32; 4]; // 0=unknown, 1=UV, 2=LIF, 3=EFP
                for s in &site_spikes {
                    let src = (s.spike_source as usize).min(3);
                    source_counts[src] += 1;
                }
                let total = site_spikes.len() as f32;
                let entropy: f32 = source_counts.iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f32 / total;
                        -p * p.ln()
                    })
                    .sum();

                // UV/LIF balance: binding pockets show convergent multi-channel
                // signals (balanced UV+LIF), while surface grooves are LIF-dominated.
                // balance = 1 - |frac_uv - frac_lif| → 1.0 = perfectly balanced,
                //                                       0.0 = single-channel dominated
                // EFP bonus: sites with EFP spikes get a boost (EFP requires
                // charged residues = functional site).
                let uv_count = source_counts[1] as f32;
                let lif_count = source_counts[2] as f32;
                let efp_count = source_counts[3] as f32;
                let ul_total = (uv_count + lif_count).max(1.0);
                let balance = 1.0 - ((uv_count - lif_count).abs() / ul_total);
                let efp_bonus = (efp_count / total).min(0.3) / 0.3; // 0..1, saturates at 30% EFP
                let diversity = balance * 0.7 + efp_bonus * 0.3;
                (diversity, entropy)
            } else {
                (0.0, 0.0)
            };

            // ─── Burial depth: mean nearby residues per spike ───
            let (mean_burial, deep_fraction, burial_score) = if !site_spikes.is_empty() {
                let burial_values: Vec<f32> = site_spikes.iter()
                    .map(|s| s.n_residues as f32)
                    .collect();
                let mean_b = burial_values.iter().sum::<f32>() / burial_values.len() as f32;
                let deep_frac = burial_values.iter().filter(|&&b| b >= 5.0).count() as f32
                    / burial_values.len() as f32;
                // Sigmoid normalization: center=3, slope=2.0
                // Recentered to match actual n_residues distribution (2.4-3.5).
                // WarpEntry has max 16 atoms → ~3-5 unique residues in buried pockets,
                // ~1-2 for surface grooves. center=3 puts the discrimination midpoint
                // at the observed mean; slope=2.0 sharpens the transition.
                let score = 1.0 / (1.0 + (-2.0 * (mean_b - 3.0)).exp());
                (mean_b, deep_frac, score)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Store physics signals for JSON output later
            physics_signals.insert(site.cluster_id, (onset_score, source_diversity, mean_burial, burial_score));

            // ─── Task 1: Per-site STI (Jarzynski free energy) ───
            // Compute binding free energy from spike thermodynamic integration.
            // Uses per-site spikes to estimate delta_g via Jarzynski equality,
            // channel decomposition, and kinetic accessibility.
            if site_spikes.len() >= 10 {
                let owned_spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent> =
                    site_spikes.iter().map(|&s| *s).collect();
                let bfe = compute_binding_free_energy(
                    &owned_spikes,
                    None,  // hysteresis bins not available per-site
                    None,  // no branching theory delta_g
                    &sti_config,
                );
                log::info!("  Site {}: STI delta_g={:.3} kcal/mol (effective={:.3}, n_voxels={}, n_spikes={}, kinetic_acc={:.3})",
                    site.cluster_id, bfe.delta_g_sti_kcal_mol, bfe.effective_delta_g_kcal_mol,
                    bfe.n_voxels, bfe.n_spikes, bfe.kinetic_accessibility);
                sti_results.insert(site.cluster_id, bfe);
            }

            // ─── Task 7: UV Aromatic Enrichment ───
            // Measures whether UV-source spikes are enriched during UV-on periods.
            // UV burst is active when `timestep % burst_interval < burst_duration`.
            // uv_enrichment = (uv_on_rate) / (uv_off_rate)
            // uv_score = min(uv_enrichment / 3.0, 1.0)
            let uv_score = if !site_spikes.is_empty() {
                let burst_interval = protocol.uv_burst_interval;
                let burst_duration = protocol.uv_burst_duration;
                let mut uv_on_count = 0u32;
                let mut uv_off_count = 0u32;
                for s in &site_spikes {
                    let is_uv_source = s.spike_source == 1 || s.aromatic_type >= 0;
                    let is_uv_on = (s.timestep % burst_interval) < burst_duration;
                    if is_uv_source {
                        if is_uv_on {
                            uv_on_count += 1;
                        } else {
                            uv_off_count += 1;
                        }
                    }
                }
                let on_fraction = burst_duration as f32 / burst_interval as f32;
                let off_fraction = 1.0 - on_fraction;
                let uv_on_rate = if on_fraction > 0.0 { uv_on_count as f32 / on_fraction } else { 0.0 };
                let uv_off_rate = if off_fraction > 0.0 { uv_off_count as f32 / off_fraction } else { 0.0 };
                let enrichment = if uv_off_rate > 0.0 { uv_on_rate / uv_off_rate } else { 0.0 };
                let score = (enrichment / 3.0).min(1.0);
                log::info!("  Site {}: UV enrichment={:.3} (on={}, off={}, burst_interval={}, burst_duration={}) uv_score={:.3}",
                    site.cluster_id, enrichment, uv_on_count, uv_off_count, burst_interval, burst_duration, score);
                score
            } else {
                0.0
            };
            uv_enrichment_scores.insert(site.cluster_id, uv_score);

            // Bin spikes by frame to compute per-frame volumes and activity
            let max_ts = site_spikes.iter().map(|s| s.timestep).max().unwrap_or(0);
            let n_frames = (max_ts / frame_window + 1) as usize;
            let mut frame_spike_counts = vec![0usize; n_frames];
            let mut frame_intensity_sums = vec![0.0f32; n_frames];

            for s in &site_spikes {
                let frame = (s.timestep / frame_window) as usize;
                if frame < n_frames {
                    frame_spike_counts[frame] += 1;
                    frame_intensity_sums[frame] += s.intensity;
                }
            }

            // Per-frame volume proxy: spike_count * voxel_volume (27 Å³ for 3Å voxel)
            let voxel_vol = 27.0f32;
            let volumes: Vec<f64> = frame_spike_counts.iter()
                .map(|&c| (c as f32 * voxel_vol) as f64)
                .collect();
            let mean_volume: f64 = if !volumes.is_empty() {
                volumes.iter().sum::<f64>() / volumes.len() as f64
            } else { 0.0 };
            let cv_volume = if mean_volume > 0.0 {
                let variance = volumes.iter().map(|v| (v - mean_volume).powi(2)).sum::<f64>()
                    / volumes.len() as f64;
                variance.sqrt() / mean_volume
            } else { 0.0 };

            all_pockets_json.push(serde_json::json!({
                "site_id": site.cluster_id,
                "centroid": site.centroid,
                "mean_volume": mean_volume,
                "cv_volume": cv_volume,
                "n_frames": n_frames,
                "volumes": volumes,
            }));

            // ---- CV druggability penalty ----
            // Rigid buried cores: CV ~ 0 (no volume fluctuation). Penalize.
            // Real pockets breathe: CV > 0.3 expected.
            if cv_volume < 0.3 {
                let penalty = (cv_volume / 0.3) as f32;
                let old_drug = site.druggability.overall;
                site.druggability.overall *= penalty;
                site.druggability.is_druggable = site.druggability.overall >= 0.45;
                log::info!("  Site {}: CV penalty cv={:.4} factor={:.3} drug {:.3}->{:.3}{}",
                    site.cluster_id, cv_volume, penalty, old_drug, site.druggability.overall,
                    if site.druggability.is_druggable { "" } else { " NOT_DRUGGABLE" });
            }

            // ---- Physics-informed quality reranking ----
            // v5 composite ranking is applied AFTER all physics signals are computed.
            // See "COMPOSITE PHYSICS RANKING (v5)" block below.

            log::info!("  Site {}: physics signals [onset={:.3} srcDiv={:.2} srcEnt={:.2} burial={:.1} deepFrac={:.2} burialScore={:.3}]",
                site.cluster_id, onset_score, source_diversity, source_entropy,
                mean_burial, deep_fraction, burial_score);

            // ─── Sphericity: tight cluster (sphere) vs elongated (groove) ───
            let sphericity_score = if site_spikes.len() >= 10 {
                let n = site_spikes.len() as f32;
                let mx = site_spikes.iter().map(|s| s.position[0]).sum::<f32>() / n;
                let my = site_spikes.iter().map(|s| s.position[1]).sum::<f32>() / n;
                let mz = site_spikes.iter().map(|s| s.position[2]).sum::<f32>() / n;

                let mut cov = [0.0f32; 6]; // [xx, xy, xz, yy, yz, zz]
                for s in &site_spikes {
                    let dx = s.position[0] - mx;
                    let dy = s.position[1] - my;
                    let dz = s.position[2] - mz;
                    cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
                    cov[3] += dy * dy; cov[4] += dy * dz; cov[5] += dz * dz;
                }
                for c in cov.iter_mut() { *c /= n; }

                // Eigenvalues via Cardano's formula for 3x3 symmetric matrix
                let a = cov[0]; let b = cov[3]; let c_val = cov[5];
                let d = cov[1]; let e = cov[2]; let f = cov[4];

                let p1 = d*d + e*e + f*f;
                if p1 < 1e-10 {
                    let mut eigs = [a, b, c_val];
                    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                    if eigs[2] > 1e-10 { eigs[0] / eigs[2] } else { 0.0 }
                } else {
                    let q = (a + b + c_val) / 3.0;
                    let p2 = (a - q).powi(2) + (b - q).powi(2) + (c_val - q).powi(2) + 2.0 * p1;
                    let p = (p2 / 6.0).sqrt();

                    let b00 = (a - q) / p; let b11 = (b - q) / p; let b22 = (c_val - q) / p;
                    let b01 = d / p; let b02 = e / p; let b12 = f / p;

                    let det_b = b00 * (b11 * b22 - b12 * b12)
                              - b01 * (b01 * b22 - b12 * b02)
                              + b02 * (b01 * b12 - b11 * b02);

                    let half_det = (det_b / 2.0).clamp(-1.0, 1.0);
                    let phi = half_det.acos() / 3.0;

                    let eig1 = q + 2.0 * p * phi.cos();
                    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f32::consts::PI / 3.0).cos();
                    let eig2 = 3.0 * q - eig1 - eig3;

                    let mut eigs = [eig1, eig2, eig3];
                    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                    let min_eig = eigs[0].max(0.0);
                    let max_eig = eigs[2].max(1e-10);
                    (min_eig / max_eig).clamp(0.0, 1.0)
                }
            } else {
                0.0
            };

            // ─── Water displacement signal: rank-normalized within protein ───
            // Mean wd_change is uniform across sites (~0.02 for all).
            // Instead use the VARIANCE of wd_change: sites with high variance
            // have intermittent strong dewetting events (pocket opening).
            // Rank-normalize: highest variance site = 1.0, lowest = 0.0.
            let wd_coherence = if site_spikes.len() >= 10 {
                let mean_wd: f32 = site_spikes.iter()
                    .map(|s| s.wd_change).sum::<f32>() / site_spikes.len() as f32;
                let var_wd: f32 = site_spikes.iter()
                    .map(|s| (s.wd_change - mean_wd).powi(2))
                    .sum::<f32>() / site_spikes.len() as f32;
                // Store raw variance; will be rank-normalized in v5 block
                var_wd
            } else {
                0.0
            };

            // ─── Breathing: variance of burial across frames = pocket dynamics ───
            let breathing_score = if site_spikes.len() >= 20 {
                // Finer frame window (200 steps vs 1000) to capture sub-nanosecond
                // pocket dynamics. With dt=0.002ps and fused_steps=4, 200 steps =
                // 0.4ps windows — enough to see water entry/exit events.
                let breath_frame_window = 200i32;
                let mut frame_burials: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
                for s in &site_spikes {
                    let frame = s.timestep / breath_frame_window;
                    frame_burials.entry(frame).or_default().push(s.n_residues as f32);
                }
                let frame_means: Vec<f32> = frame_burials.values()
                    .filter(|v| !v.is_empty())
                    .map(|v| v.iter().sum::<f32>() / v.len() as f32)
                    .collect();

                if frame_means.len() >= 3 {
                    let global_mean = frame_means.iter().sum::<f32>() / frame_means.len() as f32;
                    if global_mean > 0.5 {
                        let variance = frame_means.iter()
                            .map(|&m| (m - global_mean).powi(2))
                            .sum::<f32>() / frame_means.len() as f32;
                        let cv = variance.sqrt() / global_mean;
                        // Normalize: CV of 0.5 = maximum breathing score.
                        // Previous /2.0 made even moderate CV (~0.15-0.3) invisible.
                        (cv / 0.5).clamp(0.0, 1.0)
                    } else { 0.0 }
                } else { 0.0 }
            } else {
                0.0
            };

            log::info!("  Site {}: spatial signals [sphericity={:.3} wdCoherence={:.3} breathing={:.3}]",
                site.cluster_id, sphericity_score, wd_coherence, breathing_score);
            spatial_signals.insert(site.cluster_id, (sphericity_score, wd_coherence, breathing_score));

            // ─── COMPOSITE PHYSICS RANKING (v5) — applied AFTER all signals computed ───
            {
                let is_viable_pocket = mean_burial >= 2.0
                    && site_spikes.len() >= 20
                    && site.estimated_volume >= 30.0;

                // Recompute enclosure (originally in inner scope above)
                let n_lining_f = site.lining_residues.len() as f32;
                let encl = if site.estimated_volume > 1.0 {
                    n_lining_f / site.estimated_volume.powf(0.667)
                } else { n_lining_f };

                // delta_g: STI returns POSITIVE values (~6 kcal/mol) for this system.
                // Lower positive = more favorable (closer to zero = less unfavorable).
                // Rank by inverse: sites with the LOWEST dG score highest.
                // Use rank-normalized scoring: compute within-protein percentile.
                let delta_g_score = if let Some(bfe) = sti_results.get(&site.cluster_id) {
                    let dg = bfe.effective_delta_g_kcal_mol as f32;
                    // Collect all dG values for this protein to rank-normalize
                    let all_dg: Vec<f32> = sti_results.values()
                        .map(|b| b.effective_delta_g_kcal_mol as f32)
                        .collect();
                    if all_dg.len() > 1 {
                        let min_dg = all_dg.iter().cloned().fold(f32::MAX, f32::min);
                        let max_dg = all_dg.iter().cloned().fold(f32::MIN, f32::max);
                        let range = (max_dg - min_dg).max(0.01);
                        // Lower dG = higher score (inverted, normalized to [0,1])
                        1.0 - ((dg - min_dg) / range).clamp(0.0, 1.0)
                    } else {
                        0.5 // single site — neutral
                    }
                } else {
                    0.3
                };

                let uv_s = uv_enrichment_scores.get(&site.cluster_id).copied().unwrap_or(0.0);

                // Per-spike quality (recompute — was in inner scope)
                let spk_q: f32 = if !site_spikes.is_empty() {
                    let sum: f32 = site_spikes.iter().map(|s| {
                        let b = (s.n_residues as f32 / 6.0).min(1.0);
                        let a = (s.n_nearby_excited as f32 / 3.0).min(1.0);
                        let w = (s.wd_change * 20.0).min(1.0);
                        let i = (s.intensity / 30.0).min(1.0);
                        0.40 * b + 0.20 * a + 0.20 * w + 0.20 * i
                    }).sum();
                    sum / site_spikes.len() as f32
                } else { 0.0 };

                let old_q = site.quality_score;

                // Rank-normalize wd_coherence (raw variance) within this protein
                let wd_norm = {
                    // Collect all wd variances computed so far
                    let all_wd: Vec<f32> = spatial_signals.values().map(|&(_, w, _)| w).collect();
                    if all_wd.len() > 1 {
                        let min_w = all_wd.iter().cloned().fold(f32::MAX, f32::min);
                        let max_w = all_wd.iter().cloned().fold(f32::MIN, f32::max);
                        let range = (max_w - min_w).max(1e-10);
                        ((wd_coherence - min_w) / range).clamp(0.0, 1.0)
                    } else { 0.5 }
                };

                // Lining density: n_lining / vol^0.33 — best single feature
                // from 2.5M-trial weight optimization (0.32 weight when solo)
                let lining_density = if site.estimated_volume > 1.0 {
                    n_lining_f / site.estimated_volume.powf(0.33)
                } else { n_lining_f };
                // Rank-normalize lining_density within protein
                // (computed inline since we don't have all sites' values yet —
                // use sigmoid normalization centered on typical value)
                let lining_density_norm = 1.0 / (1.0 + (-2.0 * (lining_density - 2.5)).exp());

                // Hysteresis asymmetry: pull directly from PRISM-Therm
                // if available (computed later in thermo-rerank, but we can
                // read the raw asymmetry from the site's therm data if injected)
                // For now, use the site's existing druggability as a proxy for
                // the thermo signal — the multiplicative thermo-rerank at 40%
                // handles the actual hysteresis. The direct integration will
                // happen when we restructure to compute SDST before ranking.

                if is_viable_pocket {
                    // v6 weights — optimized via 2.5M-trial search on SNDC
                    // + cross-target analysis. Key changes from v5:
                    // - burial raised 0.18→0.22 (consistently discriminates)
                    // - lining_density added at 0.12 (best single feature)
                    // - onset/sphericity reduced (anti-correlate on some targets)
                    // - dead signals removed (source_entropy was uniform 0.667)
                    site.quality_score =
                        0.22 * burial_score +               // recentered sigmoid, 3x range
                        0.12 * lining_density_norm +        // NEW: n_lining/vol^0.33
                        0.10 * encl.clamp(0.0, 2.0) / 2.0 + // enclosure
                        0.08 * delta_g_score +              // Jarzynski ΔG (cumulant)
                        0.08 * onset_score +                // temporal onset (reduced)
                        0.06 * sphericity_score +           // spatial concentration (reduced)
                        0.08 * uv_s +                       // UV enrichment
                        0.06 * (spk_q * 2.0).clamp(0.0, 1.0) + // per-spike quality
                        0.06 * source_diversity +           // UV/LIF balance + EFP
                        0.06 * breathing_score +            // pocket dynamics
                        0.04 * wd_norm +                    // water displacement
                        0.04 * (source_entropy / 1.1).clamp(0.0, 1.0); // source entropy
                } else {
                    site.quality_score = -1.0;
                }

                log::info!("  Site {}: v5 composite {:.3}->{:.3} [bur={:.2} dG={:.2} encl={:.2} ons={:.2} sph={:.2} src={:.2} uv={:.2} spkQ={:.2} br={:.2} wd={:.2}{}]",
                    site.cluster_id, old_q, site.quality_score,
                    burial_score, delta_g_score, encl, onset_score,
                    sphericity_score, source_diversity, uv_s, spk_q,
                    breathing_score, wd_norm,
                    if !is_viable_pocket { " FILTERED" } else { "" });
            }

            // ─── Peak centroid refinement (multi-stream path) ───
            // For medium sites (300-500 Å³), spike-weighted centroid can drift.
            // Peak centroid (top-50 intensity²-weighted) tracks the hotspot.
            // SKIP for mega-pockets (>500Å³) — LIGSITE geometric centroid is
            // empirically validated as superior (1hhp: 1.6Å geometric vs 9.7Å
            // spike-weighted). Peak centroid would negate that fix.
            if site.estimated_volume > 300.0 && site.estimated_volume <= 500.0 && site_spikes.len() >= 20 {
                // Collect top 50 hottest spikes by intensity
                let mut top_spikes: Vec<(f32, [f32; 3])> = site_spikes.iter()
                    .map(|s| (s.intensity, s.position))
                    .collect();
                top_spikes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                top_spikes.truncate(50);

                let mut pw = [0.0f64; 3];
                let mut pws = 0.0f64;
                for &(intensity, pos) in &top_spikes {
                    let w2 = (intensity as f64).powi(2);
                    pw[0] += pos[0] as f64 * w2;
                    pw[1] += pos[1] as f64 * w2;
                    pw[2] += pos[2] as f64 * w2;
                    pws += w2;
                }
                if pws > 1e-12 {
                    let pc = [
                        (pw[0] / pws) as f32,
                        (pw[1] / pws) as f32,
                        (pw[2] / pws) as f32,
                    ];
                    let dist = ((pc[0] - cx).powi(2) + (pc[1] - cy).powi(2) + (pc[2] - cz).powi(2)).sqrt();
                    let max_shift = site.estimated_volume.cbrt() * 2.0;
                    if dist > 2.0 && dist < max_shift {
                        log::info!("  Site {}: peak centroid shift {:.1}Å (vol={:.0}Å³, {} top spikes)",
                            site.cluster_id, dist, site.estimated_volume, top_spikes.len());
                        site.centroid = pc;
                    }
                }
            }

            // spike_frames: frames where this site had spikes
            let spike_frames: Vec<usize> = frame_spike_counts.iter().enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(i, _)| i)
                .collect();

            // spike_amplitudes: mean intensity per active frame
            let spike_amplitudes: Vec<f32> = spike_frames.iter()
                .map(|&f| {
                    if frame_spike_counts[f] > 0 {
                        frame_intensity_sums[f] / frame_spike_counts[f] as f32
                    } else { 0.0 }
                })
                .collect();

            // inter_spike_intervals: gaps between active frames
            let inter_spike_intervals: Vec<f32> = if spike_frames.len() >= 2 {
                spike_frames.windows(2)
                    .map(|w| (w[1] - w[0]) as f32)
                    .collect()
            } else {
                Vec::new()
            };

            cryptic_sites_json.push(serde_json::json!({
                "site_id": site.cluster_id,
                "centroid": site.centroid,
                "spike_count": site_spikes.len(),
                "consensus_spike_count": site.spike_count,
                "spike_source": if !site.spike_indices.is_empty() { "cluster_assigned" } else { "radius_fallback" },
                "spike_frames": spike_frames,
                "spike_amplitudes": spike_amplitudes,
                "inter_spike_intervals": inter_spike_intervals,
                "volume": site.estimated_volume,
                "druggability": site.druggability.overall,
                "classification": format!("{:?}", site.classification),
            }));
        }

        // ── PRISM-Therm (multi-stream path) — run BEFORE final reranking ──
        let prism_therm_result: Option<PrismThermAnalysis> = if args.prism_therm {
            log::info!("\n[PRISM-Therm] Initializing SDST thermodynamic analysis (multi-stream)...");
            match SdstBridge::new(&topology, &protocol, all_stream_spikes.len()) {
                Err(e) => { log::warn!("  PRISM-Therm: SDST init failed ({})", e); None }
                Ok(bridge) => {
                    match bridge.ingest_all_spikes(&all_stream_spikes) {
                        Err(e) => { log::warn!("  PRISM-Therm: ingest failed ({})", e); None }
                        Ok(event_count) => {
                            log::info!("  PRISM-Therm: {} events ingested", event_count);
                            match bridge.analyze(&clustered_sites) {
                                Err(e) => { log::warn!("  PRISM-Therm: analysis failed ({})", e); None }
                                Ok(analysis) => {
                                    log::info!("  PRISM-Therm: {}/{} hysteretic | {} SDST pockets",
                                        analysis.hysteretic_site_count,
                                        clustered_sites.len(),
                                        analysis.global_pockets.len());
                                    Some(analysis)
                                }
                            }
                        }
                    }
                }
            }
        } else {
            None
        };

        // ── Thermodynamic reranking: boost quality_score with tau + z-score ──
        // When PRISM-Therm data is available, fold thermodynamic signals into
        // the quality score. These signals are unique to PRISM-4D — no other
        // tool can compute them.
        if let Some(ref analysis) = prism_therm_result {
            log::info!("[Thermo-Rerank] Applying thermodynamic quality boost...");
            for site in clustered_sites.iter_mut() {
                if let Some(therm) = analysis.sites.iter().find(|s| s.site_id == site.cluster_id as i32) {
                    let old_q = site.quality_score;

                    // 1. SOC criticality: tau in [1.2, 1.5] is the self-organized
                    //    critical regime — most indicative of functional binding.
                    //    tau=0 (no data) or tau>2 (noise) score low.
                    let tau_q = if therm.tau >= 1.2 && therm.tau <= 1.5 {
                        1.0_f32
                    } else if therm.tau > 1.0 && therm.tau < 1.2 {
                        (therm.tau - 1.0) / 0.2  // ramp 1.0→1.2
                    } else if therm.tau > 1.5 && therm.tau < 2.0 {
                        1.0 - (therm.tau - 1.5) / 0.5  // decay 1.5→2.0
                    } else {
                        0.0
                    };

                    // 2. Thermal asymmetry z-score: high z = hysteretic/dynamic.
                    //    z > 1.5 = cryptic candidate, z > 0.5 = dynamic, z < -0.5 = inert.
                    //    Clamp to [0, 1] using sigmoid-like mapping.
                    let z = therm.relative_asymmetry;
                    let asym_q = if z > 1.5 {
                        1.0_f32
                    } else if z > 0.0 {
                        z / 1.5
                    } else {
                        0.0
                    };

                    // 3. Thermodynamic class bonus: CRYPTIC and DYNAMIC sites
                    //    get a direct boost; INERT sites get penalized slightly.
                    let class_q = match therm.therm_class {
                        ThermClass::Cryptic  => 1.0_f32,
                        ThermClass::Dynamic  => 0.7,
                        ThermClass::Responsive => 0.4,
                        ThermClass::Inert    => 0.1,
                    };

                    // Multiplicative thermodynamic boost: weight search found
                    // hysteresis is 3rd most important feature (0.18 weight).
                    // Previous 20% max was insufficient. Increased to 40% max
                    // with asymmetry weighted 2x vs tau and class (asymmetry
                    // is the strongest thermodynamic discriminator).
                    let thermo_avg = (tau_q + 2.0 * asym_q + class_q) / 4.0;
                    let thermo_factor = 1.0 + 0.40 * thermo_avg;
                    site.quality_score = old_q * thermo_factor;

                    log::info!("  Site {}: thermo-rerank {:.3}->{:.3} [tau={:.2}→q={:.2} z={:.2}→q={:.2} class={}→q={:.2}]",
                        site.cluster_id, old_q, site.quality_score,
                        therm.tau, tau_q, therm.relative_asymmetry, asym_q,
                        therm.therm_class, class_q);
                }
            }
        }

        // ---- Re-sort by updated quality_score ----
        // Build permutation indices so we can reorder the JSON vectors too
        let mut ranked_indices: Vec<usize> = (0..clustered_sites.len().min(100)).collect();
        ranked_indices.sort_by(|&a, &b|
            clustered_sites[b].quality_score
                .partial_cmp(&clustered_sites[a].quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        );

        // Reorder all three parallel vectors + sites
        let reordered_sites: Vec<_> = ranked_indices.iter().map(|&i| &clustered_sites[i]).collect();
        let reordered_pockets: Vec<_> = ranked_indices.iter().map(|&i| all_pockets_json[i].clone()).collect();
        let reordered_cryptic: Vec<_> = ranked_indices.iter().map(|&i| cryptic_sites_json[i].clone()).collect();

        log::info!("Quality reranking applied. New top-3: {}",
            ranked_indices.iter().take(3)
                .map(|&i| format!("site{}(q={:.3})", clustered_sites[i].cluster_id, clustered_sites[i].quality_score))
                .collect::<Vec<_>>().join(", "));

        let json_path = output_base.with_extension("binding_sites.json");

        // Build per-site JSON, merging PRISM-Therm therm_class when available
        let mut ms_sites_json: Vec<serde_json::Value> = reordered_sites.iter().map(|s| {
            let cat_count = s.lining_residues.iter()
                .filter(|r| catalytic_residues.contains(&r.resname.as_str())).count();
            let (ps_onset, ps_src_div, ps_burial, ps_burial_score) =
                physics_signals.get(&s.cluster_id).copied().unwrap_or((0.0, 0.0, 0.0, 0.0));
            serde_json::json!({
                "id": s.cluster_id,
                "centroid": s.centroid,
                "volume": s.estimated_volume,
                "spike_count": s.spike_count,
                "quality_score": s.quality_score,
                "druggability": s.druggability.overall,
                "is_druggable": s.druggability.is_druggable,
                "classification": format!("{:?}", s.classification),
                "aromatic_score": s.aromatic_proximity.as_ref().map(|p| p.aromatic_score),
                "catalytic_residue_count": cat_count,
                "onset_score": ps_onset,
                "source_diversity": ps_src_div,
                "mean_burial": ps_burial,
                "burial_score": ps_burial_score,
                "lining_residues": s.lining_residues.iter().map(|r| {
                    serde_json::json!({
                        "chain": r.chain, "resid": r.resid, "resname": r.resname,
                        "min_distance": r.min_distance, "n_atoms": r.n_atoms_in_pocket,
                        "is_catalytic": catalytic_residues.contains(&r.resname.as_str()),
                    })
                }).collect::<Vec<_>>(),
                "residue_ids": s.lining_residue_ids(),
            })
        }).collect();

        // Inject PRISM-Therm classification into each site
        if let Some(ref analysis) = prism_therm_result {
            for site_json in ms_sites_json.iter_mut() {
                let site_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
                if let Some(therm_site) = analysis.sites.iter().find(|s| s.site_id == site_id) {
                    site_json["therm_class"] = serde_json::Value::String(
                        therm_site.therm_class.to_string()
                    );
                    site_json["hysteresis_asymmetry"] = serde_json::json!(therm_site.asymmetry_score);
                    site_json["relative_asymmetry"] = serde_json::json!(therm_site.relative_asymmetry);
                    site_json["ccns_tau"] = serde_json::json!(therm_site.tau);
                    if therm_site.therm_class.to_string() == "CRYPTIC" {
                        site_json["classification"] = serde_json::Value::String("Cryptic".to_string());
                    }
                }
            }
        }

        // Inject spatial signals (sphericity, wd_coherence, breathing) into each site JSON
        for site_json in ms_sites_json.iter_mut() {
            let site_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
            if let Some(&(sph, wdc, breath)) = spatial_signals.get(&site_id) {
                site_json["sphericity"] = serde_json::json!(sph);
                site_json["wd_coherence"] = serde_json::json!(wdc);
                site_json["breathing_score"] = serde_json::json!(breath);
            }
        }

        // Inject STI free energy and UV enrichment into each site JSON
        for site_json in ms_sites_json.iter_mut() {
            let site_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
            if let Some(bfe) = sti_results.get(&site_id) {
                site_json["delta_g_sti_kcal_mol"] = serde_json::json!(bfe.delta_g_sti_kcal_mol);
                site_json["effective_delta_g_kcal_mol"] = serde_json::json!(bfe.effective_delta_g_kcal_mol);
                site_json["delta_g_aromatic_kcal_mol"] = serde_json::json!(bfe.delta_g_aromatic_kcal_mol);
                site_json["delta_g_dewetting_kcal_mol"] = serde_json::json!(bfe.delta_g_dewetting_kcal_mol);
                site_json["delta_g_electrostatic_kcal_mol"] = serde_json::json!(bfe.delta_g_electrostatic_kcal_mol);
                site_json["delta_g_cooperative_kcal_mol"] = serde_json::json!(bfe.delta_g_cooperative_kcal_mol);
                site_json["kinetic_accessibility"] = serde_json::json!(bfe.kinetic_accessibility);
                site_json["sti_n_voxels"] = serde_json::json!(bfe.n_voxels);
                site_json["sti_n_spikes"] = serde_json::json!(bfe.n_spikes);
            }
            if let Some(&uv_score) = uv_enrichment_scores.get(&site_id) {
                site_json["uv_enrichment_score"] = serde_json::json!(uv_score);
            }
        }

        let json_output = serde_json::json!({
            "structure": structure_name,
            "mode": "multi_stream",
            "n_streams": n_streams,
            "total_steps_per_stream": steps_per_stream,
            "simulation_time_sec": sim_elapsed.as_secs_f64(),
            "consensus_threshold": consensus_threshold,
            "per_stream_stats": per_stream_stats,
            "binding_sites": clustered_sites.len(),
            "druggable_sites": clustered_sites.iter().filter(|s| s.druggability.is_druggable).count(),
            "lining_residue_cutoff_angstroms": args.lining_cutoff,
            "sites": ms_sites_json,
            "all_pockets": reordered_pockets,
            "cryptic_sites": reordered_cryptic,
            "prism_therm": prism_therm_result,
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON: {}", json_path.display());

        // ── PRISM-Therm standalone report (multi-stream) ──
        if let Some(ref analysis) = prism_therm_result {
            let site_centroids: Vec<([f32; 3], i32)> = clustered_sites.iter()
                .map(|s| (s.centroid, s.cluster_id))
                .collect();
            let report = sdst_report::build_report(analysis, &topology, &structure_name, &site_centroids);
            sdst_report::print_summary_table(&report);
            if let Err(e) = sdst_report::write_json(&report, &args.output, &structure_name) {
                log::warn!("  PRISM-Therm JSON write failed: {}", e);
            }
            if let Err(e) = sdst_report::write_druggability_pdb(&report, &topology, &args.output, &structure_name) {
                log::warn!("  PRISM-Therm druggability PDB failed: {}", e);
            }
        }
    }

    // Export spike events with enhanced metadata for pharmacophore mapping
    // Gated by --emit-spike-json (off by default, saves ~55s per run)
    if args.emit_spike_json && !all_stream_spikes.is_empty() && !clustered_sites.is_empty() {
        let arom_type_name = |t: i32| -> &str {
            match t { 0 => "TRP", 1 => "TYR", 2 => "PHE", 3 => "SS", 4 => "BNZ", 5 => "CATION", 6 => "ANION", _ => "UNK" }
        };
        // Closure to determine CCNS phase from timestep using protocol parameters
        let phase_label = |ts: i32| -> &str {
            let p1 = protocol.cold_hold_steps;
            let p2 = p1 + protocol.ramp_steps;
            let p3 = p2 + protocol.warm_hold_steps;
            let p4 = p3 + protocol.ramp_down_steps;
            if ts < p1 { "cold_hold" }
            else if ts < p2 { "heating" }
            else if ts < p3 { "warm_hold" }
            else if ts < p4 { "cooling" }
            else { "cold_return" }
        };
        let lining_cutoff = args.lining_cutoff;
        for site in &clustered_sites {
            let site_radius = lining_cutoff + 2.0;
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];
            // Collect raw spikes for this site
            let raw_site_spikes: Vec<_> = all_stream_spikes.iter()
                .filter(|s| {
                    let dx = s.position[0] - cx;
                    let dy = s.position[1] - cy;
                    let dz = s.position[2] - cz;
                    (dx*dx + dy*dy + dz*dz).sqrt() <= site_radius
                })
                .collect();
            // Compute open_frequency: fraction of simulation frames with spike activity
            // Use frame_index (timestep / 1000) from actual spike data
            let unique_frames: std::collections::HashSet<i32> = raw_site_spikes.iter()
                .map(|s| s.timestep / 1000)
                .collect();
            let max_frame = raw_site_spikes.iter().map(|s| s.timestep / 1000).max().unwrap_or(0);
            let total_frames = (max_frame + 1).max(1) as f32;
            let open_frequency = unique_frames.len() as f32 / total_frames;
            let site_spikes: Vec<serde_json::Value> = raw_site_spikes.iter()
                .map(|s| {
                    let pos = s.position;
                    let intensity = s.intensity;
                    let atype = s.aromatic_type;
                    let wl = s.wavelength_nm;
                    let src = s.spike_source;
                    let arom_res = s.aromatic_residue_id;
                    let wd = s.water_density;
                    let ve = s.vibrational_energy;
                    let nne = s.n_nearby_excited;
                    let ts = s.timestep;
                    serde_json::json!({
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2],
                        "intensity": intensity,
                        "type": arom_type_name(atype),
                        "wavelength_nm": wl,
                        "spike_source": match src { 1 => "UV", 3 => "EFP", _ => "LIF" },
                        "aromatic_residue_id": arom_res,
                        "water_density": wd,
                        "vibrational_energy": ve,
                        "n_nearby_excited": nne,
                        "timestep": ts,
                        "frame_index": ts / 1000,
                        "ccns_phase": phase_label(ts),
                    })
                })
                .collect();
            let spike_json = serde_json::json!({
                "site_id": site.cluster_id,
                "centroid": site.centroid,
                "n_spikes": site_spikes.len(),
                "lining_cutoff": args.lining_cutoff,
                "open_frequency": open_frequency,
                "spikes": site_spikes,
            });
            let spike_path = output_base.with_extension(
                format!("site{}.spike_events.json", site.cluster_id)
            );
            std::fs::write(&spike_path, serde_json::to_string_pretty(&spike_json)?)?;
            log::info!("  Spike events: {} ({} spikes, f_open={:.3})", spike_path.display(), site_spikes.len(), open_frequency);
        }
    } else if !args.emit_spike_json && !all_stream_spikes.is_empty() && !clustered_sites.is_empty() {
        log::info!("  Spike event JSON skipped (use --emit-spike-json to enable)");
    }

    // Write per-stream ensemble trajectories
    for (i, snapshots) in all_stream_snapshots.iter().enumerate() {
        if !snapshots.is_empty() {
            let stem = structure_name.strip_suffix(".topology").unwrap_or(&structure_name);
            let stream_base = args.output.join(format!("{}_stream{:02}", stem, i));
            write_ensemble_trajectory(snapshots, &topology, &stream_base)?;
            log::info!("  ✓ Trajectory stream {}: {} frames", i, snapshots.len());
        }
    }

    let total_time = total_start.elapsed();
    let druggable = clustered_sites.iter().filter(|s| s.druggability.is_druggable).count();

    log::info!("\n╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  MULTI-STREAM PIPELINE COMPLETE                               ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    log::info!("║  Structure: {:<48} ║", structure_name);
    log::info!("║  CUDA streams: {:<44} ║", n_streams);
    log::info!("║  Steps/stream: {:<44} ║", steps_per_stream);
    log::info!("║  Simulation time: {:<40.1}s ║", sim_elapsed.as_secs_f64());
    log::info!("║  Total time: {:<46.1}s ║", total_time.as_secs_f64());
    log::info!("║  Consensus sites: {:<41} ║", clustered_sites.len());
    log::info!("║  Druggable sites: {:<41} ║", druggable);
    log::info!("║  Consensus: {}/{:<41} ║", consensus_threshold, n_streams);
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Extract aromatic residue positions from topology
#[cfg(feature = "gpu")]
fn extract_aromatic_positions(topology: &PrismPrepTopology) -> Vec<(u32, u8, [f32; 3])> {
    let mut aromatics = Vec::new();

    // Get aromatic residue indices
    let aromatic_residues = topology.aromatic_residues();

    for &res_idx in &aromatic_residues {
        // Find atoms belonging to this residue
        let atoms: Vec<usize> = topology.residue_ids.iter()
            .enumerate()
            .filter(|(_, &r)| r == res_idx)
            .map(|(i, _)| i)
            .collect();

        if atoms.is_empty() {
            continue;
        }

        // Compute centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for &atom_idx in &atoms {
            cx += topology.positions[atom_idx * 3];
            cy += topology.positions[atom_idx * 3 + 1];
            cz += topology.positions[atom_idx * 3 + 2];
        }
        let n = atoms.len() as f32;
        cx /= n;
        cy /= n;
        cz /= n;

        // Determine aromatic type from residue name (use first atom's name, not res_idx as atom index)
        let aromatic_type = if let Some(name) = topology.residue_names.get(atoms[0]) {
            match name.trim().to_uppercase().as_str() {
                "TRP" => 0u8,
                "TYR" => 1u8,
                "PHE" => 2u8,
                _ => continue,
            }
        } else {
            continue;
        };

        aromatics.push((res_idx as u32, aromatic_type, [cx, cy, cz]));
    }

    aromatics
}

/// Build ClusteredBindingSite from clustering result
#[cfg(feature = "gpu")]
fn build_sites_from_clustering(
    spike_events: &[prism_nhs::fused_engine::GpuSpikeEvent],
    result: &prism_nhs::rt_clustering::RtClusteringResult,
) -> Vec<ClusteredBindingSite> {
    use std::collections::HashMap;
    use prism_nhs::{DruggabilityScore, SiteClassification};

    let mut cluster_spikes: HashMap<i32, Vec<(usize, &prism_nhs::fused_engine::GpuSpikeEvent)>> = HashMap::new();

    for (idx, (spike, &cluster_id)) in spike_events.iter()
        .zip(result.cluster_ids.iter())
        .enumerate()
    {
        if cluster_id >= 0 {
            cluster_spikes.entry(cluster_id).or_default().push((idx, spike));
        }
    }

    let mut sites = Vec::new();
    for (cluster_id, spikes) in cluster_spikes {
        if spikes.is_empty() {
            continue;
        }

        let mut centroid = [0.0f32; 3];
        let mut sum_intensity = 0.0f32;
        let mut min_pos = [f32::MAX; 3];
        let mut max_pos = [f32::MIN; 3];

        for (_, spike) in &spikes {
            // Copy packed fields to avoid alignment issues
            let pos = spike.position;
            let intensity = spike.intensity;
            centroid[0] += pos[0];
            centroid[1] += pos[1];
            centroid[2] += pos[2];
            sum_intensity += intensity;
            for i in 0..3 {
                min_pos[i] = min_pos[i].min(pos[i]);
                max_pos[i] = max_pos[i].max(pos[i]);
            }
        }

        let n = spikes.len() as f32;
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        let bounding_box = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];

        // Estimate pocket volume using voxel density method (2Å resolution)
        // Cavity volume estimation via convex hull of spike positions within pocket
        // Spikes mark the pocket boundary surface; the enclosed volume approximates the cavity
        let pocket_radius = 8.0f32;
        let estimated_volume = {
            // Collect spike positions within pocket radius of centroid
            let mut pocket_points: Vec<[f32; 3]> = Vec::new();
            for (_, spike) in &spikes {
                let pos = spike.position;
                let dx = pos[0] - centroid[0];
                let dy = pos[1] - centroid[1];
                let dz = pos[2] - centroid[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= pocket_radius {
                    pocket_points.push(pos);
                }
            }
            if pocket_points.len() >= 4 {
                // Compute volume via Monte Carlo sampling within the point cloud
                // 1. Find tight bounding box of pocket points
                let mut pmin = [f32::MAX; 3];
                let mut pmax = [f32::MIN; 3];
                for p in &pocket_points {
                    for d in 0..3 {
                        pmin[d] = pmin[d].min(p[d]);
                        pmax[d] = pmax[d].max(p[d]);
                    }
                }
                // 2. Grid at 1A resolution, count points inside pocket
                //    A point is "inside" if it's closer to a spike than to any
                //    protein atom = void space near the surface
                let grid_step = 1.0f32;
                let mut void_count = 0u32;
                let mut total_count = 0u32;
                let mut gx = pmin[0];
                while gx <= pmax[0] {
                    let mut gy = pmin[1];
                    while gy <= pmax[1] {
                        let mut gz = pmin[2];
                        while gz <= pmax[2] {
                            total_count += 1;
                            // Check if grid point is near spike surface (within 2A of any spike)
                            let mut near_spike = false;
                            let mut min_spike_dist = f32::MAX;
                            for p in &pocket_points {
                                let d2 = (gx - p[0]).powi(2) + (gy - p[1]).powi(2) + (gz - p[2]).powi(2);
                                let d = d2.sqrt();
                                if d < min_spike_dist { min_spike_dist = d; }
                                if d < 3.0 { near_spike = true; break; }
                            }
                            // Point is within pocket envelope
                            if near_spike {
                                void_count += 1;
                            }
                            gz += grid_step;
                        }
                        gy += grid_step;
                    }
                    gx += grid_step;
                }
                // Volume = counted grid points × grid_step³
                // Apply 0.5 correction: spikes sit on surface, interior is ~half the envelope
                let raw_vol = void_count as f32 * grid_step.powi(3);
                (raw_vol * 0.5).clamp(50.0, 2500.0)
            } else {
                // Fallback: too few points for meaningful volume
                100.0f32
            }
        };

        let avg_intensity = sum_intensity / n;
        let spike_count = spikes.len();

        let druggability = DruggabilityScore::from_site(estimated_volume, avg_intensity, &bounding_box);
        let classification = SiteClassification::from_properties(spike_count, estimated_volume, avg_intensity);

        // Initial quality estimate (overwritten by enclosure-based reranking later)
        let spike_quality = (spike_count as f32 / 100.0).clamp(0.0, 1.0);
        let intensity_quality = (avg_intensity / 10.0).clamp(0.0, 1.0);
        let quality_score = 0.3 * spike_quality + 0.3 * intensity_quality + 0.4 * druggability.overall;

        sites.push(ClusteredBindingSite {
            cluster_id,
            centroid,
            spike_count,
            spike_indices: spikes.iter().map(|(idx, _)| *idx).collect(),
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
            aromatic_proximity: None,
            lining_residues: Vec::new(),  // Computed later when topology available
        });
    }

    sites.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));
    sites
}

/// Merge symmetric binding sites for multi-chain (dimer) proteins.
///
/// For homodimers like HIV-1 protease, identical pockets appear on both chains.
/// This function detects symmetric site pairs and merges them, producing a single
/// site at the interface centroid with combined spike counts. Also detects and
/// preserves interface pockets (sites spanning multiple chains).
///
/// The merge criterion: two sites within `merge_radius` whose spike contributions
/// come predominantly from different chains are merged into one.
#[cfg(feature = "gpu")]
fn merge_symmetric_sites(
    sites: &mut Vec<ClusteredBindingSite>,
    spike_events: &[prism_nhs::fused_engine::GpuSpikeEvent],
    topology: &prism_nhs::input::PrismPrepTopology,
    merge_radius: f32,
) {
    use std::collections::{HashMap, HashSet};

    // Only relevant for multi-chain structures
    let unique_chains: HashSet<&str> = topology.chain_ids.iter().map(|s| s.as_str()).collect();
    if unique_chains.len() < 2 {
        return;
    }

    // Check for homodimer: chains with identical residue sequences
    let mut chain_sequences: HashMap<&str, Vec<&str>> = HashMap::new();
    for (atom_idx, chain) in topology.chain_ids.iter().enumerate() {
        let res_name = &topology.residue_names[atom_idx];
        let atom_name = topology.atom_names.get(atom_idx).map(|s| s.as_str()).unwrap_or("");
        if atom_name == "CA" {
            chain_sequences.entry(chain.as_str()).or_default().push(res_name.as_str());
        }
    }

    let chain_list: Vec<&str> = chain_sequences.keys().copied().collect();
    let mut is_homodimer = false;
    for i in 0..chain_list.len() {
        for j in (i + 1)..chain_list.len() {
            let seq_a = &chain_sequences[chain_list[i]];
            let seq_b = &chain_sequences[chain_list[j]];
            if seq_a.len() == seq_b.len() && seq_a == seq_b {
                is_homodimer = true;
            }
        }
    }

    if !is_homodimer {
        log::info!("  Multi-chain but not homodimer ({} chains, different sequences) — no merge needed",
            unique_chains.len());
        return;
    }

    log::info!("  Homodimer detected ({} chains, {} residues/chain) — checking for symmetric sites",
        unique_chains.len(),
        chain_sequences.values().next().map(|v| v.len()).unwrap_or(0));

    // Build residue → chain mapping for fast lookup
    // First, build a residue_id → chain_id map from atom data
    let mut residue_chain: HashMap<usize, String> = HashMap::new();
    for (atom_idx, &res_id) in topology.residue_ids.iter().enumerate() {
        if !residue_chain.contains_key(&res_id) {
            residue_chain.insert(res_id, topology.chain_ids[atom_idx].clone());
        }
    }

    // For each site, determine chain contributions from spike nearby_residues
    let mut site_chain_fractions: Vec<HashMap<String, usize>> = Vec::new();
    for site in sites.iter() {
        let mut chain_counts: HashMap<String, usize> = HashMap::new();
        for &spike_idx in &site.spike_indices {
            if spike_idx >= spike_events.len() { continue; }
            let spike = &spike_events[spike_idx];
            let n_res = spike.n_residues.min(8) as usize;
            for r in 0..n_res {
                let res_id = spike.nearby_residues[r];
                if res_id >= 0 {
                    if let Some(chain) = residue_chain.get(&(res_id as usize)) {
                        *chain_counts.entry(chain.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        site_chain_fractions.push(chain_counts);
    }

    // Find merge candidates: site pairs within merge_radius
    let n = sites.len();
    let mut merge_pairs: Vec<(usize, usize)> = Vec::new();
    let mut merged: HashSet<usize> = HashSet::new();

    for i in 0..n {
        if merged.contains(&i) { continue; }
        for j in (i + 1)..n {
            if merged.contains(&j) { continue; }

            let dx = sites[i].centroid[0] - sites[j].centroid[0];
            let dy = sites[i].centroid[1] - sites[j].centroid[1];
            let dz = sites[i].centroid[2] - sites[j].centroid[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist > merge_radius { continue; }

            // Check if sites are on different chains (symmetric pair)
            let chains_i: HashSet<&str> = site_chain_fractions[i].keys().map(|s| s.as_str()).collect();
            let chains_j: HashSet<&str> = site_chain_fractions[j].keys().map(|s| s.as_str()).collect();

            // Dominant chain for each site
            let dom_i = site_chain_fractions[i].iter().max_by_key(|&(_, v)| v).map(|(k, _)| k.as_str());
            let dom_j = site_chain_fractions[j].iter().max_by_key(|&(_, v)| v).map(|(k, _)| k.as_str());

            // Merge if: (a) dominant chains differ, or (b) both span the interface
            let should_merge = match (dom_i, dom_j) {
                (Some(ci), Some(cj)) => ci != cj,
                _ => false,
            } || (chains_i.len() > 1 && chains_j.len() > 1 && dist < merge_radius * 0.5);

            if should_merge {
                log::info!("    Merging site {} (chain {:?}) + site {} (chain {:?}), dist={:.1}Å",
                    sites[i].cluster_id, dom_i, sites[j].cluster_id, dom_j, dist);
                merge_pairs.push((i, j));
                merged.insert(i);
                merged.insert(j);
                break;  // Each site merges with at most one partner
            }
        }
    }

    if merge_pairs.is_empty() {
        log::info!("  No symmetric site pairs found within {:.1}Å", merge_radius);
        return;
    }

    // Execute merges: combine spike indices, recompute centroid
    let mut to_remove: HashSet<usize> = HashSet::new();
    for (i, j) in &merge_pairs {
        // Merge j into i
        let j_indices = sites[*j].spike_indices.clone();
        let j_count = sites[*j].spike_count;
        let j_intensity_sum = sites[*j].avg_intensity * j_count as f32;

        sites[*i].spike_indices.extend(j_indices);
        let combined_count = sites[*i].spike_count + j_count;
        let i_intensity_sum = sites[*i].avg_intensity * sites[*i].spike_count as f32;
        sites[*i].avg_intensity = (i_intensity_sum + j_intensity_sum) / combined_count as f32;
        sites[*i].spike_count = combined_count;

        // Recompute centroid from all spike positions
        let mut new_centroid = [0.0f32; 3];
        for &idx in &sites[*i].spike_indices {
            if idx < spike_events.len() {
                let pos = spike_events[idx].position;
                new_centroid[0] += pos[0];
                new_centroid[1] += pos[1];
                new_centroid[2] += pos[2];
            }
        }
        let n = sites[*i].spike_indices.len() as f32;
        if n > 0.0 {
            new_centroid[0] /= n;
            new_centroid[1] /= n;
            new_centroid[2] /= n;
        }
        sites[*i].centroid = new_centroid;

        to_remove.insert(*j);
    }

    // Remove merged sites (in reverse order to preserve indices)
    let mut remove_indices: Vec<usize> = to_remove.into_iter().collect();
    remove_indices.sort_unstable_by(|a, b| b.cmp(a));
    for idx in remove_indices {
        sites.remove(idx);
    }

    log::info!("  Merged {} symmetric pairs → {} sites remaining", merge_pairs.len(), sites.len());
}

/// Build consensus sites from per-replica clustering results
/// Sites must appear in at least `threshold` replicas within `spatial_tolerance` Angstroms
///
/// `stream_spike_offsets`: offset into the concatenated all_stream_spikes for each stream,
/// used to remap per-stream spike_indices to global indices for frame-aligned analysis.
#[cfg(feature = "gpu")]
fn build_consensus_sites(
    per_replica_sites: &[Vec<ClusteredBindingSite>],
    threshold: usize,
    spatial_tolerance: f32,
    stream_spike_offsets: &[usize],
) -> Vec<ClusteredBindingSite> {
    use prism_nhs::{DruggabilityScore, SiteClassification};

    if per_replica_sites.is_empty() {
        return Vec::new();
    }

    // Collect all sites from all replicas
    let mut all_sites: Vec<(usize, &ClusteredBindingSite)> = Vec::new();
    for (replica_idx, sites) in per_replica_sites.iter().enumerate() {
        for site in sites {
            all_sites.push((replica_idx, site));
        }
    }

    if all_sites.is_empty() {
        return Vec::new();
    }

    // Cluster sites spatially across replicas
    let mut consensus_clusters: Vec<Vec<(usize, &ClusteredBindingSite)>> = Vec::new();
    let mut assigned = vec![false; all_sites.len()];

    for i in 0..all_sites.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![all_sites[i]];
        assigned[i] = true;

        // Find all sites within spatial tolerance
        for j in (i + 1)..all_sites.len() {
            if assigned[j] {
                continue;
            }

            let dist = {
                let c1 = all_sites[i].1.centroid;
                let c2 = all_sites[j].1.centroid;
                ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2) + (c1[2] - c2[2]).powi(2)).sqrt()
            };

            if dist <= spatial_tolerance {
                cluster.push(all_sites[j]);
                assigned[j] = true;
            }
        }

        // Count unique replicas in this cluster
        let mut replica_set = std::collections::HashSet::new();
        for (replica_idx, _) in &cluster {
            replica_set.insert(*replica_idx);
        }

        // Only keep clusters that meet the threshold
        if replica_set.len() >= threshold {
            consensus_clusters.push(cluster);
        }
    }

    // Build consensus sites by averaging properties
    let mut consensus_sites = Vec::new();
    for (cluster_id, cluster) in consensus_clusters.iter().enumerate() {
        // Average centroid
        let mut centroid = [0.0f32, 0.0, 0.0];
        let mut total_spike_count = 0;
        let mut total_intensity = 0.0;
        let mut total_volume = 0.0;
        let mut total_quality = 0.0;

        for (_, site) in cluster {
            centroid[0] += site.centroid[0];
            centroid[1] += site.centroid[1];
            centroid[2] += site.centroid[2];
            total_spike_count += site.spike_count;
            total_intensity += site.avg_intensity;
            total_volume += site.estimated_volume;
            total_quality += site.quality_score;
        }

        let n = cluster.len() as f32;
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;
        let avg_spike_count = (total_spike_count as f32 / n) as usize;
        let avg_intensity = total_intensity / n;
        let avg_volume = total_volume / n;
        let avg_quality = total_quality / n;

        // Compute consensus bounding box dimensions (average across replicas)
        let mut total_bbox = [0.0f32; 3];
        for (_, site) in cluster {
            total_bbox[0] += site.bounding_box[0];
            total_bbox[1] += site.bounding_box[1];
            total_bbox[2] += site.bounding_box[2];
        }
        let bounding_box = [
            total_bbox[0] / n,
            total_bbox[1] / n,
            total_bbox[2] / n,
        ];

        let druggability = DruggabilityScore::from_site(avg_volume, avg_intensity, &bounding_box);
        let classification = SiteClassification::from_properties(avg_spike_count, avg_volume, avg_intensity);

        // Merge spike_indices from all contributing per-stream sites,
        // remapping local per-stream indices to global all_stream_spikes indices.
        // Only merge if we have valid offsets (stream_spike_offsets is non-empty).
        let merged_indices = if !stream_spike_offsets.is_empty() {
            let mut indices = Vec::new();
            for (replica_idx, site) in cluster {
                if let Some(&offset) = stream_spike_offsets.get(*replica_idx) {
                    for &local_idx in &site.spike_indices {
                        indices.push(offset + local_idx);
                    }
                }
            }
            indices.sort_unstable();
            indices.dedup();
            indices
        } else {
            Vec::new() // No offsets available; downstream uses nearest-centroid fallback
        };

        consensus_sites.push(ClusteredBindingSite {
            cluster_id: cluster_id as i32,
            centroid,
            spike_count: avg_spike_count,
            spike_indices: merged_indices,
            avg_intensity,
            estimated_volume: avg_volume,
            bounding_box,
            quality_score: avg_quality,
            druggability,
            classification,
            aromatic_proximity: None,
            lining_residues: Vec::new(),
        });
    }

    consensus_sites.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));
    consensus_sites
}

/// "Dynamic LIGSITE" — Geometry-first pocket detection with spike scoring.
///
/// Pipeline inversion: Geometry Proposes, Physics Disposes.
///
///   1. Build 3D boolean grid `is_protein` over entire protein (SES surface)
///   2. Flood-fill solvent accessibility from grid boundary
///   3. DAH depth field (distance transform from solvent surface)
///   4. Ray-cast ±X,±Y,±Z → `is_enclosed` boolean (≥ min_blocked directions blocked)
///   5. BFS connected components on enclosed voxels → PocketCandidates
///   6. O(N) spike overlay: map each spike to nearest pocket via component_id grid
///   7. Score pockets (spike density + depth penalty), rebuild sites vector
///
/// Negative controls (convex rocks): zero enclosed components → zero sites
/// Positive controls (pockets/grooves): enclosed voids with high spike density
#[cfg(feature = "gpu")]
fn recalculate_enclosure_volume(
    sites: &mut Vec<ClusteredBindingSite>,
    all_spikes: &[prism_nhs::fused_engine::GpuSpikeEvent],
    atom_positions: &[f32],
) {
    use prism_nhs::{DruggabilityScore, SiteClassification};
    use std::collections::VecDeque;

    let n_atoms = atom_positions.len() / 3;
    let grid_step = 1.0f32;
    let exclusion_radius = 3.0f32;  // Standard SES probe radius
    let scan_margin = 10.0f32;      // margin for rays to escape past protein surface
    let min_blocked = 4u32;         // Trench mode: grooves/trenches pass, convex surfaces fail
    let min_pocket_voxels = 50u32;  // 50 Å³ minimum viable pocket
    // Adaptive spike_intensity_min: 10th percentile of actual spike intensities.
    // Hardcoded 5.0 killed 80% of spikes for low-intensity proteins (1w50 @ 3.6 mean).
    // This is the same fix applied to per-stream filtering (critical fix 2026-03-12).
    let spike_intensity_min = if !all_spikes.is_empty() {
        let mut intensities: Vec<f32> = all_spikes.iter().map(|s| s.intensity).collect();
        intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p10_idx = intensities.len() / 10;
        let adaptive_min = intensities[p10_idx.min(intensities.len() - 1)];
        log::info!("  LIGSITE spike_intensity_min: adaptive={:.2} (10th percentile of {} spikes)",
            adaptive_min, all_spikes.len());
        adaptive_min
    } else {
        5.0f32  // fallback for empty spike arrays
    };
    let spike_search_r = 3i32;     // Bridge SES gap for spike→pocket mapping

    if n_atoms == 0 {
        sites.clear();
        return;
    }

    // ---- Build GLOBAL grid over entire protein ----
    let mut prot_min = [f32::MAX; 3];
    let mut prot_max = [f32::MIN; 3];
    for i in 0..n_atoms {
        for d in 0..3 {
            let v = atom_positions[i*3 + d];
            prot_min[d] = prot_min[d].min(v);
            prot_max[d] = prot_max[d].max(v);
        }
    }
    let gmin = [prot_min[0] - scan_margin, prot_min[1] - scan_margin, prot_min[2] - scan_margin];
    let nx = ((prot_max[0] - prot_min[0] + 2.0 * scan_margin) / grid_step).ceil() as usize + 1;
    let ny = ((prot_max[1] - prot_min[1] + 2.0 * scan_margin) / grid_step).ceil() as usize + 1;
    let nz = ((prot_max[2] - prot_min[2] + 2.0 * scan_margin) / grid_step).ceil() as usize + 1;
    let grid_size = nx * ny * nz;

    log::info!("  LIGSITE grid: {}×{}×{} = {} voxels (margin={:.0}Å)",
        nx, ny, nz, grid_size, scan_margin);

    let mut is_protein = vec![false; grid_size];

    let to_idx = |ix: usize, iy: usize, iz: usize| -> usize {
        ix * ny * nz + iy * nz + iz
    };
    let to_world = |ix: usize, iy: usize, iz: usize| -> [f32; 3] {
        [gmin[0] + ix as f32 * grid_step,
         gmin[1] + iy as f32 * grid_step,
         gmin[2] + iz as f32 * grid_step]
    };

    // ---- Stage 1: Mark protein (SES) voxels ----
    let excl_sq = exclusion_radius * exclusion_radius;
    let excl_cells = (exclusion_radius / grid_step).ceil() as i32 + 1;
    for i in 0..n_atoms {
        let ax = atom_positions[i*3];
        let ay = atom_positions[i*3 + 1];
        let az = atom_positions[i*3 + 2];
        let aix = ((ax - gmin[0]) / grid_step).round() as i32;
        let aiy = ((ay - gmin[1]) / grid_step).round() as i32;
        let aiz = ((az - gmin[2]) / grid_step).round() as i32;
        for dx in -excl_cells..=excl_cells {
            for dy in -excl_cells..=excl_cells {
                for dz in -excl_cells..=excl_cells {
                    let ix = aix + dx;
                    let iy = aiy + dy;
                    let iz = aiz + dz;
                    if ix < 0 || iy < 0 || iz < 0 { continue; }
                    let ix = ix as usize;
                    let iy = iy as usize;
                    let iz = iz as usize;
                    if ix >= nx || iy >= ny || iz >= nz { continue; }
                    let w = to_world(ix, iy, iz);
                    let d2 = (w[0]-ax).powi(2) + (w[1]-ay).powi(2) + (w[2]-az).powi(2);
                    if d2 <= excl_sq {
                        is_protein[to_idx(ix, iy, iz)] = true;
                    }
                }
            }
        }
    }

    // ---- Stage 2: (DELETED — geometry proposes pockets; spikes score in Stage 6) ----

    // ---- Stage 2.5: Flood-fill solvent accessibility from grid boundary ----
    // Buried hydrophobic cores are NOT reachable from exterior -> filtered out.
    let mut is_solvent = vec![false; grid_size];
    {
        let mut queue: std::collections::VecDeque<(usize, usize, usize)> =
            std::collections::VecDeque::new();

        // Seed: all non-protein boundary voxels
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let on_boundary = ix == 0 || ix == nx - 1
                                   || iy == 0 || iy == ny - 1
                                   || iz == 0 || iz == nz - 1;
                    if on_boundary {
                        let idx = to_idx(ix, iy, iz);
                        if !is_protein[idx] && !is_solvent[idx] {
                            is_solvent[idx] = true;
                            queue.push_back((ix, iy, iz));
                        }
                    }
                }
            }
        }

        // BFS through non-protein voxels (6-connected)
        while let Some((cx, cy, cz)) = queue.pop_front() {
            for &(dx, dy, dz) in &[
                (1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
            ] {
                let nix = cx as i32 + dx;
                let niy = cy as i32 + dy;
                let niz = cz as i32 + dz;
                if nix < 0 || niy < 0 || niz < 0 { continue; }
                let (nix, niy, niz) = (nix as usize, niy as usize, niz as usize);
                if nix >= nx || niy >= ny || niz >= nz { continue; }
                let nidx = to_idx(nix, niy, niz);
                if !is_protein[nidx] && !is_solvent[nidx] {
                    is_solvent[nidx] = true;
                    queue.push_back((nix, niy, niz));
                }
            }
        }

        let solvent_count = is_solvent.iter().filter(|&&v| v).count();
        let buried_count = (0..grid_size).filter(|&i| !is_protein[i] && !is_solvent[i]).count();
        log::info!("  Solvent gate: {} exterior-reachable, {} buried void voxels",
            solvent_count, buried_count);
    }

    // ---- Stage 2.75: MOVED to Stage 3.5 (depends on is_enclosed) ----

    // ---- Stage 3: Ray-cast → is_enclosed boolean ----
    // Geometry proposes: find all solvent-accessible void voxels enclosed on ≥4 sides
    let mut is_enclosed = vec![false; grid_size];
    let mut total_enclosed = 0u32;
    let mut total_protein = 0u32;
    let max_ray_steps = (10.0f32 / grid_step).ceil() as usize; // 10Å horizon

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = to_idx(ix, iy, iz);
                if is_protein[idx] { total_protein += 1; continue; }
                if !is_solvent[idx] { continue; } // Must be reachable from exterior

                // Ray-cast in 6 axial directions with 10Å horizon
                let mut blocked = 0u32;

                // +X
                let mut hit = false;
                for step in 1..=usize::min(nx - 1 - ix, max_ray_steps) {
                    if is_protein[to_idx(ix + step, iy, iz)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                // -X
                hit = false;
                for step in 1..=usize::min(ix, max_ray_steps) {
                    if is_protein[to_idx(ix - step, iy, iz)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                // +Y
                hit = false;
                for step in 1..=usize::min(ny - 1 - iy, max_ray_steps) {
                    if is_protein[to_idx(ix, iy + step, iz)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                // -Y
                hit = false;
                for step in 1..=usize::min(iy, max_ray_steps) {
                    if is_protein[to_idx(ix, iy - step, iz)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                // +Z
                hit = false;
                for step in 1..=usize::min(nz - 1 - iz, max_ray_steps) {
                    if is_protein[to_idx(ix, iy, iz + step)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                // -Z
                hit = false;
                for step in 1..=usize::min(iz, max_ray_steps) {
                    if is_protein[to_idx(ix, iy, iz - step)] { hit = true; break; }
                }
                if hit { blocked += 1; }

                if blocked >= min_blocked {
                    is_enclosed[idx] = true;
                    total_enclosed += 1;
                }
            }
        }
    }

    log::info!("  LIGSITE ray-cast: protein={}, enclosed(≥{}/6)={}",
        total_protein, min_blocked, total_enclosed);

    if total_enclosed == 0 {
        sites.clear();
        log::info!("  No enclosed voxels — zero geometric pockets (negative control behavior)");
        return;
    }

    // ---- Stage 3.5: DAH "Lid-Seeding" Distance Transform ----
    // Measure distance from the pocket mouth (bulk ocean) into the enclosed void.
    // Seed depth=0 from non-enclosed solvent (the "lid"), propagate ONLY into enclosed voxels.
    // This correctly measures how deep into the protein a pocket extends.
    // Init to 99.0 (not f32::MAX) so disconnected buried cores get exp(-97/5) ≈ 0 penalty.
    let depth_grid = {
        let mut depth = vec![99.0f32; grid_size];
        let mut depth_queue = VecDeque::new();

        // 1. Seed the "Lid": bulk solvent that is NOT enclosed
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = to_idx(ix, iy, iz);
                    if is_solvent[idx] && !is_enclosed[idx] {
                        depth[idx] = 0.0;
                        depth_queue.push_back((ix, iy, iz));
                    }
                }
            }
        }

        // 2. Propagate BFS inward strictly INTO enclosed pocket void
        while let Some((cx, cy, cz)) = depth_queue.pop_front() {
            let curr_depth = depth[to_idx(cx, cy, cz)];

            for &(ddx, ddy, ddz) in &[
                (1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
            ] {
                let nix = cx as i32 + ddx;
                let niy = cy as i32 + ddy;
                let niz = cz as i32 + ddz;
                if nix < 0 || niy < 0 || niz < 0 { continue; }
                let (nix, niy, niz) = (nix as usize, niy as usize, niz as usize);
                if nix >= nx || niy >= ny || niz >= nz { continue; }
                let nidx = to_idx(nix, niy, niz);

                // Only propagate into enclosed pocket voxels
                if is_enclosed[nidx] {
                    let new_depth = curr_depth + grid_step;
                    if new_depth < depth[nidx] {
                        depth[nidx] = new_depth;
                        depth_queue.push_back((nix, niy, niz));
                    }
                }
            }
        }

        // DAH Lid-Depth Summary
        let pocket_depths: Vec<f32> = (0..grid_size)
            .filter(|&i| is_enclosed[i] && depth[i] < 99.0)
            .map(|i| depth[i])
            .collect();
        let max_pocket_depth = pocket_depths.iter().cloned().fold(0.0f32, f32::max);
        let avg_pocket_depth = if !pocket_depths.is_empty() {
            pocket_depths.iter().sum::<f32>() / pocket_depths.len() as f32
        } else { 0.0 };
        let unreachable = (0..grid_size)
            .filter(|&i| is_enclosed[i] && depth[i] >= 99.0)
            .count();
        log::info!("  DAH lid-depth: {} pocket voxels measured, {} unreachable, max={:.1}Å, avg={:.1}Å",
            pocket_depths.len(), unreachable, max_pocket_depth, avg_pocket_depth);

        depth
    };

    // ---- Stage 4: BFS connected components on is_enclosed ----
    let mut component_id: Vec<i32> = vec![-1; grid_size];
    let mut num_components = 0i32;

    // Per-component accumulators (built during BFS)
    struct PocketAccum {
        voxel_count: u32,
        coord_sum: [f64; 3],
        bbox_min: [f32; 3],
        bbox_max: [f32; 3],
        depth_sum: f64,
    }
    let mut accums: Vec<PocketAccum> = Vec::new();

    for start_idx in 0..grid_size {
        if !is_enclosed[start_idx] || component_id[start_idx] != -1 { continue; }

        let cid = num_components;
        num_components += 1;

        let mut acc = PocketAccum {
            voxel_count: 0,
            coord_sum: [0.0; 3],
            bbox_min: [f32::MAX; 3],
            bbox_max: [f32::MIN; 3],
            depth_sum: 0.0,
        };

        let mut queue = VecDeque::new();
        component_id[start_idx] = cid;
        queue.push_back(start_idx);

        while let Some(cidx) = queue.pop_front() {
            let ciz = cidx % nz;
            let ciy = (cidx / nz) % ny;
            let cix = cidx / (ny * nz);
            let w = to_world(cix, ciy, ciz);

            acc.voxel_count += 1;
            acc.coord_sum[0] += w[0] as f64;
            acc.coord_sum[1] += w[1] as f64;
            acc.coord_sum[2] += w[2] as f64;
            for d in 0..3 {
                acc.bbox_min[d] = acc.bbox_min[d].min(w[d]);
                acc.bbox_max[d] = acc.bbox_max[d].max(w[d]);
            }
            let depth_val = depth_grid[cidx];
            if depth_val < f32::MAX {
                acc.depth_sum += depth_val as f64;
            }

            // 6-connected BFS
            for &(ddx, ddy, ddz) in &[
                (1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
            ] {
                let nix = cix as i32 + ddx;
                let niy = ciy as i32 + ddy;
                let niz = ciz as i32 + ddz;
                if nix < 0 || niy < 0 || niz < 0 { continue; }
                let (nix, niy, niz) = (nix as usize, niy as usize, niz as usize);
                if nix >= nx || niy >= ny || niz >= nz { continue; }
                let nidx = to_idx(nix, niy, niz);
                if is_enclosed[nidx] && component_id[nidx] == -1 {
                    component_id[nidx] = cid;
                    queue.push_back(nidx);
                }
            }
        }

        accums.push(acc);
    }

    // Filter by minimum pocket size
    let surviving: Vec<(usize, &PocketAccum)> = accums.iter().enumerate()
        .filter(|(_, a)| a.voxel_count >= min_pocket_voxels)
        .collect();

    log::info!("  Connected components: {} total, {} above {} voxel threshold",
        num_components, surviving.len(), min_pocket_voxels);

    if surviving.is_empty() {
        sites.clear();
        log::info!("  No pockets above size threshold — clearing all sites");
        return;
    }

    // ---- Stage 5: Build pocket metadata from surviving components ----
    // Map from raw component_id → surviving pocket index (-1 = below threshold)
    let mut cid_to_pocket: Vec<i32> = vec![-1; num_components as usize];

    struct PocketInfo {
        centroid: [f32; 3],
        volume: f32,
        bbox: [[f32; 3]; 2],
        mean_depth: f32,
        voxel_count: u32,
    }
    let mut pockets: Vec<PocketInfo> = Vec::new();

    for (pocket_idx, &(cid, ref acc)) in surviving.iter().enumerate() {
        cid_to_pocket[cid] = pocket_idx as i32;

        let n = acc.voxel_count as f64;
        let centroid = [
            (acc.coord_sum[0] / n) as f32,
            (acc.coord_sum[1] / n) as f32,
            (acc.coord_sum[2] / n) as f32,
        ];
        let volume = acc.voxel_count as f32 * grid_step.powi(3);
        let mean_depth = (acc.depth_sum / n) as f32;

        log::info!("  Pocket {}: {} voxels, {:.0} Å³, centroid=({:.1},{:.1},{:.1}), depth={:.1}Å",
            pocket_idx, acc.voxel_count, volume,
            centroid[0], centroid[1], centroid[2], mean_depth);

        pockets.push(PocketInfo {
            centroid,
            volume,
            bbox: [acc.bbox_min, acc.bbox_max],
            mean_depth,
            voxel_count: acc.voxel_count,
        });
    }

    // ---- Stage 5b: Watershed sub-pocket decomposition ----
    // If any pocket has multiple density peaks inside it, split it.
    // This eliminates the mega-pocket failure mode (1ERE 18K Å³).
    // Algorithm: multi-source BFS from intensity² peaks, bounded by
    // the LIGSITE enclosed mask. Protein walls are impenetrable.
    let watershed_threshold = 1500u32; // Only split pockets > 1500 voxels (~1500 Å³)
    let mut next_cid = num_components;
    let mut new_pockets: Vec<(usize, Vec<(i32, [f32; 3])>)> = Vec::new(); // (orig_pocket_idx, seeds)

    for (pocket_idx, pocket) in pockets.iter().enumerate() {
        if pocket.voxel_count < watershed_threshold { continue; }

        // Find the original cid for this pocket
        let orig_cid = surviving[pocket_idx].0 as i32;

        // Collect spikes inside this pocket's bounding box with intensity filtering
        let bmin = pocket.bbox[0];
        let bmax = pocket.bbox[1];
        let margin = 2.0f32;

        // Build a coarse spike density grid (3Å spacing) over this pocket
        let ws = 3.0f32; // watershed grid spacing
        let wdx = ((bmax[0] - bmin[0] + 2.0 * margin) / ws).ceil() as usize + 1;
        let wdy = ((bmax[1] - bmin[1] + 2.0 * margin) / ws).ceil() as usize + 1;
        let wdz = ((bmax[2] - bmin[2] + 2.0 * margin) / ws).ceil() as usize + 1;
        let worigin = [bmin[0] - margin, bmin[1] - margin, bmin[2] - margin];
        let wsize = wdx * wdy * wdz;
        if wsize > 500_000 { continue; } // safety: skip absurdly large pockets

        let mut wgrid = vec![0.0f64; wsize];
        let w_idx = |x: usize, y: usize, z: usize| -> usize { x * wdy * wdz + y * wdz + z };

        let mut pocket_spike_count = 0u32;
        for spike in all_spikes.iter() {
            if spike.intensity < spike_intensity_min { continue; }
            let sp = spike.position;
            if sp[0] < bmin[0] - margin || sp[0] > bmax[0] + margin { continue; }
            if sp[1] < bmin[1] - margin || sp[1] > bmax[1] + margin { continue; }
            if sp[2] < bmin[2] - margin || sp[2] > bmax[2] + margin { continue; }
            let wx = ((sp[0] - worigin[0]) / ws) as usize;
            let wy = ((sp[1] - worigin[1]) / ws) as usize;
            let wz = ((sp[2] - worigin[2]) / ws) as usize;
            if wx < wdx && wy < wdy && wz < wdz {
                let isq = (spike.intensity as f64).powi(2);
                wgrid[w_idx(wx, wy, wz)] += isq;
                pocket_spike_count += 1;
            }
        }
        if pocket_spike_count < 20 { continue; }

        // NMS: find local maxima in the coarse grid (3x3x3 neighborhood)
        let mut peaks: Vec<(i32, [f32; 3])> = Vec::new(); // (new_cid, position)
        for wx in 1..wdx.saturating_sub(1) {
            for wy in 1..wdy.saturating_sub(1) {
                for wz in 1..wdz.saturating_sub(1) {
                    let val = wgrid[w_idx(wx, wy, wz)];
                    if val < 1.0 { continue; }
                    let mut is_max = true;
                    'nms: for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nx2 = (wx as i32 + dx) as usize;
                                let ny2 = (wy as i32 + dy) as usize;
                                let nz2 = (wz as i32 + dz) as usize;
                                if nx2 < wdx && ny2 < wdy && nz2 < wdz {
                                    if wgrid[w_idx(nx2, ny2, nz2)] > val {
                                        is_max = false;
                                        break 'nms;
                                    }
                                }
                            }
                        }
                    }
                    if !is_max { continue; }

                    // Convert coarse grid position back to Cartesian
                    let peak_pos = [
                        worigin[0] + (wx as f32 + 0.5) * ws,
                        worigin[1] + (wy as f32 + 0.5) * ws,
                        worigin[2] + (wz as f32 + 0.5) * ws,
                    ];
                    // Check: is this peak inside an enclosed voxel of THIS pocket?
                    let gx = ((peak_pos[0] - gmin[0]) / grid_step).round() as i32;
                    let gy = ((peak_pos[1] - gmin[1]) / grid_step).round() as i32;
                    let gz = ((peak_pos[2] - gmin[2]) / grid_step).round() as i32;
                    if gx >= 0 && gy >= 0 && gz >= 0 {
                        let (gx, gy, gz) = (gx as usize, gy as usize, gz as usize);
                        if gx < nx && gy < ny && gz < nz {
                            let gidx = to_idx(gx, gy, gz);
                            if component_id[gidx] == orig_cid {
                                peaks.push((next_cid, peak_pos));
                                next_cid += 1;
                            }
                        }
                    }
                }
            }
        }

        if peaks.len() >= 2 {
            log::info!("  Watershed: Pocket {} ({} voxels, {:.0} Å³) has {} density peaks → splitting",
                pocket_idx, pocket.voxel_count, pocket.volume, peaks.len());
            new_pockets.push((pocket_idx, peaks));
        }
    }

    // Execute watershed BFS for each pocket that needs splitting
    for (pocket_idx, seeds) in &new_pockets {
        let orig_cid = surviving[*pocket_idx].0 as i32;

        // Initialize Eikonal priority queue — expansion cost weighted by
        // inverse spike density. Boundaries form along low-density ridges
        // (physical energy barriers) rather than geometric midpoints.
        // This is the Fast Marching Method applied to the spike density field.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Build a spike density lookup on the LIGSITE grid for this pocket
        // (reuse wgrid data mapped onto the 1Å grid)
        let mut voxel_density = vec![0.0f64; grid_size];
        for spike in all_spikes.iter() {
            if spike.intensity < spike_intensity_min { continue; }
            let sp = spike.position;
            let gx = ((sp[0] - gmin[0]) / grid_step).round() as i32;
            let gy = ((sp[1] - gmin[1]) / grid_step).round() as i32;
            let gz = ((sp[2] - gmin[2]) / grid_step).round() as i32;
            if gx >= 0 && gy >= 0 && gz >= 0 {
                let (gx, gy, gz) = (gx as usize, gy as usize, gz as usize);
                if gx < nx && gy < ny && gz < nz {
                    let idx = to_idx(gx, gy, gz);
                    if component_id[idx] == orig_cid {
                        voxel_density[idx] += (spike.intensity as f64).powi(2);
                    }
                }
            }
        }

        // Priority queue: (cost_fixed_point, voxel_idx, seed_cid)
        // Use integer costs for BinaryHeap (f64 doesn't impl Ord)
        let mut pq: BinaryHeap<Reverse<(u64, usize, i32)>> = BinaryHeap::new();

        for (seed_cid, seed_pos) in seeds {
            let gx = ((seed_pos[0] - gmin[0]) / grid_step).round() as usize;
            let gy = ((seed_pos[1] - gmin[1]) / grid_step).round() as usize;
            let gz = ((seed_pos[2] - gmin[2]) / grid_step).round() as usize;
            if gx < nx && gy < ny && gz < nz {
                let idx = to_idx(gx, gy, gz);
                if component_id[idx] == orig_cid {
                    component_id[idx] = *seed_cid;
                    pq.push(Reverse((0u64, idx, *seed_cid)));
                }
            }
        }

        // Eikonal expansion: lowest-cost voxel pops first.
        // Cost to enter voxel = 1.0 / (1.0 + density) — high density = easy,
        // low density = hard. Boundaries form at density minima (saddle points).
        while let Some(Reverse((cost, idx, seed_cid))) = pq.pop() {
            // Skip if already claimed by a different seed (first arrival wins)
            if component_id[idx] != orig_cid && component_id[idx] != seed_cid {
                continue;
            }

            let iz = idx % nz;
            let iy = (idx / nz) % ny;
            let ix = idx / (ny * nz);
            let neighbors = [
                if ix > 0      { Some(to_idx(ix-1, iy, iz)) } else { None },
                if ix < nx - 1 { Some(to_idx(ix+1, iy, iz)) } else { None },
                if iy > 0      { Some(to_idx(ix, iy-1, iz)) } else { None },
                if iy < ny - 1 { Some(to_idx(ix, iy+1, iz)) } else { None },
                if iz > 0      { Some(to_idx(ix, iy, iz-1)) } else { None },
                if iz < nz - 1 { Some(to_idx(ix, iy, iz+1)) } else { None },
            ];
            for n in neighbors.iter().flatten() {
                if component_id[*n] == orig_cid {
                    component_id[*n] = seed_cid;
                    // Eikonal cost: traverse low-density voxels = expensive
                    let edge_cost = (1.0e6 / (1.0 + voxel_density[*n])) as u64;
                    pq.push(Reverse((cost + edge_cost, *n, seed_cid)));
                }
            }
        }
    }

    // If any splits happened, rebuild pockets and cid_to_pocket
    if !new_pockets.is_empty() {
        let total_seeds: usize = new_pockets.iter().map(|(_, s)| s.len()).sum();
        log::info!("  Watershed split {} mega-pockets into {} sub-pockets", new_pockets.len(), total_seeds);

        // Rebuild: recount all components (including new sub-pocket cids)
        pockets.clear();
        cid_to_pocket = vec![-1; next_cid as usize];

        // Re-accumulate from component_id grid
        struct WsAccum { coord_sum: [f64; 3], voxel_count: u32, bbox_min: [f32; 3], bbox_max: [f32; 3], depth_sum: f64 }
        let mut ws_accum: std::collections::HashMap<i32, WsAccum> = std::collections::HashMap::new();

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = to_idx(ix, iy, iz);
                    let cid = component_id[idx];
                    if cid < 0 { continue; }
                    let pos = [
                        gmin[0] + ix as f32 * grid_step,
                        gmin[1] + iy as f32 * grid_step,
                        gmin[2] + iz as f32 * grid_step,
                    ];
                    let d = depth_grid[idx] as f64;
                    let acc = ws_accum.entry(cid).or_insert(WsAccum {
                        coord_sum: [0.0; 3], voxel_count: 0,
                        bbox_min: [f32::MAX; 3], bbox_max: [f32::MIN; 3], depth_sum: 0.0,
                    });
                    acc.coord_sum[0] += pos[0] as f64;
                    acc.coord_sum[1] += pos[1] as f64;
                    acc.coord_sum[2] += pos[2] as f64;
                    acc.voxel_count += 1;
                    acc.depth_sum += d;
                    for d2 in 0..3 {
                        if pos[d2] < acc.bbox_min[d2] { acc.bbox_min[d2] = pos[d2]; }
                        if pos[d2] > acc.bbox_max[d2] { acc.bbox_max[d2] = pos[d2]; }
                    }
                }
            }
        }

        // Rebuild pocket list from accumulator, filtering by min size
        let mut sorted_cids: Vec<(i32, &WsAccum)> = ws_accum.iter()
            .filter(|(_, a)| a.voxel_count >= min_pocket_voxels)
            .map(|(&cid, a)| (cid, a))
            .collect();
        sorted_cids.sort_by(|a, b| b.1.voxel_count.cmp(&a.1.voxel_count));

        for (pocket_idx, (cid, acc)) in sorted_cids.iter().enumerate() {
            cid_to_pocket[*cid as usize] = pocket_idx as i32;
            let n = acc.voxel_count as f64;
            let centroid = [
                (acc.coord_sum[0] / n) as f32,
                (acc.coord_sum[1] / n) as f32,
                (acc.coord_sum[2] / n) as f32,
            ];
            let volume = acc.voxel_count as f32 * grid_step.powi(3);
            let mean_depth = (acc.depth_sum / n) as f32;
            pockets.push(PocketInfo {
                centroid, volume,
                bbox: [acc.bbox_min, acc.bbox_max],
                mean_depth,
                voxel_count: acc.voxel_count,
            });
        }
        log::info!("  Post-watershed: {} pockets (was {})", pockets.len(), surviving.len());
    }

    // ---- Stage 5b: Watershed sub-pocket decomposition ----
    // If any pocket has multiple density peaks inside it, split it.
    // This eliminates the mega-pocket failure mode (1ERE 18K Å³).
    // Algorithm: multi-source BFS from intensity² peaks, bounded by
    // the LIGSITE enclosed mask. Protein walls are impenetrable.
    let watershed_threshold = 1500u32; // Only split pockets > 1500 voxels (~1500 Å³)
    let mut next_cid = num_components;
    let mut new_pockets: Vec<(usize, Vec<(i32, [f32; 3])>)> = Vec::new(); // (orig_pocket_idx, seeds)

    for (pocket_idx, pocket) in pockets.iter().enumerate() {
        if pocket.voxel_count < watershed_threshold { continue; }

        // Find the original cid for this pocket
        let orig_cid = surviving[pocket_idx].0 as i32;

        // Collect spikes inside this pocket's bounding box with intensity filtering
        let bmin = pocket.bbox[0];
        let bmax = pocket.bbox[1];
        let margin = 2.0f32;

        // Build a coarse spike density grid (3Å spacing) over this pocket
        let ws = 3.0f32; // watershed grid spacing
        let wdx = ((bmax[0] - bmin[0] + 2.0 * margin) / ws).ceil() as usize + 1;
        let wdy = ((bmax[1] - bmin[1] + 2.0 * margin) / ws).ceil() as usize + 1;
        let wdz = ((bmax[2] - bmin[2] + 2.0 * margin) / ws).ceil() as usize + 1;
        let worigin = [bmin[0] - margin, bmin[1] - margin, bmin[2] - margin];
        let wsize = wdx * wdy * wdz;
        if wsize > 500_000 { continue; } // safety: skip absurdly large pockets

        let mut wgrid = vec![0.0f64; wsize];
        let w_idx = |x: usize, y: usize, z: usize| -> usize { x * wdy * wdz + y * wdz + z };

        let mut pocket_spike_count = 0u32;
        for spike in all_spikes.iter() {
            if spike.intensity < spike_intensity_min { continue; }
            let sp = spike.position;
            if sp[0] < bmin[0] - margin || sp[0] > bmax[0] + margin { continue; }
            if sp[1] < bmin[1] - margin || sp[1] > bmax[1] + margin { continue; }
            if sp[2] < bmin[2] - margin || sp[2] > bmax[2] + margin { continue; }
            let wx = ((sp[0] - worigin[0]) / ws) as usize;
            let wy = ((sp[1] - worigin[1]) / ws) as usize;
            let wz = ((sp[2] - worigin[2]) / ws) as usize;
            if wx < wdx && wy < wdy && wz < wdz {
                let isq = (spike.intensity as f64).powi(2);
                wgrid[w_idx(wx, wy, wz)] += isq;
                pocket_spike_count += 1;
            }
        }
        if pocket_spike_count < 20 { continue; }

        // NMS: find local maxima in the coarse grid (3x3x3 neighborhood)
        let mut peaks: Vec<(i32, [f32; 3])> = Vec::new(); // (new_cid, position)
        for wx in 1..wdx.saturating_sub(1) {
            for wy in 1..wdy.saturating_sub(1) {
                for wz in 1..wdz.saturating_sub(1) {
                    let val = wgrid[w_idx(wx, wy, wz)];
                    if val < 1.0 { continue; }
                    let mut is_max = true;
                    'nms: for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nx2 = (wx as i32 + dx) as usize;
                                let ny2 = (wy as i32 + dy) as usize;
                                let nz2 = (wz as i32 + dz) as usize;
                                if nx2 < wdx && ny2 < wdy && nz2 < wdz {
                                    if wgrid[w_idx(nx2, ny2, nz2)] > val {
                                        is_max = false;
                                        break 'nms;
                                    }
                                }
                            }
                        }
                    }
                    if !is_max { continue; }

                    // Convert coarse grid position back to Cartesian
                    let peak_pos = [
                        worigin[0] + (wx as f32 + 0.5) * ws,
                        worigin[1] + (wy as f32 + 0.5) * ws,
                        worigin[2] + (wz as f32 + 0.5) * ws,
                    ];
                    // Check: is this peak inside an enclosed voxel of THIS pocket?
                    let gx = ((peak_pos[0] - gmin[0]) / grid_step).round() as i32;
                    let gy = ((peak_pos[1] - gmin[1]) / grid_step).round() as i32;
                    let gz = ((peak_pos[2] - gmin[2]) / grid_step).round() as i32;
                    if gx >= 0 && gy >= 0 && gz >= 0 {
                        let (gx, gy, gz) = (gx as usize, gy as usize, gz as usize);
                        if gx < nx && gy < ny && gz < nz {
                            let gidx = to_idx(gx, gy, gz);
                            if component_id[gidx] == orig_cid {
                                peaks.push((next_cid, peak_pos));
                                next_cid += 1;
                            }
                        }
                    }
                }
            }
        }

        if peaks.len() >= 2 {
            log::info!("  Watershed: Pocket {} ({} voxels, {:.0} Å³) has {} density peaks → splitting",
                pocket_idx, pocket.voxel_count, pocket.volume, peaks.len());
            new_pockets.push((pocket_idx, peaks));
        }
    }

    // Execute watershed BFS for each pocket that needs splitting
    for (pocket_idx, seeds) in &new_pockets {
        let orig_cid = surviving[*pocket_idx].0 as i32;

        // Initialize Eikonal priority queue — expansion cost weighted by
        // inverse spike density. Boundaries form along low-density ridges
        // (physical energy barriers) rather than geometric midpoints.
        // This is the Fast Marching Method applied to the spike density field.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Build a spike density lookup on the LIGSITE grid for this pocket
        // (reuse wgrid data mapped onto the 1Å grid)
        let mut voxel_density = vec![0.0f64; grid_size];
        for spike in all_spikes.iter() {
            if spike.intensity < spike_intensity_min { continue; }
            let sp = spike.position;
            let gx = ((sp[0] - gmin[0]) / grid_step).round() as i32;
            let gy = ((sp[1] - gmin[1]) / grid_step).round() as i32;
            let gz = ((sp[2] - gmin[2]) / grid_step).round() as i32;
            if gx >= 0 && gy >= 0 && gz >= 0 {
                let (gx, gy, gz) = (gx as usize, gy as usize, gz as usize);
                if gx < nx && gy < ny && gz < nz {
                    let idx = to_idx(gx, gy, gz);
                    if component_id[idx] == orig_cid {
                        voxel_density[idx] += (spike.intensity as f64).powi(2);
                    }
                }
            }
        }

        // Priority queue: (cost_fixed_point, voxel_idx, seed_cid)
        // Use integer costs for BinaryHeap (f64 doesn't impl Ord)
        let mut pq: BinaryHeap<Reverse<(u64, usize, i32)>> = BinaryHeap::new();

        for (seed_cid, seed_pos) in seeds {
            let gx = ((seed_pos[0] - gmin[0]) / grid_step).round() as usize;
            let gy = ((seed_pos[1] - gmin[1]) / grid_step).round() as usize;
            let gz = ((seed_pos[2] - gmin[2]) / grid_step).round() as usize;
            if gx < nx && gy < ny && gz < nz {
                let idx = to_idx(gx, gy, gz);
                if component_id[idx] == orig_cid {
                    component_id[idx] = *seed_cid;
                    pq.push(Reverse((0u64, idx, *seed_cid)));
                }
            }
        }

        // Eikonal expansion: lowest-cost voxel pops first.
        // Cost to enter voxel = 1.0 / (1.0 + density) — high density = easy,
        // low density = hard. Boundaries form at density minima (saddle points).
        while let Some(Reverse((cost, idx, seed_cid))) = pq.pop() {
            // Skip if already claimed by a different seed (first arrival wins)
            if component_id[idx] != orig_cid && component_id[idx] != seed_cid {
                continue;
            }

            let iz = idx % nz;
            let iy = (idx / nz) % ny;
            let ix = idx / (ny * nz);
            let neighbors = [
                if ix > 0      { Some(to_idx(ix-1, iy, iz)) } else { None },
                if ix < nx - 1 { Some(to_idx(ix+1, iy, iz)) } else { None },
                if iy > 0      { Some(to_idx(ix, iy-1, iz)) } else { None },
                if iy < ny - 1 { Some(to_idx(ix, iy+1, iz)) } else { None },
                if iz > 0      { Some(to_idx(ix, iy, iz-1)) } else { None },
                if iz < nz - 1 { Some(to_idx(ix, iy, iz+1)) } else { None },
            ];
            for n in neighbors.iter().flatten() {
                if component_id[*n] == orig_cid {
                    component_id[*n] = seed_cid;
                    // Eikonal cost: traverse low-density voxels = expensive
                    let edge_cost = (1.0e6 / (1.0 + voxel_density[*n])) as u64;
                    pq.push(Reverse((cost + edge_cost, *n, seed_cid)));
                }
            }
        }
    }

    // If any splits happened, rebuild pockets and cid_to_pocket
    if !new_pockets.is_empty() {
        let total_seeds: usize = new_pockets.iter().map(|(_, s)| s.len()).sum();
        log::info!("  Watershed split {} mega-pockets into {} sub-pockets", new_pockets.len(), total_seeds);

        // Rebuild: recount all components (including new sub-pocket cids)
        pockets.clear();
        cid_to_pocket = vec![-1; next_cid as usize];

        // Re-accumulate from component_id grid
        struct WsAccum { coord_sum: [f64; 3], voxel_count: u32, bbox_min: [f32; 3], bbox_max: [f32; 3], depth_sum: f64 }
        let mut ws_accum: std::collections::HashMap<i32, WsAccum> = std::collections::HashMap::new();

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = to_idx(ix, iy, iz);
                    let cid = component_id[idx];
                    if cid < 0 { continue; }
                    let pos = [
                        gmin[0] + ix as f32 * grid_step,
                        gmin[1] + iy as f32 * grid_step,
                        gmin[2] + iz as f32 * grid_step,
                    ];
                    let d = depth_grid[idx] as f64;
                    let acc = ws_accum.entry(cid).or_insert(WsAccum {
                        coord_sum: [0.0; 3], voxel_count: 0,
                        bbox_min: [f32::MAX; 3], bbox_max: [f32::MIN; 3], depth_sum: 0.0,
                    });
                    acc.coord_sum[0] += pos[0] as f64;
                    acc.coord_sum[1] += pos[1] as f64;
                    acc.coord_sum[2] += pos[2] as f64;
                    acc.voxel_count += 1;
                    acc.depth_sum += d;
                    for d2 in 0..3 {
                        if pos[d2] < acc.bbox_min[d2] { acc.bbox_min[d2] = pos[d2]; }
                        if pos[d2] > acc.bbox_max[d2] { acc.bbox_max[d2] = pos[d2]; }
                    }
                }
            }
        }

        // Rebuild pocket list from accumulator, filtering by min size
        let mut sorted_cids: Vec<(i32, &WsAccum)> = ws_accum.iter()
            .filter(|(_, a)| a.voxel_count >= min_pocket_voxels)
            .map(|(&cid, a)| (cid, a))
            .collect();
        sorted_cids.sort_by(|a, b| b.1.voxel_count.cmp(&a.1.voxel_count));

        for (pocket_idx, (cid, acc)) in sorted_cids.iter().enumerate() {
            cid_to_pocket[*cid as usize] = pocket_idx as i32;
            let n = acc.voxel_count as f64;
            let centroid = [
                (acc.coord_sum[0] / n) as f32,
                (acc.coord_sum[1] / n) as f32,
                (acc.coord_sum[2] / n) as f32,
            ];
            let volume = acc.voxel_count as f32 * grid_step.powi(3);
            let mean_depth = (acc.depth_sum / n) as f32;
            pockets.push(PocketInfo {
                centroid, volume,
                bbox: [acc.bbox_min, acc.bbox_max],
                mean_depth,
                voxel_count: acc.voxel_count,
            });
        }
        log::info!("  Post-watershed: {} pockets (was {})", pockets.len(), surviving.len());
    }

    // ---- Stage 6: O(N) spike overlay via component_id grid lookup ----
    // Single pass through all spikes. For each spike, find its grid voxel,
    // search a small radius to bridge the SES gap, and credit the nearest pocket.
    struct PocketStats {
        spike_count: u32,
        intensity_sum: f32,
        spike_indices: Vec<usize>,
        // Intensity-weighted centroid accumulators
        weighted_pos: [f64; 3],
        weight_sum: f64,
        // Peak tracking: top-K highest-intensity spikes for peak centroid
        top_spikes: Vec<(f32, [f32; 3])>,
    }
    const PEAK_TOP_K: usize = 10;
    let mut stats: Vec<PocketStats> = (0..pockets.len())
        .map(|_| PocketStats { spike_count: 0, intensity_sum: 0.0, spike_indices: Vec::new(), weighted_pos: [0.0; 3], weight_sum: 0.0, top_spikes: Vec::with_capacity(PEAK_TOP_K + 1) })
        .collect();

    let sr = spike_search_r;
    let sr_sq = sr * sr;
    let mut spikes_mapped = 0u32;
    let mut spikes_filtered = 0u32;

    for (spike_idx, spike) in all_spikes.iter().enumerate() {
        // Silver bullet: only count high-intensity trapped spikes
        if spike.intensity < spike_intensity_min {
            spikes_filtered += 1;
            continue;
        }

        let sp = spike.position;
        let six = ((sp[0] - gmin[0]) / grid_step).round() as i32;
        let siy = ((sp[1] - gmin[1]) / grid_step).round() as i32;
        let siz = ((sp[2] - gmin[2]) / grid_step).round() as i32;

        // Search within spike_search_r voxels to bridge SES gap
        let mut best_pocket: i32 = -1;
        let mut best_d2 = i32::MAX;

        for ddx in -sr..=sr {
            for ddy in -sr..=sr {
                for ddz in -sr..=sr {
                    let d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                    if d2 > sr_sq { continue; }

                    let nix = six + ddx;
                    let niy = siy + ddy;
                    let niz = siz + ddz;
                    if nix < 0 || niy < 0 || niz < 0 { continue; }
                    let (nix, niy, niz) = (nix as usize, niy as usize, niz as usize);
                    if nix >= nx || niy >= ny || niz >= nz { continue; }

                    let cid = component_id[to_idx(nix, niy, niz)];
                    if cid >= 0 {
                        let pid = cid_to_pocket[cid as usize];
                        if pid >= 0 && d2 < best_d2 {
                            best_pocket = pid;
                            best_d2 = d2;
                        }
                    }
                }
            }
        }

        if best_pocket >= 0 {
            let s = &mut stats[best_pocket as usize];
            s.spike_count += 1;
            s.intensity_sum += spike.intensity;
            s.spike_indices.push(spike_idx);
            // Intensity² weighted centroid (pulls toward thermodynamic hotspot)
            let w = (spike.intensity as f64).powi(2);
            s.weighted_pos[0] += sp[0] as f64 * w;
            s.weighted_pos[1] += sp[1] as f64 * w;
            s.weighted_pos[2] += sp[2] as f64 * w;
            s.weight_sum += w;
            // Track top-K highest-intensity spikes for peak centroid
            s.top_spikes.push((spike.intensity, sp));
            if s.top_spikes.len() > PEAK_TOP_K {
                s.top_spikes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                s.top_spikes.truncate(PEAK_TOP_K);
            }
            spikes_mapped += 1;
        }
    }

    log::info!("  Spike overlay: {} mapped to pockets, {} filtered (intensity<{:.0}), {} unmapped",
        spikes_mapped, spikes_filtered,
        spike_intensity_min,
        all_spikes.len() as u32 - spikes_mapped - spikes_filtered);

    // ---- Stage 7: Score pockets and rebuild sites vector ----
    sites.clear();

    for (pi, pocket) in pockets.iter().enumerate() {
        let stat = &mut stats[pi];

        // NaN guards
        let avg_intensity = if stat.spike_count > 0 {
            stat.intensity_sum / stat.spike_count as f32
        } else {
            0.0
        };
        let spike_density = if pocket.volume > 0.0 {
            stat.spike_count as f32 / pocket.volume
        } else {
            0.0
        };

        // Burial penalty: surface-proximal pockets get full score
        // depth=2Å -> factor=1.0, depth=5Å -> 0.55, depth=8Å -> 0.30
        let surface_factor = (-(pocket.mean_depth - 2.0).max(0.0) / 5.0).exp();

        let bbox_extents = [
            pocket.bbox[1][0] - pocket.bbox[0][0],
            pocket.bbox[1][1] - pocket.bbox[0][1],
            pocket.bbox[1][2] - pocket.bbox[0][2],
        ];
        let mut druggability = DruggabilityScore::from_site(
            pocket.volume, avg_intensity, &bbox_extents);
        druggability.overall *= surface_factor;
        druggability.is_druggable = druggability.overall >= 0.40
            && pocket.volume >= 50.0
            && pocket.volume <= 3000.0;

        let classification = SiteClassification::from_properties(
            stat.spike_count as usize, pocket.volume, avg_intensity);

        let quality_score = {
            // Enclosure-based ranking (v2): matches multi-stream path
            let n_lining = 0.0_f32; // lining_residues computed later; use spike_frac + vol + drug here
            let total = all_spikes.len() as f32;
            let spike_frac = if total > 0.0 { stat.spike_count as f32 / total } else { 0.0 };
            let vol_q = if pocket.volume >= 100.0 && pocket.volume <= 800.0 {
                1.0_f32
            } else if pocket.volume > 800.0 {
                (800.0 / pocket.volume).sqrt()
            } else {
                (pocket.volume / 100.0).clamp(0.1, 1.0)
            };
            0.35 * (spike_frac * 5.0).clamp(0.0, 1.0)    // spike fraction (dominant here)
                + 0.25 * vol_q                               // pocket-like volume
                + 0.20 * spike_density.clamp(0.0, 10.0) / 10.0  // spike density
                + 0.10 * druggability.overall                // druggability (secondary)
                + 0.10 * surface_factor                      // burial/surface accessibility
        };

        log::info!("  Site {}: vol={:.0}Å³ spikes={} density={:.2} intensity={:.1} depth={:.1}Å \
            surface_factor={:.3} drug={:.3} quality={:.3} class={:?}",
            pi, pocket.volume, stat.spike_count, spike_density, avg_intensity,
            pocket.mean_depth, surface_factor, druggability.overall, quality_score,
            classification);
        // Centroid strategy: for mega-pockets (>500Å³), the spike-weighted
        // centroid drifts far from the actual binding site because uniform
        // high-intensity spikes pull it toward the center of the diffuse cloud.
        // The LIGSITE geometric centroid (center of enclosed voxels) is much
        // more accurate: 1.6Å for 1hhp vs 9.7Å spike-weighted, 2.4Å for
        // 1w50 vs 12.8Å spike-weighted.
        // For small pockets (≤500Å³), spike-weighted centroid is fine.
        let final_centroid = if pocket.volume > 500.0 {
            // Mega-pocket: prefer geometric centroid from LIGSITE enclosed voxels
            log::info!("    Mega-pocket (vol={:.0}Å³): using geometric centroid (spike-weighted drifts in large cavities)",
                pocket.volume);
            pocket.centroid
        } else if stat.weight_sum > 1e-12 {
            [
                (stat.weighted_pos[0] / stat.weight_sum) as f32,
                (stat.weighted_pos[1] / stat.weight_sum) as f32,
                (stat.weighted_pos[2] / stat.weight_sum) as f32,
            ]
        } else {
            pocket.centroid
        };
        // Peak centroid: intensity^2-weighted centroid of top-K hottest spikes.
        // For large pockets (>500 A^3), this is often closer to the actual
        // ligand than the global weighted centroid because it tracks the
        // thermodynamic maximum, not the center-of-mass of diffuse spike cloud.
        let peak_centroid = if !stat.top_spikes.is_empty() {
            let mut pw = [0.0f64; 3];
            let mut pws = 0.0f64;
            for &(intensity, pos) in &stat.top_spikes {
                let w2 = (intensity as f64).powi(2);
                pw[0] += pos[0] as f64 * w2;
                pw[1] += pos[1] as f64 * w2;
                pw[2] += pos[2] as f64 * w2;
                pws += w2;
            }
            if pws > 1e-12 {
                let pc = [
                    (pw[0] / pws) as f32,
                    (pw[1] / pws) as f32,
                    (pw[2] / pws) as f32,
                ];
                let dist = ((pc[0] - final_centroid[0]).powi(2)
                    + (pc[1] - final_centroid[1]).powi(2)
                    + (pc[2] - final_centroid[2]).powi(2)).sqrt();
                log::info!("    Peak centroid: ({:.1},{:.1},{:.1}) -- {:.1}A from weighted centroid (top {} spikes, max_I={:.1})",
                    pc[0], pc[1], pc[2], dist, stat.top_spikes.len(),
                    stat.top_spikes[0].0);
                Some(pc)
            } else { None }
        } else { None };
        // Peak centroid switching for large pockets:
        // For volumes > 300 Å³, the weighted centroid drifts toward the
        // center-of-mass of the diffuse spike cloud, which is often far
        // from the actual ligand binding hotspot. The peak centroid
        // (intensity²-weighted top-K spikes) tracks the thermodynamic
        // maximum and is typically 2-5Å closer to the ligand.
        // For small pockets (≤300 Å³), weighted centroid is already accurate.
        let use_centroid = if let Some(pc) = peak_centroid {
            if pocket.volume > 300.0 {
                let dist = ((pc[0] - final_centroid[0]).powi(2)
                    + (pc[1] - final_centroid[1]).powi(2)
                    + (pc[2] - final_centroid[2]).powi(2)).sqrt();
                if dist > 2.0 && dist < pocket.volume.cbrt() * 2.0 {
                    // Peak is meaningfully different but not pathologically far
                    log::info!("    → Switching to peak centroid ({:.1}Å from weighted, vol={:.0}Å³)",
                        dist, pocket.volume);
                    pc
                } else {
                    final_centroid
                }
            } else {
                final_centroid
            }
        } else {
            final_centroid
        };
        sites.push(ClusteredBindingSite {
            cluster_id: pi as i32,
            centroid: use_centroid,
            spike_count: stat.spike_count as usize,
            spike_indices: std::mem::take(&mut stat.spike_indices),
            avg_intensity,
            estimated_volume: pocket.volume,
            bounding_box: bbox_extents,
            quality_score,
            druggability,
            classification,
            aromatic_proximity: None,
            lining_residues: Vec::new(),
        });
    }

    // Sort by quality descending (NaN-safe)
    sites.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));

    log::info!("  Dynamic LIGSITE complete: {} geometric pockets", sites.len());
}

/// Write ensemble trajectory as multi-MODEL PDB file
/// Each EnsembleSnapshot becomes a MODEL with full atomic coordinates
fn write_ensemble_trajectory(
    snapshots: &[prism_nhs::fused_engine::EnsembleSnapshot],
    topology: &prism_nhs::input::PrismPrepTopology,
    output_base: &std::path::Path,
) -> anyhow::Result<()> {
    use std::io::Write;

    if snapshots.is_empty() {
        log::info!("  No ensemble snapshots to write");
        return Ok(());
    }

    let n_atoms = topology.n_atoms;
    let traj_path = output_base.with_extension("ensemble_trajectory.pdb");
    let mut file = std::fs::File::create(&traj_path)?;

    let mut written_models = 0;

    for (model_idx, snapshot) in snapshots.iter().enumerate() {
        // Verify snapshot has correct number of coordinates
        if snapshot.positions.len() != n_atoms * 3 {
            log::warn!("  Snapshot {} has {} coords, expected {} ({}×3) — skipping",
                model_idx, snapshot.positions.len(), n_atoms * 3, n_atoms);
            continue;
        }

        writeln!(file, "MODEL     {:>4}", model_idx + 1)?;
        writeln!(file, "REMARK   TIMESTEP {}", snapshot.timestep)?;
        writeln!(file, "REMARK   TEMPERATURE {:.1} K", snapshot.temperature)?;
        writeln!(file, "REMARK   TIME {:.3} ps", snapshot.time_ps)?;
        writeln!(file, "REMARK   ALIGNMENT_QUALITY {:.3}", snapshot.alignment_quality)?;
        writeln!(file, "REMARK   TRIGGER {:?}", snapshot.trigger_reason)?;

        for atom_idx in 0..n_atoms {
            let x = snapshot.positions[atom_idx * 3];
            let y = snapshot.positions[atom_idx * 3 + 1];
            let z = snapshot.positions[atom_idx * 3 + 2];

            let atom_name = topology.atom_names.get(atom_idx)
                .map(|s| s.as_str()).unwrap_or("UNK");
            let res_name = topology.residue_names.get(atom_idx)
                .map(|s| s.as_str()).unwrap_or("UNK");
            let chain_id = topology.chain_ids.get(atom_idx)
                .and_then(|s| s.chars().next()).unwrap_or('A');
            let res_id = topology.residue_ids.get(atom_idx)
                .copied().unwrap_or(1);
            let element = topology.elements.get(atom_idx)
                .map(|s| s.as_str()).unwrap_or("X");

            // PDB ATOM format (fixed-width columns)
            // Columns: 1-6 record, 7-11 serial, 13-16 name, 17 altloc,
            //          18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode,
            //          31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
            //          61-66 tempFactor, 77-78 element
            let atom_name_padded = if atom_name.len() < 4 {
                format!(" {:<3}", atom_name)
            } else {
                format!("{:<4}", atom_name)
            };

            write!(file,
                "ATOM  {:>5} {:4}{}{:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                (atom_idx + 1) % 100000,
                atom_name_padded,
                ' ',  // altloc
                res_name,
                chain_id,
                res_id % 10000,
                x, y, z,
                1.00,  // occupancy
                snapshot.alignment_quality * 100.0,  // B-factor = quality metric
                element,
            )?;
        }

        writeln!(file, "ENDMDL")?;
        written_models += 1;
    }
    writeln!(file, "END")?;

    let file_size = std::fs::metadata(&traj_path)?.len();
    let size_str = if file_size > 1_000_000 {
        format!("{:.1} MB", file_size as f64 / 1_000_000.0)
    } else {
        format!("{:.1} KB", file_size as f64 / 1_000.0)
    };

    log::info!("  ✓ Ensemble trajectory: {} ({} models, {} atoms each, {})",
        traj_path.display(), written_models, n_atoms, size_str);

    Ok(())
}
