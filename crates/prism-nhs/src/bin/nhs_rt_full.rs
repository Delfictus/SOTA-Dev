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
    let topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

    let structure_name = topology_path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "structure".to_string());

    log::info!("  Atoms: {}", topology.n_atoms);
    log::info!("  Residues: {}", topology.residue_ids.iter().max().unwrap_or(&0) + 1);

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
    let config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
        ..Default::default()
    };

    let mut engine = PersistentNhsEngine::new(&config)?;
    engine.load_topology(&topology)?;

    // Check RT clustering availability
    let has_rt = engine.has_rt_clustering();
    log::info!("  RT-core clustering: {}", if has_rt { "✓ Available" } else { "✗ Fallback mode" });

    // Configure cryo-UV protocol
    let protocol = if args.fast {
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

    // --fast uses full protocol length (respects hysteresis), otherwise user --steps
    let steps_per_replica = if args.fast {
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
                _padding: 0,
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
        let site_radius = args.lining_cutoff + 2.0;
        let frame_window = 1000i32;
        let mut all_pockets_json = Vec::new();
        let mut cryptic_sites_json = Vec::new();

        for site in clustered_sites.iter().take(100) {
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];

            let site_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> = accumulated_spikes.iter()
                .filter(|s| {
                    let dx = s.position[0] - cx;
                    let dy = s.position[1] - cy;
                    let dz = s.position[2] - cz;
                    (dx*dx + dy*dy + dz*dz).sqrt() <= site_radius
                })
                .collect();

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
                "spike_frames": spike_frames,
                "spike_amplitudes": spike_amplitudes,
                "inter_spike_intervals": inter_spike_intervals,
                "volume": site.estimated_volume,
                "druggability": site.druggability.overall,
                "classification": format!("{:?}", site.classification),
            }));
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
            "all_pockets": all_pockets_json,
            "cryptic_sites": cryptic_sites_json,
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON summary: {}", json_path.display());

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
    let protocol = if args.fast {
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

    // Determine steps: --fast uses full protocol length (respects hysteresis)
    let steps_per_structure = if args.fast {
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
    let config = PersistentBatchConfig {
        max_atoms: max_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
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
        let clustered_sites = build_consensus_sites(&per_replica_sites, consensus_threshold, 5.0);
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
                    _padding: 0,
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
    let topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

    let structure_name = topology_path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "structure".to_string());

    let aromatic_positions = extract_aromatic_positions(&topology);
    log::info!("  Structure: {} ({} atoms, {} aromatics)",
        structure_name, topology.n_atoms, aromatic_positions.len());

    if topology.n_atoms < 500 {
        anyhow::bail!("Protein too small for GPU analysis ({} atoms, min 500)", topology.n_atoms);
    }

    let config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(15000),
        survey_steps: args.steps / 2,
        convergence_steps: args.steps / 4,
        precision_steps: args.steps / 4,
        temperature: args.temperature,
        cryo_temp: args.cryo_temp,
        cryo_hold: 50000,
        ..Default::default()
    };

    let protocol = if args.fast {
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

    // Steps per stream: use full protocol length for --fast (respects hysteresis),
    // otherwise use user-specified --steps
    let steps_per_stream = if args.fast {
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

                s.spawn(move || -> Result<(Vec<prism_nhs::fused_engine::GpuSpikeEvent>, Vec<prism_nhs::fused_engine::EnsembleSnapshot>)> {
                    log::info!("    [stream {}] Starting (seed: {})...", i, seed);

                    let mut engine = PersistentNhsEngine::new_on_stream(
                        config_ref, ctx, mod_, stream_i,
                    )?;
                    engine.load_topology(topo_ref)?;
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
        build_consensus_sites(&per_stream_sites, consensus_threshold, 10.0)
    };
    log::info!("  Consensus binding sites: {}", clustered_sites.len());

    // Recalculate volumes using buriedness-aware enclosure method
    // This replaces the naive spike-cloud volume with true pocket enclosure volume
    if !clustered_sites.is_empty() {
        log::info!("  Recalculating volumes with enclosure analysis...");
        recalculate_enclosure_volume(&mut clustered_sites, &all_stream_spikes, &topology.positions);
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
        let mut all_pockets_json = Vec::new();
        let mut cryptic_sites_json = Vec::new();

        for site in clustered_sites.iter().take(100) {
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];

            // Collect spikes for this site
            let site_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> = all_stream_spikes.iter()
                .filter(|s| {
                    let dx = s.position[0] - cx;
                    let dy = s.position[1] - cy;
                    let dz = s.position[2] - cz;
                    (dx*dx + dy*dy + dz*dz).sqrt() <= site_radius
                })
                .collect();

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
                "spike_frames": spike_frames,
                "spike_amplitudes": spike_amplitudes,
                "inter_spike_intervals": inter_spike_intervals,
                "volume": site.estimated_volume,
                "druggability": site.druggability.overall,
                "classification": format!("{:?}", site.classification),
            }));
        }

        let json_path = output_base.with_extension("binding_sites.json");
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
            "sites": clustered_sites.iter().take(100).map(|s| {
                let cat_count = s.lining_residues.iter()
                    .filter(|r| catalytic_residues.contains(&r.resname.as_str())).count();
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
                    "lining_residues": s.lining_residues.iter().map(|r| {
                        serde_json::json!({
                            "chain": r.chain, "resid": r.resid, "resname": r.resname,
                            "min_distance": r.min_distance, "n_atoms": r.n_atoms_in_pocket,
                            "is_catalytic": catalytic_residues.contains(&r.resname.as_str()),
                        })
                    }).collect::<Vec<_>>(),
                    "residue_ids": s.lining_residue_ids(),
                })
            }).collect::<Vec<_>>(),
            "all_pockets": all_pockets_json,
            "cryptic_sites": cryptic_sites_json,
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON: {}", json_path.display());
    }


    // Export spike events with enhanced metadata for pharmacophore mapping
    if !all_stream_spikes.is_empty() && !clustered_sites.is_empty() {
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

/// Build consensus sites from per-replica clustering results
/// Sites must appear in at least `threshold` replicas within `spatial_tolerance` Angstroms
#[cfg(feature = "gpu")]
fn build_consensus_sites(
    per_replica_sites: &[Vec<ClusteredBindingSite>],
    threshold: usize,
    spatial_tolerance: f32,
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

        consensus_sites.push(ClusteredBindingSite {
            cluster_id: cluster_id as i32,
            centroid,
            spike_count: avg_spike_count,
            spike_indices: Vec::new(), // Not meaningful for consensus
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

/// Recalculate site volumes using LIGSITE-style grid scanline method.
///
/// Scans the ENTIRE protein to find enclosed void regions, then intersects
/// with spike activity. Grid covers the full protein bounding box + margin.
///
///   1. Build 3D boolean grid `is_protein` over entire protein (SES surface)
///   2. Build 3D boolean grid `is_active` from ALL spikes
///   3. For each non-protein, active voxel:
///      - Ray-cast ±X, ±Y, ±Z through `is_protein` grid
///      - Blocked = ray hits protein before grid boundary
///   4. Voxel is pocket void if blocked in ≥ 5 of 6 directions
///   5. Assign enclosed voxels to nearest site; sum per site
///
/// Surface noise: spikes exist, void exists, but rays escape → volume ≈ 0
/// Real pocket: spikes exist, void exists, rays hit walls → volume > 0
#[cfg(feature = "gpu")]
fn recalculate_enclosure_volume(
    sites: &mut [ClusteredBindingSite],
    all_spikes: &[prism_nhs::fused_engine::GpuSpikeEvent],
    atom_positions: &[f32],
) {
    use prism_nhs::{DruggabilityScore, SiteClassification};

    let n_atoms = atom_positions.len() / 3;
    let grid_step = 1.0f32;
    let exclusion_radius = 3.0f32;  // Standard SES probe radius
    let spike_reach = 5.0f32;       // active zone around spike positions
    let scan_margin = 10.0f32;      // margin for rays to escape past protein surface
    let min_blocked = 4u32;         // Trench/groove mode: 4 of 6 directions blocked catches shallow binding grooves

    if n_atoms == 0 || sites.is_empty() {
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
    let mut is_active = vec![false; grid_size];

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

    // ---- Stage 2: Mark active voxels (near ANY spike, whole protein) ----
    let spike_reach_sq = spike_reach * spike_reach;
    let spike_cells = (spike_reach / grid_step).ceil() as i32 + 1;
    for spike in all_spikes {
        let sp = spike.position;
        // Quick reject: spike outside grid
        let six = ((sp[0] - gmin[0]) / grid_step).round() as i32;
        let siy = ((sp[1] - gmin[1]) / grid_step).round() as i32;
        let siz = ((sp[2] - gmin[2]) / grid_step).round() as i32;
        if six < -spike_cells || siy < -spike_cells || siz < -spike_cells { continue; }
        if six > nx as i32 + spike_cells || siy > ny as i32 + spike_cells || siz > nz as i32 + spike_cells { continue; }
        for dx in -spike_cells..=spike_cells {
            for dy in -spike_cells..=spike_cells {
                for dz in -spike_cells..=spike_cells {
                    let ix = six + dx;
                    let iy = siy + dy;
                    let iz = siz + dz;
                    if ix < 0 || iy < 0 || iz < 0 { continue; }
                    let ix = ix as usize;
                    let iy = iy as usize;
                    let iz = iz as usize;
                    if ix >= nx || iy >= ny || iz >= nz { continue; }
                    let idx = to_idx(ix, iy, iz);
                    if !is_active[idx] {
                        let w = to_world(ix, iy, iz);
                        let d2 = (w[0]-sp[0]).powi(2) + (w[1]-sp[1]).powi(2) + (w[2]-sp[2]).powi(2);
                        if d2 <= spike_reach_sq {
                            is_active[idx] = true;
                        }
                    }
                }
            }
        }
    }

    // ---- Stage 3: Find enclosed active void voxels, assign to nearest site ----
    let mut per_site_enclosed: Vec<u32> = vec![0; sites.len()];
    let mut total_active_void = 0u32;
    let mut total_enclosed = 0u32;
    let mut total_protein = 0u32;

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = to_idx(ix, iy, iz);
                if is_protein[idx] { total_protein += 1; continue; }
                if !is_active[idx] { continue; }
                total_active_void += 1;

                // Ray-cast in 6 axial directions with a 10Å HORIZON
                // This prevents tangent rays from hitting distant protein parts on convex surfaces
                let mut blocked = 0u32;
                let max_ray_dist = 10.0f32;
                let max_ray_steps = (max_ray_dist / grid_step).ceil() as usize;

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
                    total_enclosed += 1;

                    // Assign to nearest site, BUT ONLY if within the 12Å local pocket leash.
                    // This prevents "Global Vacuum" where distant surface roughness is summed into a site.
                    let w = to_world(ix, iy, iz);
                    let mut best_site = None;
                    let mut best_d2 = f32::MAX;
                    let max_assignment_dist = 20.0f32; // Expanded to capture long trenches/grooves
                    let max_site_d2 = max_assignment_dist * max_assignment_dist;

                    for (si, site) in sites.iter().enumerate() {
                        let d2 = (w[0]-site.centroid[0]).powi(2) +
                                 (w[1]-site.centroid[1]).powi(2) +
                                 (w[2]-site.centroid[2]).powi(2);

                        // Strict assignment: must be within 12Å of site centroid
                        if d2 < best_d2 && d2 <= max_site_d2 {
                            best_d2 = d2;
                            best_site = Some(si);
                        }
                    }

                    if let Some(si) = best_site {
                        per_site_enclosed[si] += 1;
                    }
                }
            }
        }
    }

    log::info!("  LIGSITE totals: protein={}, active_void={}, enclosed(≥{}/6)={}",
        total_protein, total_active_void, min_blocked, total_enclosed);

    // Update each site's volume
    for (si, site) in sites.iter_mut().enumerate() {
        let new_volume = per_site_enclosed[si] as f32 * grid_step.powi(3);
        log::info!("  Site {}: LIGSITE volume = {:.1} Å³ (old={:.1}, {} enclosed voxels)",
            site.cluster_id, new_volume, site.estimated_volume, per_site_enclosed[si]);

        site.estimated_volume = new_volume;
        site.druggability = DruggabilityScore::from_site(new_volume, site.avg_intensity, &site.bounding_box);
        site.classification = SiteClassification::from_properties(site.spike_count, new_volume, site.avg_intensity);
        site.quality_score = {
            let spike_quality = (site.spike_count as f32 / 100.0).clamp(0.0, 1.0);
            let intensity_quality = (site.avg_intensity / 10.0).clamp(0.0, 1.0);
            0.3 * spike_quality + 0.3 * intensity_quality + 0.4 * site.druggability.overall
        };
    }
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
