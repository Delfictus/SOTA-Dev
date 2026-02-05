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
    #[arg(long, default_value = "100.0")]
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

    /// Fast 50K protocol - uses ultra-cold start for faster equilibration
    /// ~60% faster than standard while maintaining detection quality
    #[arg(long, default_value = "false")]
    fast: bool,

    /// Enable true parallel replica execution via AmberSimdBatch
    /// All replicas run simultaneously on GPU (vs sequential when disabled)
    #[arg(long, default_value = "false")]
    parallel: bool,

    /// Enable adaptive epsilon selection from k-NN distribution
    /// Automatically determines optimal clustering scales per structure
    #[arg(long, default_value = "true")]
    adaptive_epsilon: bool,

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
#[cfg(feature = "gpu")]
fn run_from_manifest(args: &Args, manifest_path: &PathBuf) -> Result<()> {
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

    // Override replicas from manifest if not explicitly set (backward compat - now using per-batch replicas)
    let _replicas = if args.replicas == 1 && manifest.replicas > 1 {
        manifest.replicas
    } else {
        args.replicas
    };

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // PARALLEL BATCH EXECUTION: Run all batches concurrently on shared GPU
    // Total memory (92MB) << GPU capacity (16GB), so launch all batches at once
    log::info!("Launching {} batches in parallel on shared GPU...", manifest.total_batches);

    use std::thread;
    use std::sync::{Arc, Mutex};

    let results_mutex = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for batch in &manifest.batches {
        let batch = batch.clone();
        let args = args.clone();
        let results_clone = Arc::clone(&results_mutex);

        let handle = thread::spawn(move || {
            let batch_replicas = batch.replicas_per_structure;
            log::info!("═══ Batch {} ({} tier, {} structures, {} replicas) ═══",
                batch.batch_id, batch.memory_tier, batch.structures.len(), batch_replicas);

            log::info!("  GPU-concurrent execution: {} structures × {} replicas = {} total in AmberSimdBatch",
                batch.structures.len(), batch_replicas, batch.structures.len() * batch_replicas);

            // Run entire batch concurrently on GPU with replicas
            match run_batch_gpu_concurrent(&batch.structures, &args, batch_replicas) {
                Ok(batch_results) => {
                    results_clone.lock().unwrap().extend(batch_results);
                }
                Err(e) => {
                    log::error!("  ✗ Batch {} execution failed: {}", batch.batch_id, e);
                    // Mark all structures in batch as failed
                    let mut failed_results = Vec::new();
                    for structure in &batch.structures {
                        failed_results.push(StructureRunResult {
                            name: structure.name.clone(),
                            success: false,
                            error: Some(format!("Batch GPU execution failed: {}", e)),
                            elapsed_seconds: 0.0,
                            sites_found: None,
                            druggable_sites: None,
                        });
                    }
                    results_clone.lock().unwrap().extend(failed_results);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all batches to complete
    log::info!("Waiting for all {} batches to complete...", handles.len());
    for handle in handles {
        handle.join().unwrap();
    }

    // Extract final results
    let results = results_mutex.lock().unwrap().clone();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.iter().filter(|r| !r.success).count();

    // Write summary
    let total_elapsed = total_start.elapsed().as_secs_f64();
    let summary = ManifestRunSummary {
        manifest_path: manifest_path.to_string_lossy().to_string(),
        total_structures: manifest.total_structures,
        successful,
        failed,
        total_elapsed_seconds: total_elapsed,
        results,
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
    println!("║ Total time:       {:>6.1}s                                    ║", total_elapsed);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Summary written to: {}", summary_path.display());

    Ok(())
}

/// Run single structure (original behavior)
#[cfg(feature = "gpu")]
fn run_single_structure(args: &Args, topology_path: &PathBuf) -> Result<()> {
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
        log::info!("  Protocol: Fast 50K (60% faster, full aromatic coverage)");
        CryoUvProtocol::fast_50k()
    } else {
        // Standard protocol with user-configurable temperatures
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: config.cryo_hold,
            ramp_steps: config.convergence_steps / 2,
            warm_hold_steps: config.convergence_steps / 2,
            current_step: 0,
            uv_burst_energy: 30.0,
            uv_burst_interval: 500,
            uv_burst_duration: 50,
            // Full aromatic coverage: TRP, TYR, PHE, HIS (all protonation states)
            scan_wavelengths: vec![280.0, 274.0, 258.0, 211.0],
            wavelength_dwell_steps: 500,
        }
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

    // --fast overrides step count to use optimized 50K protocol
    let steps_per_replica = if args.fast {
        // fast_50k protocol: 20K cold + 8K ramp + 2K warm = 30K, but we add margin
        50_000  // Total 50K steps for fast validation
    } else {
        args.steps
    };

    log::info!("\n[3/6] Running MD simulation ({} steps x {} replicas)...",
        steps_per_replica, n_replicas);

    let sim_start = Instant::now();
    let mut all_spikes = Vec::new();
    let mut final_temperature = 0.0f32;
    let mut total_snapshots = 0usize;

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
                voxel_idx: 0, // Not tracked in parallel mode
                position: spike.position,
                intensity: spike.intensity,
                nearby_residues: [0; 8], // Not tracked in parallel mode
                n_residues: 0,
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
            total_snapshots += engine.get_snapshots().len();

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

    // Intensity pre-filtering: keep only top 20% by intensity to reduce noise
    let accumulated_spikes = if all_spikes.len() > 1000 {
        // Compute intensity threshold (80th percentile)
        let mut intensities: Vec<f32> = all_spikes.iter()
            .map(|s| s.intensity)
            .collect();
        intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (intensities.len() as f32 * 0.80) as usize;
        let intensity_threshold = intensities.get(threshold_idx).copied().unwrap_or(0.0);

        let filtered: Vec<_> = all_spikes.into_iter()
            .filter(|s| s.intensity >= intensity_threshold)
            .collect();
        log::info!("  Intensity filter: kept {} spikes (top 20%, threshold={:.2})",
            filtered.len(), intensity_threshold);
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
                Some(vec![5.0f32, 7.0, 10.0, 14.0]) // Fixed default values
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

                    // Post-filter: keep only significant clusters (>50 spikes)
                    let min_spikes = 50;
                    let sites: Vec<_> = all_sites.into_iter()
                        .filter(|s| s.spike_count >= min_spikes)
                        .collect();
                    log::info!("  Binding sites: {} (filtered, min {} spikes)",
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
                Ok(result) => {
                    log::info!("  ✓ Clustering complete: {} clusters, {} neighbor pairs, {:.2}ms",
                        result.num_clusters, result.total_neighbors, result.gpu_time_ms);

                    // Build clustered binding sites
                    let all_sites = build_sites_from_clustering(&accumulated_spikes, &result);

                    // Post-filter: keep only significant clusters (>50 spikes)
                    let min_spikes = 50;
                    let sites: Vec<_> = all_sites.into_iter()
                        .filter(|s| s.spike_count >= min_spikes)
                        .collect();
                    log::info!("  Binding sites: {} (filtered from {} clusters, min {} spikes)",
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

        // Compute lining residues for top sites (limit to top 100 for performance)
        let lining_cutoff = args.lining_cutoff;
        for site in clustered_sites.iter_mut().take(100) {
            site.compute_lining_residues(
                &topology.positions,
                &topology.residue_ids,
                &topology.residue_names,
                &topology.chain_ids,
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
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON summary: {}", json_path.display());
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

/// Run batch of structures concurrently on GPU using AmberSimdBatch
#[cfg(feature = "gpu")]
fn run_batch_gpu_concurrent(
    structures: &[ManifestStructure],
    args: &Args,
    replicas: usize,
) -> Result<Vec<StructureRunResult>> {
    use prism_gpu::{AmberSimdBatch, OptimizationConfig};
    use cudarc::driver::CudaContext;
    use prism_nhs::fused_engine::GpuSpikeEvent;

    let batch_start = Instant::now();

    // Find max atoms for batch sizing
    let max_atoms = structures.iter().map(|s| s.atoms).max().unwrap_or(0);
    let n_structures = structures.len();
    let total_entries = n_structures * replicas;

    log::info!("    Creating AmberSimdBatch: {} structures × {} replicas = {} total entries, max {} atoms",
        n_structures, replicas, total_entries, max_atoms);

    // Create CUDA context and batch
    let context = CudaContext::new(0)?;
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
    let mut structure_ids = Vec::new();
    let mut topologies = Vec::new();
    let mut aromatic_positions_per_structure = Vec::new();
    let mut aromatic_indices_per_structure = Vec::new();

    for (struct_idx, structure) in structures.iter().enumerate() {
        let topology_path = PathBuf::from(&structure.topology_path);
        let topology = PrismPrepTopology::load(&topology_path)
            .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

        // Extract aromatic positions for UV burst targeting
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

    // Determine steps
    let steps_per_structure = if args.fast { 50_000 } else { args.steps as usize };

    // Configure protocol
    let protocol = if args.fast {
        CryoUvProtocol::fast_50k()
    } else {
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: 50000,
            ramp_steps: args.steps / 4,
            warm_hold_steps: args.steps / 4,
            current_step: 0,
            uv_burst_energy: 30.0,
            uv_burst_interval: 500,
            uv_burst_duration: 50,
            scan_wavelengths: vec![280.0, 274.0, 258.0, 211.0],
            wavelength_dwell_steps: 500,
        }
    };

    // Compute simulation phases
    let cold_hold = protocol.cold_hold_steps as usize;
    let ramp = protocol.ramp_steps as usize;
    let warm_hold = protocol.warm_hold_steps as usize;
    let total_protocol_steps = cold_hold + ramp + warm_hold;

    let scale = if steps_per_structure < total_protocol_steps {
        steps_per_structure as f64 / total_protocol_steps as f64
    } else {
        1.0
    };

    let cold_steps = ((cold_hold as f64 * scale) as usize).max(100);
    let ramp_steps = ((ramp as f64 * scale) as usize).max(100);
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
            // Apply intensity filtering per replica (top 20%)
            let filtered_spikes = if replica_spikes.len() > 1000 {
                let mut intensities: Vec<f32> = replica_spikes.iter()
                    .map(|s| s.intensity)
                    .collect();
                intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let threshold_idx = (intensities.len() as f32 * 0.80) as usize;
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
                        Some(vec![5.0f32, 7.0, 10.0, 14.0])
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
                            let min_spikes = 50;
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
                            let min_spikes = 50;
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

        // Compute lining residues
        let lining_cutoff = args.lining_cutoff;
        for site in clustered_sites.iter_mut().take(100) {
            site.compute_lining_residues(
                &topology.positions,
                &topology.residue_ids,
                &topology.residue_names,
                &topology.chain_ids,
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
            apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy)?;
        }

        // Extract positions and detect spikes per entry (structure × replica)
        let all_positions = batch.get_positions()?;
        for (entry_idx, &(struct_idx, _replica_idx)) in entry_mapping.iter().enumerate() {
            let topology = &topologies[struct_idx];
            let n_atoms = topology.n_atoms;
            let aromatic_indices = &aromatic_indices_per_structure[struct_idx];

            // Extract per-entry positions from batch
            let start_idx = entry_idx * n_atoms * 3;
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
            apply_batch_uv_burst(batch, aromatic_indices_per_structure, topologies, uv_energy)?;
        }

        // Extract positions and detect spikes per entry (structure × replica)
        let all_positions = batch.get_positions()?;
        for (entry_idx, &(struct_idx, _replica_idx)) in entry_mapping.iter().enumerate() {
            let topology = &topologies[struct_idx];
            let n_atoms = topology.n_atoms;
            let aromatic_indices = &aromatic_indices_per_structure[struct_idx];

            // Extract per-entry positions from batch
            let start_idx = entry_idx * n_atoms * 3;
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
    energy: f32,
) -> Result<()> {
    // Download current velocities
    let mut velocities = batch.get_velocities()?;

    let velocity_boost = (2.0 * energy / 12.0).sqrt(); // Approximate for carbon mass

    for (struct_idx, aromatic_indices) in aromatic_indices_per_structure.iter().enumerate() {
        if aromatic_indices.is_empty() {
            continue;
        }

        let topology = &topologies[struct_idx];
        let n_atoms = topology.n_atoms;
        let offset = struct_idx * n_atoms * 3;

        for &atom_idx in aromatic_indices {
            let base = offset + atom_idx * 3;
            if base + 2 < velocities.len() {
                // Add random direction burst
                let theta = (struct_idx as f32 * 0.7 + atom_idx as f32 * 1.3) % std::f32::consts::TAU;
                let phi = (struct_idx as f32 * 1.1 + atom_idx as f32 * 0.9) % std::f32::consts::PI;
                velocities[base] += velocity_boost * theta.sin() * phi.cos();
                velocities[base + 1] += velocity_boost * theta.sin() * phi.sin();
                velocities[base + 2] += velocity_boost * theta.cos();
            }
        }
    }

    // Upload modified velocities
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
                });
            }
        }
    }

    spikes
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

        // Determine aromatic type from residue name
        let aromatic_type = if let Some(name) = topology.residue_names.get(res_idx) {
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
        let voxel_size = 2.0f32;
        let estimated_volume = if bounding_box[0] > 0.0 && bounding_box[1] > 0.0 && bounding_box[2] > 0.0 {
            let mut occupied_voxels = std::collections::HashSet::new();
            for (_, spike) in &spikes {
                let pos = spike.position;
                let vx = ((pos[0] - min_pos[0]) / voxel_size) as i32;
                let vy = ((pos[1] - min_pos[1]) / voxel_size) as i32;
                let vz = ((pos[2] - min_pos[2]) / voxel_size) as i32;

                // Mark voxel and immediate neighbors
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            occupied_voxels.insert((vx + dx, vy + dy, vz + dz));
                        }
                    }
                }
            }

            let voxel_volume = voxel_size.powi(3);
            let raw_volume = occupied_voxels.len() as f32 * voxel_volume;
            // Packing efficiency correction, clamped to reasonable pocket sizes
            (raw_volume * 0.74).clamp(50.0, 5000.0)
        } else {
            (spikes.len() as f32 * 15.0).clamp(50.0, 2000.0)
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
