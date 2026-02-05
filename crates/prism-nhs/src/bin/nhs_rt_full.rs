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
//!   nhs-rt-full -t topology.json -o output_dir --steps 500000

use anyhow::{Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{
    PersistentBatchConfig, PersistentNhsEngine, CryoUvProtocol,
    ClusteredBindingSite, SitePersistenceTracker, PersistenceAnalysis,
    AromaticProximityInfo, enhance_sites_with_aromatics,
    BindingSiteFormatter, write_binding_site_visualizations,
    PrismPrepTopology,
};

#[derive(Parser)]
#[command(name = "nhs-rt-full")]
#[command(about = "Full NHS pipeline with RT-core acceleration")]
struct Args {
    /// Topology JSON file
    #[arg(short, long)]
    topology: PathBuf,

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

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    {
        run_full_pipeline(&args)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        anyhow::bail!("GPU feature required for nhs-rt-full");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_full_pipeline(args: &Args) -> Result<()> {
    let start_time = Instant::now();

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║     PRISM4D NHS RT-FULL PIPELINE                              ║");
    log::info!("║     RT-Core Accelerated Binding Site Detection                ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // Load topology
    log::info!("\n[1/6] Loading topology: {}", args.topology.display());
    let topology = PrismPrepTopology::load(&args.topology)
        .with_context(|| format!("Failed to load: {}", args.topology.display()))?;

    let structure_name = args.topology.file_stem()
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
    engine.set_cryo_uv_protocol(protocol)?;

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
    let n_replicas = args.replicas.max(1);
    let steps_per_replica = args.steps;

    log::info!("\n[3/6] Running MD simulation ({} steps x {} replicas)...",
        steps_per_replica, n_replicas);

    let sim_start = Instant::now();
    let mut all_spikes = Vec::new();
    let mut final_temperature = 0.0f32;
    let mut total_snapshots = 0usize;

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

    let sim_time = sim_start.elapsed();
    let total_steps = steps_per_replica as usize * n_replicas;

    log::info!("  ✓ Completed in {:.1}s ({:.0} steps/sec)",
        sim_time.as_secs_f64(),
        total_steps as f64 / sim_time.as_secs_f64());

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
            match engine.multi_scale_cluster_spikes(&positions) {
                Ok(ms_result) => {
                    log::info!("  ✓ Multi-scale clustering complete: {} persistent clusters",
                        ms_result.num_clusters());

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
    let output_base = args.output.join(&structure_name);

    if !clustered_sites.is_empty() {
        write_binding_site_visualizations(&clustered_sites, &output_base, &structure_name)?;

        // Also write JSON summary (with lining residues for top 100 sites)
        let json_path = output_base.with_extension("binding_sites.json");
        let catalytic_residues = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
        let json_output = serde_json::json!({
            "structure": structure_name,
            "total_steps": args.steps,
            "simulation_time_sec": sim_time.as_secs_f64(),
            "spike_count": accumulated_spikes.len(),
            "binding_sites": clustered_sites.len(),
            "druggable_sites": clustered_sites.iter().filter(|s| s.druggability.is_druggable).count(),
            "lining_residue_cutoff_angstroms": args.lining_cutoff,
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
