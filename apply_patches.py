#!/usr/bin/env python3
"""
PRISM4D Multi-Stream Patcher
Adds true multi-stream CUDA concurrency to nhs_rt_full.

Patches 3 files:
  1. fused_engine.rs      — new_on_stream() constructor
  2. persistent_engine.rs — new_on_stream() + stream passthrough
  3. nhs_rt_full.rs       — --multi-stream N flag + pipeline
"""

import sys
import os

def patch_file(path, old, new, label):
    with open(path, "r") as f:
        src = f.read()
    if old not in src:
        print(f"✗ {label} FAILED: match text not found in {path}")
        # Write first 200 chars around expected location for debugging
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"✓ {label}: {path}")
    return True


def main():
    # Detect repo root
    if os.path.exists("crates/prism-nhs/src/fused_engine.rs"):
        root = "."
    elif os.path.exists(os.path.expanduser("~/Desktop/Prism4D-bio/crates/prism-nhs/src/fused_engine.rs")):
        root = os.path.expanduser("~/Desktop/Prism4D-bio")
    else:
        print("ERROR: Can't find crates/prism-nhs/src/fused_engine.rs")
        print("Run this script from the Prism4D-bio directory")
        sys.exit(1)

    os.chdir(root)
    print(f"Working in: {os.getcwd()}\n")

    ok = True

    # ═══════════════════════════════════════════════════════════
    # PATCH 1: fused_engine.rs — add new_on_stream()
    # ═══════════════════════════════════════════════════════════
    ok &= patch_file(
        "crates/prism-nhs/src/fused_engine.rs",

        '''    pub fn new(
        context: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        log::info!("Creating NHS-AMBER Fused Engine: {} atoms, grid {}\\u{b3}",
            topology.n_atoms, grid_dim);
        if grid_dim > MAX_GRID_DIM {
            bail!("Grid dimension {} exceeds maximum {}", grid_dim, MAX_GRID_DIM);
        }
        let stream = context.default_stream();''',

        '''    pub fn new(
        context: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        let stream = context.default_stream();
        Self::new_on_stream(context, stream, topology, grid_dim, grid_spacing)
    }

    /// Create new fused engine with explicit CUDA stream (for multi-stream concurrency).
    /// Each stream gets independent kernel execution — GPU overlaps work across streams.
    pub fn new_on_stream(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        log::info!("Creating NHS-AMBER Fused Engine: {} atoms, grid {}\\u{b3}",
            topology.n_atoms, grid_dim);
        if grid_dim > MAX_GRID_DIM {
            bail!("Grid dimension {} exceeds maximum {}", grid_dim, MAX_GRID_DIM);
        }''',

        "PATCH 1: new_on_stream()"
    )

    # ═══════════════════════════════════════════════════════════
    # PATCH 2a: persistent_engine.rs — add new_on_stream()
    # ═══════════════════════════════════════════════════════════
    ok &= patch_file(
        "crates/prism-nhs/src/persistent_engine.rs",

        '''            context_init_time_ms,
            module_init_time_ms,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }''',

        '''            context_init_time_ms,
            module_init_time_ms,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Create persistent engine on an explicit CUDA stream (for multi-stream concurrency).
    /// Shares context + module with other engines — each gets its own stream for GPU overlap.
    pub fn new_on_stream(
        config: &PersistentBatchConfig,
        context: Arc<CudaContext>,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        log::info!("Persistent NHS Engine on dedicated stream (max_atoms: {})", config.max_atoms);
        Ok(Self {
            context,
            module,
            stream,
            engine: None,
            rt_engine: None,
            max_atoms: config.max_atoms,
            grid_dim: config.grid_dim,
            grid_spacing: config.grid_spacing,
            current_topology_id: None,
            context_init_time_ms: 0,
            module_init_time_ms: 0,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Access the shared CUDA context (for creating additional streams)
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Access the compiled PTX module (for sharing across engines)
    pub fn cuda_module(&self) -> &Arc<CudaModule> {
        &self.module
    }''',

        "PATCH 2a: persistent new_on_stream()"
    )

    # ═══════════════════════════════════════════════════════════
    # PATCH 2b: persistent_engine.rs — load_topology passes stream
    # ═══════════════════════════════════════════════════════════
    ok &= patch_file(
        "crates/prism-nhs/src/persistent_engine.rs",

        '''        let engine = NhsAmberFusedEngine::new(
            self.context.clone(),
            topology,
            self.grid_dim,
            self.grid_spacing,
        )?;''',

        '''        let engine = NhsAmberFusedEngine::new_on_stream(
            self.context.clone(),
            self.stream.clone(),
            topology,
            self.grid_dim,
            self.grid_spacing,
        )?;''',

        "PATCH 2b: load_topology stream passthrough"
    )

    # ═══════════════════════════════════════════════════════════
    # PATCH 3a: nhs_rt_full.rs — add --multi-stream arg
    # ═══════════════════════════════════════════════════════════
    ok &= patch_file(
        "crates/prism-nhs/src/bin/nhs_rt_full.rs",

        '''    /// Enable true parallel replica execution via AmberSimdBatch
    /// All replicas run simultaneously on GPU (vs sequential when disabled)
    #[arg(long, default_value = "false")]
    parallel: bool,''',

        '''    /// Enable true parallel replica execution via AmberSimdBatch
    /// All replicas run simultaneously on GPU (vs sequential when disabled)
    #[arg(long, default_value = "false")]
    parallel: bool,

    /// True multi-stream concurrency: N independent CUDA streams each running
    /// the FULL cryo-UV-BNZ-RT pipeline. Creates N PersistentNhsEngine instances
    /// on separate streams for maximum GPU utilization. Results are aggregated
    /// via consensus clustering across all streams.
    #[arg(long, default_value = "0")]
    multi_stream: usize,''',

        "PATCH 3a: --multi-stream arg"
    )

    # ═══════════════════════════════════════════════════════════
    # PATCH 3b: nhs_rt_full.rs — dispatch to multi-stream
    # ═══════════════════════════════════════════════════════════
    ok &= patch_file(
        "crates/prism-nhs/src/bin/nhs_rt_full.rs",

        '''fn run_single_structure(args: &Args, topology_path: &PathBuf) -> Result<()> {
    run_single_structure_internal(topology_path, &args.output, args, args.replicas)?;
    Ok(())
}''',

        '''fn run_single_structure(args: &Args, topology_path: &PathBuf) -> Result<()> {
    if args.multi_stream > 1 {
        return run_multi_stream_pipeline(args, topology_path, args.multi_stream);
    }
    run_single_structure_internal(topology_path, &args.output, args, args.replicas)?;
    Ok(())
}''',

        "PATCH 3b: multi-stream dispatch"
    )

    # ═══════════════════════════════════════════════════════════
    # PATCH 3c: nhs_rt_full.rs — add run_multi_stream_pipeline()
    # ═══════════════════════════════════════════════════════════

    MULTI_STREAM_FN = r'''/// True multi-stream pipeline: N independent CUDA streams, each running
/// the full cryo-UV-BNZ-RT stack. One CudaContext, one PTX module, N streams.
/// Results aggregated via consensus clustering.
#[cfg(feature = "gpu")]
fn run_multi_stream_pipeline(
    args: &Args,
    topology_path: &PathBuf,
    n_streams: usize,
) -> Result<()> {
    use cudarc::driver::{CudaContext, Ptx};
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

    let steps_per_stream = if args.fast { 35_000i32 } else { args.steps };

    let protocol = if args.fast {
        CryoUvProtocol::fast_35k()
    } else {
        CryoUvProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            cold_hold_steps: config.cryo_hold,
            ramp_steps: config.convergence_steps / 2,
            warm_hold_steps: config.convergence_steps / 2,
            current_step: 0,
            uv_burst_energy: 50.0,
            uv_burst_interval: 250,
            uv_burst_duration: 50,
            scan_wavelengths: vec![280.0, 274.0, 258.0, 254.0, 211.0],
            wavelength_dwell_steps: 500,
        }
    };
    let _target_end_temp = protocol.end_temp;

    // ── Run N engines on N threads (scoped for safe borrowing) ──
    log::info!("\n  🚀 Launching {} independent trajectories...", n_streams);
    let sim_start = Instant::now();

    let stream_results: Vec<Result<Vec<prism_nhs::fused_engine::GpuSpikeEvent>>> =
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

                s.spawn(move || -> Result<Vec<prism_nhs::fused_engine::GpuSpikeEvent>> {
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

                    log::info!("    [stream {}] Complete: {} spikes, T={:.1}K",
                        i, spikes.len(), summary.end_temperature);
                    Ok(spikes)
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

    for (i, result) in stream_results.into_iter().enumerate() {
        let raw_spikes = match result {
            Ok(spikes) => spikes,
            Err(e) => {
                log::error!("    Stream {} failed: {}", i, e);
                per_stream_stats.push(serde_json::json!({
                    "stream_id": i, "error": e.to_string(),
                }));
                per_stream_sites.push(Vec::new());
                continue;
            }
        };

        let filtered = if raw_spikes.len() > 1000 {
            let mut intensities: Vec<f32> = raw_spikes.iter().map(|s| s.intensity).collect();
            intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (intensities.len() as f32 * 0.98) as usize;
            let threshold = intensities.get(idx).copied().unwrap_or(0.0);
            raw_spikes.into_iter().filter(|s| s.intensity >= threshold).collect::<Vec<_>>()
        } else {
            raw_spikes
        };

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
    }

    let consensus_threshold = if n_streams >= 3 {
        (n_streams as f32 * 0.67).ceil() as usize
    } else {
        1
    };
    log::info!("  Consensus threshold: {}/{} streams", consensus_threshold, n_streams);

    let mut clustered_sites = if n_streams == 1 && !per_stream_sites.is_empty() {
        per_stream_sites.into_iter().next().unwrap_or_default()
    } else {
        build_consensus_sites(&per_stream_sites, consensus_threshold, 5.0)
    };
    log::info!("  Consensus binding sites: {}", clustered_sites.len());

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
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_output)?)?;
        log::info!("  ✓ JSON: {}", json_path.display());
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

'''

    ok &= patch_file(
        "crates/prism-nhs/src/bin/nhs_rt_full.rs",

        '''/// Extract aromatic residue positions from topology
#[cfg(feature = "gpu")]
fn extract_aromatic_positions''',

        MULTI_STREAM_FN + '''/// Extract aromatic residue positions from topology
#[cfg(feature = "gpu")]
fn extract_aromatic_positions''',

        "PATCH 3c: run_multi_stream_pipeline()"
    )

    print()
    if ok:
        print("═══ ALL PATCHES APPLIED SUCCESSFULLY ═══")
        print()
        print("Next steps:")
        print("  cargo build --release -p prism-nhs --bin nhs_rt_full")
        print()
        print("Then run:")
        print("  target/release/nhs_rt_full \\")
        print("    -t e2e_validation_test/prep/1nkp_dimer.topology.json \\")
        print("    -o e2e_validation_test/results_myc_multistream \\")
        print("    --multi-stream 4 --multi-scale --rt-clustering \\")
        print("    --lining-cutoff 8.0 --fast --ultimate-mode -v")
    else:
        print("═══ SOME PATCHES FAILED — check output above ═══")
        sys.exit(1)


if __name__ == "__main__":
    main()
