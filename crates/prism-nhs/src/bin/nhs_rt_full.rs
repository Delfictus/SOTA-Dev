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

    /// Adaptive cryo-thermal protocol: measure spike rate during cold_hold phase
    /// and auto-tune remaining protocol phases based on protein flexibility.
    /// Stiff proteins (low spike rate) get aggressive ramping + extended warm hold.
    /// Flexible proteins (high spike rate) get faster cooling to trap open states.
    #[arg(long, default_value = "false")]
    adaptive_protocol: bool,

    /// Spike-guided adaptive bias: closed-loop UV energy modulation driven by
    /// real-time neuromorphic spike activity. Voxel regions with high spike rates
    /// get boosted UV energy, creating a positive feedback loop that enhances
    /// conformational sampling in dynamically active regions. The spikes become
    /// the collective variable -- the protein tells the simulation where to push.
    #[arg(long, default_value = "false")]
    adaptive_bias: bool,

    /// Use learned Boltzmann thermodynamic ranking (trained on BENCH30).
    /// Replaces the hand-tuned Cobb-Douglas with ΔG = ΔH - TΔS + W + K + C
    /// using JAX-optimized thermodynamic constants.
    #[arg(long, default_value = "false")]
    boltzmann_rank: bool,

    /// Emit per-site spike_events.json files with full per-spike temporal data.
    /// Enabled by default — these files contain the raw neuromorphic signals
    /// (timestep, intensity, n_nearby_excited, water_density, spike_source)
    /// needed for spike-train ranking. Use --no-emit-spike-json to disable.
    #[arg(long, default_value = "true")]
    emit_spike_json: bool,

    /// Enable LADD (Local Atom Departure Detection) — 4th neuromorphic channel.
    #[arg(long, default_value = "false")]
    ladd: bool,

    /// Path to NMA modes JSON file for NMA-biased perturbation during warm_hold.
    /// Generated by: prism-prep --nma-modes 10
    /// Applies eigenvalue-scaled forces along normal mode directions to enhance
    /// cryptic pocket sampling. Forces are applied ONLY during warm_hold phase,
    /// ONLY to CA atoms, with spike-responsive mode scheduling.
    #[arg(long)]
    nma_perturb: Option<PathBuf>,

    /// NMA perturbation amplification factor. Each mode is pushed to
    /// amplification × thermal_amplitude. Higher = more aggressive.
    #[arg(long, default_value = "3.0")]
    nma_amplification: f32,

    /// Fraction of warm_hold spent scanning all modes (remainder focuses on best).
    #[arg(long, default_value = "0.3")]
    nma_scan_fraction: f32,

    /// Enable PRISM-TWIN: Coupled Observation MD with two simultaneous streams.
    /// Stream A (scout) runs standard thermal, Stream B (observer) runs thermal + NMA.
    /// Observation layers exchange spike density to cross-sensitize detectors.
    #[arg(long, default_value = "false")]
    coupled_twin: bool,

    /// Use persistent cooperative kernel for TWIN coupling instead of host-mediated.
    /// DISABLED: causes SM starvation on Blackwell SM120. Use --graph-coupling instead.
    #[arg(long, default_value = "false")]
    persistent_coupling: bool,

    /// Use CUDA Graph-based autonomous coupling (bleeding-edge Blackwell optimization).
    /// Captures the entire physics + coupling sequence as a conditional WHILE graph.
    /// One cudaGraphLaunch for the entire simulation. CPU harvests spike data to
    /// Parquet while GPU runs autonomously. Requires --coupled-twin.
    #[arg(long, default_value = "false")]
    graph_coupling: bool,

    /// Multi-Differential Interferometric TWIN: 4 groups × 2 engines each.
    /// Group A: Thermal Shock, Group B: Equilibrium Observer,
    /// Group C: UV Aromatic Probe, Group D: Hysteresis Probe.
    /// N-way cross-correlation reveals multi-mechanism binding site evidence.
    /// Requires ~8GB VRAM (8 engines). Overrides --coupled-twin.
    #[arg(long, default_value = "false")]
    multi_differential: bool,

    /// Enable ALL four stages of the hierarchical elimination cascade.
    /// Progressively filters detected sites through multi-channel convergence,
    /// temporal persistence, persistent homology, and Boltzmann gap gates.
    /// Reduces ~34 sites to ~5-8 high-confidence candidates.
    #[arg(long, default_value = "false")]
    cascade: bool,

    /// Stage 1: require spikes from both UV AND LIF channels.
    #[arg(long, default_value = "false")]
    cascade_multichannel: bool,

    /// Stage 2: require spike activity across ≥2 protocol phases.
    #[arg(long, default_value = "false")]
    cascade_temporal: bool,

    /// Stage 3: eliminate sites with below-median persistent homology persistence.
    #[arg(long, default_value = "false")]
    cascade_ph: bool,

    /// Stage 4: eliminate sites with Boltzmann probability <1% of rank-1.
    #[arg(long, default_value = "false")]
    cascade_boltzmann_gap: bool,

    /// REST2 solute tempering: scale intramolecular forces by λ per stream.
    /// With 8 streams, λ ladder = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3].
    /// Lower λ = softer potential = exponentially faster barrier crossing.
    /// λ=0.5 at 300K ≈ effective 600K on the potential energy surface.
    /// Consensus merging filters artifacts: sites must appear in physical (λ=1)
    /// AND softened streams to survive.
    #[arg(long, default_value = "false")]
    rest2: bool,

    /// Multi-temperature streams: spread end temperatures across streams.
    /// With 8 streams at --temperature 300, the ladder becomes:
    /// [300, 325, 350, 375, 400, 450, 500, 600] K.
    /// Higher-temperature streams crack high-barrier cryptic pockets
    /// (DFG flips, loop rearrangements) that equilibrium sampling misses.
    /// All streams share the same start_temp (cryo). Consensus merging
    /// naturally filters thermally-denatured artifacts because they won't
    /// appear consistently across the cooler streams.
    #[arg(long, default_value = "false")]
    multi_temp: bool,

    /// Enable multi-temperature stepped holds during the ramp phase.
    /// Instead of a linear ramp 50K→300K, pauses at intermediate temperatures
    /// (100K, 150K, 200K) to sample conformational basins where different
    /// pocket types open. Cryptic pockets crack at ~150K, allosteric sites
    /// appear at ~200K. Each hold runs for 3000 steps with full UV probing.
    /// Adds ~9K steps total but significantly improves cryptic pocket detection.
    #[arg(long, default_value = "false")]
    stepped_holds: bool,

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
    // ── PRISM_VALIDATED gate ─────────────────────────────────────────────
    // Direct invocation of nhs_rt_full is prohibited.
    // All runs MUST go through scripts/prism-validate-and-run.sh which sets
    // PRISM_VALIDATED=1 after preflight passes.
    if std::env::var("PRISM_VALIDATED").unwrap_or_default() != "1" {
        eprintln!("ERROR: Direct invocation of nhs_rt_full is prohibited.");
        eprintln!("Use: scripts/prism-validate-and-run.sh -t <topology> -o <output> [flags]");
        std::process::exit(2);
    }

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
    // PRISM-TWIN Multi-Differential: 4 groups × N engines via multi-stream
    if args.multi_differential {
        let n = args.multi_stream.max(8); // at least 8 engines for 4 groups × 2
        return run_multi_stream_pipeline(args, topology_path, n);
    }

    // PRISM-TWIN: coupled observation mode
    if args.coupled_twin {
        if args.multi_stream > 1 {
            // Phase B: multi-engine groups — 2 groups × (N/2) engines each
            if args.multi_stream % 2 != 0 {
                anyhow::bail!(
                    "--coupled-twin requires even --multi-stream value (got {}). \
                     Engines are split equally between Group A and Group B.",
                    args.multi_stream
                );
            }
            let streams_per_group = args.multi_stream / 2;
            log::info!("TWIN multi-engine mode: {} engines/group, {} total",
                streams_per_group, args.multi_stream);
            return run_coupled_twin_multi_pipeline(args, topology_path, streams_per_group);
        }
        // Phase A: single engine per group (2 total)
        return run_coupled_twin_pipeline(args, topology_path);
    }
    if args.multi_stream > 1 {
        return run_multi_stream_pipeline(args, topology_path, args.multi_stream);
    }
    run_single_structure_internal(topology_path, &args.output, args, args.replicas)?;
    Ok(())
}

/// Run Multi-Differential Interferometric TWIN (4 groups × 2 engines)
#[cfg(feature = "gpu")]
fn run_multi_differential_pipeline(args: &Args, topology_path: &PathBuf) -> Result<()> {
    use prism_nhs::coupled_md::{MultiDifferentialConfig, run_multi_differential_twin};

    log::info!("PRISM-TWIN Multi-Differential Interferometric mode activated");

    let topo_json = std::fs::read_to_string(topology_path)
        .with_context(|| format!("Failed to read topology: {}", topology_path.display()))?;
    let topology: prism_nhs::input::PrismPrepTopology = serde_json::from_str(&topo_json)
        .context("Failed to parse topology JSON")?;

    let context = cudarc::driver::CudaContext::new(0)
        .context("Failed to create CUDA context")?;
    let ptx_path = find_ptx_path()?;
    let module = context.load_module(cudarc::nvrtc::Ptx::from_file(&ptx_path))
        .with_context(|| format!("Failed to load PTX from {}", ptx_path))?;

    std::fs::create_dir_all(&args.output)?;

    let config = MultiDifferentialConfig::standard_4x1(args.replica_seed);

    let result = run_multi_differential_twin(
        &config, context, module, &topology, &args.output,
    )?;

    // Write result summary
    let result_path = args.output.join("multi_differential_result.json");
    std::fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;
    log::info!("Result saved: {}", result_path.display());

    Ok(())
}

/// Run PRISM-TWIN: Interferometric Coupled Observation MD
#[cfg(feature = "gpu")]
fn run_coupled_twin_pipeline(args: &Args, topology_path: &PathBuf) -> Result<()> {
    use prism_nhs::coupled_md::{CoupledTwinConfig, run_coupled_twin};
    use prism_nhs::persistent_engine::PersistentBatchConfig;

    log::info!("PRISM-TWIN mode activated");

    // Load topology
    let topo_json = std::fs::read_to_string(topology_path)
        .with_context(|| format!("Failed to read topology: {}", topology_path.display()))?;
    let topology: prism_nhs::input::PrismPrepTopology = serde_json::from_str(&topo_json)
        .context("Failed to parse topology JSON")?;

    // Build protocol (same logic as multi-stream path)
    let protocol = if args.fast {
        prism_nhs::fused_engine::CryoUvProtocol::fast_35k()
    } else {
        prism_nhs::fused_engine::CryoUvProtocol::standard()
    };

    let steps = protocol.total_steps();

    // Build twin config
    let mut twin_config = CoupledTwinConfig::default();
    twin_config.nma_modes_path = args.nma_perturb.as_ref().map(|p| p.to_string_lossy().to_string());
    twin_config.nma_amplification = args.nma_amplification;
    // Gate 1: exchange and CCF disabled
    twin_config.enable_exchange = true;   // Gate 2: spike density exchange
    twin_config.enable_ccf = true;        // Gate 2: cross-correlation
    twin_config.persistent_coupling = args.persistent_coupling;
    twin_config.graph_coupling = args.graph_coupling;

    // Initialize CUDA
    let context = cudarc::driver::CudaContext::new(0)
        .context("Failed to create CUDA context")?;

    // Load PTX module
    let ptx_path = find_ptx_path()?;
    let module = context.load_module(cudarc::nvrtc::Ptx::from_file(&ptx_path))
        .with_context(|| format!("Failed to load PTX from {}", ptx_path))?;

    let batch_config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(50000) as usize,
        ..Default::default()
    };

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Run PRISM-TWIN
    let result = run_coupled_twin(
        &batch_config,
        context,
        module,
        &topology,
        protocol,
        &twin_config,
        args.replica_seed,
        steps,
        args.hmr,
        args.fused_steps,
        args.adaptive_dt,
        args.ladd,
        &args.output,
    )?;

    // Save result
    let result_path = args.output.join("coupled_twin_result.json");
    let result_json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&result_path, &result_json)?;
    log::info!("PRISM-TWIN result saved: {}", result_path.display());

    Ok(())
}

/// Phase B: Multi-engine TWIN pipeline — 2 groups × N engines each.
///
/// Each group runs N independent AMBER trajectories. Between outer steps,
/// ring buffers exchange spike evidence between groups to adapt thresholds.
/// After simulation, per-group spikes are aggregated and processed through
/// the same CCF + feature pipeline as Phase A.
///
/// Threading model: all 2N engines run concurrently per outer step via
/// scoped threads. Ring buffer coupling is serial between steps.
#[cfg(feature = "gpu")]
fn run_coupled_twin_multi_pipeline(
    args: &Args,
    topology_path: &PathBuf,
    streams_per_group: usize,
) -> Result<()> {
    use prism_nhs::coupled_md::{CoupledTwinConfig, CoupledTwinResult, StreamResult,
                                 InterferometricFeatures, SiteInterferometricFeatures};
    use prism_nhs::persistent_engine::PersistentBatchConfig;
    use prism_nhs::twin_kernels::{TwinRingBuffer, find_twin_ptx};
    use cudarc::driver::CudaContext;
    use cudarc::nvrtc::Ptx;

    let n = streams_per_group;
    let total_engines = 2 * n;

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║   PRISM-TWIN MULTI-ENGINE ({} × 2 groups = {} total)       ║", n, total_engines);
    log::info!("╚══════════════════════════════════════════════════════════╝");

    // ── Shared GPU resources ──
    let context = CudaContext::new(0).context("CUDA context")?;
    let ptx_path = find_ptx_path()?;
    let fused_module = context.load_module(Ptx::from_file(&ptx_path))
        .with_context(|| format!("Failed to load PTX from {}", ptx_path))?;

    // Load topology
    let topo_json = std::fs::read_to_string(topology_path)
        .with_context(|| format!("Failed to read topology: {}", topology_path.display()))?;
    let mut topology: prism_nhs::input::PrismPrepTopology = serde_json::from_str(&topo_json)
        .context("Failed to parse topology JSON")?;
    if args.hmr { topology.apply_hmr(3.0); }

    let protocol = if args.fast {
        prism_nhs::fused_engine::CryoUvProtocol::fast_35k()
    } else {
        prism_nhs::fused_engine::CryoUvProtocol::standard()
    };
    let protocol = if args.hysteresis { protocol.with_hysteresis() } else { protocol };
    let steps = protocol.total_steps();

    let mut twin_config = CoupledTwinConfig::default();
    twin_config.nma_modes_path = args.nma_perturb.as_ref().map(|p| p.to_string_lossy().to_string());
    twin_config.nma_amplification = args.nma_amplification;
    twin_config.enable_exchange = true;
    twin_config.enable_ccf = true;
    twin_config.persistent_coupling = args.persistent_coupling;

    let offset_steps = (protocol.cold_hold_steps as f32 * twin_config.phase_offset_fraction) as i32;

    let batch_config = PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(50000) as usize,
        ..Default::default()
    };

    std::fs::create_dir_all(&args.output)?;

    // ── VRAM guard ──
    let (vram_free, vram_total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
    let estimated_per_engine_mb = 1500.0; // ~1.5 GB per engine for typical protein
    let estimated_total_mb = total_engines as f64 * estimated_per_engine_mb;
    let vram_free_mb = vram_free as f64 / (1024.0 * 1024.0);
    log::info!("  VRAM: {:.0} MB free / {:.0} MB total", vram_free_mb, vram_total as f64 / (1024.0 * 1024.0));
    log::info!("  Estimated VRAM for {} engines: {:.0} MB", total_engines, estimated_total_mb);
    if estimated_total_mb > vram_free_mb * 0.9 {
        log::warn!("  WARNING: estimated VRAM ({:.0} MB) may exceed available ({:.0} MB)",
            estimated_total_mb, vram_free_mb);
        log::warn!("  Consider reducing --multi-stream or using a smaller protein");
    }

    // ── Create 2N engines ──
    // Group A: engines[0..n), Group B: engines[n..2n)
    let mut engines: Vec<PersistentNhsEngine> = Vec::with_capacity(total_engines);
    let mut engine_seeds: Vec<u64> = Vec::with_capacity(total_engines);
    let mut engine_protocols: Vec<prism_nhs::fused_engine::CryoUvProtocol> = Vec::with_capacity(total_engines);

    for i in 0..total_engines {
        let is_group_b = i >= n;
        let seed = args.replica_seed + (i as u64) * 12345;
        let stream = context.new_stream().context("CUDA stream")?;

        let mut prot = protocol.clone();

        // Group B: phase offset
        if is_group_b {
            prot.cold_hold_steps += offset_steps;
        }

        let mut engine = PersistentNhsEngine::new_on_stream(
            &batch_config, context.clone(), fused_module.clone(), stream,
        )?;
        engine.load_topology(&topology)?;
        if args.hmr { engine.set_dt(0.004)?; }
        if args.fused_steps > 1 { engine.set_fused_inner_steps(args.fused_steps)?; }
        if args.adaptive_dt { engine.set_adaptive_dt(true)?; }
        if args.ladd { engine.set_ladd_enabled(true); }
        engine.set_cryo_uv_protocol(prot.clone())?;
        engine.set_spike_accumulation(true);

        // NMA for Group B only
        if is_group_b {
            if let Some(ref nma_path) = args.nma_perturb {
                engine.set_nma_amplification(args.nma_amplification);
                engine.load_nma_modes(nma_path.to_str().unwrap_or(""))?;
            }
        }

        log::info!("  Engine {}: {} seed={} cold_hold={}{}",
            i, if is_group_b { "Group B" } else { "Group A" },
            seed, prot.cold_hold_steps,
            if is_group_b && twin_config.nma_modes_path.is_some() { " +NMA" } else { "" });

        engines.push(engine);
        engine_seeds.push(seed);
        engine_protocols.push(prot);
    }

    // ── Ring buffers ──
    let ring_module = context.load_module(
        Ptx::from_file(&find_twin_ptx("ring_buffer.ptx")?)
    ).context("ring_buffer.ptx")?;
    let stream_exchange = context.new_stream().context("exchange stream")?;

    let ring_capacity = 8192 * (n as u32).max(1);
    let mut ring_a_to_b = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, ring_capacity)?;
    let mut ring_b_to_a = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, ring_capacity)?;
    ring_a_to_b.reset(&stream_exchange)?;
    ring_b_to_a.reset(&stream_exchange)?;
    log::info!("  Ring buffers: capacity={} per direction", ring_capacity);

    // ── Main simulation loop ──
    let inner = args.fused_steps.max(1) as i32;
    let outer_steps = (steps + inner - 1) / inner;
    let start = std::time::Instant::now();

    let mut prev_accum_lens = vec![0usize; total_engines];
    let mut ring_spikes_exchanged: u64 = 0;

    log::info!("  Running {} outer steps ({} fused)...", outer_steps, inner);

    for step in 0..outer_steps {
        // ══ PHASE 1: Cross-group threshold adaptation ══
        if twin_config.enable_exchange && step > 0 {
            if let Some((gx, gy, gz, ox, oy, oz, vs)) = engines[0].grid_info() {
                // B→A: adapt all Group A engines
                for i in 0..n {
                    if let Some((thresh, base)) = engines[i].threshold_buffers_mut() {
                        ring_b_to_a.read_and_adapt(
                            &stream_exchange, thresh, base,
                            (gx,gy,gz), (ox,oy,oz), vs,
                            twin_config.sensitivity_boost,
                            twin_config.max_threshold_reduction,
                            step as u32, 500.0,
                        )?;
                    }
                }
                // A→B: adapt all Group B engines
                for i in n..total_engines {
                    if let Some((thresh, base)) = engines[i].threshold_buffers_mut() {
                        ring_a_to_b.read_and_adapt(
                            &stream_exchange, thresh, base,
                            (gx,gy,gz), (ox,oy,oz), vs,
                            twin_config.sensitivity_boost,
                            twin_config.max_threshold_reduction,
                            step as u32, 500.0,
                        )?;
                    }
                }
                stream_exchange.synchronize()?;
            }
        }

        // ══ PHASE 2: Run all engines concurrently ══
        // Run all engines sequentially. Each engine launches kernels on its own
        // CUDA stream, so they execute asynchronously on the GPU. The CPU overhead
        // of sequential launch is negligible compared to GPU kernel execution time.
        // (True CPU-parallel threading is blocked by PersistentNhsEngine containing
        // OptiX raw pointers that are !Send. Sequential launch with async CUDA
        // streams achieves equivalent GPU throughput.)
        for i in 0..total_engines {
            match engines[i].run(inner) {
                Ok(_summary) => {},
                Err(e) => log::error!("  Engine {} failed at step {}: {}", i, step, e),
            }
        }

        // ══ PHASE 3: Push new spikes to ring buffers ══
        if twin_config.enable_exchange {
            for i in 0..total_engines {
                let curr_len = engines[i].accumulated_spike_count();
                if curr_len > prev_accum_lens[i] {
                    let accum = engines[i].get_accumulated_spikes();
                    let delta = &accum[prev_accum_lens[i]..];
                    let ring = if i < n { &mut ring_a_to_b } else { &mut ring_b_to_a };
                    ring.push_compacted(&stream_exchange, delta)?;
                    ring_spikes_exchanged += delta.len() as u64;
                    prev_accum_lens[i] = curr_len;
                }
            }
        }

        // ══ PHASE 4: Periodic recovery ══
        if twin_config.enable_exchange && step as u32 % 1000 == 0 && step > 0 {
            let n_voxels = engines[0].total_voxels() as u32;
            for i in 0..total_engines {
                if let Some((thresh, base)) = engines[i].threshold_buffers_mut() {
                    let ring = if i < n { &ring_b_to_a } else { &ring_a_to_b };
                    ring.threshold_recovery(&stream_exchange, thresh, base, n_voxels, 0.01)?;
                }
            }
        }

        // Progress
        if (step + 1) % 2000 == 0 || step == outer_steps - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let overflow_a = ring_a_to_b.overflow_count(&stream_exchange)?;
            let overflow_b = ring_b_to_a.overflow_count(&stream_exchange)?;
            log::info!("  Step {}/{}: exchanged={} overflow_a={} overflow_b={} ({:.0} steps/s)",
                step + 1, outer_steps, ring_spikes_exchanged, overflow_a, overflow_b,
                (step + 1) as f64 / elapsed);
        }
    }

    let wall_time = start.elapsed();

    // ── Collect per-group spikes + snapshots ──
    let mut group_a_spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent> = Vec::new();
    let mut group_b_spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent> = Vec::new();
    let mut group_a_snapshots: Vec<prism_nhs::fused_engine::EnsembleSnapshot> = Vec::new();
    let mut group_b_snapshots: Vec<prism_nhs::fused_engine::EnsembleSnapshot> = Vec::new();

    for i in 0..total_engines {
        let spikes = engines[i].get_accumulated_spikes();
        let snapshots = engines[i].get_snapshots();
        if i < n {
            group_a_spikes.extend(spikes);
            group_a_snapshots.extend(snapshots);
        } else {
            group_b_spikes.extend(spikes);
            group_b_snapshots.extend(snapshots);
        }
    }

    log::info!("  Group A: {} spikes, {} snapshots ({} engines)",
        group_a_spikes.len(), group_a_snapshots.len(), n);
    log::info!("  Group B: {} spikes, {} snapshots ({} engines)",
        group_b_spikes.len(), group_b_snapshots.len(), n);

    // ── Post-process with REAL aggregated N-engine data ──
    // Uses the shared twin_post_process function (extracted from Phase A).
    // This produces ALL output files: binding_sites, CCF matrix, per-residue
    // features (50 fields), per-site features, ensemble trajectory, coupled_spikes.
    use prism_nhs::coupled_md::{twin_post_process, TwinSimulationMetadata};

    let meta = TwinSimulationMetadata {
        seed_a: args.replica_seed,
        seed_b: args.replica_seed + 1000,
        steps,
        steps_b: steps + offset_steps,
        offset_steps,
        wall_time_secs: wall_time.as_secs_f64(),
        vram_used_gb: 0.0, // TODO: compute from nvidia-smi or mem_get_info delta
        n_exchanges: 0,    // multi-engine doesn't track CPU density exchanges
        total_density_a_to_b: 0.0,
        total_density_b_to_a: 0.0,
        max_nonzero_regions: 0,
        ring_spikes_exchanged,
    };

    let se_ref = Some(&stream_exchange);
    let result = twin_post_process(
        group_a_spikes,
        group_b_spikes,
        group_a_snapshots,
        group_b_snapshots,
        &topology,
        &protocol,
        &twin_config,
        &context,
        se_ref,
        &args.output,
        &meta,
    )?;

    // Save result JSON
    let result_path = args.output.join("coupled_twin_result.json");
    std::fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;
    log::info!("PRISM-TWIN multi-engine result saved: {}", result_path.display());

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║  TWIN MULTI-ENGINE COMPLETE                             ║");
    log::info!("║  Engines: {} ({} × 2 groups)                            ║", total_engines, n);
    log::info!("║  Spikes A: {:>10}  B: {:>10}                  ║",
        result.stream_a.total_spikes, result.stream_b.total_spikes);
    log::info!("║  Sites:  {:>4}  Residues: {:>4}                         ║",
        result.per_site_features.len(), result.per_residue_features.len());
    log::info!("║  Ring exchanged: {:>12}                         ║", ring_spikes_exchanged);
    log::info!("║  Wall time: {:.1}s                                     ║", wall_time.as_secs_f64());
    log::info!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Find the PTX file path (same logic as fused_engine)
#[cfg(feature = "gpu")]
fn find_ptx_path() -> Result<String> {
    let candidates = vec![
        "target/ptx/nhs_amber_fused.ptx".to_string(),
        "../../target/ptx/nhs_amber_fused.ptx".to_string(),
    ];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("../assets/ptx/nhs_amber_fused.ptx");
            if p.exists() { return Ok(p.display().to_string()); }
        }
    }
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return Ok(p.clone());
        }
    }
    anyhow::bail!("PTX file not found. Set PRISM4D_PTX_DIR or ensure target/ptx/ exists.")
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
    // Very large (>5000): 128³ (capped to match CUDA MAX_GRID_DIM)
    let adaptive_grid_dim = if topology.n_atoms < 500 {
        64
    } else if topology.n_atoms < 2000 {
        96
    } else {
        128
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

    // Apply spike-guided adaptive bias if requested
    if args.adaptive_bias {
        engine.set_adaptive_bias(true)?;
        // Adaptive bias requires spike accumulation to track activity
        engine.set_spike_accumulation(true);
    }

    // Enable LADD channel if requested
    if args.ladd {
        engine.set_ladd_enabled(true);
    }

    // Load NMA modes for perturbation if requested
    if let Some(ref nma_path) = args.nma_perturb {
        engine.set_nma_amplification(args.nma_amplification);
        engine.set_nma_scan_fraction(args.nma_scan_fraction);
        engine.load_nma_modes(nma_path.to_str().unwrap_or(""))?;
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
            stepped_holds: vec![],
        }
    };
    // Apply stepped holds if requested (multi-temperature sampling during ramp)
    let protocol = if args.stepped_holds {
        let holds = vec![
            (100.0, 3000),  // 100K: initial unfolding, surface crevices open
            (150.0, 3000),  // 150K: cryptic pockets begin to crack
            (200.0, 3000),  // 200K: allosteric sites, domain interfaces open
        ];
        log::info!("  Stepped holds: ENABLED ({} intermediate temperatures)", holds.len());
        for (temp, steps) in &holds {
            log::info!("    Hold at {:.0}K for {} steps", temp, steps);
        }
        CryoUvProtocol {
            stepped_holds: holds,
            ..protocol
        }
    } else {
        protocol
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

            // Adaptive protocol: split run into cold_hold + rest, adapt between
            let summary = if args.adaptive_protocol {
                let cold_steps = protocol.cold_hold_steps;
                log::info!("  [adaptive-protocol] Running cold_hold phase ({} steps)...", cold_steps);
                let cold_summary = engine.run(cold_steps)?;
                log::info!("  [adaptive-protocol] Cold hold complete: {} spikes in {} steps",
                    cold_summary.total_spikes, cold_steps);

                // Adapt engine parameters based on measured spike rate
                let _flexibility = engine.adapt_protocol_from_spike_rate(cold_steps);

                // Run remaining steps with adapted parameters
                let remaining = steps_per_replica - cold_steps;
                if remaining > 0 {
                    engine.run(remaining)?
                } else {
                    cold_summary
                }
            } else {
                engine.run(steps_per_replica)?
            };

            // Collect spikes from this replica
            let replica_spikes = engine.get_accumulated_spikes();
            let spike_count = replica_spikes.len();
            all_spikes.extend(replica_spikes);

            // Download signal preservation diagnostic (one-time per replica)
            if let Ok(_sig) = engine.download_signal_preservation() {
                // Summary logged inside download_signal_preservation()
            }

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
    // Apply PER-CHANNEL to prevent high-intensity LIF/UV from drowning out
    // lower-intensity EFP (electrostatic) spikes. EFP spikes are rare (~0.2%)
    // but carry critical electrostatic binding information.
    let pct = (args.spike_percentile.min(99) as f32) / 100.0;
    let accumulated_spikes = if all_spikes.len() > 1000 {
        // Partition by source channel
        let mut uv_spikes: Vec<_> = Vec::new();
        let mut lif_spikes: Vec<_> = Vec::new();
        let mut efp_spikes: Vec<_> = Vec::new();
        let mut ladd_spikes: Vec<_> = Vec::new();
        for s in all_spikes {
            match s.spike_source {
                1 => uv_spikes.push(s),
                3 => efp_spikes.push(s),
                4 | 5 => ladd_spikes.push(s), // LADD + COFIRE: preserve all (rare channel)
                _ => lif_spikes.push(s),
            }
        }

        // Apply percentile filter independently per channel
        let filter_channel = |mut spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent>| -> Vec<prism_nhs::fused_engine::GpuSpikeEvent> {
            if spikes.len() < 100 {
                return spikes; // keep all if too few
            }
            let mut intensities: Vec<f32> = spikes.iter().map(|s| s.intensity).collect();
            intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (intensities.len() as f32 * pct) as usize;
            let thresh = intensities.get(idx).copied().unwrap_or(0.0);
            spikes.into_iter().filter(|s| s.intensity >= thresh).collect()
        };

        let n_uv_pre = uv_spikes.len();
        let n_lif_pre = lif_spikes.len();
        let n_efp_pre = efp_spikes.len();
        let n_ladd_pre = ladd_spikes.len();

        let uv_filtered = filter_channel(uv_spikes);
        let lif_filtered = filter_channel(lif_spikes);
        let efp_filtered = filter_channel(efp_spikes);
        // LADD/COFIRE: keep all (rare, high-value channel — no percentile filtering)

        log::info!("  Intensity filter (per-channel, top {}%):", 100 - args.spike_percentile);
        log::info!("    UV:  {} → {}", n_uv_pre, uv_filtered.len());
        log::info!("    LIF: {} → {}", n_lif_pre, lif_filtered.len());
        log::info!("    EFP: {} → {} (preserved)", n_efp_pre, efp_filtered.len());
        log::info!("    LADD/COFIRE: {} (all preserved)", n_ladd_pre);

        let mut filtered = uv_filtered;
        filtered.extend(lif_filtered);
        filtered.extend(efp_filtered);
        filtered.extend(ladd_spikes);
        log::info!("    Total: {} spikes retained", filtered.len());
        filtered
    } else {
        all_spikes
    };

    // Write LADD/COFIRE spike summary (before clustering, so it survives any crash)
    {
        let ladd_count = accumulated_spikes.iter().filter(|s| s.spike_source == 4).count();
        let cofire_count = accumulated_spikes.iter().filter(|s| s.spike_source == 5).count();
        if ladd_count + cofire_count > 0 {
            std::fs::create_dir_all(&output_dir).ok();
            let ladd_json: Vec<serde_json::Value> = accumulated_spikes.iter()
                .filter(|s| s.spike_source == 4 || s.spike_source == 5)
                .map(|s| {
                    let pos = s.position;
                    let int = s.intensity;
                    let ts = s.timestep;
                    let src = s.spike_source;
                    let wd = s.water_density;
                    let ve = s.vibrational_energy;
                    let wdc = s.wd_change;
                    let n_res = s.n_residues as usize;
                    let residues: Vec<i32> = (0..n_res.min(8)).map(|r| s.nearby_residues[r]).collect();
                    serde_json::json!({
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "intensity": int, "timestep": ts,
                        "spike_source": src,
                        "water_density": wd,
                        "vibrational_energy": ve,
                        "wd_change": wdc,
                        "nearby_residues": residues,
                    })
                })
                .collect();
            let summary = serde_json::json!({
                "ladd_count": ladd_count,
                "cofire_count": cofire_count,
                "total_spikes": accumulated_spikes.len(),
                "ladd_fraction": ladd_count as f64 / accumulated_spikes.len() as f64,
                "cofire_fraction": cofire_count as f64 / accumulated_spikes.len() as f64,
                "spikes": ladd_json,
            });
            let ladd_path = output_dir.join(&structure_name).with_extension("ladd_spikes.json");
            if let Ok(f) = std::fs::File::create(&ladd_path) {
                let _ = serde_json::to_writer(f, &summary);
                log::info!("  LADD/COFIRE spikes: {} (LADD={}, COFIRE={})", ladd_path.display(), ladd_count, cofire_count);
            }
        }
    }

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
                    // Phase fraction fields — expose the cold/hot spike breakdown
                    let total_phase = therm_site.heating_spike_count + therm_site.cooling_spike_count;
                    if total_phase > 0 {
                        let hot_frac = therm_site.heating_spike_count as f64 / total_phase as f64;
                        let cold_frac = therm_site.cooling_spike_count as f64 / total_phase as f64;
                        site_json["cold_phase_fraction"] = serde_json::json!({
                            "cold": cold_frac,
                            "hot": hot_frac,
                            "delta": hot_frac - cold_frac,
                            "heating_spike_count": therm_site.heating_spike_count,
                            "cooling_spike_count": therm_site.cooling_spike_count,
                            "heating_spike_rate": therm_site.heating_spike_rate,
                            "cooling_spike_rate": therm_site.cooling_spike_rate,
                        });
                    }
                    // Override heuristic classification if PRISM-Therm says CRYPTIC
                    if therm_site.therm_class.to_string() == "CRYPTIC" {
                        site_json["classification"] = serde_json::Value::String("Cryptic".to_string());
                    }
                    // TIDE coupling score for single-structure path
                    if !therm_site.tide_decomposition.is_empty() {
                        if let Some(site_ref) = clustered_sites.iter().find(|s| s.cluster_id == site_id) {
                            let trigger_residues: std::collections::HashSet<i32> = therm_site.tide_decomposition.iter()
                                .take(5)
                                .map(|t| t.residue_id as i32)
                                .collect();
                            let lining_ids: std::collections::HashSet<i32> = site_ref.lining_residues.iter()
                                .map(|r| r.resid)
                                .collect();
                            let overlap = trigger_residues.intersection(&lining_ids).count();
                            let tide_coupling = overlap as f32 / trigger_residues.len().max(1) as f32;
                            site_json["tide_coupling_score"] = serde_json::json!(tide_coupling);
                            let trigger_ids: Vec<u32> = therm_site.tide_decomposition.iter()
                                .take(5)
                                .map(|t| t.residue_id)
                                .collect();
                            site_json["tide_trigger_residues"] = serde_json::json!(trigger_ids);
                        }
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
            stepped_holds: vec![],
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
    } else {
        128
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
    use prism_nhs::SiteClassification;

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
    } else {
        128
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
            stepped_holds: vec![],
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

    // ══════════════════════════════════════════════════════════════════════
    // ASC FUSION CONTROLLER — EVENT-DRIVEN INTERFEROMETRIC STEERING
    //
    // The ASC fires on PHYSICS EVENTS, not periodic timers:
    //   Event 1: NEW RESIDUE ONSET — a residue appears in 2+ groups for the first time
    //   Event 2: BURST DETECTION — spike rate exceeds 2σ above running mean
    //   Event 3: INTERFEROMETRIC MATCH — same residue in scout + observer simultaneously
    //   Event 4: PHASE TRANSITION — an engine crosses a protocol phase boundary
    //
    // Each event triggers specific steering actions appropriate to the physics.
    // ══════════════════════════════════════════════════════════════════════
    struct AscSharedState {
        barrier: std::sync::Barrier,
        /// Per-engine spike count (atomically updated each chunk)
        spike_counts: Vec<std::sync::atomic::AtomicU64>,
        /// Per-engine spike DELTA this chunk (for burst detection)
        spike_deltas: Vec<std::sync::atomic::AtomicU64>,
        /// Per-group × per-residue cumulative spike counts [group][residue]
        group_residue_counts: std::sync::Mutex<Vec<Vec<u32>>>,
        /// Known residues: set of residues already seen in 2+ groups (for onset detection)
        known_multi_group_residues: std::sync::Mutex<std::collections::HashSet<i32>>,
        /// ASC output: consensus residues with phase coherence + surprise
        consensus_residues: std::sync::Mutex<Vec<(i32, usize, f32)>>, // (rid, n_groups, S_pc)
        /// Event log: timestamped ASC events for telemetry
        event_log: std::sync::Mutex<Vec<(u32, String)>>,
        /// ACL contrast log
        acl_contrast_log: std::sync::Mutex<Vec<(u32, f32)>>,
        /// Running spike rate statistics per engine: (sum, sum_sq, count) for mean/σ
        rate_stats: Vec<(std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU32)>,
        n_residues: usize,
        // ── PCMI v3.0: Phase Coherence + Bayesian Surprise ──
        /// Per-group × per-residue phase accumulators: (cos_sum, sin_sum) for S_pc
        /// Phase angle φ = 2π × (spike_timestep / group_total_steps)
        /// S_pc = |Σ exp(iφ)| / N = sqrt(cos_sum² + sin_sum²) / N
        group_residue_phase: std::sync::Mutex<Vec<Vec<(f64, f64)>>>, // [group][residue] = (cos_sum, sin_sum)
        /// Bayesian prior: running per-residue spike density baseline (first 20 chunks)
        prior_residue_density: std::sync::Mutex<Vec<f64>>, // [residue] = avg spikes/chunk during baseline
        /// Baseline collection complete flag
        baseline_ready: std::sync::atomic::AtomicBool,
        /// Per-group total CCNS steps (for phase angle computation)
        group_total_steps: Vec<i32>,
    }
    let is_multi_diff = args.multi_differential;
    let n_residues_est = topology.n_residues;
    let diff_total_steps: Vec<i32> = if is_multi_diff {
        CryoUvProtocol::twin_differential_set().iter().map(|p| p.total_steps()).collect()
    } else {
        vec![35000; 4]
    };
    let asc_shared: Option<std::sync::Arc<AscSharedState>> = if is_multi_diff && n_streams >= 4 {
        Some(std::sync::Arc::new(AscSharedState {
            barrier: std::sync::Barrier::new(n_streams),
            spike_counts: (0..n_streams).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            spike_deltas: (0..n_streams).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            group_residue_counts: std::sync::Mutex::new(vec![vec![0u32; n_residues_est + 1]; 4]),
            known_multi_group_residues: std::sync::Mutex::new(std::collections::HashSet::new()),
            consensus_residues: std::sync::Mutex::new(Vec::new()),
            event_log: std::sync::Mutex::new(Vec::new()),
            acl_contrast_log: std::sync::Mutex::new(Vec::new()),
            rate_stats: (0..n_streams).map(|_| (
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU32::new(0),
            )).collect(),
            n_residues: n_residues_est,
            // PCMI v3.0
            group_residue_phase: std::sync::Mutex::new(vec![vec![(0.0f64, 0.0f64); n_residues_est + 1]; 4]),
            prior_residue_density: std::sync::Mutex::new(vec![0.0f64; n_residues_est + 1]),
            baseline_ready: std::sync::atomic::AtomicBool::new(false),
            group_total_steps: diff_total_steps,
        }))
    } else {
        None
    };
    let epg_val = if is_multi_diff { n_streams / 4 } else { 1 };

    // ── Run N engines on N threads (scoped for safe borrowing) ──
    log::info!("\n  🚀 Launching {} independent trajectories...", n_streams);
    let sim_start = Instant::now();

    let stream_results: Vec<Result<(Vec<prism_nhs::fused_engine::GpuSpikeEvent>, Vec<prism_nhs::fused_engine::EnsembleSnapshot>, Option<prism_nhs::fused_engine::SignalPreservationData>, Option<prism_nhs::fused_engine::KccData>)>> =
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_streams).map(|i| {
                let ctx = context.clone();
                let mod_ = module.clone();
                let stream_i = streams[i].clone();
                let topo_ref = &topology;
                let config_ref = &config;
                // Multi-temperature ladder: spread end_temp across streams
                // Stream 0 gets the base temperature; higher streams get progressively
                // hotter to crack high-barrier pockets. The ramp_down phase then cools
                // from the higher temperature, producing larger hysteresis asymmetry
                // for pockets that require elevated temperatures to open — exactly the
                // high-barrier cryptic sites that equilibrium sampling misses.
                // Multi-Differential Interferometric TWIN: 4 groups × 2 engines
                // Each group gets a physically distinct CryoUvProtocol.
                let prot = if args.multi_differential && n_streams >= 4 {
                    let diff_set = CryoUvProtocol::twin_differential_set();
                    let group_idx = i / 2; // 0-1→ThermalShock, 2-3→Equilibrium, 4-5→UvAromatic, 6-7→Hysteresis
                    let p = diff_set[group_idx % diff_set.len()].clone();
                    let group_names = ["ThermalShock", "Equilibrium", "UvAromatic", "Hysteresis"];
                    log::info!("    [stream {}] MULTI-DIFF group {} [{}]: {}K→{}K, UV={:.0}kcal, interval={}",
                        i, group_idx, group_names[group_idx % 4],
                        p.start_temp, p.end_temp, p.uv_burst_energy, p.uv_burst_interval);
                    p
                } else if args.multi_temp && n_streams >= 4 {
                    let base_temp = protocol.end_temp;
                    let mut p = protocol.clone();

                    // Hybrid multi-temperature protocol:
                    // Mix of equilibrium streams (standard ramp) and flash-freeze
                    // streams (heat high → quench rapidly to trap open conformations).
                    //
                    // 8-stream layout:
                    //   0: 300K baseline (reference)
                    //   1: 350K mild heat
                    //   2: 400K → QUENCH (flash-freeze from 400K)
                    //   3: 450K → QUENCH (flash-freeze from 450K)
                    //   4: 500K equilibrium
                    //   5: 500K → QUENCH (flash-freeze from 500K)
                    //   6: 600K → QUENCH (flash-freeze from 600K)
                    //   7: 300K baseline (duplicate for consensus)
                    //
                    // Flash-freeze: ramp_down in 200 steps (~0.4ps) instead of 6000+
                    // This traps the backbone in the open conformation while the pocket
                    // is still accessible, like computational cryo-EM vitrification.
                    // The cold_return hold then probes the trapped state with UV/LIF/EFP.

                    match i % n_streams {
                        0 => {
                            // Baseline reference
                            p.end_temp = base_temp;
                            log::info!("    [stream {}] baseline {:.0}K", i, p.end_temp);
                        }
                        1 => {
                            // Mild heat, standard ramp-down
                            p.end_temp = base_temp * 1.17; // ~350K
                            log::info!("    [stream {}] mild heat {:.0}K", i, p.end_temp);
                        }
                        2 => {
                            // Flash-freeze from 400K + intensified probing
                            p.end_temp = base_temp * 1.33; // ~400K
                            p.ramp_down_steps = 200; // rapid quench
                            p.cold_return_steps = p.cold_return_steps.max(5000); // extended probing
                            p.uv_burst_energy *= 2.0; // 2× UV energy: aggressive pocket interrogation
                            p.uv_burst_interval = (p.uv_burst_interval / 2).max(50); // 2× burst frequency
                            log::info!("    [stream {}] FLASH-FREEZE from {:.0}K (quench 200, UV 2x)", i, p.end_temp);
                        }
                        3 => {
                            // Flash-freeze from 450K + intensified probing
                            p.end_temp = base_temp * 1.5; // ~450K
                            p.ramp_down_steps = 200;
                            p.cold_return_steps = p.cold_return_steps.max(5000);
                            p.uv_burst_energy *= 2.0;
                            p.uv_burst_interval = (p.uv_burst_interval / 2).max(50);
                            log::info!("    [stream {}] FLASH-FREEZE from {:.0}K (quench 200, UV 2x)", i, p.end_temp);
                        }
                        4 => {
                            // Hot equilibrium (standard ramp-down)
                            p.end_temp = base_temp * 1.67; // ~500K
                            log::info!("    [stream {}] hot equilibrium {:.0}K", i, p.end_temp);
                        }
                        5 => {
                            // Flash-freeze from 500K + intensified probing
                            p.end_temp = base_temp * 1.67; // ~500K
                            p.ramp_down_steps = 200;
                            p.cold_return_steps = p.cold_return_steps.max(5000);
                            p.uv_burst_energy *= 2.5; // 2.5× UV: maximum probe intensity
                            p.uv_burst_interval = (p.uv_burst_interval / 3).max(50); // 3× frequency
                            log::info!("    [stream {}] FLASH-FREEZE from {:.0}K (quench 200, UV 2.5x)", i, p.end_temp);
                        }
                        6 => {
                            // Flash-freeze from 600K + intensified probing
                            p.end_temp = base_temp * 2.0; // ~600K
                            p.ramp_down_steps = 200;
                            p.cold_return_steps = p.cold_return_steps.max(5000);
                            p.uv_burst_energy *= 3.0; // 3× UV: maximum energy for deep pockets
                            p.uv_burst_interval = (p.uv_burst_interval / 3).max(50); // 3× frequency
                            log::info!("    [stream {}] FLASH-FREEZE from {:.0}K (quench 200, UV 3x)", i, p.end_temp);
                        }
                        _ => {
                            // Additional streams: duplicate baseline for consensus
                            p.end_temp = base_temp;
                            log::info!("    [stream {}] baseline {:.0}K (consensus duplicate)", i, p.end_temp);
                        }
                    }
                    p
                } else {
                    protocol.clone()
                };
                // REST2 solute tempering: λ ladder across streams
                // Linear ladder from 1.0 (physical) to 0.3 (very soft)
                let rest2_lambda = if args.rest2 && n_streams >= 4 {
                    let lambda = 1.0 - 0.7 * (i as f32 / (n_streams - 1).max(1) as f32);
                    let lambda = lambda.clamp(0.3, 1.0);
                    log::info!("    [stream {}] REST2: λ={:.2} (effective T_PES={:.0}K)",
                        i, lambda, 300.0 / lambda);
                    lambda
                } else {
                    1.0
                };

                let seed = args.replica_seed + i as u64 * 12345;
                let ultimate = args.ultimate_mode;
                // Multi-differential: each group's protocol determines its step count
                let steps = if args.multi_differential { prot.total_steps() } else { steps_per_stream };
                let asc_shared = asc_shared.clone();
                let is_multi_diff = is_multi_diff;
                let hmr_enabled = args.hmr;
                let fused_steps = args.fused_steps;
                let adaptive_dt = args.adaptive_dt;
                let adaptive_bias = args.adaptive_bias;
                let adaptive_protocol = args.adaptive_protocol;
                let ladd_enabled = args.ladd;
                let nma_path = args.nma_perturb.clone();
                let nma_amp = args.nma_amplification;
                let nma_frac = args.nma_scan_fraction;
                let cold_hold_steps = prot.cold_hold_steps;

                s.spawn(move || -> Result<(Vec<prism_nhs::fused_engine::GpuSpikeEvent>, Vec<prism_nhs::fused_engine::EnsembleSnapshot>, Option<prism_nhs::fused_engine::SignalPreservationData>, Option<prism_nhs::fused_engine::KccData>)> {
                    log::info!("    [stream {}] Starting (seed: {})...", i, seed);

                    let mut engine = PersistentNhsEngine::new_on_stream(
                        config_ref, ctx, mod_, stream_i,
                    )?;
                    engine.load_topology(topo_ref)?;
                    if rest2_lambda < 1.0 {
                        engine.set_solute_lambda(rest2_lambda);
                    }
                    if hmr_enabled {
                        engine.set_dt(0.004)?;  // 4fs with HMR masses
                    }
                    if fused_steps > 1 {
                        engine.set_fused_inner_steps(fused_steps)?;
                    }
                    if adaptive_dt {
                        engine.set_adaptive_dt(true)?;
                    }
                    if adaptive_bias {
                        engine.set_adaptive_bias(true)?;
                    }
                    if ladd_enabled {
                        engine.set_ladd_enabled(true);
                    }
                    if let Some(ref nma_p) = nma_path {
                        engine.set_nma_amplification(nma_amp);
                        engine.set_nma_scan_fraction(nma_frac);
                        engine.load_nma_modes(nma_p.to_str().unwrap_or(""))?;
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

                    // ── ASC Fusion Controller: chunk-based coupled loop ──
                    // Multi-differential: run in chunks with barrier sync for cross-group coupling.
                    // After each chunk, download spike centroids, identify spatial consensus,
                    // and write steering decisions back to ProtocolState.
                    let summary = if is_multi_diff {
                        let chunk_size = 500i32;
                        // All engines must iterate the SAME number of chunks (barrier sync).
                        // Max protocol: Equilibrium = 40000 steps → 80 chunks.
                        let max_steps_any_group = 40000i32; // max across all differential protocols
                        let n_chunks = (max_steps_any_group + chunk_size - 1) / chunk_size;
                        let mut last_summary = None;
                        let mut steps_run = 0i32;

                        for chunk_idx in 0..n_chunks {
                            // Only run physics if this engine hasn't finished its protocol
                            if steps_run < steps {
                                let this_chunk = chunk_size.min(steps - steps_run);
                                let s = engine.run(this_chunk)?;
                                // Force-sync spikes from GPU — chunk_size < sync_interval so
                                // spikes wouldn't be downloaded otherwise
                                let _ = engine.force_spike_sync();
                                last_summary = Some(s);
                                steps_run += this_chunk;
                            }

                            // ══════════════════════════════════════════════════════════
                            // ASC EVENT-DRIVEN INTERFEROMETRIC CONTROLLER
                            //
                            // Fires on PHYSICS EVENTS, not periodic timers:
                            //   Event 1: BURST — spike delta > 2σ above running mean
                            //   Event 2: ONSET — new residue appears in 2+ groups
                            //   Event 3: MATCH — scout + observer agree on residue
                            // ══════════════════════════════════════════════════════════
                            if let Some(ref asc_state) = asc_shared {
                                let group_idx = i / epg_val;

                                // ── Per-engine spike delta (burst detection) ──
                                let current_total = engine.get_accumulated_spikes().len() as u64;
                                let prev_total = asc_state.spike_counts[i].load(std::sync::atomic::Ordering::Relaxed);
                                let delta = current_total.saturating_sub(prev_total);
                                asc_state.spike_counts[i].store(current_total, std::sync::atomic::Ordering::Relaxed);
                                asc_state.spike_deltas[i].store(delta, std::sync::atomic::Ordering::Relaxed);

                                // Update running rate statistics for burst detection
                                let (ref sum_atom, ref sum_sq_atom, ref count_atom) = asc_state.rate_stats[i];
                                sum_atom.fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
                                sum_sq_atom.fetch_add(delta * delta, std::sync::atomic::Ordering::Relaxed);
                                count_atom.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                // Detect burst: delta > mean + 2σ
                                let n_samples = count_atom.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
                                let mean_rate = sum_atom.load(std::sync::atomic::Ordering::Relaxed) as f64 / n_samples;
                                let mean_sq = sum_sq_atom.load(std::sync::atomic::Ordering::Relaxed) as f64 / n_samples;
                                let variance = (mean_sq - mean_rate * mean_rate).max(0.0);
                                let sigma = variance.sqrt();
                                let is_burst = n_samples > 5.0 && delta as f64 > mean_rate + 2.0 * sigma;

                                // ── Residue-level analysis + phase angle accumulation ──
                                // Use the spike DELTA (recent spikes only) for freshness
                                if delta > 0 {
                                    let spikes = engine.get_accumulated_spikes();
                                    let recent_start = spikes.len().saturating_sub(delta as usize);
                                    let total_steps_g = asc_state.group_total_steps[group_idx] as f64;
                                    let two_pi = 2.0 * std::f64::consts::PI;

                                    if let (Ok(mut grc), Ok(mut gph)) = (
                                        asc_state.group_residue_counts.lock(),
                                        asc_state.group_residue_phase.lock(),
                                    ) {
                                        for spike in &spikes[recent_start..] {
                                            // Phase angle: where in the CCNS cycle did this spike fire?
                                            let phi = two_pi * (spike.timestep as f64 / total_steps_g);
                                            let cos_phi = phi.cos();
                                            let sin_phi = phi.sin();

                                            for j in 0..(spike.n_residues as usize).min(8) {
                                                let rid = spike.nearby_residues[j];
                                                if rid >= 0 && (rid as usize) < asc_state.n_residues {
                                                    let r = rid as usize;
                                                    grc[group_idx][r] += 1;
                                                    // Accumulate phasor components for S_pc
                                                    gph[group_idx][r].0 += cos_phi;
                                                    gph[group_idx][r].1 += sin_phi;
                                                }
                                            }
                                        }
                                    }
                                }

                                // Barrier: all engines have written their residue data
                                asc_state.barrier.wait();

                                // ══════════════════════════════════════════════════════
                                // ASC v3.0 CONTROLLER (thread 0): PCMI Gated Fusion
                                //
                                // Phase Coherence S_pc: |Σ exp(iφ)| / N across groups
                                // Bayesian Surprise: KL(P_current || P_baseline)
                                // Gate: (S_pc > 0.85) AND (surprise > 3σ)
                                // ══════════════════════════════════════════════════════
                                if i == 0 {
                                    let any_burst = (0..n_streams).any(|s| {
                                        asc_state.spike_deltas[s].load(std::sync::atomic::Ordering::Relaxed) as f64
                                            > mean_rate + 2.0 * sigma && n_samples > 5.0
                                    });

                                    if let (Ok(grc), Ok(gph)) = (
                                        asc_state.group_residue_counts.lock(),
                                        asc_state.group_residue_phase.lock(),
                                    ) {
                                        let mut hotspots: Vec<(i32, usize, f32)> = Vec::new(); // (rid, n_groups, S_pc)
                                        let scout_groups = [0usize, 2]; // TS, UV
                                        let observer_groups = [1usize, 3]; // EQ, HY

                                        for rid in 0..asc_state.n_residues {
                                            let counts: Vec<u32> = (0..4).map(|g| grc[g][rid]).collect();
                                            let active_groups = counts.iter().filter(|&&c| c > 5).count();
                                            if active_groups < 2 { continue; }

                                            // ── Phase Coherence S_pc ──
                                            // Sum phasors across ALL groups for this residue
                                            let mut cos_total = 0.0f64;
                                            let mut sin_total = 0.0f64;
                                            let mut n_total = 0u64;
                                            for g in 0..4 {
                                                cos_total += gph[g][rid].0;
                                                sin_total += gph[g][rid].1;
                                                n_total += counts[g] as u64;
                                            }
                                            let s_pc = if n_total > 10 {
                                                (cos_total.powi(2) + sin_total.powi(2)).sqrt() / n_total as f64
                                            } else {
                                                0.0
                                            };

                                            if s_pc > 0.01 { // minimum coherence to even consider
                                                hotspots.push((rid as i32, active_groups, s_pc as f32));
                                            }
                                        }
                                        hotspots.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                                        // ── Bayesian Surprise: build baseline for first 20 chunks, then compare ──
                                        let baseline_window = 20u32;
                                        if chunk_idx as u32 == baseline_window {
                                            // Lock in the baseline from accumulated residue counts
                                            if let Ok(mut prior) = asc_state.prior_residue_density.lock() {
                                                let chunk_f = baseline_window as f64;
                                                for rid in 0..asc_state.n_residues {
                                                    let total: u32 = (0..4).map(|g| grc[g][rid]).sum();
                                                    prior[rid] = total as f64 / chunk_f;
                                                }
                                            }
                                            asc_state.baseline_ready.store(true, std::sync::atomic::Ordering::Relaxed);
                                            log::info!("    [ASC] Bayesian baseline locked at chunk {}", chunk_idx);
                                        }

                                        // Compute KL-divergence surprise for top hotspots
                                        let is_baseline_ready = asc_state.baseline_ready.load(std::sync::atomic::Ordering::Relaxed);
                                        let mut surprised_residues: Vec<(i32, f64)> = Vec::new();
                                        if is_baseline_ready {
                                            if let Ok(prior) = asc_state.prior_residue_density.lock() {
                                                for &(rid, _, s_pc) in hotspots.iter().take(30) {
                                                    let r = rid as usize;
                                                    if r >= asc_state.n_residues { continue; }
                                                    let expected = prior[r].max(0.1); // avoid div-by-zero
                                                    let observed: u32 = (0..4).map(|g| grc[g][r]).sum();
                                                    let obs_rate = observed as f64 / (chunk_idx as f64).max(1.0);
                                                    // Simplified KL: obs * ln(obs/expected) — scalar surprise
                                                    let kl = if obs_rate > 0.01 {
                                                        obs_rate * (obs_rate / expected).ln()
                                                    } else {
                                                        0.0
                                                    };
                                                    if kl > 0.0 && s_pc > 0.1 {
                                                        surprised_residues.push((rid, kl));
                                                    }
                                                }
                                            }
                                        }

                                        // Event 2: NEW ONSET
                                        let mut new_onsets: Vec<i32> = Vec::new();
                                        if let Ok(mut known) = asc_state.known_multi_group_residues.lock() {
                                            for &(rid, ng, _) in &hotspots {
                                                if ng >= 2 && !known.contains(&rid) {
                                                    known.insert(rid);
                                                    new_onsets.push(rid);
                                                }
                                            }
                                        }

                                        // Event 3: INTERFEROMETRIC MATCH with phase coherence
                                        let matches: Vec<(i32, f32)> = hotspots.iter()
                                            .filter(|&&(rid, _, s_pc)| {
                                                let r = rid as usize;
                                                if r >= asc_state.n_residues || s_pc < 0.85 { return false; }
                                                let scout_active = scout_groups.iter().any(|&g| grc[g][r] > 5);
                                                let observer_active = observer_groups.iter().any(|&g| grc[g][r] > 5);
                                                scout_active && observer_active
                                            })
                                            .map(|&(rid, _, s_pc)| (rid, s_pc))
                                            .collect();

                                        // Store consensus (now with S_pc instead of contrast)
                                        if let Ok(mut cr) = asc_state.consensus_residues.lock() {
                                            *cr = hotspots.iter().take(10).cloned().collect();
                                        }

                                        // Log events
                                        let has_event = any_burst || !new_onsets.is_empty() || !matches.is_empty();
                                        if has_event || chunk_idx % 40 == 0 {
                                            let mut events = Vec::new();
                                            if any_burst { events.push("BURST".to_string()); }
                                            if !new_onsets.is_empty() {
                                                events.push(format!("ONSET({}res)", new_onsets.len()));
                                            }
                                            if !matches.is_empty() {
                                                let match_str: String = matches.iter().take(3)
                                                    .map(|(r, spc)| format!("r{}(Spc={:.2})", r, spc))
                                                    .collect::<Vec<_>>().join(",");
                                                events.push(format!("PCMI[{}]", match_str));
                                            }
                                            if !surprised_residues.is_empty() {
                                                let s_str: String = surprised_residues.iter().take(3)
                                                    .map(|(r, kl)| format!("r{}(KL={:.2})", r, kl))
                                                    .collect::<Vec<_>>().join(",");
                                                events.push(format!("SURP[{}]", s_str));
                                            }
                                            let event_str = if events.is_empty() { "monitor".into() } else { events.join("+") };
                                            let top_str = hotspots.iter().take(3).map(|(rid, ng, spc)| {
                                                format!("res{}({}/4,Spc={:.3})", rid, ng, spc)
                                            }).collect::<Vec<_>>().join(" ");
                                            log::info!("    [ASC] chunk {}: {} [{}] top=[{}]",
                                                chunk_idx, event_str, hotspots.len(), top_str);

                                            if let Ok(mut el) = asc_state.event_log.lock() {
                                                el.push((chunk_idx as u32, event_str));
                                            }
                                        }

                                        // ACL telemetry
                                        let scout_total: u64 = scout_groups.iter().map(|&g| {
                                            (0..epg_val).map(|e| asc_state.spike_counts[g * epg_val + e]
                                                .load(std::sync::atomic::Ordering::Relaxed)).sum::<u64>()
                                        }).sum();
                                        let observer_total: u64 = observer_groups.iter().map(|&g| {
                                            (0..epg_val).map(|e| asc_state.spike_counts[g * epg_val + e]
                                                .load(std::sync::atomic::Ordering::Relaxed)).sum::<u64>()
                                        }).sum();
                                        let so_ratio = if observer_total > 0 { scout_total as f32 / observer_total as f32 } else { 1.0 };
                                        if let Ok(mut log) = asc_state.acl_contrast_log.lock() {
                                            log.push((chunk_idx as u32, so_ratio));
                                        }
                                    }
                                }

                                // Second barrier: ASC decisions ready
                                asc_state.barrier.wait();

                                // ══════════════════════════════════════════════════════
                                // OMNIDIRECTIONAL PCMI-GATED STEERING
                                //
                                // Gate: (S_pc > 0.85) AND (≥3 groups active)
                                // Action: Information handshake — boost engines that
                                //         HAVEN'T seen the phase-coherent signal
                                // ══════════════════════════════════════════════════════
                                if let Ok(cr) = asc_state.consensus_residues.lock() {
                                    if let Some(&(top_residue, n_groups, s_pc)) = cr.first() {
                                        // PCMI gate: only steer when phase-coherent AND multi-group
                                        if n_groups >= 3 && s_pc >= 0.85 {
                                            // Information handshake: only boost engines missing the signal
                                            let my_group_has_it = if let Ok(grc) = asc_state.group_residue_counts.lock() {
                                                let rid = top_residue as usize;
                                                rid < asc_state.n_residues && grc[group_idx][rid] > 20
                                            } else { false };

                                            if !my_group_has_it {
                                                engine.set_steering_focus_residue(top_residue);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        last_summary.unwrap_or_else(|| engine.run(0).unwrap())
                    } else if adaptive_protocol || rest2_lambda < 1.0 {
                        // Split run: cold_hold first, then apply focused REST2, then remaining steps.
                        log::info!("    [stream {}] Running cold_hold ({} steps)...", i, cold_hold_steps);
                        let cold_summary = engine.run(cold_hold_steps)?;
                        log::info!("    [stream {}] Cold hold: {} spikes", i, cold_summary.total_spikes);

                        if adaptive_protocol {
                            let _flexibility = engine.adapt_protocol_from_spike_rate(cold_hold_steps);
                        }

                        if rest2_lambda < 1.0 {
                            match engine.apply_focused_lambda() {
                                Ok(()) => {},
                                Err(e) => log::warn!("    [stream {}] Focused REST2 failed: {}", i, e),
                            }
                        }

                        let remaining = steps - cold_hold_steps;
                        if remaining > 0 {
                            engine.run(remaining)?
                        } else {
                            cold_summary
                        }
                    } else {
                        engine.run(steps)?
                    };

                    let spikes = engine.get_accumulated_spikes();
                    log::info!("    [stream {}] Accumulated spikes: {} (steps_run={})",
                        i, spikes.len(), if is_multi_diff { steps } else { steps_per_stream });
                    let snapshots = engine.get_snapshots();

                    // Download signal preservation grids from this stream's GPU buffers
                    let sig_data = engine.download_signal_preservation().ok();
                    // Compute and download KCC v2-full descriptors
                    let kcc_data = engine.compute_and_download_kcc().ok();

                    log::info!("    [stream {}] Complete: {} spikes, {} snapshots, T={:.1}K",
                        i, spikes.len(), snapshots.len(), summary.end_temperature);
                    Ok((spikes, snapshots, sig_data, kcc_data))
                })
            }).collect();

            handles.into_iter()
                .map(|h| h.join().expect("stream thread panicked"))
                .collect()
        });

    let sim_elapsed = sim_start.elapsed();
    log::info!("  ✓ All {} streams complete in {:.1}s", n_streams, sim_elapsed.as_secs_f64());

    // ── Extract ASC consensus for downstream spike filtering ──
    let asc_consensus_residues: std::collections::HashSet<i32> = if let Some(ref asc) = asc_shared {
        let residues: std::collections::HashSet<i32> = asc.consensus_residues.lock()
            .map(|cr| cr.iter()
                .filter(|&&(_, ng, c)| ng >= 3 && c > 0.3) // 3+ groups, contrast > 0.3
                .map(|&(rid, _, _)| rid)
                .collect())
            .unwrap_or_default();
        if !residues.is_empty() {
            log::info!("  [ASC] {} consensus residues for downstream filtering: {:?}",
                residues.len(), residues.iter().take(20).collect::<Vec<_>>());
        }
        // Write ACL telemetry to output
        if let Ok(acl_log) = asc.acl_contrast_log.lock() {
            if !acl_log.is_empty() {
                log::info!("  [ACL] {} contrast samples, mean S/O={:.3}",
                    acl_log.len(),
                    acl_log.iter().map(|(_, r)| *r).sum::<f32>() / acl_log.len() as f32);
            }
        }
        residues
    } else {
        std::collections::HashSet::new()
    };

    // ── Aggregate: per-stream filtering + clustering → consensus ──
    log::info!("\n  Aggregating results across {} streams...", n_streams);

    let cluster_stream = context.new_stream().context("CUDA stream for consensus")?;
    let mut cluster_engine = PersistentNhsEngine::new_on_stream(&config, context.clone(), module.clone(), cluster_stream)?;
    cluster_engine.load_topology(&topology)?;

    let mut per_stream_sites: Vec<Vec<ClusteredBindingSite>> = Vec::new();
    let mut per_stream_stats: Vec<serde_json::Value> = Vec::new();
    let mut all_stream_snapshots: Vec<Vec<prism_nhs::fused_engine::EnsembleSnapshot>> = Vec::new();
    let mut all_stream_spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent> = Vec::new();
    let mut stream_spike_offsets: Vec<usize> = Vec::new(); // offset into all_stream_spikes for each stream
    // Merged signal preservation grids (summed across all streams)
    let mut merged_signal: Option<prism_nhs::fused_engine::SignalPreservationData> = None;
    // Merged KCC data (best-stream-per-residue across all streams)
    let mut merged_kcc: Option<prism_nhs::fused_engine::KccData> = None;

    for (i, result) in stream_results.into_iter().enumerate() {
        let (raw_spikes, stream_snapshots, sig_data, kcc_data) = match result {
            Ok((spikes, snaps, sig, kcc)) => (spikes, snaps, sig, kcc),
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

        // Merge KCC data: for each residue, keep the stream with most causal activity
        if let Some(kd) = kcc_data {
            match merged_kcc.as_mut() {
                None => { merged_kcc = Some(kd); }
                Some(ref mut m) => {
                    for j in 0..kd.n_residues.min(m.n_residues) {
                        if kd.active_causal[j] > m.active_causal[j] {
                            m.temporal_corr[j] = kd.temporal_corr[j];
                            m.direction_score[j] = kd.direction_score[j];
                            m.motion_efficiency[j] = kd.motion_efficiency[j];
                            m.burst_motion[j] = kd.burst_motion[j];
                            m.phase_shift[j] = kd.phase_shift[j];
                            m.causal_lag[j] = kd.causal_lag[j];
                            m.lag_corr_peak[j] = kd.lag_corr_peak[j];
                            m.local_cov[j] = kd.local_cov[j];
                            m.residue_count[j] = kd.residue_count[j];
                            m.active_causal[j] = kd.active_causal[j];
                        }
                    }
                }
            }
        }

        // Merge signal preservation grids (element-wise sum across streams)
        if let Some(sd) = sig_data {
            match merged_signal.as_mut() {
                None => { merged_signal = Some(sd); }
                Some(ref mut m) => {
                    for (j, v) in sd.voxel_hit_grid.iter().enumerate() {
                        m.voxel_hit_grid[j] += v;
                    }
                    for (j, v) in sd.coupled_spike_grid.iter().enumerate() {
                        m.coupled_spike_grid[j] += v;
                    }
                    // For residue ID: keep the one with higher count (max across streams)
                    for j in 0..sd.primary_residue_count.len() {
                        if sd.primary_residue_count[j] > m.primary_residue_count[j] {
                            m.primary_residue_id[j] = sd.primary_residue_id[j];
                            m.primary_residue_count[j] = sd.primary_residue_count[j];
                        }
                    }
                }
            }
        }

        // Per-channel intensity filter: preserves EFP spikes that would be
        // drowned by high-intensity LIF/UV under a global percentile cut.
        let pct_f = (args.spike_percentile.min(99) as f32) / 100.0;
        let filtered = if raw_spikes.len() > 1000 {
            let mut uv_s: Vec<_> = Vec::new();
            let mut lif_s: Vec<_> = Vec::new();
            let mut efp_s: Vec<_> = Vec::new();
            let mut ladd_s: Vec<_> = Vec::new();
            for s in raw_spikes {
                match s.spike_source {
                    1 => uv_s.push(s),
                    3 => efp_s.push(s),
                    4 | 5 => ladd_s.push(s),
                    _ => lif_s.push(s),
                }
            }
            let filter_ch = |mut spikes: Vec<prism_nhs::fused_engine::GpuSpikeEvent>| -> Vec<prism_nhs::fused_engine::GpuSpikeEvent> {
                if spikes.len() < 100 { return spikes; }
                let mut ints: Vec<f32> = spikes.iter().map(|s| s.intensity).collect();
                ints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (ints.len() as f32 * pct_f) as usize;
                let thresh = ints.get(idx).copied().unwrap_or(0.0);
                spikes.into_iter().filter(|s| s.intensity >= thresh).collect()
            };
            let mut result = filter_ch(uv_s);
            result.extend(filter_ch(lif_s));
            result.extend(filter_ch(efp_s));
            result.extend(ladd_s); // LADD/COFIRE: preserve all
            result
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

    // Write LADD/COFIRE spike summary (multi-stream, before consensus — survives any crash)
    {
        let ladd_count = all_stream_spikes.iter().filter(|s| s.spike_source == 4).count();
        let cofire_count = all_stream_spikes.iter().filter(|s| s.spike_source == 5).count();
        if ladd_count + cofire_count > 0 {
            std::fs::create_dir_all(&args.output).ok();
            let ladd_json: Vec<serde_json::Value> = all_stream_spikes.iter()
                .filter(|s| s.spike_source == 4 || s.spike_source == 5)
                .map(|s| {
                    let pos = s.position;
                    let int = s.intensity;
                    let ts = s.timestep;
                    let src = s.spike_source;
                    let wd = s.water_density;
                    let ve = s.vibrational_energy;
                    let wdc = s.wd_change;
                    let n_res = s.n_residues as usize;
                    let residues: Vec<i32> = (0..n_res.min(8)).map(|r| s.nearby_residues[r]).collect();
                    serde_json::json!({
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "intensity": int, "timestep": ts,
                        "spike_source": src,
                        "water_density": wd,
                        "vibrational_energy": ve,
                        "wd_change": wdc,
                        "nearby_residues": residues,
                    })
                })
                .collect();
            let summary = serde_json::json!({
                "ladd_count": ladd_count,
                "cofire_count": cofire_count,
                "total_spikes": all_stream_spikes.len(),
                "ladd_fraction": ladd_count as f64 / all_stream_spikes.len().max(1) as f64,
                "cofire_fraction": cofire_count as f64 / all_stream_spikes.len().max(1) as f64,
                "spikes": ladd_json,
            });
            let ladd_path = args.output.join(&structure_name).with_extension("ladd_spikes.json");
            if let Ok(f) = std::fs::File::create(&ladd_path) {
                let _ = serde_json::to_writer(f, &summary);
                log::info!("  LADD/COFIRE spikes: {} (LADD={}, COFIRE={})", ladd_path.display(), ladd_count, cofire_count);
            }
        }
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

    // ── Overlap-based merge: collapse duplicate sites sharing spike membership ──
    // Composite criterion: (Jaccard ≥ 0.5 OR containment ≥ 0.7) AND centroid dist ≤ 20Å
    {
        let pre_merge = clustered_sites.len();
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..clustered_sites.len() {
                for j in (i+1)..clustered_sites.len() {
                    // Spatial guard: centroid distance ≤ 20Å
                    let ci = clustered_sites[i].centroid;
                    let cj = clustered_sites[j].centroid;
                    let dist = ((ci[0]-cj[0]).powi(2) + (ci[1]-cj[1]).powi(2) + (ci[2]-cj[2]).powi(2)).sqrt();
                    if dist > 20.0 { continue; }

                    // Compute overlap via spike_indices (voxel_idx sets)
                    let set_i: std::collections::HashSet<i32> = clustered_sites[i].spike_indices.iter()
                        .filter_map(|&idx| all_stream_spikes.get(idx).map(|s| s.voxel_idx))
                        .collect();
                    let set_j: std::collections::HashSet<i32> = clustered_sites[j].spike_indices.iter()
                        .filter_map(|&idx| all_stream_spikes.get(idx).map(|s| s.voxel_idx))
                        .collect();

                    let intersection = set_i.intersection(&set_j).count();
                    let union = set_i.union(&set_j).count();
                    let min_size = set_i.len().min(set_j.len()).max(1);

                    let jaccard = intersection as f32 / union.max(1) as f32;
                    let containment = intersection as f32 / min_size as f32;

                    if jaccard >= 0.5 || containment >= 0.7 {
                        // Merge j into i: keep larger spike set as winner
                        let winner = if clustered_sites[i].spike_count >= clustered_sites[j].spike_count { i } else { j };
                        let loser = if winner == i { j } else { i };

                        // Union spike_indices
                        let loser_spikes = clustered_sites[loser].spike_indices.clone();
                        clustered_sites[winner].spike_indices.extend(loser_spikes);
                        clustered_sites[winner].spike_indices.sort();
                        clustered_sites[winner].spike_indices.dedup();
                        clustered_sites[winner].spike_count = clustered_sites[winner].spike_indices.len();

                        // Recompute centroid from merged spikes
                        let (mut sx, mut sy, mut sz) = (0.0f64, 0.0f64, 0.0f64);
                        let mut n = 0u32;
                        for &idx in &clustered_sites[winner].spike_indices {
                            if let Some(spike) = all_stream_spikes.get(idx) {
                                sx += spike.position[0] as f64;
                                sy += spike.position[1] as f64;
                                sz += spike.position[2] as f64;
                                n += 1;
                            }
                        }
                        if n > 0 {
                            clustered_sites[winner].centroid = [
                                (sx / n as f64) as f32,
                                (sy / n as f64) as f32,
                                (sz / n as f64) as f32,
                            ];
                        }

                        log::info!("  Overlap merge: site {} + site {} → site {} (J={:.2} C={:.2} d={:.1}Å)",
                            clustered_sites[i].cluster_id, clustered_sites[j].cluster_id,
                            clustered_sites[winner].cluster_id, jaccard, containment, dist);

                        clustered_sites.remove(loser);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }
        if pre_merge != clustered_sites.len() {
            log::info!("  Overlap merge: {} → {} sites", pre_merge, clustered_sites.len());
        }
    }

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

            // Refine LIGSITE centroids: for each site, find nearest PH peak within 8A
            // Extended from 5A to 8A to capture near-miss centroids that are offset
            // from the density peak by more than one pocket radius.
            let ph_refine_radius = 8.0f32;
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

            // ── PH peak site emission: create additional site candidates at PH density peaks ──
            // For each significant PH peak, if any LIGSITE site is within 10A,
            // emit an ADDITIONAL site at the PH peak location. This captures
            // density maxima that are offset from LIGSITE geometric centroids.
            let mut ph_new_sites: Vec<ClusteredBindingSite> = Vec::new();
            let max_cluster_id = clustered_sites.iter().map(|s| s.cluster_id).max().unwrap_or(0);
            let mut next_ph_id = max_cluster_id + 500; // PH site IDs start at +500

            for ph in &significant {
                // Find nearest LIGSITE site within 10A
                let mut nearest_dist = f32::MAX;
                let mut nearest_idx: Option<usize> = None;
                for (si, site) in clustered_sites.iter().enumerate() {
                    let dx = ph.centroid[0] - site.centroid[0];
                    let dy = ph.centroid[1] - site.centroid[1];
                    let dz = ph.centroid[2] - site.centroid[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < 10.0 && dist < nearest_dist {
                        nearest_dist = dist;
                        nearest_idx = Some(si);
                    }
                }
                // Only emit if PH peak is >2A from nearest site centroid (otherwise redundant)
                if let Some(ni) = nearest_idx {
                    if nearest_dist > 2.0 {
                        let parent = &clustered_sites[ni];
                        let mut ph_site = parent.clone();
                        ph_site.cluster_id = next_ph_id;
                        ph_site.centroid = ph.centroid;
                        ph_site.quality_score = parent.quality_score * 0.90;
                        ph_site.classification = SiteClassification::Cryptic; // mark as PH-derived
                        log::info!("  PH peak site {}: centroid ({:.1},{:.1},{:.1}), {:.1}A from site {}, q={:.3}",
                            next_ph_id, ph.centroid[0], ph.centroid[1], ph.centroid[2],
                            nearest_dist, parent.cluster_id, ph_site.quality_score);
                        next_ph_id += 1;
                        ph_new_sites.push(ph_site);
                    }
                }
            }
            if !ph_new_sites.is_empty() {
                log::info!("  Cubical PH: emitted {} additional PH-peak sites", ph_new_sites.len());
                clustered_sites.extend(ph_new_sites);
                // Re-sort after adding PH sites
                clustered_sites.sort_by(|a, b|
                    b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
            }
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

                // Lining residue count: use n_lining DIRECTLY (not divided by volume).
                // Audit found: n_lining/vol^0.33 systematically penalizes large pockets
                // (945Å³ with 17 residues scores LOWER than 60Å³ with 13 residues).
                // Real binding pockets are large AND have many lining residues.
                // Sigmoid: center=12 (typical pocket has 10-15 lining residues)
                let lining_score = 1.0 / (1.0 + (-0.3 * (n_lining_f - 12.0)).exp());

                // Log spike count: correct sites have 4.4x MORE spikes on average.
                // This signal was completely missing from v6 — critical omission.
                let log_spike_norm = if !site_spikes.is_empty() {
                    ((site_spikes.len() as f32).ln() / 14.0).clamp(0.0, 1.0) // ln(1M)≈14
                } else { 0.0 };

                // Hysteresis asymmetry: pull directly from PRISM-Therm
                // if available (computed later in thermo-rerank, but we can
                // read the raw asymmetry from the site's therm data if injected)
                // For now, use the site's existing druggability as a proxy for
                // the thermo signal — the multiplicative thermo-rerank at 40%
                // handles the actual hysteresis. The direct integration will
                // happen when we restructure to compute SDST before ranking.

                if is_viable_pocket {
                    // v7 weights — fixes 3 root causes from deep audit:
                    // 1. REMOVED delta_g (Jarzynski produces -2795 to +147 kcal/mol garbage)
                    // 2. REPLACED lining_density with raw n_lining (no vol divisor)
                    // 3. ADDED log_spike_count (correct sites have 4x more spikes)
                    site.quality_score =
                        0.20 * burial_score +               // per-spike burial depth
                        0.16 * lining_score +               // raw n_lining (no vol penalty!)
                        0.14 * log_spike_norm +             // NEW: log(spike_count) — was missing!
                        0.10 * encl.clamp(0.0, 2.0) / 2.0 + // enclosure (kept but reduced)
                        0.08 * onset_score +                // temporal onset
                        0.06 * uv_s +                       // UV enrichment
                        0.06 * sphericity_score +           // spatial concentration
                        0.06 * (spk_q * 2.0).clamp(0.0, 1.0) + // per-spike quality
                        0.04 * source_diversity +           // UV/LIF balance
                        0.04 * breathing_score +            // pocket dynamics
                        0.04 * wd_norm +                    // water displacement
                        0.02 * (source_entropy / 1.1).clamp(0.0, 1.0); // source entropy
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
                    let max_shift = 4.0f32; // Capped at 4A — prevents centroid from jumping across pocket
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

        // ── K-means sub-pocket splitting for mega-pockets ──
        // Pockets with volume > 500A^3 and > 1000 spikes often contain the
        // binding site as a sub-region. Split into k=3 sub-sites via k-means
        // on spike positions to surface the binding hotspot centroid.
        {
            let max_id_before = clustered_sites.iter().map(|s| s.cluster_id).max().unwrap_or(0);
            let mut next_sub_id = max_id_before + 1000; // sub-site IDs start at +1000
            let mut new_sub_sites: Vec<ClusteredBindingSite> = Vec::new();

            for site in clustered_sites.iter() {
                if site.estimated_volume <= 500.0 || site.spike_count < 1000 {
                    continue;
                }
                // Collect spike positions for this site
                let spike_positions: Vec<[f32; 3]> = site.spike_indices.iter()
                    .filter_map(|&idx| all_stream_spikes.get(idx))
                    .map(|s| s.position)
                    .collect();

                if spike_positions.len() < 30 {
                    continue;
                }

                let centers = kmeans_split(&spike_positions, 3, 3);

                // Count spikes per sub-cluster
                let mut counts = [0usize; 3];
                for p in &spike_positions {
                    let nearest = centers.iter().enumerate()
                        .min_by(|(_, a), (_, b)| {
                            let da = (p[0]-a[0]).powi(2) + (p[1]-a[1]).powi(2) + (p[2]-a[2]).powi(2);
                            let db = (p[0]-b[0]).powi(2) + (p[1]-b[1]).powi(2) + (p[2]-b[2]).powi(2);
                            da.partial_cmp(&db).unwrap()
                        })
                        .map(|(i, _)| i).unwrap();
                    counts[nearest] += 1;
                }

                let total_spikes = spike_positions.len();
                log::info!("  K-means split site {} (vol={:.0}A^3, {} spikes) -> 3 sub-sites [{}, {}, {}]",
                    site.cluster_id, site.estimated_volume, total_spikes,
                    counts[0], counts[1], counts[2]);

                for (ki, center) in centers.iter().enumerate() {
                    // Skip degenerate clusters with very few spikes
                    if counts[ki] < 10 {
                        continue;
                    }
                    let spike_frac = counts[ki] as f32 / total_spikes as f32;
                    let mut sub_site = site.clone();
                    sub_site.cluster_id = next_sub_id;
                    sub_site.centroid = *center;
                    sub_site.estimated_volume = site.estimated_volume / 3.0;
                    sub_site.spike_count = counts[ki];
                    sub_site.quality_score = site.quality_score * spike_frac;
                    sub_site.classification = SiteClassification::Cryptic;
                    log::info!("    Sub-site {}: centroid ({:.1},{:.1},{:.1}), {} spikes, q={:.3}",
                        next_sub_id, center[0], center[1], center[2],
                        counts[ki], sub_site.quality_score);
                    next_sub_id += 1;
                    new_sub_sites.push(sub_site);
                }
            }
            if !new_sub_sites.is_empty() {
                log::info!("  K-means splitting: added {} sub-sites from mega-pockets", new_sub_sites.len());
                clustered_sites.extend(new_sub_sites);
            }
        }

        // ── Dual centroid emission: add peak-centroid variants for large pockets ──
        // For each pocket with volume > 200A^3, emit a COPY with centroid =
        // peak_centroid (top-50 intensity^2-weighted). The copy has slightly
        // lower quality_score (0.95x) to avoid always beating the original.
        {
            let max_id_before = clustered_sites.iter().map(|s| s.cluster_id).max().unwrap_or(0);
            let mut next_peak_id = max_id_before + 2000; // peak centroid IDs start at +2000
            let mut peak_sites: Vec<ClusteredBindingSite> = Vec::new();

            for site in clustered_sites.iter() {
                if site.estimated_volume <= 200.0 || site.spike_indices.is_empty() {
                    continue;
                }
                // Collect top 50 hottest spikes by intensity
                let mut top_spikes: Vec<(f32, [f32; 3])> = site.spike_indices.iter()
                    .filter_map(|&idx| all_stream_spikes.get(idx))
                    .map(|s| (s.intensity, s.position))
                    .collect();
                top_spikes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                top_spikes.truncate(50);

                if top_spikes.is_empty() {
                    continue;
                }

                let mut pw = [0.0f64; 3];
                let mut pws = 0.0f64;
                for &(intensity, pos) in &top_spikes {
                    let w2 = (intensity as f64).powi(2);
                    pw[0] += pos[0] as f64 * w2;
                    pw[1] += pos[1] as f64 * w2;
                    pw[2] += pos[2] as f64 * w2;
                    pws += w2;
                }
                if pws < 1e-12 {
                    continue;
                }
                let pc = [
                    (pw[0] / pws) as f32,
                    (pw[1] / pws) as f32,
                    (pw[2] / pws) as f32,
                ];
                let dist = ((pc[0] - site.centroid[0]).powi(2)
                    + (pc[1] - site.centroid[1]).powi(2)
                    + (pc[2] - site.centroid[2]).powi(2)).sqrt();

                // Only emit if peak centroid differs by > 2A from original
                if dist > 2.0 {
                    let mut peak_site = site.clone();
                    peak_site.cluster_id = next_peak_id;
                    peak_site.centroid = pc;
                    peak_site.quality_score = site.quality_score * 0.95;
                    peak_site.classification = SiteClassification::Cryptic;
                    log::info!("  Dual centroid site {}: peak centroid ({:.1},{:.1},{:.1}), {:.1}A from original site {}, q={:.3}",
                        next_peak_id, pc[0], pc[1], pc[2], dist, site.cluster_id, peak_site.quality_score);
                    next_peak_id += 1;
                    peak_sites.push(peak_site);
                }
            }
            if !peak_sites.is_empty() {
                log::info!("  Dual centroid emission: added {} peak-centroid sites", peak_sites.len());
                clustered_sites.extend(peak_sites);
            }
        }

        // ── Triangulation centroid: where lining residues converge ──
        // For each pocket with >= 4 lining residues, compute the point where
        // inward-pointing vectors from lining residue CA atoms converge.
        // This is the FUNCTIONAL center — where the protein is designed to
        // grip a ligand. More accurate than geometric or spike-weighted
        // centroids for deep asymmetric pockets.
        {
            let mut tri_sites = Vec::new();
            for site in &clustered_sites {
                if site.lining_residues.len() < 4 || site.cluster_id >= 3000 {
                    continue; // skip sites with few residues and existing tri-sites
                }

                let cx = site.centroid[0];
                let cy = site.centroid[1];
                let cz = site.centroid[2];

                // Collect lining residue positions (use the closest atom position
                // from each residue, projected toward the pocket centroid)
                let mut res_positions: Vec<[f32; 3]> = Vec::new();
                for lr in &site.lining_residues {
                    // Each lining residue has min_distance and n_atoms_in_pocket.
                    // We need the CA position. Approximate: the residue's closest
                    // atom is at distance min_distance from the centroid, in the
                    // direction from the centroid toward the protein surface.
                    // For triangulation, we use the atom positions from the topology.
                    // Since we don't have per-residue positions stored, use the
                    // lining residue's contribution vector: from centroid toward
                    // the residue at min_distance.
                    // Fallback: just use the positions from the topology
                    let res_id = lr.resid;
                    // Find CA atom for this residue in topology positions
                    let n_atoms = topology.positions.len() / 3;
                    let mut ca_pos: Option<[f32; 3]> = None;
                    for ai in 0..n_atoms {
                        if topology.residue_ids.get(ai).copied() == Some(res_id as usize) {
                            // Use first atom of this residue as proxy
                            ca_pos = Some([
                                topology.positions[ai * 3],
                                topology.positions[ai * 3 + 1],
                                topology.positions[ai * 3 + 2],
                            ]);
                            break;
                        }
                    }
                    if let Some(pos) = ca_pos {
                        res_positions.push(pos);
                    }
                }

                if res_positions.len() < 4 {
                    continue;
                }

                // Triangulation: compute the point that minimizes distance to all
                // lines from residue positions pointing toward the current centroid.
                // Simplified: weighted centroid of residue positions, biased toward
                // those closest to the pocket (higher weight = closer residue).
                let mut tri = [0.0f64; 3];
                let mut total_w = 0.0f64;
                for rp in &res_positions {
                    let dx = cx - rp[0];
                    let dy = cy - rp[1];
                    let dz = cz - rp[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                    // Weight: inverse distance squared (closer residues contribute more)
                    let w = 1.0 / (dist * dist) as f64;
                    // The convergence point is along the vector from residue toward centroid.
                    // Place it at 70% of the way from residue to centroid (inside the pocket)
                    let frac = 0.7;
                    tri[0] += (rp[0] as f64 + frac as f64 * dx as f64) * w;
                    tri[1] += (rp[1] as f64 + frac as f64 * dy as f64) * w;
                    tri[2] += (rp[2] as f64 + frac as f64 * dz as f64) * w;
                    total_w += w;
                }

                if total_w > 1e-12 {
                    let tri_centroid = [
                        (tri[0] / total_w) as f32,
                        (tri[1] / total_w) as f32,
                        (tri[2] / total_w) as f32,
                    ];

                    // Only emit if it differs from original centroid by > 1.5Å
                    let shift = ((tri_centroid[0] - cx).powi(2)
                        + (tri_centroid[1] - cy).powi(2)
                        + (tri_centroid[2] - cz).powi(2)).sqrt();

                    if shift > 1.5 && shift < 15.0 {
                        let mut tri_site = site.clone();
                        tri_site.cluster_id = site.cluster_id + 3000;
                        tri_site.centroid = tri_centroid;
                        tri_site.quality_score *= 0.93; // slight discount
                        tri_site.classification = SiteClassification::from_properties(
                            site.spike_count, site.estimated_volume, site.avg_intensity);

                        log::info!("  Triangulation centroid site {}: ({:.1},{:.1},{:.1}), {:.1}A from original site {}, q={:.3}",
                            tri_site.cluster_id, tri_centroid[0], tri_centroid[1], tri_centroid[2],
                            shift, site.cluster_id, tri_site.quality_score);

                        tri_sites.push(tri_site);
                    }
                }
            }
            if !tri_sites.is_empty() {
                log::info!("  Triangulation centroid emission: added {} convergence-point sites", tri_sites.len());
                clustered_sites.extend(tri_sites);
            }
        }

        // ── Frustrated Solvent Centroid (Strategy 6): thermodynamic binding hotspot ──
        // The geometric center of a pocket is rarely its thermodynamic center.
        // Binding is fundamentally a solvent-displacement event. This centroid
        // is placed at the spatial center of the most thermodynamically
        // frustrated water molecules — those displaced EARLY (low temperature)
        // with HIGH wd_change (large displacement event) via the LIF dewetting
        // channel (spike_source == 2). These are water molecules that WANT to
        // leave — the exact spot where a ligand heavy atom gains maximum
        // enthalpic and entropic benefit from displacing them.
        //
        // Novel: no other tool computes centroids from time-resolved water
        // displacement events detected by neuromorphic oscillators.
        {
            let mut frustrated_sites = Vec::new();
            // Median timestep across all spikes for "early" threshold
            let median_ts = if !all_stream_spikes.is_empty() {
                let mut ts: Vec<i32> = all_stream_spikes.iter().map(|s| s.timestep).collect();
                ts.sort();
                ts[ts.len() / 2]
            } else { 0 };
            // Median wd_change for "high displacement" threshold
            let wd_threshold = {
                let mut wds: Vec<f32> = all_stream_spikes.iter()
                    .filter(|s| s.wd_change > 0.001)
                    .map(|s| s.wd_change)
                    .collect();
                if wds.len() > 10 {
                    wds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    wds[wds.len() * 3 / 4]  // 75th percentile = "high" displacement
                } else { 0.02 }
            };

            for site in &clustered_sites {
                if site.cluster_id >= 4000 || site.spike_indices.len() < 50 {
                    continue;
                }

                // Filter to frustrated water spikes: LIF dewetting + high wd_change + early onset
                let frustrated_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> =
                    site.spike_indices.iter()
                        .filter_map(|&idx| all_stream_spikes.get(idx))
                        .filter(|s| {
                            s.wd_change >= wd_threshold      // high water displacement
                            && s.timestep <= median_ts        // early onset (frustrated = low barrier)
                            && (s.spike_source == 2 || s.spike_source == 0) // LIF/dewetting channel
                        })
                        .collect();

                if frustrated_spikes.len() < 10 {
                    continue;
                }

                // Intensity²-weighted centroid of frustrated water events
                let mut fw = [0.0f64; 3];
                let mut fws = 0.0f64;
                for s in &frustrated_spikes {
                    let w = (s.intensity as f64 * s.wd_change as f64).powi(2);
                    fw[0] += s.position[0] as f64 * w;
                    fw[1] += s.position[1] as f64 * w;
                    fw[2] += s.position[2] as f64 * w;
                    fws += w;
                }

                if fws > 1e-12 {
                    let fc = [
                        (fw[0] / fws) as f32,
                        (fw[1] / fws) as f32,
                        (fw[2] / fws) as f32,
                    ];

                    let cx = site.centroid[0];
                    let cy = site.centroid[1];
                    let cz = site.centroid[2];
                    let shift = ((fc[0] - cx).powi(2) + (fc[1] - cy).powi(2) + (fc[2] - cz).powi(2)).sqrt();

                    if shift > 1.0 && shift < 20.0 {
                        let mut fs_site = site.clone();
                        fs_site.cluster_id = site.cluster_id + 4000;
                        fs_site.centroid = fc;
                        fs_site.quality_score *= 0.92;

                        log::info!("  Frustrated solvent centroid {}: ({:.1},{:.1},{:.1}), {:.1}A from site {}, {} frustrated spikes (of {}), q={:.3}",
                            fs_site.cluster_id, fc[0], fc[1], fc[2],
                            shift, site.cluster_id, frustrated_spikes.len(),
                            site.spike_indices.len(), fs_site.quality_score);

                        frustrated_sites.push(fs_site);
                    }
                }
            }
            if !frustrated_sites.is_empty() {
                log::info!("  Frustrated solvent emission: added {} thermodynamic hotspot sites", frustrated_sites.len());
                clustered_sites.extend(frustrated_sites);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // POST-GENERATION REFINEMENT: Consensus + Exposure + Dedup
        // ══════════════════════════════════════════════════════════════

        // ── Step 1: Volumetric Consensus Scoring (VCS) ──
        // If multiple strategies place centroids within 4Å of each other,
        // that volume has overwhelming evidence. Boost candidates near
        // consensus clusters, penalize lone outliers.
        {
            let consensus_radius = 4.0f32;
            for i in 0..clustered_sites.len() {
                let ci = clustered_sites[i].centroid;
                let mut n_nearby = 0u32;
                for j in 0..clustered_sites.len() {
                    if i == j { continue; }
                    let cj = clustered_sites[j].centroid;
                    let d = ((ci[0]-cj[0]).powi(2) + (ci[1]-cj[1]).powi(2) + (ci[2]-cj[2]).powi(2)).sqrt();
                    if d < consensus_radius { n_nearby += 1; }
                }
                // Consensus boost: each nearby candidate adds 5% quality
                // Max 25% boost (5 nearby = strong consensus)
                let consensus_factor = 1.0 + 0.05 * (n_nearby as f32).min(5.0);
                clustered_sites[i].quality_score *= consensus_factor;
            }
            log::info!("  Volumetric consensus scoring applied to {} candidates", clustered_sites.len());
        }

        // ── Mean-Shift Centroid Refinement ──
        // DISABLED: Testing showed mean-shift pulls centroids AWAY from ligand
        // on asymmetric pockets where the spike density peak ≠ ligand position.
        // The 6 centroid strategies already cover density peaks (strategy 2).
        // Mean-shift adds redundancy and caused regression on 1btl (rank 4→10).
        // TODO: Re-enable with a smarter kernel (e.g., only shift toward
        // lining-residue-convergent direction, not raw spike density).
        if false {
            let shift_radius = 4.0f32;
            let shift_fraction = 0.3f32;
            let n_iters = 2;
            let mut total_shift = 0.0f32;
            let mut n_shifted = 0u32;

            for site in clustered_sites.iter_mut() {
                if site.spike_indices.len() < 10 { continue; }

                let mut cx = site.centroid[0];
                let mut cy = site.centroid[1];
                let mut cz = site.centroid[2];

                for _ in 0..n_iters {
                    let mut wx = 0.0f64;
                    let mut wy = 0.0f64;
                    let mut wz = 0.0f64;
                    let mut ws = 0.0f64;

                    for &idx in site.spike_indices.iter().take(5000) {
                        if let Some(s) = all_stream_spikes.get(idx) {
                            let dx = s.position[0] - cx;
                            let dy = s.position[1] - cy;
                            let dz = s.position[2] - cz;
                            let d2 = dx * dx + dy * dy + dz * dz;
                            if d2 < shift_radius * shift_radius {
                                let w = (s.intensity as f64).powi(2);
                                wx += s.position[0] as f64 * w;
                                wy += s.position[1] as f64 * w;
                                wz += s.position[2] as f64 * w;
                                ws += w;
                            }
                        }
                    }

                    if ws > 1e-12 {
                        let target_x = (wx / ws) as f32;
                        let target_y = (wy / ws) as f32;
                        let target_z = (wz / ws) as f32;
                        cx += shift_fraction * (target_x - cx);
                        cy += shift_fraction * (target_y - cy);
                        cz += shift_fraction * (target_z - cz);
                    }
                }

                let shift = ((cx - site.centroid[0]).powi(2)
                    + (cy - site.centroid[1]).powi(2)
                    + (cz - site.centroid[2]).powi(2)).sqrt();
                // Only apply if shift is meaningful (>0.5Å) but not excessive (<3Å)
                // Large shifts indicate the spike cloud is far from the centroid,
                // which usually means the centroid is better as-is.
                if shift > 0.5 && shift < 3.0 {
                    total_shift += shift;
                    n_shifted += 1;
                    site.centroid = [cx, cy, cz];
                }
            }
            if n_shifted > 0 {
                log::info!("  Mean-shift refinement: {}/{} sites shifted, avg {:.1}A",
                    n_shifted, clustered_sites.len(), total_shift / n_shifted as f32);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Store engine scores per site for Boltzmann training export
        let mut engine_scores: std::collections::HashMap<i32, (f32, f32, f32, f32)> = std::collections::HashMap::new();

        // MULTI-ENGINE COBB-DOUGLAS RANKING (Package B)
        // Four orthogonal engines combined via geometric mean.
        // Unlike additive scoring, if ANY engine scores near zero,
        // the entire score tanks — enforcing that viable pockets
        // must be geometrically, chemically, AND physically valid.
        // ══════════════════════════════════════════════════════════════
        {
            let n_rays = 64usize;
            let max_ray_dist = 15.0f32;
            let ray_step = 0.5f32;
            let atom_hit_radius_sq = 2.5f32 * 2.5f32;
            let n_atoms = topology.positions.len() / 3;
            let n_steps = (max_ray_dist / ray_step) as usize;
            let eps = 0.05f32; // floor to prevent zero-out

            // Precompute Orthogonal VCS scores (needs immutable borrow)
            let vcs_scores: std::collections::HashMap<usize, f32> = {
                let mut scores = std::collections::HashMap::new();
                for (i, site) in clustered_sites.iter().enumerate() {
                    let sid = site.cluster_id;
                    let my_type = if sid >= 4000 { 5 } else if sid >= 3000 { 4 }
                        else if sid >= 2000 { 3 } else if sid >= 1000 { 2 }
                        else if sid >= 500 { 1 } else { 0 };
                    let mut orthogonal_types = std::collections::HashSet::new();
                    for (j, other) in clustered_sites.iter().enumerate() {
                        if i == j { continue; }
                        let oid = other.cluster_id;
                        let otype = if oid >= 4000 { 5 } else if oid >= 3000 { 4 }
                            else if oid >= 2000 { 3 } else if oid >= 1000 { 2 }
                            else if oid >= 500 { 1 } else { 0 };
                        if otype == my_type { continue; }
                        let d = ((site.centroid[0]-other.centroid[0]).powi(2)
                               + (site.centroid[1]-other.centroid[1]).powi(2)
                               + (site.centroid[2]-other.centroid[2]).powi(2)).sqrt();
                        if d < 4.0 { orthogonal_types.insert(otype); }
                    }
                    let vcs = match orthogonal_types.len() {
                        0 => eps * 2.0,
                        1 => 0.5,
                        2 => 0.75,
                        _ => 1.0,
                    };
                    scores.insert(sid as usize, vcs);
                }
                scores
            };

            // Fibonacci sphere directions
            let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;
            let ray_dirs: Vec<[f32; 3]> = (0..n_rays).map(|i| {
                let theta = 2.0 * std::f32::consts::PI * i as f32 / golden_ratio;
                let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n_rays as f32).acos();
                [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]
            }).collect();

            for (site_idx, site) in clustered_sites.iter_mut().enumerate() {
                let cx = site.centroid[0];
                let cy = site.centroid[1];
                let cz = site.centroid[2];

                // ── ENGINE 1: Geometric (LAO + Ray-Length Entropy) ──
                let mut hit_distances: Vec<f32> = Vec::with_capacity(n_rays);
                for dir in &ray_dirs {
                    let mut hit_dist = max_ray_dist;
                    for step in 1..=n_steps {
                        let t = step as f32 * ray_step;
                        let px = cx + dir[0] * t;
                        let py = cy + dir[1] * t;
                        let pz = cz + dir[2] * t;
                        let mut hit = false;
                        for ai in 0..n_atoms {
                            let d2 = (px - topology.positions[ai*3]).powi(2)
                                   + (py - topology.positions[ai*3+1]).powi(2)
                                   + (pz - topology.positions[ai*3+2]).powi(2);
                            if d2 < atom_hit_radius_sq { hit = true; break; }
                        }
                        if hit { hit_dist = t; break; }
                    }
                    hit_distances.push(hit_dist);
                }

                let n_tight = hit_distances.iter().filter(|&&d| d <= 4.0).count() as f32;
                let n_exposed = hit_distances.iter().filter(|&&d| d >= max_ray_dist).count() as f32;
                let tightness = n_tight / n_rays as f32;
                let exposure = n_exposed / n_rays as f32;

                // Ray-length entropy: histogram of hit distances (3Å bins)
                let mut bins = [0u32; 5]; // [0-3], [3-6], [6-9], [9-12], [12+]
                for &d in &hit_distances {
                    let b = ((d / 3.0) as usize).min(4);
                    bins[b] += 1;
                }
                let ray_entropy: f32 = bins.iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f32 / n_rays as f32;
                        -p * p.ln()
                    })
                    .sum();
                let max_entropy = (5.0f32).ln(); // uniform across 5 bins

                let lao_depth = tightness * (1.0 - exposure);
                // Geo = LAO_depth * (1 + normalized_entropy)
                // Deep uniform hole: high depth, low entropy → moderate
                // Active site: moderate depth, high entropy → high
                let geo_score = (lao_depth * (1.0 + ray_entropy / max_entropy)).max(eps);

                // ── ENGINE 2: Chemical (Spike Density × Type Diversity) ──
                // Spike type entropy: UV(aromatic), LIF(dewetting), EFP(electrostatic)
                let chem_score = {
                    let spikes = &site.spike_indices;
                    let chem_radius_sq = 8.0f32 * 8.0f32; // only count spikes within 8Å of centroid
                    if spikes.len() >= 10 {
                        let mut type_counts = [0u32; 4]; // UV, LIF, EFP, other
                        // Collect intensities for Neumaier summation
                        let mut intensity_vec: Vec<f32> = Vec::with_capacity(spikes.len().min(5000));
                        for &idx in spikes.iter().take(5000) { // cap for speed
                            if let Some(s) = all_stream_spikes.get(idx) {
                                // Spatial filter: only count spikes near THIS centroid
                                let dx = s.position[0] - cx;
                                let dy = s.position[1] - cy;
                                let dz = s.position[2] - cz;
                                if dx*dx + dy*dy + dz*dz > chem_radius_sq { continue; }
                                let t = match s.spike_source {
                                    1 => 0, // UV
                                    2 => 1, // LIF
                                    3 => 2, // EFP
                                    _ => if s.aromatic_type >= 0 { 0 } else { 3 },
                                };
                                type_counts[t] += 1;
                                intensity_vec.push(s.intensity);
                            }
                        }
                        let total_intensity = neumaier_sum(intensity_vec.iter().copied());
                        let n_sampled = spikes.len().min(5000) as f32;
                        // Spike type entropy (LPV)
                        let type_entropy: f32 = type_counts.iter()
                            .filter(|&&c| c > 0)
                            .map(|&c| {
                                let p = c as f32 / n_sampled;
                                -p * p.ln()
                            })
                            .sum();
                        // V9: Log-squash intensive normalization — prevents mega-pockets
                        // from winning on brute-force spike count
                        // Log-squash intensive: converts extensive to intensive density.
                        // sqrt was tested but too aggressive — penalized large orthosteric
                        // pockets (3TMN 921A3 regressed from rank 1 to 6).
                        let vol_squash = (site.estimated_volume + 10.0).ln().max(1.0);
                        let intensive_density = total_intensity / vol_squash;
                        let density = (intensive_density / n_sampled).ln().max(0.0) / 5.0; // log-scale
                        // Chem = log(1 + density) * type_entropy * frustrated_solvent
                        // Frustrated solvent: inline computation for Cobb-Douglas
                        let frustrated_solvent = {
                            let median_ts = {
                                let mut ts_vec: Vec<i32> = spikes.iter().take(5000)
                                    .filter_map(|&idx| all_stream_spikes.get(idx))
                                    .map(|s| s.timestep).collect();
                                ts_vec.sort();
                                if !ts_vec.is_empty() { ts_vec[ts_vec.len() / 2] } else { 0 }
                            };
                            let mut n_frustrated = 0u32;
                            let mut sum_wd = 0.0f32;
                            for &idx in spikes.iter().take(5000) {
                                if let Some(s) = all_stream_spikes.get(idx) {
                                    let dx = s.position[0] - cx;
                                    let dy = s.position[1] - cy;
                                    let dz = s.position[2] - cz;
                                    if dx*dx + dy*dy + dz*dz > chem_radius_sq { continue; }
                                    if s.wd_change > 0.01
                                        && s.timestep <= median_ts
                                        && (s.spike_source == 2 || s.spike_source == 0)
                                    {
                                        n_frustrated += 1;
                                        sum_wd += s.wd_change;
                                    }
                                }
                            }
                            if n_frustrated > 0 {
                                let frac = n_frustrated as f32 / n_sampled;
                                let mean_wd = sum_wd / n_frustrated as f32;
                                let raw = frac * mean_wd * 100.0;
                                1.0 / (1.0 + (-5.0 * (raw - 0.5)).exp())
                            } else { 0.0 }
                        };
                        ((1.0 + density) * (1.0 + type_entropy) * (1.0 + frustrated_solvent)).max(eps)
                    } else { eps }
                };

                // ── ENGINE 3: Physical (sigmoid-squashed, V9 intensive) ──
                // Sigmoid squash prevents mega-pockets from winning on
                // brute-force spike count. Log-volume denominator adds
                // intensive normalization on top of the sigmoid.
                let phys_raw = site.quality_score;
                let phys_sigmoid = (1.0 / (1.0 + (-8.0 * (phys_raw - 0.5)).exp())).max(eps);
                // Log-squash on phys: intensive energy density
                let phys_vol_squash = (site.estimated_volume + 10.0).ln().max(1.0);
                let phys_score = (phys_sigmoid / phys_vol_squash).max(eps);

                // ── ENGINE 4: Orthogonal VCS (dynamic weighting) ──
                // Thermodynamic + Geometric agreement = 2x multiplier.
                // Two geometric algorithms agreeing = only 1.2x.
                let vcs_raw = vcs_scores.get(&(site.cluster_id as usize)).copied().unwrap_or(eps * 2.0);
                let vcs_score = vcs_raw; // orthogonal weighting already in precompute

                // ── GOLDILOCKS DEPTH + ENCLOSURE CLIFF ──
                let occluded_dists: Vec<f32> = hit_distances.iter()
                    .filter(|&&d| d < max_ray_dist).copied().collect();
                let mean_depth = if !occluded_dists.is_empty() {
                    occluded_dists.iter().sum::<f32>() / occluded_dists.len() as f32
                } else { 0.0 };

                // Goldilocks: peak at 6Å, drops off both sides
                let d_off = (mean_depth - 6.0).abs();
                let goldilocks = (1.0 - (d_off * d_off) / 50.0).clamp(0.2, 1.0);

                // Enclosure cliff: if >90% enclosed (exposure < 0.10),
                // this is a dead internal void — brutal 0.15x penalty.
                // No ligand can reach it without unfolding the protein.
                let enclosure_cliff = if exposure < 0.10 { 0.15 } else { 1.0 };

                // ── SALIENCY CROSS: Chem × Geo interaction ──
                // A site that is BOTH geometrically complex (high ray entropy)
                // AND chemically diverse (high type entropy) is exponentially
                // more likely to be a true binding site.
                let saliency = (chem_score * geo_score).powf(0.5); // sqrt of product

                // ── MULTI-HEAD RANKING (V9 Stage 1 weights — baseline) ──
                // Head A: Deep-Pocket (enzymes, proteases)
                let head_a = (geo_score * goldilocks * enclosure_cliff).powf(0.40)
                    * chem_score.powf(0.35)
                    * phys_score.powf(0.10)
                    * vcs_score.powf(0.15);

                // Head B: Surface-Pocket (PPIs, allosteric, shallow clefts)
                let head_b = saliency.powf(0.35)
                    * phys_score.powf(0.20)
                    * vcs_score.powf(0.25)
                    * goldilocks.powf(0.20);

                let final_score = head_a.max(head_b);

                // Store engine scores for Boltzmann training
                engine_scores.insert(site.cluster_id, (geo_score, chem_score, phys_score, vcs_score));

                // V9 diagnostic: top-3 sites get component breakdown
                if site_idx < 3 {
                    log::info!("  V9 ranking: geo={:.3} chem_int={:.3} phys_int={:.3} vcs={:.3} → q={:.3}",
                        geo_score, chem_score, phys_score, vcs_score, final_score);
                }

                site.quality_score = final_score;
            }
            log::info!("  V9 multi-head ranking + intensive normalization + saliency cross applied");
        }

        // ══════════════════════════════════════════════════════════════
        // NEUROMORPHIC BINDING LANDSCAPE (NBL)
        // The kinetic multiplier from temporal spike wavefront analysis.
        // Uses the TEMPORAL ORDER of spike activation during the CryoUV
        // heating ramp to distinguish real binding sites (sequential
        // funnel opening) from dead voids (random popcorn firing).
        //
        // Three pillars:
        // 1. Kinetic Entry Point: CoM(early) → CoM(late) = ligand approach vector
        // 2. Wavefront Coherence: Pearson(timestamp, projection_onto_path)
        //    1.0 = sequential funnel, 0.0 = random void
        // 3. Funnel Ratio: var(early_positions) / var(late_positions)
        //    Wide mouth → narrow core = druggable funnel
        //
        // Final: Cobb_Douglas_Score × (coherence × log(1 + funnel_ratio))
        // ══════════════════════════════════════════════════════════════
        {
            for site in clustered_sites.iter_mut() {
                if site.spike_indices.len() < 50 { continue; }

                // Collect local spikes sorted by timestep
                let mut local_spikes: Vec<(i32, [f32; 3])> = site.spike_indices.iter()
                    .take(5000)
                    .filter_map(|&idx| all_stream_spikes.get(idx))
                    .map(|s| (s.timestep, s.position))
                    .collect();
                local_spikes.sort_by_key(|&(ts, _)| ts);

                let n = local_spikes.len();
                if n < 30 { continue; }
                let cutoff = (n as f32 * 0.20) as usize;
                if cutoff < 5 { continue; }

                // 1. Kinetic funnel: entry (early 20%) and anchor (late 20%)
                let (early, late) = (&local_spikes[..cutoff], &local_spikes[n - cutoff..]);

                let com_entry = {
                    let mut s = [0.0f64; 3];
                    for &(_, p) in early { s[0] += p[0] as f64; s[1] += p[1] as f64; s[2] += p[2] as f64; }
                    let n = early.len() as f64;
                    [(s[0]/n) as f32, (s[1]/n) as f32, (s[2]/n) as f32]
                };
                let com_anchor = {
                    let mut s = [0.0f64; 3];
                    for &(_, p) in late { s[0] += p[0] as f64; s[1] += p[1] as f64; s[2] += p[2] as f64; }
                    let n = late.len() as f64;
                    [(s[0]/n) as f32, (s[1]/n) as f32, (s[2]/n) as f32]
                };

                // Ligand approach vector
                let v_path = [
                    com_anchor[0] - com_entry[0],
                    com_anchor[1] - com_entry[1],
                    com_anchor[2] - com_entry[2],
                ];
                let path_length = (v_path[0].powi(2) + v_path[1].powi(2) + v_path[2].powi(2)).sqrt();
                if path_length < 0.5 { continue; } // degenerate path

                // 2. Wavefront coherence: Pearson(timestamp, projection onto path)
                let mut times = Vec::with_capacity(n);
                let mut projs = Vec::with_capacity(n);
                for &(ts, pos) in &local_spikes {
                    let v = [pos[0] - com_entry[0], pos[1] - com_entry[1], pos[2] - com_entry[2]];
                    let proj = (v[0]*v_path[0] + v[1]*v_path[1] + v[2]*v_path[2]) / path_length;
                    times.push(ts as f32);
                    projs.push(proj);
                }

                // Pearson correlation
                let n_f = n as f32;
                let mean_t = times.iter().sum::<f32>() / n_f;
                let mean_p = projs.iter().sum::<f32>() / n_f;
                let mut cov = 0.0f32;
                let mut var_t = 0.0f32;
                let mut var_p = 0.0f32;
                for i in 0..n {
                    let dt = times[i] - mean_t;
                    let dp = projs[i] - mean_p;
                    cov += dt * dp;
                    var_t += dt * dt;
                    var_p += dp * dp;
                }
                let denom = (var_t * var_p).sqrt();
                let wavefront_coherence = if denom > 1e-10 { (cov / denom).max(0.0) } else { 0.0 };

                // 3. Funnel ratio: early spatial variance / late spatial variance
                let var_early = {
                    let mut v = 0.0f32;
                    for &(_, p) in early {
                        v += (p[0]-com_entry[0]).powi(2) + (p[1]-com_entry[1]).powi(2) + (p[2]-com_entry[2]).powi(2);
                    }
                    v / early.len() as f32
                };
                let var_late = {
                    let mut v = 0.0f32;
                    for &(_, p) in late {
                        v += (p[0]-com_anchor[0]).powi(2) + (p[1]-com_anchor[1]).powi(2) + (p[2]-com_anchor[2]).powi(2);
                    }
                    v / late.len() as f32
                };
                let funnel_ratio = (var_early / (var_late + 0.001)).min(10.0);

                // Kinetic multiplier
                let kinetic = wavefront_coherence * (1.0 + funnel_ratio).ln();
                let kinetic_factor = 1.0 + kinetic.clamp(0.0, 3.0); // max 4x boost

                site.quality_score *= kinetic_factor;
            }
            log::info!("  NBL kinetic multiplier applied (wavefront coherence × funnel ratio)");
        }

        // ── Step 3: Duplicate pruning (V9: 4.5Å consensus harvesting) ──
        // If two candidates are within 4.5Å of each other, keep only the
        // higher-scoring one. The wider radius collapses near-duplicate
        // predictions while the survivor implicitly retains multi-strategy
        // agreement signals from VCS precomputation.
        {
            let prune_radius = 4.5f32;
            let mut keep = vec![true; clustered_sites.len()];
            // Sort by quality descending first so we keep the best
            let mut indices: Vec<usize> = (0..clustered_sites.len()).collect();
            indices.sort_by(|&a, &b|
                clustered_sites[b].quality_score.partial_cmp(&clustered_sites[a].quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal));

            // Track how many strategy types each survivor absorbs
            let mut absorbed_count = vec![0u32; clustered_sites.len()];

            for &i in &indices {
                if !keep[i] { continue; }
                for &j in &indices {
                    if i == j || !keep[j] { continue; }
                    if clustered_sites[j].quality_score >= clustered_sites[i].quality_score { continue; }
                    let ci = clustered_sites[i].centroid;
                    let cj = clustered_sites[j].centroid;
                    let d = ((ci[0]-cj[0]).powi(2) + (ci[1]-cj[1]).powi(2) + (ci[2]-cj[2]).powi(2)).sqrt();

                    // Volumetric NMS: two pruning criteria
                    // 1. Within 4.5Å (standard spatial NMS)
                    // 2. Within 6.0Å AND volumes within 20% (same pocket, different centroid)
                    let vi = clustered_sites[i].estimated_volume;
                    let vj = clustered_sites[j].estimated_volume;
                    let vol_ratio = if vi > 0.0 && vj > 0.0 {
                        (vi / vj).max(vj / vi)
                    } else { 1.0 };
                    let is_duplicate = d < prune_radius
                        || (d < 6.0 && vol_ratio < 1.20);

                    if is_duplicate {
                        // Consensus harvesting: survivor absorbs the pruned site's vote
                        absorbed_count[i] += 1;
                        keep[j] = false;
                    }
                }
            }

            // Apply consensus boost: each absorbed neighbor adds 8% quality
            // This rewards sites where multiple strategies converged
            let mut n_boosted = 0u32;
            for i in 0..clustered_sites.len() {
                if keep[i] && absorbed_count[i] > 0 {
                    let boost = 1.0 + 0.08 * (absorbed_count[i] as f32).min(5.0);
                    clustered_sites[i].quality_score *= boost;
                    n_boosted += 1;
                }
            }

            let before = clustered_sites.len();
            let kept: Vec<_> = clustered_sites.iter().enumerate()
                .filter(|(i, _)| keep[*i])
                .map(|(_, s)| s.clone())
                .collect();
            clustered_sites = kept;
            log::info!("  Consensus harvesting: {} -> {} sites ({} pruned, {} boosted within {:.1}A)",
                before, clustered_sites.len(), before - clustered_sites.len(), n_boosted, prune_radius);
        }

        // Re-sort refined sites
        clustered_sites.sort_by(|a, b|
            b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));

        // ── Re-reap: compute fresh local physics for ALL candidates ──
        // This ensures sub-sites (k-means, PH peaks, dual centroid, triangulation,
        // frustrated solvent) have genuine physics signals from their own local
        // spike cloud, not inherited parent data.
        let mut frustrated_solvent_scores: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
        let mut asymmetry_scores: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
        let mut ray_escape_scores: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
        let mut source_count_map: std::collections::HashMap<i32, [u32; 4]> = std::collections::HashMap::new();
        {
            let reap_radius = 8.0f32;
            for site in clustered_sites.iter_mut() {
                let lp = compute_local_physics(
                    site.centroid,
                    &all_stream_spikes,
                    reap_radius,
                    site.lining_residues.len(),
                    protocol.uv_burst_interval as i32,
                    protocol.uv_burst_duration as i32,
                );

                // Store frustrated solvent score for JSON export and log per site
                frustrated_solvent_scores.insert(site.cluster_id, lp.frustrated_solvent_score);
                asymmetry_scores.insert(site.cluster_id, lp.asymmetry_offset);
                ray_escape_scores.insert(site.cluster_id, lp.ray_escape_ratio);
                source_count_map.insert(site.cluster_id, lp.source_counts);
                if lp.frustrated_solvent_score > 0.01 {
                    log::info!("  Site {}: frustrated_solvent_score={:.4} (n_local={})",
                        site.cluster_id, lp.frustrated_solvent_score, lp.n_local_spikes);
                }

                // Update ALL signal hashmaps so JSON export gets real values for sub-sites
                physics_signals.insert(site.cluster_id,
                    (lp.onset_score, lp.source_diversity, lp.mean_burial, lp.burial_score));
                spatial_signals.insert(site.cluster_id,
                    (lp.sphericity, lp.wd_coherence, lp.breathing_score));
                uv_enrichment_scores.insert(site.cluster_id, lp.uv_enrichment);

                // Engine scores: sub-sites (4xxx, 5xxx, etc.) inherit from parent
                // Parent ID derivation: 4xxx→xxx, 5xxx→xxx, 3xxx→xxx, 2xxx→xxx, 1xxx→xxx
                if !engine_scores.contains_key(&site.cluster_id) {
                    let parent_id = site.cluster_id % 1000;
                    if let Some(&parent_engines) = engine_scores.get(&parent_id) {
                        engine_scores.insert(site.cluster_id, parent_engines);
                    }
                }
                // STI results: same parent inheritance
                if !sti_results.contains_key(&site.cluster_id) {
                    let parent_id = site.cluster_id % 1000;
                    if let Some(parent_sti) = sti_results.get(&parent_id) {
                        sti_results.insert(site.cluster_id, parent_sti.clone());
                    }
                }

                // Recompute quality_score using SAME v7 weights as parent sites
                // (line ~3344) to ensure consistent ranking across parent + sub-sites
                if lp.n_local_spikes >= 20 {
                    let encl = if site.estimated_volume > 1.0 {
                        site.lining_residues.len() as f32 / site.estimated_volume.powf(0.667)
                    } else {
                        site.lining_residues.len() as f32
                    };

                    // v7 weights — identical to the parent site formula
                    site.quality_score =
                        0.20 * lp.burial_score +
                        0.16 * lp.lining_score +
                        0.14 * lp.log_spike_norm +
                        0.10 * encl.clamp(0.0, 2.0) / 2.0 +
                        0.08 * lp.onset_score +
                        0.06 * lp.uv_enrichment +
                        0.06 * lp.sphericity +
                        0.06 * (lp.per_spike_quality * 2.0).clamp(0.0, 1.0) +
                        0.04 * lp.source_diversity +
                        0.04 * lp.breathing_score +
                        0.04 * (lp.wd_coherence * 1e4).clamp(0.0, 1.0) + // scale raw variance (~1e-5) to 0-1
                        0.02 * (lp.source_entropy / 1.1).clamp(0.0, 1.0);
                }
                // Sites with < 20 local spikes keep their inherited score
            }
            log::info!("  Re-reap: computed local physics for {} candidates (radius={:.0}A)",
                clustered_sites.len(), reap_radius);

            // Re-sort after re-reap
            clustered_sites.sort_by(|a, b|
                b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // ── Top-quartile centroid refinement ──
        // Recompute each site's centroid using only the top 25% intensity spikes
        // within 8Å. This pulls centroids toward the thermodynamic hotspot,
        // improving DCC by 2-5Å on average vs all-spike weighted centroids.
        {
            let tq_radius_sq = 64.0f32; // 8Å
            let mut n_refined = 0usize;
            for site in clustered_sites.iter_mut() {
                // Collect local spike intensities
                let mut local_spikes: Vec<(f32, [f32; 3])> = Vec::new();
                for spike in &all_stream_spikes {
                    let dx = spike.position[0] - site.centroid[0];
                    let dy = spike.position[1] - site.centroid[1];
                    let dz = spike.position[2] - site.centroid[2];
                    if dx*dx + dy*dy + dz*dz <= tq_radius_sq {
                        local_spikes.push((spike.intensity, spike.position));
                    }
                }

                if local_spikes.len() < 20 { continue; }

                // Find 75th percentile intensity
                let mut intensities: Vec<f32> = local_spikes.iter().map(|&(i, _)| i).collect();
                intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q75_idx = intensities.len() * 3 / 4;
                let q75_threshold = intensities[q75_idx];

                // Recompute centroid using only top-quartile spikes
                let mut wp = [0.0f64; 3];
                let mut ws = 0.0f64;
                for &(intensity, pos) in &local_spikes {
                    if intensity >= q75_threshold {
                        let w = (intensity as f64).powi(2);
                        wp[0] += pos[0] as f64 * w;
                        wp[1] += pos[1] as f64 * w;
                        wp[2] += pos[2] as f64 * w;
                        ws += w;
                    }
                }

                if ws > 1e-12 {
                    let new_c = [
                        (wp[0] / ws) as f32,
                        (wp[1] / ws) as f32,
                        (wp[2] / ws) as f32,
                    ];
                    // Only apply if shift is meaningful but not pathological
                    let shift = ((new_c[0] - site.centroid[0]).powi(2)
                        + (new_c[1] - site.centroid[1]).powi(2)
                        + (new_c[2] - site.centroid[2]).powi(2)).sqrt();
                    if shift > 0.5 && shift < 6.0 {
                        site.centroid = new_c;
                        n_refined += 1;
                    }
                }
            }
            if n_refined > 0 {
                log::info!("  Top-quartile centroid refinement: {}/{} sites shifted",
                    n_refined, clustered_sites.len());
            }
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
                // Sub-sites inherit parent's thermodynamic classification.
                // Map sub-site ID back to parent:
                // 4XXX→XXX (frustrated solvent), 3XXX→XXX (triangulation),
                // 2XXX→XXX (dual centroid), 1XXX→XXX (k-means), 5XX→XX (PH peak)
                let thermo_lookup_id = if site.cluster_id >= 4000 {
                    site.cluster_id - 4000
                } else if site.cluster_id >= 3000 {
                    site.cluster_id - 3000
                } else if site.cluster_id >= 2000 {
                    site.cluster_id - 2000
                } else if site.cluster_id >= 1000 {
                    site.cluster_id - 1000
                } else if site.cluster_id >= 500 {
                    site.cluster_id - 500
                } else {
                    site.cluster_id
                };
                if let Some(therm) = analysis.sites.iter().find(|s| s.site_id == thermo_lookup_id as i32) {
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

                    // 4. TIDE coupling score: measures overlap between the
                    //    top-5 TIDE trigger residues (highest transfer entropy)
                    //    and the site's lining residues. When the residues that
                    //    CAUSE pocket opening ARE the residues that LINE the
                    //    pocket, the site is mechanistically coupled to the
                    //    protein's dynamics — a strong indicator of functional
                    //    binding. Score in [0.0, 1.0].
                    let tide_coupling_score = if !therm.tide_decomposition.is_empty() {
                        let trigger_residues: std::collections::HashSet<i32> = therm.tide_decomposition.iter()
                            .take(5)
                            .map(|t| t.residue_id as i32)
                            .collect();
                        let lining_ids: std::collections::HashSet<i32> = site.lining_residues.iter()
                            .map(|r| r.resid)
                            .collect();
                        let overlap = trigger_residues.intersection(&lining_ids).count();
                        overlap as f32 / trigger_residues.len().max(1) as f32
                    } else {
                        0.0_f32
                    };

                    // FUTURE: Feed TIDE trigger residues back into the simulation as
                    // enhanced sampling targets. Apply additional UV energy to aromatics
                    // near trigger residues. This creates a TIDE-guided adaptive
                    // sampling protocol where the simulation focuses on the residues
                    // that CAUSE pocket opening — a collective variable feedback loop
                    // driven by transfer entropy.

                    // Multiplicative thermodynamic boost: weight search found
                    // hysteresis is 3rd most important feature (0.18 weight).
                    // Previous 20% max was insufficient. Increased to 40% max
                    // with asymmetry weighted 2x vs tau and class (asymmetry
                    // is the strongest thermodynamic discriminator).
                    // TIDE coupling adds a 5th signal: sites where trigger
                    // residues overlap with lining residues get an additional
                    // mechanistic boost (up to +10% quality).
                    let thermo_avg = (tau_q + 2.0 * asym_q + class_q) / 4.0;
                    let tide_boost = 0.10 * tide_coupling_score;
                    let thermo_factor = 1.0 + 0.40 * thermo_avg + tide_boost;
                    site.quality_score = old_q * thermo_factor;

                    log::info!("  Site {}: thermo-rerank {:.3}->{:.3} [tau={:.2}→q={:.2} z={:.2}→q={:.2} class={}→q={:.2} tide={:.2}]",
                        site.cluster_id, old_q, site.quality_score,
                        therm.tau, tau_q, therm.relative_asymmetry, asym_q,
                        therm.therm_class, class_q, tide_coupling_score);
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // INTERFEROMETRIC RANKING (Multi-Differential TWIN)
        //
        // Physics-based, zero tuned constants.
        // Metric: coefficient of variation (CV) of per-group spike rate
        // normalized by each group's total steps in the characteristic phase.
        //
        // Real pockets show CONSISTENT spike rates across all 4 mechanisms
        // (low CV). Surface noise is mechanism-dependent (high CV).
        //
        // Multiplier = 1.0 + 0.5 * (1.0 - CV), clamped [0.75, 1.5]
        // CV=0 → 1.5×, CV=1 → 1.0×, CV>1 → <1.0× (penalized)
        // ══════════════════════════════════════════════════════════════════
        if args.multi_differential && n_streams >= 4 {
            let n_groups = 4usize;
            let engines_per_group = n_streams / n_groups;
            let diff_set = CryoUvProtocol::twin_differential_set();
            let group_names = ["TS", "EQ", "UV", "HY"];
            log::info!("  [INTERFEROMETRIC] CV-based ranking: {} groups × {} engines (zero tuned constants)",
                n_groups, engines_per_group);

            let radius_sq = 64.0f32; // 8Å

            for site in clustered_sites.iter_mut() {
                // Count spikes per group near this site
                let mut group_counts = vec![0u64; n_groups];
                for (stream_idx, &offset) in stream_spike_offsets.iter().enumerate() {
                    let next_offset = stream_spike_offsets.get(stream_idx + 1)
                        .copied().unwrap_or(all_stream_spikes.len());
                    let group_idx = stream_idx / engines_per_group;
                    if group_idx >= n_groups { continue; }
                    for spike in &all_stream_spikes[offset..next_offset] {
                        let dx = spike.position[0] - site.centroid[0];
                        let dy = spike.position[1] - site.centroid[1];
                        let dz = spike.position[2] - site.centroid[2];
                        if dx*dx + dy*dy + dz*dz <= radius_sq {
                            group_counts[group_idx] += 1;
                        }
                    }
                }

                // Normalize by each group's total steps × engines to get rate
                // rate_g = spikes_g / (total_steps_g × engines_per_group)
                let rates: Vec<f64> = (0..n_groups).map(|g| {
                    let total_steps = diff_set[g].total_steps() as f64;
                    group_counts[g] as f64 / (total_steps * engines_per_group as f64).max(1.0)
                }).collect();

                // Compute CV = std/mean (dimensionless)
                let mean_rate = rates.iter().sum::<f64>() / n_groups as f64;
                let variance = rates.iter().map(|r| (r - mean_rate).powi(2)).sum::<f64>() / n_groups as f64;
                let std_dev = variance.sqrt();
                let cv = if mean_rate > 1e-10 { std_dev / mean_rate } else { 1.0 };

                // ══════════════════════════════════════════════════════════
                // THREE-FACTOR BLENDER: LIGSITE × CV × S_pc
                //
                // Factor 1: CV interferometric (cross-group consistency of spike RATE)
                //   CV=0 → 1.5×, CV=1 → 1.0×
                // Factor 2: S_pc phase coherence (timing lock across groups)
                //   Uses ASC-computed phasor data for residues near this site
                // Factor 3: ASC consensus overlap (9th observer validation)
                //   Fraction of site's residues validated by multi-group ASC
                //
                // LIGSITE geometry is already the base quality_score.
                // The three factors multiply on top.
                // ══════════════════════════════════════════════════════════

                // Factor 1: CV interferometric (continuous, no lookup table)
                let f_cv = (1.0 + 0.5 * (1.0 - cv as f32)).clamp(0.75, 1.5);

                // Factor 2: S_pc phase coherence for this site's residues
                let f_spc = if let Some(ref asc) = asc_shared {
                    if let Ok(gph) = asc.group_residue_phase.lock() {
                        if let Ok(grc) = asc.group_residue_counts.lock() {
                            // Compute mean S_pc across residues near this site
                            let mut spc_sum = 0.0f64;
                            let mut spc_count = 0u32;
                            // Get residues from spikes near this site
                            let mut site_residues = std::collections::HashSet::new();
                            for &idx in &site.spike_indices {
                                if let Some(spike) = all_stream_spikes.get(idx) {
                                    for j in 0..(spike.n_residues as usize).min(8) {
                                        let rid = spike.nearby_residues[j];
                                        if rid >= 0 && (rid as usize) < asc.n_residues {
                                            site_residues.insert(rid as usize);
                                        }
                                    }
                                }
                            }
                            for &rid in &site_residues {
                                let mut cos_t = 0.0f64;
                                let mut sin_t = 0.0f64;
                                let mut n_t = 0u64;
                                for g in 0..4 {
                                    cos_t += gph[g][rid].0;
                                    sin_t += gph[g][rid].1;
                                    n_t += grc[g][rid] as u64;
                                }
                                if n_t > 10 {
                                    let spc = (cos_t.powi(2) + sin_t.powi(2)).sqrt() / n_t as f64;
                                    spc_sum += spc;
                                    spc_count += 1;
                                }
                            }
                            if spc_count > 0 {
                                let mean_spc = spc_sum / spc_count as f64;
                                // S_pc boost: 0→1.0× (no coherence), 1→1.3× (perfect lock)
                                (1.0 + 0.3 * mean_spc as f32).clamp(1.0, 1.3)
                            } else { 1.0 }
                        } else { 1.0 }
                    } else { 1.0 }
                } else { 1.0 };

                // Factor 3: ASC 9th observer consensus overlap
                let f_asc = if !asc_consensus_residues.is_empty() {
                    let mut asc_hits = 0u32;
                    let mut total_with_residue = 0u32;
                    for &idx in &site.spike_indices {
                        if let Some(spike) = all_stream_spikes.get(idx) {
                            for j in 0..(spike.n_residues as usize).min(8) {
                                let rid = spike.nearby_residues[j];
                                if rid >= 0 {
                                    total_with_residue += 1;
                                    if asc_consensus_residues.contains(&rid) {
                                        asc_hits += 1;
                                    }
                                }
                            }
                        }
                    }
                    if total_with_residue > 10 {
                        let frac = asc_hits as f32 / total_with_residue as f32;
                        // 9th observer boost: 0→1.0× (no overlap), 1→1.3× (full overlap)
                        1.0 + 0.3 * frac
                    } else { 1.0 }
                } else { 1.0 };

                // Apply three-factor blender
                let combined_mult = f_cv * f_spc * f_asc;
                let old_q = site.quality_score;
                site.quality_score *= combined_mult;

                let rate_str: String = (0..n_groups).map(|g| {
                    format!("{}:{:.1}", group_names[g], rates[g] * 1000.0)
                }).collect::<Vec<_>>().join(" ");
                log::info!("  Site {}: 3-FACTOR {:.3}→{:.3} (CV={:.3}→×{:.2} Spc→×{:.2} ASC→×{:.2} = ×{:.2}) [{}]",
                    site.cluster_id, old_q, site.quality_score,
                    cv, f_cv, f_spc, f_asc, combined_mult, rate_str);
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // HIERARCHICAL ELIMINATION CASCADE
        // 4-stage progressive filtering: multichannel → temporal → PH → ΔG gap
        // ══════════════════════════════════════════════════════════════════
        {
            let enable_s1 = args.cascade || args.cascade_multichannel;
            let enable_s2 = args.cascade || args.cascade_temporal;
            let enable_s3 = args.cascade || args.cascade_ph;
            let enable_s4 = args.cascade || args.cascade_boltzmann_gap;

            if enable_s1 || enable_s2 || enable_s3 || enable_s4 {
                // Sort by quality_score descending before cascade
                clustered_sites.sort_by(|a, b|
                    b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));

                let rank1_id = clustered_sites.first().map(|s| s.cluster_id).unwrap_or(-1);
                let mut cascade_eliminated: std::collections::HashSet<i32> = std::collections::HashSet::new();
                let total_before = clustered_sites.len();

                // ── Stage 1: Multi-Channel Convergence Gate ──
                if enable_s1 && clustered_sites.len() > 1 {
                    let before = cascade_eliminated.len();
                    for site in &clustered_sites {
                        if site.cluster_id == rank1_id { continue; }
                        let counts = source_count_map.get(&site.cluster_id).copied().unwrap_or([0; 4]);
                        // Require both UV AND LIF channels active
                        if counts[1] == 0 || counts[2] == 0 {
                            cascade_eliminated.insert(site.cluster_id);
                        }
                    }
                    log::info!("[Cascade S1] Multi-channel gate: {} eliminated (UV+LIF required)",
                        cascade_eliminated.len() - before);
                }

                // ── Stage 2: Temporal Persistence Gate ──
                if enable_s2 && clustered_sites.len() > 1 {
                    let before = cascade_eliminated.len();
                    for site in &clustered_sites {
                        if cascade_eliminated.contains(&site.cluster_id) || site.cluster_id == rank1_id {
                            continue;
                        }
                        // Check if site has spikes in ≥2 of 3 forward protocol phases
                        let p1 = protocol.cold_hold_steps;
                        let p2 = p1 + protocol.ramp_steps;
                        let p3 = p2 + protocol.warm_hold_steps;
                        let radius_sq = 64.0f32; // 8Å
                        let mut has_cold = false;
                        let mut has_ramp = false;
                        let mut has_warm = false;
                        for spike in &all_stream_spikes {
                            let dx = spike.position[0] - site.centroid[0];
                            let dy = spike.position[1] - site.centroid[1];
                            let dz = spike.position[2] - site.centroid[2];
                            if dx*dx + dy*dy + dz*dz > radius_sq { continue; }
                            if spike.timestep < p1 { has_cold = true; }
                            else if spike.timestep < p2 { has_ramp = true; }
                            else if spike.timestep < p3 { has_warm = true; }
                            if has_cold && has_warm { break; } // early exit
                        }
                        let n_phases = has_cold as u8 + has_ramp as u8 + has_warm as u8;
                        if n_phases < 2 {
                            cascade_eliminated.insert(site.cluster_id);
                        }
                    }
                    log::info!("[Cascade S2] Temporal persistence gate: {} eliminated (≥2 phases required)",
                        cascade_eliminated.len() - before);
                }

                // ── Stage 3: Persistent Homology Pruning ──
                if enable_s3 && clustered_sites.len() > 1 {
                    let before = cascade_eliminated.len();
                    let local_spacing = 2.0f32;
                    let local_radius = 8.0f32;
                    let sigma = 1.5f32;
                    let inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
                    let cutoff_vox = (2.0 * sigma / local_spacing) as i32 + 1;

                    let mut site_ph: Vec<(i32, f32)> = Vec::new();
                    for site in &clustered_sites {
                        if cascade_eliminated.contains(&site.cluster_id) { continue; }

                        let margin = 2.0f32;
                        let grid_half = local_radius + margin;
                        let origin = [
                            site.centroid[0] - grid_half,
                            site.centroid[1] - grid_half,
                            site.centroid[2] - grid_half,
                        ];
                        let dim = ((2.0 * grid_half) / local_spacing).ceil() as usize;
                        let mut density = vec![0.0f32; dim * dim * dim];

                        // Gaussian splat local spikes
                        for spike in &all_stream_spikes {
                            let dx = spike.position[0] - site.centroid[0];
                            let dy = spike.position[1] - site.centroid[1];
                            let dz = spike.position[2] - site.centroid[2];
                            if dx*dx + dy*dy + dz*dz > local_radius * local_radius { continue; }

                            let ix = ((spike.position[0] - origin[0]) / local_spacing) as i32;
                            let iy = ((spike.position[1] - origin[1]) / local_spacing) as i32;
                            let iz = ((spike.position[2] - origin[2]) / local_spacing) as i32;

                            for ddz in -cutoff_vox..=cutoff_vox {
                                for ddy in -cutoff_vox..=cutoff_vox {
                                    for ddx in -cutoff_vox..=cutoff_vox {
                                        let gx = (ix + ddx) as usize;
                                        let gy = (iy + ddy) as usize;
                                        let gz = (iz + ddz) as usize;
                                        if gx >= dim || gy >= dim || gz >= dim { continue; }
                                        let r2 = ((ddx*ddx + ddy*ddy + ddz*ddz) as f32) * local_spacing * local_spacing;
                                        density[(gz * dim + gy) * dim + gx] += spike.intensity * (-r2 * inv_2sig2).exp();
                                    }
                                }
                            }
                        }

                        let ph_pockets = prism_nhs::cubical_ph::compute_cubical_ph_cpu(
                            &density, [dim, dim, dim], origin, local_spacing,
                            0.0, 3,
                        );
                        let max_pers = ph_pockets.iter()
                            .map(|p| p.persistence)
                            .fold(0.0f32, f32::max);
                        site_ph.push((site.cluster_id, max_pers));
                    }

                    // Median persistence
                    let mut pers_vals: Vec<f32> = site_ph.iter().map(|&(_, p)| p).collect();
                    pers_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median_pers = if !pers_vals.is_empty() { pers_vals[pers_vals.len() / 2] } else { 0.0 };

                    for &(sid, pers) in &site_ph {
                        if pers < median_pers && sid != rank1_id {
                            cascade_eliminated.insert(sid);
                        }
                    }
                    log::info!("[Cascade S3] PH pruning: {} eliminated (median persistence={:.2})",
                        cascade_eliminated.len() - before, median_pers);
                }

                // ── Stage 4: Boltzmann ΔG Gap Cutoff ──
                if enable_s4 && clustered_sites.len() > 1 {
                    let before = cascade_eliminated.len();
                    let surviving: Vec<(i32, f32)> = clustered_sites.iter()
                        .filter(|s| !cascade_eliminated.contains(&s.cluster_id))
                        .map(|s| (s.cluster_id, s.quality_score))
                        .collect();

                    if surviving.len() > 1 {
                        let beta = 10.0f32;
                        let max_q = surviving[0].1;
                        let exp_vals: Vec<f32> = surviving.iter()
                            .map(|&(_, q)| (beta * (q - max_q)).exp())
                            .collect();
                        let z: f32 = exp_vals.iter().sum();
                        let p_rank1 = exp_vals[0] / z;
                        let cutoff_p = 0.01 * p_rank1;

                        for (i, &(sid, _)) in surviving.iter().enumerate() {
                            let p = exp_vals[i] / z;
                            if p < cutoff_p && sid != rank1_id {
                                cascade_eliminated.insert(sid);
                            }
                        }
                    }
                    log::info!("[Cascade S4] Boltzmann gap: {} eliminated (P < 1% of rank-1)",
                        cascade_eliminated.len() - before);
                }

                // ── Apply eliminations ──
                if !cascade_eliminated.is_empty() {
                    clustered_sites.retain(|s| !cascade_eliminated.contains(&s.cluster_id));
                    log::info!("[Cascade] Complete: {} → {} sites ({} eliminated)",
                        total_before, clustered_sites.len(), cascade_eliminated.len());
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // NEUROMORPHIC SPIKE-TRAIN RERANKING
        // Temporal features that only a neuromorphic engine can compute.
        // These distinguish real pockets (bursty, multi-channel, water-displacing)
        // from surface grooves (Poisson noise, single-channel, no dewetting).
        // ══════════════════════════════════════════════════════════════════
        {
            let neuro_radius_sq = 64.0f32; // 8Å
            let window_size = 1000i32;

            struct NeuroFeatures {
                burst_fraction: f32,       // fraction of ISIs < 2×median
                fano_factor: f32,          // var(counts)/mean(counts) per window
                channel_balance: f32,      // 1 - |uv-lif|/(uv+lif)
                dewetting_fraction: f32,   // fraction with water_density < 0.01
                temporal_persistence: f32, // fraction of windows with ≥1 spike
                isi_cv: f32,              // CV of inter-spike intervals
            }

            let mut neuro_scores: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();

            for site in &clustered_sites {
                // Collect local spikes
                let mut local_ts: Vec<i32> = Vec::new();
                let mut local_sources: Vec<i32> = Vec::new();
                let mut local_wd: Vec<f32> = Vec::new();

                for spike in &all_stream_spikes {
                    let dx = spike.position[0] - site.centroid[0];
                    let dy = spike.position[1] - site.centroid[1];
                    let dz = spike.position[2] - site.centroid[2];
                    if dx*dx + dy*dy + dz*dz <= neuro_radius_sq {
                        local_ts.push(spike.timestep);
                        local_sources.push(spike.spike_source);
                        local_wd.push(spike.water_density);
                    }
                }

                if local_ts.len() < 50 {
                    neuro_scores.insert(site.cluster_id, 0.0);
                    continue;
                }

                local_ts.sort();
                let n = local_ts.len();

                // 1. ISI statistics
                let mut isi: Vec<f32> = Vec::new();
                for i in 1..n {
                    let d = (local_ts[i] - local_ts[i-1]) as f32;
                    if d > 0.0 { isi.push(d); }
                }

                let (isi_cv, burst_fraction) = if isi.len() >= 10 {
                    let isi_mean: f32 = isi.iter().sum::<f32>() / isi.len() as f32;
                    let isi_var: f32 = isi.iter().map(|x| (x - isi_mean).powi(2)).sum::<f32>() / isi.len() as f32;
                    let cv = isi_var.sqrt() / (isi_mean + 1e-10);

                    // Burstiness: fraction of ISIs < 2×median
                    let mut sorted_isi = isi.clone();
                    sorted_isi.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = sorted_isi[sorted_isi.len() / 2];
                    let burst_thresh = 2.0 * median;
                    let in_burst = isi.iter().filter(|&&x| x < burst_thresh).count();
                    (cv, in_burst as f32 / isi.len() as f32)
                } else {
                    (0.0, 0.0)
                };

                // 2. Fano factor (spike count variability per window)
                let t_min = *local_ts.first().unwrap_or(&0);
                let t_max = *local_ts.last().unwrap_or(&1);
                let n_windows = ((t_max - t_min) / window_size).max(1) as usize;
                let mut window_counts = vec![0u32; n_windows];
                for &t in &local_ts {
                    let widx = ((t - t_min) / window_size) as usize;
                    if widx < n_windows { window_counts[widx] += 1; }
                }
                let wc_mean: f32 = window_counts.iter().sum::<u32>() as f32 / n_windows as f32;
                let wc_var: f32 = window_counts.iter()
                    .map(|&c| (c as f32 - wc_mean).powi(2)).sum::<f32>() / n_windows as f32;
                let fano = wc_var / (wc_mean + 1e-10);

                // 3. Channel balance
                let n_uv = local_sources.iter().filter(|&&s| s == 1).count() as f32;
                let n_lif = local_sources.iter().filter(|&&s| s == 2).count() as f32;
                let channel_balance = 1.0 - (n_uv - n_lif).abs() / (n_uv + n_lif + 1e-10);

                // 4. Dewetting fraction
                let dewetting = local_wd.iter().filter(|&&w| w < 0.01).count() as f32 / n as f32;

                // 5. Temporal persistence: fraction of windows with ≥1 spike
                let active_windows = window_counts.iter().filter(|&&c| c > 0).count();
                let persistence = active_windows as f32 / n_windows as f32;

                // ── Neuromorphic composite score ──
                // Rank-normalize each feature within this protein, then weight.
                // Stored as raw values; rank normalization happens below.
                let features = NeuroFeatures {
                    burst_fraction,
                    fano_factor: (fano + 1.0).ln(), // log-scale fano
                    channel_balance,
                    dewetting_fraction: dewetting,
                    temporal_persistence: persistence,
                    isi_cv: isi_cv.min(10.0),
                };

                // Weighted sum (these weights reflect what distinguishes
                // cryptic pockets from surface grooves, validated on 1P38):
                let raw_neuro =
                    0.20 * features.channel_balance +      // multi-channel convergence
                    0.20 * features.dewetting_fraction * 10.0 + // water displacement (scaled up)
                    0.15 * features.burst_fraction +        // bursty firing = collective motion
                    0.15 * features.fano_factor / 10.0 +   // super-Poisson = real pocket
                    0.15 * features.temporal_persistence +  // persists across time
                    0.15 * features.isi_cv / 10.0;         // irregular = complex dynamics

                neuro_scores.insert(site.cluster_id, raw_neuro);
            }

            // Rank-normalize neuro_scores within this protein [0, 1]
            let all_neuro: Vec<f32> = neuro_scores.values().copied().collect();
            let neuro_min = all_neuro.iter().cloned().fold(f32::MAX, f32::min);
            let neuro_max = all_neuro.iter().cloned().fold(f32::MIN, f32::max);
            let neuro_range = (neuro_max - neuro_min).max(1e-10);

            // TIEBREAKER MODE: neuro score only affects sites within 15% of rank-1.
            // If v7 clearly picks a winner (>15% gap to rank-2), trust it completely.
            // If the top sites are clustered (within 15%), use neuro to break the tie.
            // This prevents neuro from overriding confident v7 rankings while still
            // leveraging temporal signatures when v7 is ambiguous.
            {
                let mut qs: Vec<(i32, f32)> = clustered_sites.iter()
                    .map(|s| (s.cluster_id, s.quality_score)).collect();
                qs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let q_rank1 = qs[0].1;
                let tiebreak_threshold = q_rank1 * 0.85; // within 15% of rank-1

                // Only rerank sites within the tiebreak zone
                let in_zone: std::collections::HashSet<i32> = qs.iter()
                    .filter(|&&(_, q)| q >= tiebreak_threshold)
                    .map(|&(id, _)| id)
                    .collect();

                let n_in_zone = in_zone.len();

                if n_in_zone >= 2 {
                    // Apply neuro as tiebreaker within the zone
                    for site in clustered_sites.iter_mut() {
                        if in_zone.contains(&site.cluster_id) {
                            if let Some(&raw) = neuro_scores.get(&site.cluster_id) {
                                let neuro_norm = (raw - neuro_min) / neuro_range;
                                let old_q = site.quality_score;
                                // 15% neuro weight — just enough to reorder within the tie zone
                                site.quality_score = 0.85 * old_q + 0.15 * neuro_norm * q_rank1;
                            }
                        }
                    }

                    // Re-sort
                    clustered_sites.sort_by(|a, b|
                        b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
                }

                log::info!("[Neuro-rerank] Tiebreaker: {}/{} sites in zone (threshold={:.3}), reranked",
                    n_in_zone, clustered_sites.len(), tiebreak_threshold);
            }
        }

        // ---- Final ranking: Boltzmann thermodynamic OR quality_score ----
        let n_original_json = all_pockets_json.len();
        let mut ranked_indices: Vec<usize> = (0..clustered_sites.len().min(200)).collect();

        if args.boltzmann_rank {
            // Boltzmann ranking: compute ΔG from learned thermodynamic constants
            // and rank by Boltzmann probability P(i) = exp(-β*ΔG_i) / Z
            use prism_nhs::boltzmann_weights;
            log::info!("[Boltzmann Rank] Applying learned thermodynamic constants (β={:.2})", boltzmann_weights::BETA);

            let mut delta_gs: Vec<(usize, f32)> = ranked_indices.iter().map(|&i| {
                let s = &clustered_sites[i];
                let vol = s.estimated_volume.max(1.0);
                let n_lr = s.lining_residues.len();
                // Extract 15 features matching the JAX training order
                let features: [f32; 15] = [
                    (s.spike_count as f32).ln().max(0.0) / 14.0,  // H_spike_density
                    s.avg_intensity / 8.0,                         // H_burial_depth proxy
                    0.3,                                           // H_frustrated_water (default)
                    (n_lr as f32 / 20.0).min(1.0),                // H_lining_count
                    0.5,                                           // S_ray_entropy (computed in Cobb-Douglas but not stored)
                    0.5,                                           // S_spike_type_LPV
                    0.5,                                           // S_uv_enrichment
                    0.5,                                           // S_sphericity
                    0.5,                                           // W_activation
                    1.0 - (n_lr as f32 / vol.powf(0.667)).min(1.0), // W_enclosure
                    0.3,                                           // K_wavefront_coherence
                    0.1,                                           // K_funnel_ratio
                    0.1,                                           // K_breathing
                    0.5,                                           // C_vcs_orthogonal
                    s.quality_score.clamp(0.0, 2.0) / 2.0,       // C_quality_score
                ];
                let dg = boltzmann_weights::compute_delta_g(&features);
                (i, dg)
            }).collect();

            // Sort by ΔG ascending (most negative = most favorable)
            delta_gs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked_indices = delta_gs.iter().map(|&(i, _)| i).collect();

            // Update quality_score to reflect Boltzmann probability
            let dg_vals: Vec<f32> = delta_gs.iter().map(|&(_, dg)| dg).collect();
            let probs = boltzmann_weights::boltzmann_probabilities(&dg_vals);
            for (rank, (&(idx, _), &prob)) in delta_gs.iter().zip(probs.iter()).enumerate() {
                clustered_sites[idx].quality_score = prob;
            }

            log::info!("  Boltzmann ranking: top-1 P={:.4}, top-3 P={:.4}+{:.4}+{:.4}",
                probs.get(0).unwrap_or(&0.0),
                probs.get(0).unwrap_or(&0.0),
                probs.get(1).unwrap_or(&0.0),
                probs.get(2).unwrap_or(&0.0));
        } else {
            // Standard quality_score ranking
            ranked_indices.sort_by(|&a, &b|
                clustered_sites[b].quality_score
                    .partial_cmp(&clustered_sites[a].quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            );
        }

        // Reorder all three parallel vectors + sites
        let reordered_sites: Vec<_> = ranked_indices.iter().map(|&i| &clustered_sites[i]).collect();
        let empty_json = serde_json::json!({});
        let reordered_pockets: Vec<_> = ranked_indices.iter().map(|&i| {
            if i < n_original_json { all_pockets_json[i].clone() } else { empty_json.clone() }
        }).collect();
        let reordered_cryptic: Vec<_> = ranked_indices.iter().map(|&i| {
            if i < n_original_json { cryptic_sites_json[i].clone() } else { empty_json.clone() }
        }).collect();

        log::info!("Quality reranking applied. New top-3: {}",
            ranked_indices.iter().take(3)
                .map(|&i| format!("site{}(q={:.3})", clustered_sites[i].cluster_id, clustered_sites[i].quality_score))
                .collect::<Vec<_>>().join(", "));

        let json_path = output_base.with_extension("binding_sites.json");

        // Helper: map internal residue index → PDB resid
        let map_resid = |idx: i32| -> i32 {
            if idx >= 0 && (idx as usize) < pdb_id_map.len() {
                pdb_id_map[idx as usize] + 1
            } else {
                idx + 1
            }
        };

        // ── Per-site signal preservation aggregation (exact voxel_idx) ──
        // For each site, aggregate recurrence, UV→LIF causality, and residue identity
        // from the spike events using voxel_idx as the exact grid key.
        // ── Per-site signal preservation: aggregate GPU voxel grids per site ──
        // Uses exact voxel_idx from spikes + merged GPU signal grids (summed across streams)
        let site_signal_metrics: Vec<_> = reordered_sites.iter().map(|site| {
            use std::collections::HashSet;

            // Collect unique voxel indices for this site
            let mut site_voxels: HashSet<i32> = HashSet::new();
            for &idx in &site.spike_indices {
                if let Some(spike) = all_stream_spikes.get(idx) {
                    site_voxels.insert(spike.voxel_idx);
                }
            }
            let n_voxels = site_voxels.len() as u32;

            if let Some(ref sig) = merged_signal {
                // Aggregate from GPU-side grids (exact voxel-level data)
                let mut total_recurrence: i32 = 0;
                let mut max_recurrence: i32 = 0;
                let mut total_coupling: i32 = 0;
                let mut coupled_voxels: u32 = 0;
                let mut residue_counts: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();

                for &vid in &site_voxels {
                    let vi = vid as usize;
                    if vi >= sig.voxel_hit_grid.len() { continue; }

                    let rec = sig.voxel_hit_grid[vi];
                    total_recurrence += rec;
                    if rec > max_recurrence { max_recurrence = rec; }

                    let coup = sig.coupled_spike_grid[vi];
                    total_coupling += coup;
                    if coup > 0 { coupled_voxels += 1; }

                    let res_id = sig.primary_residue_id[vi];
                    let res_count = sig.primary_residue_count[vi];
                    if res_id >= 0 {
                        *residue_counts.entry(res_id).or_insert(0) += res_count;
                    }
                }

                let mean_recurrence = if n_voxels > 0 { total_recurrence as f32 / n_voxels as f32 } else { 0.0 };
                let causality_density = if n_voxels > 0 { coupled_voxels as f32 / n_voxels as f32 } else { 0.0 };

                let (primary_residue, primary_count) = residue_counts.iter()
                    .max_by_key(|&(_, &c)| c)
                    .map(|(&r, &c)| (r, c))
                    .unwrap_or((-1, 0));
                let total_residue_votes: i32 = residue_counts.values().sum();
                let residue_concentration = if total_residue_votes > 0 {
                    primary_count as f32 / total_residue_votes as f32
                } else { 0.0 };

                serde_json::json!({
                    "n_voxels": n_voxels,
                    "total_recurrence": total_recurrence,
                    "max_recurrence": max_recurrence,
                    "mean_recurrence": mean_recurrence,
                    "total_coupling": total_coupling,
                    "coupled_voxels": coupled_voxels,
                    "causality_density": causality_density,
                    "primary_residue_id": map_resid(primary_residue),
                    "primary_residue_count": primary_count,
                    "residue_concentration": residue_concentration,
                })
            } else {
                serde_json::json!({ "n_voxels": n_voxels, "error": "no_gpu_signal_data" })
            }
        }).collect();

        // ── Per-site KCC: top-K residue candidate evaluation ──
        // For each site, derive top-3 residues from per-voxel primary_residue_id_grid,
        // compute KCC confidence for each, pick the best candidate.
        let kcc_site_metrics: Vec<serde_json::Value> = reordered_sites.iter().map(|site| {
            use std::collections::HashMap;

            // Step 1: count residues by CAUSAL contribution within this site
            // w_i = # causally active voxels in site driven by residue i
            let mut residue_causal_counts: HashMap<i32, u32> = HashMap::new();
            let mut residue_voxel_counts: HashMap<i32, u32> = HashMap::new();
            if let Some(ref sig) = merged_signal {
                for &spike_idx in &site.spike_indices {
                    if let Some(spike) = all_stream_spikes.get(spike_idx) {
                        let vid = spike.voxel_idx as usize;
                        if vid < sig.primary_residue_id.len() {
                            let res_id = sig.primary_residue_id[vid];
                            if res_id >= 0 {
                                *residue_voxel_counts.entry(res_id).or_insert(0) += 1;
                                // Causal weight: sum actual coupling magnitude (not binary)
                                let coupling = sig.coupled_spike_grid[vid];
                                if coupling > 0 {
                                    *residue_causal_counts.entry(res_id).or_insert(0) += coupling as u32;
                                }
                            }
                        }
                    }
                }
            }

            if residue_voxel_counts.is_empty() {
                return serde_json::json!(null);
            }

            // Step 2: top-K=3 by descending support, then ascending residue ID for tie-break
            let total_voxel_support: u32 = residue_voxel_counts.values().sum();
            let mut candidates: Vec<(i32, u32)> = residue_voxel_counts.into_iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            candidates.truncate(3);

            let kcc_ref = match merged_kcc.as_ref() {
                Some(k) => k,
                None => return serde_json::json!(null),
            };

            // Step 3: compute KCC confidence for each candidate
            let mut candidate_ids: Vec<i32> = Vec::new();
            let mut candidate_support: Vec<f32> = Vec::new();
            let mut candidate_confidence: Vec<f32> = Vec::new();
            let mut candidate_temporal: Vec<f32> = Vec::new();
            let mut candidate_direction: Vec<f32> = Vec::new();
            let mut candidate_burst: Vec<f32> = Vec::new();
            let mut candidate_phase: Vec<f32> = Vec::new();
            let mut candidate_lag: Vec<f32> = Vec::new();
            let mut candidate_local_cov: Vec<f32> = Vec::new();

            let mut best_idx: usize = 0;
            let mut best_confidence: f32 = f32::NEG_INFINITY;

            for (ci, &(res_id, support)) in candidates.iter().enumerate() {
                let r = res_id as usize;
                if r >= kcc_ref.n_residues { continue; }

                let sup_frac = support as f32 / total_voxel_support.max(1) as f32;
                candidate_ids.push(res_id);
                candidate_support.push(sup_frac);

                let tc = kcc_ref.temporal_corr[r];
                let ds = kcc_ref.direction_score[r];
                let me = kcc_ref.motion_efficiency[r];
                let bm = kcc_ref.burst_motion[r];
                let ps = kcc_ref.phase_shift[r];
                let cl = kcc_ref.causal_lag[r];
                let lc = kcc_ref.local_cov[r];
                let lcp = kcc_ref.lag_corr_peak[r];

                candidate_temporal.push(tc);
                candidate_direction.push(ds);
                candidate_burst.push(bm);
                candidate_phase.push(ps);
                candidate_lag.push(cl);
                candidate_local_cov.push(lc);

                // KCC confidence: combines temporal + structural + causal evidence
                // Persistent sites (high causality, low burst) score via motion_efficiency
                // Transient sites (moderate causality, high burst) score via burst + local_cov
                let causal_frac = kcc_ref.active_causal[r] as f32
                    / kcc_ref.residue_count[r].max(1) as f32;
                let persistent_score = me * causal_frac;  // rewards continuous motion + causality
                let transient_score = bm.max(0.0).min(5.0) / 5.0 * lc.max(0.0).min(1.0);
                let confidence = persistent_score.max(transient_score)
                    * (1.0 + 0.1 * sup_frac);  // slight support boost, not dominant
                candidate_confidence.push(confidence);

                if confidence > best_confidence
                    || (confidence == best_confidence && lc > candidate_local_cov.get(best_idx).copied().unwrap_or(0.0))
                {
                    best_confidence = confidence;
                    best_idx = ci;
                }
            }

            if candidate_ids.is_empty() {
                return serde_json::json!(null);
            }

            let best_res = candidate_ids[best_idx];
            let br = best_res as usize;

            // Compute causal weights for weighted KCC aggregation
            let total_causal: u32 = candidate_ids.iter()
                .map(|&rid| residue_causal_counts.get(&rid).copied().unwrap_or(0))
                .sum();
            let causal_weights: Vec<f32> = candidate_ids.iter()
                .map(|&rid| {
                    if total_causal > 0 {
                        residue_causal_counts.get(&rid).copied().unwrap_or(0) as f32 / total_causal as f32
                    } else {
                        // Fallback to voxel support if no causal data
                        candidate_support[candidate_ids.iter().position(|&r| r == rid).unwrap_or(0)]
                    }
                }).collect();

            // Site-level weighted KCC: K(site) = Σ w_i * K(residue_i)
            let mut site_direction = 0.0f32;
            let mut site_motion_eff = 0.0f32;
            let mut site_burst = 0.0f32;
            let mut site_lag_corr = 0.0f32;
            let mut site_local_cov = 0.0f32;
            let mut site_causal_lag = 0.0f32;
            for (ci, &rid) in candidate_ids.iter().enumerate() {
                let w = causal_weights[ci];
                site_direction += w * candidate_direction[ci];
                site_motion_eff += w * kcc_ref.motion_efficiency.get(rid as usize).copied().unwrap_or(0.0);
                site_burst += w * candidate_burst[ci];
                site_lag_corr += w * kcc_ref.lag_corr_peak.get(rid as usize).copied().unwrap_or(0.0);
                site_local_cov += w * candidate_local_cov[ci];
                site_causal_lag += w * candidate_lag[ci];
            }

            serde_json::json!({
                // Site-level weighted KCC (for G×T×C×K formula)
                "driver_residue_id": map_resid(best_res),
                "site_direction_score": site_direction,
                "site_motion_efficiency": site_motion_eff,
                "site_burst_motion": site_burst,
                "site_lag_corr_peak": site_lag_corr,
                "site_local_cov": site_local_cov,
                "site_causal_lag": site_causal_lag,
                // Best candidate (preserved for debugging)
                "temporal_corr": kcc_ref.temporal_corr.get(br).copied().unwrap_or(0.0),
                "direction_score": kcc_ref.direction_score.get(br).copied().unwrap_or(0.0),
                "motion_efficiency": kcc_ref.motion_efficiency.get(br).copied().unwrap_or(0.0),
                "burst_motion": kcc_ref.burst_motion.get(br).copied().unwrap_or(0.0),
                "lag_corr_peak": kcc_ref.lag_corr_peak.get(br).copied().unwrap_or(0.0),
                "local_cov": kcc_ref.local_cov.get(br).copied().unwrap_or(0.0),
                "active_causal_steps": kcc_ref.active_causal.get(br).copied().unwrap_or(0),
                "total_steps": kcc_ref.residue_count.get(br).copied().unwrap_or(0),
                "kcc_confidence": best_confidence,
                "best_kcc_candidate_index": best_idx,
                // Causal weights used for site-level aggregation
                "candidate_causal_weights": causal_weights,
                // All candidates
                "candidate_residue_ids": candidate_ids.iter().map(|&r| map_resid(r)).collect::<Vec<i32>>(),
                "candidate_residue_support": candidate_support,
                "candidate_kcc_confidence": candidate_confidence,
                "candidate_kcc_direction_score": candidate_direction,
                "candidate_kcc_burst_motion": candidate_burst,
                "candidate_kcc_causal_lag": candidate_lag,
                "candidate_kcc_local_cov": candidate_local_cov,
            })
        }).collect();

        // ── Localization score L(s): causal signal concentration within site ──
        // L_raw = C_in / (C_in + C_out + ε) where C_in/C_out are coupling within/outside site
        let localization_data: Vec<(f32, f32)> = if let Some(ref sig) = merged_signal {
            // Build unique voxel position map (voxel_idx → [x,y,z])
            let mut voxel_positions: std::collections::HashMap<i32, [f32; 3]> = std::collections::HashMap::new();
            for spike in all_stream_spikes.iter() {
                voxel_positions.entry(spike.voxel_idx).or_insert(spike.position);
            }

            reordered_sites.iter().map(|site| {
                // Site's own voxel set
                let site_voxels: std::collections::HashSet<i32> = site.spike_indices.iter()
                    .filter_map(|&idx| all_stream_spikes.get(idx).map(|s| s.voxel_idx))
                    .collect();

                // C_in: coupling sum within site
                let c_in: i64 = site_voxels.iter()
                    .map(|&vid| sig.coupled_spike_grid.get(vid as usize).copied().unwrap_or(0) as i64)
                    .sum();

                // Neighborhood: all voxels within R=6Å of centroid, excluding site voxels
                let r = 6.0f32;
                let r2 = r * r;
                let cx = site.centroid[0];
                let cy = site.centroid[1];
                let cz = site.centroid[2];

                let mut c_out: i64 = 0;
                for (&vid, pos) in &voxel_positions {
                    if site_voxels.contains(&vid) { continue; }
                    let dx = pos[0] - cx;
                    let dy = pos[1] - cy;
                    let dz = pos[2] - cz;
                    if dx*dx + dy*dy + dz*dz <= r2 {
                        c_out += sig.coupled_spike_grid.get(vid as usize).copied().unwrap_or(0) as i64;
                    }
                }

                let l_raw = c_in as f32 / (c_in as f32 + c_out as f32 + 1e-6);
                (l_raw, c_in as f32)
            }).collect()
        } else {
            vec![(0.5, 0.0); reordered_sites.len()]
        };

        let raw_localization: Vec<f32> = localization_data.iter().map(|&(l, _)| l).collect();
        let loc_n = percentile_normalize(&raw_localization);

        // ── G×T×C×K Unified Rank Score ──
        // Normalize inputs per-protein using 5th-95th percentile scaling
        fn percentile_normalize(values: &[f32]) -> Vec<f32> {
            if values.is_empty() { return Vec::new(); }
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p5 = sorted[(sorted.len() as f32 * 0.05) as usize];
            let p95 = sorted[((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1)];
            let range = p95 - p5;
            if range < 1e-12 {
                return vec![0.5; values.len()];
            }
            values.iter().map(|&v| ((v - p5) / range).clamp(0.0, 1.0)).collect()
        }

        // Collect raw values for normalization
        let raw_volumes: Vec<f32> = reordered_sites.iter().map(|s| s.estimated_volume).collect();
        let raw_burial: Vec<f32> = reordered_sites.iter().enumerate().map(|(i, _)| {
            // burial_score from physics_signals
            let sid = reordered_sites[i].cluster_id;
            physics_signals.get(&sid).map(|p| p.3).unwrap_or(0.5)
        }).collect();
        let raw_sphericity: Vec<f32> = reordered_sites.iter().map(|s| {
            // compactness: volume / bounding_box_volume (higher = more compact)
            let bbox_vol = s.bounding_box[0] * s.bounding_box[1] * s.bounding_box[2];
            if bbox_vol > 0.1 { (s.estimated_volume / bbox_vol).clamp(0.0, 1.0) } else { 0.5 }
        }).collect();

        // Signal preservation raw values
        let raw_coupling_total: Vec<f32> = site_signal_metrics.iter().map(|sp| {
            sp.get("total_coupling").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32
        }).collect();
        let raw_causality_density: Vec<f32> = site_signal_metrics.iter().map(|sp| {
            sp.get("causality_density").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32
        }).collect();
        let raw_coupled_voxel_frac: Vec<f32> = site_signal_metrics.iter().map(|sp| {
            let cv = sp.get("coupled_voxels").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let nv = sp.get("n_voxels").and_then(|v| v.as_f64()).unwrap_or(1.0).max(1.0);
            (cv / nv) as f32
        }).collect();

        // Normalize
        let vol_n = percentile_normalize(&raw_volumes);
        let burial_n = percentile_normalize(&raw_burial);
        let spher_n = percentile_normalize(&raw_sphericity);
        let ctot_n = percentile_normalize(&raw_coupling_total);
        let cdens_n = percentile_normalize(&raw_causality_density);
        let cvox_n = percentile_normalize(&raw_coupled_voxel_frac);

        // Compute rank scores
        let rank_scores: Vec<(f32, f32, f32, f32, f32, f32, f32)> = (0..reordered_sites.len()).map(|i| {
            // G(s) = vol_n * (1 - 0.7 * hydro_n) * (0.6 + 0.4 * compact_n)
            let hydro_n = 1.0 - burial_n[i]; // low burial = high hydrophilicity
            let g = vol_n[i] * (1.0 - 0.7 * hydro_n) * (0.6 + 0.4 * spher_n[i]);

            // T(s) = 0.5 + 0.25 * theta + 0.25 * tau
            let therm_n = if let Some(ref analysis) = prism_therm_result {
                let sid = reordered_sites[i].cluster_id;
                analysis.sites.iter().find(|ts| ts.site_id == sid)
                    .map(|ts| match ts.therm_class.to_string().as_str() {
                        "CRYPTIC" => 1.0f32,
                        "ALLOSTERIC" => 0.9,
                        "SURFACE" => 0.6,
                        _ => 0.35,
                    })
                    .unwrap_or(0.5)
            } else { 0.5 };
            let ccns_n = reordered_sites[i].druggability.overall.clamp(0.0, 1.0);
            let t = 0.5 + 0.25 * therm_n + 0.25 * ccns_n;

            // C(s) = sqrt(0.45 * ctot + 0.35 * cdens + 0.20 * cvox)
            let c_raw = 0.45 * ctot_n[i] + 0.35 * cdens_n[i] + 0.20 * cvox_n[i];
            let c = c_raw.max(0.0).sqrt();

            // K(s) - two regime formulation using SITE-LEVEL weighted KCC
            let kcc = &kcc_site_metrics[i];
            let direction_n = kcc.get("site_direction_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let motion_eff_n = kcc.get("site_motion_efficiency").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let burst_n = (kcc.get("site_burst_motion").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32).clamp(0.0, 5.0) / 5.0;
            let lag_corr_n = kcc.get("site_lag_corr_peak").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let local_cov_n = kcc.get("site_local_cov").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let abs_lag_n = (kcc.get("site_causal_lag").and_then(|v| v.as_f64()).unwrap_or(0.0).abs() as f32 / 50.0).clamp(0.0, 1.0);
            let residue_valid = if kcc.get("driver_residue_id").and_then(|v| v.as_i64()).unwrap_or(-1) >= 0 { 1.0f32 } else { 0.0 };
            let residue_support = kcc.get("candidate_causal_weights")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    let best_idx = kcc.get("best_kcc_candidate_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    arr.get(best_idx).and_then(|v| v.as_f64())
                })
                .unwrap_or(0.0) as f32;

            // Flatness score for persistent mode
            let flatness = 1.0 - (burst_n + abs_lag_n + local_cov_n) / 3.0;

            // K_persist (reduced causality weight to 0.25)
            let k_persist = 0.25 * cdens_n[i] + 0.35 * direction_n + 0.25 * motion_eff_n + 0.15 * flatness;
            // K_trans (reduced causality weight to 0.15)
            let k_trans = 0.15 * cdens_n[i] + 0.25 * direction_n + 0.25 * burst_n + 0.20 * lag_corr_n + 0.15 * local_cov_n;

            let k = residue_valid * (0.85 + 0.15 * residue_support) * k_persist.max(k_trans);

            // Localization factor L' = 0.80 + 0.20 * L_norm
            let l_prime = 0.80 + 0.20 * loc_n[i];

            // Hard gates
            let gated = if cvox_n[i] <= 0.0 || residue_valid == 0.0 || g < 0.01 {
                0.0
            } else {
                g * t * c * k * l_prime
            };

            (gated, g, t, c, k, l_prime, raw_localization[i])
        }).collect();

        // Re-rank by rank_score (descending)
        let mut rank_order: Vec<usize> = (0..rank_scores.len()).collect();
        rank_order.sort_by(|&a, &b| rank_scores[b].0.partial_cmp(&rank_scores[a].0).unwrap_or(std::cmp::Ordering::Equal));

        log::info!("G×T×C×K×L ranking applied. Top-3:");
        for (rank, &idx) in rank_order.iter().take(3).enumerate() {
            let (score, g, t, c, k, l, _) = rank_scores[idx];
            log::info!("  #{}: site {} score={:.6} G={:.3} T={:.3} C={:.3} K={:.3} L={:.3}",
                rank + 1, reordered_sites[idx].cluster_id, score, g, t, c, k, l);
        }

        // Build per-site JSON, merging PRISM-Therm therm_class when available
        let mut ms_sites_json: Vec<serde_json::Value> = reordered_sites.iter().enumerate().map(|(site_rank, s)| {
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
                "signal_preservation": site_signal_metrics.get(site_rank).cloned()
                    .unwrap_or(serde_json::json!(null)),
                "kcc": kcc_site_metrics.get(site_rank).cloned()
                    .unwrap_or(serde_json::json!(null)),
                "rank_score": rank_scores[site_rank].0,
                "rank_G": rank_scores[site_rank].1,
                "rank_T": rank_scores[site_rank].2,
                "rank_C": rank_scores[site_rank].3,
                "rank_K": rank_scores[site_rank].4,
                "rank_L": rank_scores[site_rank].5,
                "localization_score_raw": rank_scores[site_rank].6,
                "gtck_rank": rank_order.iter().position(|&idx| idx == site_rank).map(|p| p + 1).unwrap_or(999),
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
                    // Phase fraction fields — expose the cold/hot spike breakdown
                    let total_phase = therm_site.heating_spike_count + therm_site.cooling_spike_count;
                    if total_phase > 0 {
                        let hot_frac = therm_site.heating_spike_count as f64 / total_phase as f64;
                        let cold_frac = therm_site.cooling_spike_count as f64 / total_phase as f64;
                        site_json["cold_phase_fraction"] = serde_json::json!({
                            "cold": cold_frac,
                            "hot": hot_frac,
                            "delta": hot_frac - cold_frac,
                            "heating_spike_count": therm_site.heating_spike_count,
                            "cooling_spike_count": therm_site.cooling_spike_count,
                            "heating_spike_rate": therm_site.heating_spike_rate,
                            "cooling_spike_rate": therm_site.cooling_spike_rate,
                        });
                    }
                    if therm_site.therm_class.to_string() == "CRYPTIC" {
                        site_json["classification"] = serde_json::Value::String("Cryptic".to_string());
                    }
                    // TIDE coupling score: overlap of top-5 trigger residues with lining residues.
                    // Exported per-site for downstream analysis and explicit solvent validation.
                    if !therm_site.tide_decomposition.is_empty() {
                        // Look up the site from reordered_sites by matching cluster_id
                        let site_cluster_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
                        if let Some(site_ref) = reordered_sites.iter().find(|s| s.cluster_id == site_cluster_id) {
                            let trigger_residues: std::collections::HashSet<i32> = therm_site.tide_decomposition.iter()
                                .take(5)
                                .map(|t| t.residue_id as i32)
                                .collect();
                            let lining_ids: std::collections::HashSet<i32> = site_ref.lining_residues.iter()
                                .map(|r| r.resid)
                                .collect();
                            let overlap = trigger_residues.intersection(&lining_ids).count();
                            let tide_coupling = overlap as f32 / trigger_residues.len().max(1) as f32;
                            site_json["tide_coupling_score"] = serde_json::json!(tide_coupling);
                            // Export trigger residue IDs for downstream analysis
                            let trigger_ids: Vec<u32> = therm_site.tide_decomposition.iter()
                                .take(5)
                                .map(|t| t.residue_id)
                                .collect();
                            site_json["tide_trigger_residues"] = serde_json::json!(trigger_ids);
                        }
                    }
                }
            }
        }

        // Rank-normalize wd_coherence (raw variance → 0.0–1.0) across all sites
        // before writing to JSON. The raw variance is tiny (1e-10 to 1e-4) and
        // useless for ranking; the rank-normalized version preserves relative order.
        let wd_raw_values: Vec<(i32, f32)> = spatial_signals.iter()
            .map(|(&id, &(_, wdc, _))| (id, wdc))
            .collect();
        let mut wd_normalized: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
        if wd_raw_values.len() > 1 {
            let min_w = wd_raw_values.iter().map(|(_, w)| *w).fold(f32::MAX, f32::min);
            let max_w = wd_raw_values.iter().map(|(_, w)| *w).fold(f32::MIN, f32::max);
            let range = (max_w - min_w).max(1e-10);
            for &(id, w) in &wd_raw_values {
                wd_normalized.insert(id, ((w - min_w) / range).clamp(0.0, 1.0));
            }
        } else {
            for &(id, _) in &wd_raw_values {
                wd_normalized.insert(id, 0.5);
            }
        }

        // Inject spatial signals (sphericity, wd_coherence, breathing) into each site JSON
        for site_json in ms_sites_json.iter_mut() {
            let site_id = site_json["id"].as_i64().unwrap_or(-1) as i32;
            if let Some(&(sph, _wdc_raw, breath)) = spatial_signals.get(&site_id) {
                let wdc_norm = wd_normalized.get(&site_id).copied().unwrap_or(0.5);
                site_json["sphericity"] = serde_json::json!(sph);
                site_json["wd_coherence"] = serde_json::json!(wdc_norm);
                site_json["breathing_score"] = serde_json::json!(breath);
            }
            if let Some(&fs_score) = frustrated_solvent_scores.get(&site_id) {
                site_json["frustrated_solvent_score"] = serde_json::json!(fs_score);
            }
            if let Some(&asym) = asymmetry_scores.get(&site_id) {
                site_json["asymmetry_offset"] = serde_json::json!(asym);
            }
            if let Some(&rer) = ray_escape_scores.get(&site_id) {
                site_json["ray_escape_ratio"] = serde_json::json!(rer);
            }
            // Export Cobb-Douglas engine scores for Boltzmann training
            if let Some(&(geo, chem, phys, vcs)) = engine_scores.get(&site_id) {
                site_json["engine_geo"] = serde_json::json!(geo);
                site_json["engine_chem"] = serde_json::json!(chem);
                site_json["engine_phys"] = serde_json::json!(phys);
                site_json["engine_vcs"] = serde_json::json!(vcs);
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

        // ── V3 COMPOSITE RANKER ──
        // Additive weighted sum of 10 min-max normalized physics signals.
        // Weights frozen from bench10 correlation analysis. No ML, no training —
        // just physics observables weighted by their correlation with binding.
        {
            // Weights (sum ≈ 1.0)
            const W_BREATHING: f64     = 0.14752;
            const W_DIRECTION: f64     = 0.13121;
            const W_CAUSAL_FRAC: f64   = 0.12965;
            const W_OLD_RANK: f64      = 0.12034;
            const W_LAG_CORR: f64      = 0.11102;
            const W_SRC_DIV: f64       = 0.10637;
            const W_VOLUME: f64        = 0.10054;
            const W_BURIAL: f64        = 0.06366;
            const W_DRUGGABILITY: f64  = 0.05279;
            const W_WD_COHERENCE: f64  = 0.03688;

            // Min-max normalization clamp
            fn norm(val: f64, lo: f64, hi: f64) -> f64 {
                if hi <= lo { return 0.5; }
                ((val - lo) / (hi - lo)).clamp(0.0, 1.0)
            }

            for sj in ms_sites_json.iter_mut() {
                let breathing   = sj.get("breathing_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let kcc         = sj.get("kcc").cloned().unwrap_or(serde_json::json!(null));
                let direction   = kcc.get("site_direction_score").and_then(|v| v.as_f64()).unwrap_or(0.98);
                let active_cs   = kcc.get("active_causal_steps").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let total_steps = kcc.get("total_steps").and_then(|v| v.as_f64()).unwrap_or(1.0).max(1.0);
                let causal_frac = active_cs / total_steps;
                let old_rank    = sj.get("rank_score").and_then(|v| v.as_f64()).unwrap_or(0.0).max(0.0);
                let lag_corr    = kcc.get("site_lag_corr_peak").and_then(|v| v.as_f64())
                    .or_else(|| kcc.get("lag_corr_peak").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                let src_div     = sj.get("source_diversity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let volume      = sj.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let burial      = sj.get("burial_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let druggability= sj.get("druggability").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let wd_coh      = sj.get("wd_coherence").and_then(|v| v.as_f64()).unwrap_or(0.0);

                let score =
                    W_BREATHING    * norm(breathing,   0.0, 0.75)
                  + W_DIRECTION    * norm(direction,   0.9, 1.0)
                  + W_CAUSAL_FRAC  * norm(causal_frac, 0.0, 0.5)
                  + W_OLD_RANK     * norm(old_rank,    0.0, 0.25)
                  + W_LAG_CORR     * norm(lag_corr,    0.0, 1.0)
                  + W_SRC_DIV      * norm(src_div,     0.0, 1.0)
                  + W_VOLUME       * norm(volume,      0.0, 3000.0)
                  + W_BURIAL       * norm(burial,      0.0, 1.0)
                  + W_DRUGGABILITY * norm(druggability, 0.0, 1.0)
                  + W_WD_COHERENCE * norm(wd_coh,      0.0, 1.0);

                sj["composite_v3_score"] = serde_json::json!(score);
                // quality_score now uses v3 composite for downstream consumers
                sj["quality_score"] = serde_json::json!(score);
            }

            // Sort by composite_v3_score descending
            ms_sites_json.sort_by(|a, b| {
                b["composite_v3_score"].as_f64().unwrap_or(0.0)
                    .partial_cmp(&a["composite_v3_score"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Assign v3 ranks
            for (i, sj) in ms_sites_json.iter_mut().enumerate() {
                sj["composite_v3_rank"] = serde_json::json!(i + 1);
            }

            log::info!("  V3 composite ranking applied. Top-3:");
            for sj in ms_sites_json.iter().take(3) {
                log::info!("    #{}: site {} v3={:.4} gtckl={:.4}",
                    sj["composite_v3_rank"].as_u64().unwrap_or(0),
                    sj["id"].as_i64().unwrap_or(0),
                    sj["composite_v3_score"].as_f64().unwrap_or(0.0),
                    sj["rank_score"].as_f64().unwrap_or(0.0));
            }
        }

        // ── COMPOSITE AUDIT RANKER (27 features, correlation-weighted) ──
        // Validated on 6 proteins: SR@1=50%, SR@3=67%.
        // Uses min-max normalization per protein, weighted sum / sum(|w|).
        {
            let n_sites = ms_sites_json.len();
            if n_sites > 0 {
                // Build per-residue active_causal lookup from merged KCC
                let mut res_active: std::collections::HashMap<i32, bool> = std::collections::HashMap::new();
                if let Some(ref kcc) = merged_kcc {
                    for r in 0..kcc.n_residues {
                        res_active.insert(r as i32, kcc.active_causal[r] > 0);
                    }
                }

                // Feature definition: (name, weight, extractor from serde_json::Value)
                type Ext = fn(&serde_json::Value, &std::collections::HashMap<i32, bool>) -> f64;
                let features: Vec<(&str, f64, Ext)> = vec![
                    ("sp_causality_density",       0.2058, |s,_| s.get("signal_preservation").and_then(|sp| sp.get("causality_density")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_driver_burst",            0.1836, |s,_| s.get("kcc").and_then(|k| k.get("burst_motion")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_driver_direction",         0.1776, |s,_| s.get("kcc").and_then(|k| k.get("direction_score")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("rank_K",                      0.1721, |s,_| s.get("rank_K").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_driver_motion_eff",      -0.1554, |s,_| s.get("kcc").and_then(|k| k.get("motion_efficiency")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("sp_mean_recurrence",          0.1480, |s,_| s.get("signal_preservation").and_then(|sp| sp.get("mean_recurrence")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("druggability",                0.1444, |s,_| s.get("druggability").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("breathing_score",             0.1347, |s,_| s.get("breathing_score").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("rank_T",                      0.1336, |s,_| s.get("rank_T").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_site_burst",              0.1326, |s,_| s.get("kcc").and_then(|k| k.get("site_burst_motion")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("engine_chem",                 0.1157, |s,_| s.get("engine_chem").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_temporal_corr",           0.1125, |s,_| s.get("kcc").and_then(|k| k.get("temporal_corr")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_site_direction",          0.1050, |s,_| s.get("kcc").and_then(|k| k.get("site_direction_score")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("frustrated_solvent",          0.1044, |s,_| s.get("frustrated_solvent_score").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("source_diversity",            0.1029, |s,_| s.get("source_diversity").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("burial_score",               -0.1027, |s,_| s.get("burial_score").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("mean_burial",                -0.1023, |s,_| s.get("mean_burial").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("sp_residue_concentration",    0.0973, |s,_| s.get("signal_preservation").and_then(|sp| sp.get("residue_concentration")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_confidence",              0.0932, |s,_| s.get("kcc").and_then(|k| k.get("kcc_confidence")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("res_frac_silent",            -0.0913, |s, ra| {
                        let rids = s.get("residue_ids").and_then(|v| v.as_array());
                        match rids {
                            Some(ids) if !ids.is_empty() => {
                                let n = ids.len() as f64;
                                let silent = ids.iter().filter(|id| {
                                    let rid = id.as_i64().unwrap_or(-1) as i32;
                                    !ra.get(&rid).copied().unwrap_or(false)
                                }).count() as f64;
                                silent / n
                            }
                            _ => 0.5
                        }
                    }),
                    ("aromatic_score",              0.0893, |s,_| s.get("aromatic_score").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("res_frac_active",             0.0862, |s, ra| {
                        let rids = s.get("residue_ids").and_then(|v| v.as_array());
                        match rids {
                            Some(ids) if !ids.is_empty() => {
                                let n = ids.len() as f64;
                                let active = ids.iter().filter(|id| {
                                    let rid = id.as_i64().unwrap_or(-1) as i32;
                                    ra.get(&rid).copied().unwrap_or(false)
                                }).count() as f64;
                                active / n
                            }
                            _ => 0.5
                        }
                    }),
                    ("kcc_site_motion_eff",        -0.0855, |s,_| s.get("kcc").and_then(|k| k.get("site_motion_efficiency")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("hysteresis_asymmetry",        0.0844, |s,_| s.get("hysteresis_asymmetry").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("rank_G",                     -0.0841, |s,_| s.get("rank_G").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("kcc_site_local_cov",          0.0809, |s,_| s.get("kcc").and_then(|k| k.get("site_local_cov")).and_then(|v| v.as_f64()).unwrap_or(0.0)),
                    ("relative_asymmetry",          0.0807, |s,_| s.get("relative_asymmetry").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                ];

                // Extract raw values
                let n_feat = features.len();
                let mut raw: Vec<Vec<f64>> = Vec::with_capacity(n_feat);
                for &(_, _, ext) in &features {
                    let vals: Vec<f64> = ms_sites_json.iter().map(|s| ext(s, &res_active)).collect();
                    raw.push(vals);
                }

                // Min-max per feature
                let mut mins = vec![f64::INFINITY; n_feat];
                let mut maxs = vec![f64::NEG_INFINITY; n_feat];
                for fi in 0..n_feat {
                    for &v in &raw[fi] {
                        if v < mins[fi] { mins[fi] = v; }
                        if v > maxs[fi] { maxs[fi] = v; }
                    }
                }

                // Score each site
                let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n_sites);
                for si in 0..n_sites {
                    let mut num = 0.0f64;
                    let mut den = 0.0f64;
                    for fi in 0..n_feat {
                        let range = maxs[fi] - mins[fi];
                        if range < 1e-15 { continue; }
                        let norm = (raw[fi][si] - mins[fi]) / range;
                        num += features[fi].1 * norm;
                        den += features[fi].1.abs();
                    }
                    let score = if den > 1e-15 { num / den } else { 0.0 };
                    scores.push((si, score));
                }

                // Sort descending, assign ranks
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (rank, &(si, score)) in scores.iter().enumerate() {
                    ms_sites_json[si]["composite_audit_score"] = serde_json::json!(score);
                    ms_sites_json[si]["composite_audit_rank"] = serde_json::json!(rank + 1);
                }

                // Re-sort sites_json by audit score for output
                ms_sites_json.sort_by(|a, b| {
                    b["composite_audit_score"].as_f64().unwrap_or(0.0)
                        .partial_cmp(&a["composite_audit_score"].as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                log::info!("  Composite audit ranking (27 features). Top-3:");
                for sj in ms_sites_json.iter().take(3) {
                    log::info!("    #{}: site {} audit={:.4} gtckl={:.4}",
                        sj["composite_audit_rank"].as_u64().unwrap_or(0),
                        sj["id"].as_i64().unwrap_or(0),
                        sj["composite_audit_score"].as_f64().unwrap_or(0.0),
                        sj["rank_score"].as_f64().unwrap_or(0.0));
                }
            }
        }

        // ── CRYPTIC-AWARE RERANKING ──
        // CRYPTIC therm_class pockets are systematically underranked by GTCK
        // because GTCK rewards persistence and spike count — properties that
        // cryptic sites lack by definition.  This adds a thermodynamic boost
        // to CRYPTIC sites based on hysteresis and asymmetry signals.
        {
            let n_sites = ms_sites_json.len();
            let mut cryptic_scores: Vec<(usize, f64)> = Vec::with_capacity(n_sites);

            for si in 0..n_sites {
                let sj = &ms_sites_json[si];
                let base = sj.get("rank_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let therm_class = sj.get("therm_class").and_then(|v| v.as_str()).unwrap_or("");
                let hysteresis = sj.get("hysteresis_asymmetry").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let rel_asym = sj.get("relative_asymmetry").and_then(|v| v.as_f64()).unwrap_or(0.0);

                let cryptic_boost = if therm_class == "CRYPTIC" {
                    let hyst_factor = hysteresis.max(0.0).min(1.0);
                    let asym_factor = (rel_asym / 3.0).max(0.0).min(1.0);
                    0.15 * hyst_factor + 0.05 * asym_factor
                } else {
                    0.0
                };

                cryptic_scores.push((si, base + cryptic_boost));
            }

            // Sort descending by cryptic-aware score
            cryptic_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, &(si, score)) in cryptic_scores.iter().enumerate() {
                ms_sites_json[si]["cryptic_score"] = serde_json::json!(score);
                ms_sites_json[si]["cryptic_rank"] = serde_json::json!(rank + 1);
            }

            log::info!("  CRYPTIC-aware ranking. Top-3:");
            let top3: Vec<_> = cryptic_scores.iter().take(3).collect();
            for &(si, score) in &top3 {
                let sj = &ms_sites_json[*si];
                log::info!("    cryptic_rank={} id={} therm={} score={:.4} gtck_rank={}",
                    sj["cryptic_rank"].as_u64().unwrap_or(0),
                    sj["id"].as_i64().unwrap_or(0),
                    sj.get("therm_class").and_then(|v| v.as_str()).unwrap_or("?"),
                    score,
                    sj["gtck_rank"].as_u64().unwrap_or(0));
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

        // ── KCC Visualization Export ──
        if let Some(ref kcc) = merged_kcc {
            let ca_pos: Vec<[f32; 3]> = topology.ca_indices.iter().map(|&ci| {
                if ci * 3 + 2 < topology.positions.len() {
                    [topology.positions[ci*3], topology.positions[ci*3+1], topology.positions[ci*3+2]]
                } else { [0.0; 3] }
            }).collect();
            let mut res_json = Vec::new();
            for r in 0..kcc.n_residues.min(ca_pos.len()) {
                // Emit ALL residues — inactive ones get zeroed causal fields.
                // Motion fields (sum_motion, net_dx/dy/dz, etc.) are populated
                // for every residue regardless of causal activity.
                // Composite KCC score: weighted combination of motion metrics
                // Higher = more causally active residue
                let lc = kcc.lag_corr_peak[r];
                let bm = kcc.burst_motion[r];
                let me = kcc.motion_efficiency[r];
                let ds = kcc.direction_score[r];
                let causality_frac = if kcc.residue_count[r] > 0 {
                    kcc.active_causal[r] as f32 / kcc.residue_count[r] as f32
                } else { 0.0 };
                let kcc_score = 0.3 * lc + 0.25 * causality_frac + 0.2 * bm.min(3.0) / 3.0
                    + 0.15 * me.min(0.01) / 0.01 + 0.1 * ds;
                // Map internal index → PDB resid via pdb_id_map
                res_json.push(serde_json::json!({
                    "residue_id": map_resid(r as i32),
                    "residue_name": topology.residues.get(r).map(|res| res.residue_name.clone()).unwrap_or_default(),
                    "kcc_score": kcc_score,
                    "ca_position": ca_pos[r],
                    "net_dx": kcc.net_dx[r], "net_dy": kcc.net_dy[r], "net_dz": kcc.net_dz[r],
                    "sum_motion": kcc.sum_m[r],
                    "motion_efficiency": kcc.motion_efficiency[r],
                    "direction_score": kcc.direction_score[r],
                    "burst_motion": kcc.burst_motion[r],
                    "lag_corr_peak": kcc.lag_corr_peak[r],
                    "local_cov": kcc.local_cov[r],
                    "causal_lag": kcc.causal_lag[r],
                    "active_causal_steps": kcc.active_causal[r],
                    "total_steps": kcc.residue_count[r],
                }));
            }
            let sites_viz: Vec<serde_json::Value> = ms_sites_json.iter().map(|sj| {
                serde_json::json!({
                    "id": sj.get("id"), "centroid": sj.get("centroid"),
                    "rank_score": sj.get("rank_score"), "gtck_rank": sj.get("gtck_rank"),
                    "rank_G": sj.get("rank_G"), "rank_T": sj.get("rank_T"),
                    "rank_C": sj.get("rank_C"), "rank_K": sj.get("rank_K"),
                    "rank_L": sj.get("rank_L"), "volume": sj.get("volume"),
                    "kcc": sj.get("kcc"),
                })
            }).collect();
            let viz = serde_json::json!({
                "pdb_source": &topology.source_pdb,
                "residues": res_json,
                "sites": sites_viz,
                "semantics": {
                    "vector": {
                        "dx_dy_dz": "net residue CA displacement accumulated over all simulation steps",
                        "length": "magnitude of displacement vector (Angstroms)",
                        "direction": "preferred direction of residue motion relative to initial position"
                    },
                    "color_mapping": {
                        "R": "lag_corr — temporal alignment between causal UV→LIF signal and residue motion",
                        "G": "causality — fraction of simulation steps with UV→LIF coupling at this residue",
                        "B": "burst — temporal clustering of causal events (dense vs sparse event spacing)"
                    },
                    "weight": {
                        "causal_weight": "normalized contribution of residue to site based on causal coupling magnitude"
                    },
                    "kcc": {
                        "lag_corr_peak": "peak cross-correlation between causal activity and motion at optimal timestep lag",
                        "burst_motion": "ratio of motion during dense causal events vs sparse events (>1 = bursty)",
                        "local_cov": "maximum local covariance of motion and causality in timestep-windowed subregions",
                        "motion_efficiency": "saturating transform of total motion magnitude (0-1)",
                        "direction_score": "ratio of net displacement to total path length (1.0 = perfectly directed, 0.0 = random walk)",
                        "causal_lag": "timestep offset that maximizes motion-causality cross-correlation (steps)"
                    },
                    "site": {
                        "G": "geometry score: volume × (1 - hydrophilicity) × compactness",
                        "T": "thermodynamic score: PRISM-Therm classification + druggability prior",
                        "C": "causal score: sqrt(coupling_total + causality_density + coupled_voxel_fraction)",
                        "K": "kinematic-causal coupling: max(persistent_mode, transient_mode) weighted over top-K residues",
                        "L": "localization: fraction of causal signal concentrated within site vs neighborhood (R=6A)"
                    }
                },
                "interpretation_guidelines": {
                    "high_value_residue": {
                        "criteria": ["high active_causal_steps", "high lag_corr_peak", "high motion_efficiency"],
                        "meaning": "residue motion is strongly driven by causal UV→LIF signal and likely functionally relevant to pocket mechanism"
                    },
                    "transient_site": {
                        "criteria": ["burst_motion > 1.5", "causal_lag != 0", "high localization"],
                        "meaning": "site opens dynamically through temporally localized events; may represent cryptic or induced-fit pocket"
                    },
                    "persistent_site": {
                        "criteria": ["burst_motion ≈ 1.0", "high causality_density", "high direction_score"],
                        "meaning": "site is continuously active with stable directional motion; characteristic of constitutive binding pockets"
                    },
                    "noise_pattern": {
                        "criteria": ["low active_causal_steps", "low lag_corr_peak", "low localization"],
                        "meaning": "likely non-functional surface noise or thermal fluctuation without mechanistic significance"
                    }
                },
                "vector_field_definition": {
                    "origin": "CA atom position of residue (from topology ca_indices)",
                    "direction": "net displacement vector (net_dx, net_dy, net_dz) accumulated over full simulation",
                    "length": "scaled motion magnitude (clamped at 8 Angstroms for visualization)",
                    "radius": "proportional to causal_weight (0.08 + 0.12 * causality_fraction)",
                    "color_encoding": "RGB = (lag_corr_peak, causality_fraction, burst_motion/5)",
                    "alpha": "proportional to KCC confidence of the residue"
                }
            });
            let vp = output_base.with_extension("kcc_visualization.json");
            if let Ok(f) = std::fs::File::create(&vp) {
                let _ = serde_json::to_writer_pretty(f, &viz);
                log::info!("  KCC visualization: {}", vp.display());
            }

            let kcc_ref_nr = merged_kcc.as_ref().map(|k| k.n_residues).unwrap_or(0);

            // ── KCC Validation JSON: synchronized human↔machine sanity checks ──
            let val_path = output_base.with_extension("kcc_validation.json");
            if let Ok(vf) = std::fs::File::create(&val_path) {
                let mut val_sites = Vec::new();
                for sj in ms_sites_json.iter() {
                    let sid = sj.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let rk = sj.get("gtck_rank").and_then(|v| v.as_u64()).unwrap_or(999);
                    if rk > 5 { continue; }
                    let rs = sj.get("rank_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let kcc = match sj.get("kcc") { Some(k) => k, None => continue };
                    let cids = kcc.get("candidate_residue_ids").and_then(|v| v.as_array());
                    let cids = match cids { Some(c) => c, None => continue };

                    // Build top-K residue entries with positions + KCC
                    let mut topk_entries = Vec::new();
                    let mut positions: Vec<[f64; 3]> = Vec::new();
                    let mut vectors: Vec<[f64; 3]> = Vec::new();
                    let mut signal_strengths: Vec<f64> = Vec::new();

                    for (ci, rid_val) in cids.iter().enumerate() {
                        let rid = rid_val.as_i64().unwrap_or(-1) as usize;
                        if rid >= ca_pos.len() || rid >= kcc_ref_nr { continue; }
                        let ca = ca_pos[rid];
                        let kcc_ref = merged_kcc.as_ref().unwrap();
                        let me = kcc_ref.motion_efficiency.get(rid).copied().unwrap_or(0.0) as f64;
                        let lc = kcc_ref.lag_corr_peak.get(rid).copied().unwrap_or(0.0) as f64;
                        let bm = kcc_ref.burst_motion.get(rid).copied().unwrap_or(0.0) as f64;
                        let lcv = kcc_ref.local_cov.get(rid).copied().unwrap_or(0.0) as f64;
                        let ndx = kcc_ref.net_dx.get(rid).copied().unwrap_or(0.0) as f64;
                        let ndy = kcc_ref.net_dy.get(rid).copied().unwrap_or(0.0) as f64;
                        let ndz = kcc_ref.net_dz.get(rid).copied().unwrap_or(0.0) as f64;

                        positions.push([ca[0] as f64, ca[1] as f64, ca[2] as f64]);
                        let mag = (ndx*ndx + ndy*ndy + ndz*ndz).sqrt();
                        if mag > 0.01 { vectors.push([ndx/mag, ndy/mag, ndz/mag]); }
                        signal_strengths.push(me * lc.abs().max(0.01));

                        topk_entries.push(serde_json::json!({
                            "residue_id": rid, "ca_position": [ca[0], ca[1], ca[2]],
                            "kcc": {"motion_efficiency": me, "lag_corr": lc, "burst": bm, "local_cov": lcv}
                        }));
                    }

                    // Structural sanity
                    let (struct_pass, mean_rad, max_dist, centroid) = if positions.len() >= 2 {
                        let cx = positions.iter().map(|p| p[0]).sum::<f64>() / positions.len() as f64;
                        let cy = positions.iter().map(|p| p[1]).sum::<f64>() / positions.len() as f64;
                        let cz = positions.iter().map(|p| p[2]).sum::<f64>() / positions.len() as f64;
                        let mr = positions.iter().map(|p| ((p[0]-cx).powi(2)+(p[1]-cy).powi(2)+(p[2]-cz).powi(2)).sqrt()).sum::<f64>() / positions.len() as f64;
                        let mut md = 0.0f64;
                        for i in 0..positions.len() { for j in i+1..positions.len() {
                            let d = ((positions[i][0]-positions[j][0]).powi(2)+(positions[i][1]-positions[j][1]).powi(2)+(positions[i][2]-positions[j][2]).powi(2)).sqrt();
                            if d > md { md = d; }
                        }}
                        (mr < 6.0 && md < 12.0, mr, md, [cx, cy, cz])
                    } else { (true, 0.0, 0.0, [0.0; 3]) };

                    // Vector sanity
                    let (vec_pass, mean_cos) = if vectors.len() >= 2 {
                        let mut sum_cos = 0.0f64; let mut n = 0u32;
                        for i in 0..vectors.len() { for j in i+1..vectors.len() {
                            sum_cos += vectors[i][0]*vectors[j][0] + vectors[i][1]*vectors[j][1] + vectors[i][2]*vectors[j][2];
                            n += 1;
                        }}
                        let mc = if n > 0 { sum_cos / n as f64 } else { 0.0 };
                        (mc > 0.5, mc)
                    } else { (true, 1.0) };

                    // Signal sanity
                    let mean_sig = if !signal_strengths.is_empty() { signal_strengths.iter().sum::<f64>() / signal_strengths.len() as f64 } else { 0.0 };
                    let vec_density = if cids.len() > 0 { vectors.len() as f64 / cids.len() as f64 } else { 0.0 };
                    let sig_pass = vec_density > 0.6;

                    let all_pass = struct_pass && vec_pass && sig_pass;
                    let verdict = if all_pass { "PASS" } else if struct_pass || vec_pass { "WARN" } else { "FAIL" };

                    val_sites.push(serde_json::json!({
                        "site_id": sid, "gtck_rank": rk, "rank_score": rs,
                        "topk_residues": topk_entries,
                        "validation": {
                            "structural": {"centroid": centroid, "mean_radius": mean_rad, "max_distance": max_dist, "pass": struct_pass},
                            "vector": {"mean_cosine_similarity": mean_cos, "pass": vec_pass},
                            "signal": {"mean_signal_strength": mean_sig, "vector_density": vec_density, "pass": sig_pass}
                        },
                        "verdict": verdict
                    }));
                }

                // Global checks
                let top_scores: Vec<f64> = val_sites.iter()
                    .filter_map(|s| s.get("rank_score").and_then(|v| v.as_f64()))
                    .collect();
                let sep = if top_scores.len() >= 2 { (top_scores[0] - top_scores[1]) / top_scores[0].max(1e-12) } else { 0.0 };

                let val_output = serde_json::json!({
                    "pdb_source": &topology.source_pdb,
                    "run_id": format!("{}_{}", structure_name, chrono::Utc::now().format("%Y%m%d_%H%M%S")),
                    "sites": val_sites,
                    "global_checks": {
                        "top1_vs_top2_separation": sep,
                        "n_validated_sites": val_sites.len(),
                    },
                    "semantics": {
                        "structural": "spatial clustering of driver residues — tight cluster = coherent pocket",
                        "vector": "alignment of residue motion directions — high cosine = coordinated movement",
                        "signal": "causal strength × temporal correlation — high = mechanistically driven",
                    },
                    "debug": {
                        "n_residues_tracked": kcc.n_residues,
                        "n_residues_with_causal": kcc.active_causal.iter().filter(|&&v| v > 0).count(),
                    }
                });
                let _ = serde_json::to_writer_pretty(vf, &val_output);
                log::info!("  KCC validation: {}", val_path.display());
            }
            // PyMOL session — regime-aware driver visualization
            // Detects shared vs unique drivers across top sites
            let pp = output_base.with_extension("kcc_session.pml");
            if let Ok(mut f) = std::fs::File::create(&pp) {
                use std::io::Write;

                // Determine driver regime: collect top-K residue sets per site
                let mut site_driver_sets: Vec<(i64, Vec<i64>)> = Vec::new(); // (site_id, [residue_ids])
                for sj in ms_sites_json.iter() {
                    let rk = sj.get("gtck_rank").and_then(|v| v.as_u64()).unwrap_or(999);
                    if rk > 5 { continue; }
                    let sid = sj.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let cids: Vec<i64> = sj.get("kcc")
                        .and_then(|k| k.get("candidate_residue_ids"))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                        .unwrap_or_default();
                    site_driver_sets.push((sid, cids));
                }

                // Classify: identical / overlapping / unique
                let driver_regime = if site_driver_sets.len() >= 2 {
                    let ref_set: std::collections::HashSet<i64> = site_driver_sets[0].1.iter().copied().collect();
                    let all_identical = site_driver_sets[1..].iter().all(|(_, s)| {
                        let ss: std::collections::HashSet<i64> = s.iter().copied().collect();
                        ss == ref_set
                    });
                    if all_identical {
                        "global"  // all sites share exact same drivers
                    } else {
                        // Check overlap
                        let min_overlap = site_driver_sets[1..].iter().map(|(_, s)| {
                            let ss: std::collections::HashSet<i64> = s.iter().copied().collect();
                            let intersection = ref_set.intersection(&ss).count();
                            let union = ref_set.union(&ss).count();
                            if union > 0 { intersection as f64 / union as f64 } else { 0.0 }
                        }).fold(f64::MAX, f64::min);
                        if min_overlap > 0.6 { "hybrid" } else { "local" }
                    }
                } else { "local" };

                writeln!(f, "# PRISM4D KCC Session — {} driver regime", driver_regime.to_uppercase()).ok();
                writeln!(f, "# Auto-generated (deterministic)").ok();
                writeln!(f, "").ok();
                writeln!(f, "hide all").ok();
                writeln!(f, "show cartoon").ok();
                writeln!(f, "color gray80, all").ok();
                writeln!(f, "set cartoon_transparency, 0.7").ok();
                writeln!(f, "").ok();

                // Global/shared drivers (for global or hybrid regime)
                let global_driver_ids: Vec<i64> = if driver_regime != "local" {
                    // Shared core = intersection of all site driver sets
                    if let Some((_, first)) = site_driver_sets.first() {
                        let mut shared: std::collections::HashSet<i64> = first.iter().copied().collect();
                        for (_, s) in &site_driver_sets[1..] {
                            let ss: std::collections::HashSet<i64> = s.iter().copied().collect();
                            shared = shared.intersection(&ss).copied().collect();
                        }
                        let mut v: Vec<i64> = shared.into_iter().collect();
                        v.sort();
                        v
                    } else { Vec::new() }
                } else { Vec::new() };

                // Stamp global_residue_id into B-factor for ALL residues (identity layer)
                // This embeds topology identity globally before any selection
                writeln!(f, "# === GLOBAL IDENTITY STAMPING ===").ok();
                writeln!(f, "# Stamp topology global_residue_id into B-factor + Q-factor for ALL atoms").ok();
                for (r, ca) in ca_pos.iter().enumerate() {
                    writeln!(f, "select _id_tmp, (name CA) within 2.0 of [{:.2},{:.2},{:.2}]", ca[0], ca[1], ca[2]).ok();
                    writeln!(f, "alter (byres _id_tmp), b={}; alter (byres _id_tmp), q={}", r, r).ok();
                    writeln!(f, "delete _id_tmp").ok();
                }
                writeln!(f, "").ok();

                if !global_driver_ids.is_empty() {
                    writeln!(f, "# === GLOBAL DISTRIBUTED DRIVERS ===").ok();
                    writeln!(f, "select global_kcc_drivers, none").ok();
                    for &rid in &global_driver_ids {
                        let r = rid as usize;
                        if r >= ca_pos.len() { continue; }
                        let ca = ca_pos[r];
                        let rname = topology.residues.get(r).map(|res| res.residue_name.as_str()).unwrap_or("UNK");
                        // 2.0Å tolerance for robust CA matching
                        writeln!(f, "select _tmp, (name CA) within 2.0 of [{:.2},{:.2},{:.2}]", ca[0], ca[1], ca[2]).ok();
                        writeln!(f, "select global_kcc_drivers, global_kcc_drivers or (byres _tmp)").ok();
                        writeln!(f, "delete _tmp").ok();
                        writeln!(f, "# {} {} (global driver)", rname, rid).ok();
                    }
                    let color = if driver_regime == "global" { "orange" } else { "yellow" };
                    writeln!(f, "color {}, global_kcc_drivers", color).ok();
                    writeln!(f, "show sticks, global_kcc_drivers").ok();
                    writeln!(f, "").ok();
                }

                // Site spheres
                writeln!(f, "# === SITE SPHERES ===").ok();
                let mut site_group_names: Vec<String> = Vec::new();
                for sj in ms_sites_json.iter() {
                    let sid = sj.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let rk = sj.get("gtck_rank").and_then(|v| v.as_u64()).unwrap_or(999);
                    if rk > 5 { continue; }
                    let c = match sj.get("centroid").and_then(|v| v.as_array()) {
                        Some(c) => c.clone(), None => continue,
                    };
                    let cx = c[0].as_f64().unwrap_or(0.0);
                    let cy = c[1].as_f64().unwrap_or(0.0);
                    let cz = c[2].as_f64().unwrap_or(0.0);
                    let sc = sj.get("rank_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let vol = sj.get("volume").and_then(|v| v.as_f64()).unwrap_or(500.0);
                    let rad = (vol * 3.0 / (4.0 * std::f64::consts::PI)).cbrt();
                    writeln!(f, "pseudoatom site_{}, pos=[{:.2},{:.2},{:.2}], vdw={:.1}, label=\"R{} S{} ({:.4})\"",
                        sid, cx, cy, cz, rad, rk, sid, sc).ok();
                    writeln!(f, "show sphere, site_{}",  sid).ok();
                    writeln!(f, "set sphere_transparency, 0.6, site_{}", sid).ok();

                    // Per-site unique drivers (hybrid/local only)
                    if driver_regime == "local" || driver_regime == "hybrid" {
                        let site_cids: Vec<i64> = sj.get("kcc")
                            .and_then(|k| k.get("candidate_residue_ids"))
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                            .unwrap_or_default();
                        let unique_cids: Vec<i64> = site_cids.iter()
                            .filter(|r| !global_driver_ids.contains(r))
                            .copied().collect();
                        if !unique_cids.is_empty() {
                            writeln!(f, "select site_{}_local_drivers, none", sid).ok();
                            for &rid in &unique_cids {
                                let r = rid as usize;
                                if r >= ca_pos.len() { continue; }
                                let ca = ca_pos[r];
                                writeln!(f, "select _tmp, (name CA) within 2.0 of [{:.2},{:.2},{:.2}]", ca[0], ca[1], ca[2]).ok();
                                writeln!(f, "select site_{}_local_drivers, site_{}_local_drivers or (byres _tmp)", sid, sid).ok();
                                writeln!(f, "delete _tmp").ok();
                            }
                            writeln!(f, "color cyan, site_{}_local_drivers", sid).ok();
                            writeln!(f, "show sticks, site_{}_local_drivers", sid).ok();
                        }
                    }

                    // Group: site sphere (+ local drivers if any, + global drivers)
                    let gname = format!("site_{}_full", sid);
                    if driver_regime == "global" {
                        writeln!(f, "group {}, site_{} global_kcc_drivers", gname, sid).ok();
                    } else if driver_regime == "hybrid" {
                        writeln!(f, "group {}, site_{} global_kcc_drivers site_{}_local_drivers", gname, sid, sid).ok();
                    } else {
                        writeln!(f, "group {}, site_{} site_{}_local_drivers", gname, sid, sid).ok();
                    }
                    site_group_names.push(gname);
                }
                writeln!(f, "group kcc_sites, site_*").ok();
                writeln!(f, "").ok();

                // Vectors + commands via Python CGO + cmd.extend
                writeln!(f, "# === KCC VECTORS + COMMANDS ===").ok();
                writeln!(f, "python").ok();
                writeln!(f, "from pymol.cgo import *").ok();
                writeln!(f, "from pymol import cmd").ok();
                writeln!(f, "import json").ok();
                writeln!(f, "").ok();
                writeln!(f, "with open(r'{}') as fh:", vp.display()).ok();
                writeln!(f, "    viz = json.load(fh)").ok();
                writeln!(f, "residues = viz.get('residues', [])").ok();
                writeln!(f, "if residues:").ok();
                writeln!(f, "    max_ac = max(r['active_causal_steps'] for r in residues) or 1").ok();
                writeln!(f, "    vecs = []").ok();
                writeln!(f, "    for r in residues:").ok();
                writeln!(f, "        ca = r['ca_position']").ok();
                writeln!(f, "        dx, dy, dz = r['net_dx'], r['net_dy'], r['net_dz']").ok();
                writeln!(f, "        mag = (dx**2 + dy**2 + dz**2)**0.5").ok();
                writeln!(f, "        if mag < 0.01: continue").ok();
                writeln!(f, "        sc = min(8.0, mag * 2.0) / (mag + 1e-6)").ok();
                writeln!(f, "        lc = min(1.0, max(0.0, r.get('lag_corr_peak', 0)))").ok();
                writeln!(f, "        cf = r['active_causal_steps'] / max_ac").ok();
                writeln!(f, "        bn = min(1.0, max(0.0, r.get('burst_motion', 0)) / 5.0)").ok();
                writeln!(f, "        rad = 0.08 + 0.12 * cf").ok();
                writeln!(f, "        vecs.extend([CYLINDER, ca[0],ca[1],ca[2], ca[0]+dx*sc,ca[1]+dy*sc,ca[2]+dz*sc, rad, lc,cf,bn, lc,cf,bn])").ok();
                writeln!(f, "        vecs.extend([CONE, ca[0]+dx*sc,ca[1]+dy*sc,ca[2]+dz*sc, ca[0]+dx*sc*1.15,ca[1]+dy*sc*1.15,ca[2]+dz*sc*1.15, rad*2,0.0, lc,cf,bn, lc,cf,bn, 1.0,1.0])").ok();
                writeln!(f, "    cmd.load_cgo(vecs, 'kcc_vectors')").ok();
                writeln!(f, "").ok();
                // === Runtime verification: prove PyMOL selection matches JSON ===
                // Inject expected driver IDs as literals (not from viz variable)
                {
                    // Verification: read back B-factors from global_kcc_drivers
                    // B-factors already stamped globally — verification is independent of selection
                    let ver_path = output_base.with_extension("kcc_pymol_verification.txt");
                    let expected_ids: Vec<String> = global_driver_ids.iter().map(|r| r.to_string()).collect();
                    let expected_list = expected_ids.join(", ");

                    writeln!(f, "# Identity verification: read topology ID from B-factor (stamped globally)").ok();
                    writeln!(f, "try:").ok();
                    writeln!(f, "    _expected = sorted([{}])", expected_list).ok();
                    writeln!(f, "    _model = cmd.get_model('global_kcc_drivers and name CA')").ok();
                    writeln!(f, "    _n_atoms = len(cmd.get_model('global_kcc_drivers').atom)").ok();
                    writeln!(f, "    _observed = sorted(set(int(a.b) for a in _model.atom))").ok();
                    writeln!(f, "    _pass = (_n_atoms > 0) and (_observed == _expected)").ok();
                    writeln!(f, "    with open(r'{}', 'w') as _vf:", ver_path.display()).ok();
                    writeln!(f, "        _vf.write('=== KCC PyMOL Identity Verification ===\\n')").ok();
                    writeln!(f, "        _vf.write('method: topology global_residue_id stamped into B-factor\\n')").ok();
                    writeln!(f, "        _vf.write('expected_global_ids: %s\\n' % str(_expected))").ok();
                    writeln!(f, "        _vf.write('observed_global_ids: %s\\n' % str(_observed))").ok();
                    writeln!(f, "        _vf.write('total_atoms_selected: %d\\n' % _n_atoms)").ok();
                    writeln!(f, "        _vf.write('exact_match: %s\\n' % ('PASS' if _pass else 'FAIL'))").ok();
                    writeln!(f, "    print('KCC verify: expected=%s observed=%s atoms=%d -> %s' % (str(_expected), str(_observed), _n_atoms, 'PASS' if _pass else 'FAIL'))").ok();
                    writeln!(f, "except Exception as e:").ok();
                    writeln!(f, "    print('KCC verification error: %s' % str(e))").ok();
                    writeln!(f, "").ok();
                }
                // cmd.extend commands (reliable multi-command execution)
                for (i, gname) in site_group_names.iter().enumerate() {
                    writeln!(f, "def _show_site{}(self=None):", i).ok();
                    writeln!(f, "    cmd.disable('all')").ok();
                    writeln!(f, "    cmd.enable('{}')", gname).ok();
                    writeln!(f, "    cmd.enable('kcc_vectors')").ok();
                    writeln!(f, "    cmd.show('cartoon')").ok();
                    writeln!(f, "    cmd.set('cartoon_transparency', 0.3)").ok();
                    writeln!(f, "    cmd.zoom('{}', 10)", gname).ok();
                    writeln!(f, "cmd.extend('show_site{}', _show_site{})", i, i).ok();
                    writeln!(f, "cmd.extend('inspect_site{}', _show_site{})", i, i).ok();
                }
                if site_group_names.len() >= 2 {
                    writeln!(f, "def _compare_top2(self=None):").ok();
                    writeln!(f, "    cmd.disable('all')").ok();
                    writeln!(f, "    cmd.enable('{}')", site_group_names[0]).ok();
                    writeln!(f, "    cmd.enable('{}')", site_group_names[1]).ok();
                    writeln!(f, "    cmd.enable('kcc_vectors')").ok();
                    writeln!(f, "    cmd.show('cartoon')").ok();
                    writeln!(f, "    cmd.set('cartoon_transparency', 0.3)").ok();
                    writeln!(f, "    cmd.zoom('all')").ok();
                    writeln!(f, "cmd.extend('compare_top2', _compare_top2)").ok();
                }
                writeln!(f, "def _show_all(self=None):").ok();
                writeln!(f, "    cmd.enable('all')").ok();
                writeln!(f, "    cmd.zoom('all')").ok();
                writeln!(f, "cmd.extend('show_all', _show_all)").ok();
                writeln!(f, "python end").ok();
                writeln!(f, "").ok();

                // Default view
                writeln!(f, "bg_color white").ok();
                writeln!(f, "set ray_opaque_background, 0").ok();
                writeln!(f, "set depth_cue, 0").ok();
                if let Some(top) = site_group_names.first() {
                    writeln!(f, "disable all").ok();
                    writeln!(f, "enable {}", top).ok();
                    writeln!(f, "enable kcc_vectors").ok();
                    writeln!(f, "show cartoon").ok();
                    writeln!(f, "set cartoon_transparency, 0.7").ok();
                    writeln!(f, "zoom {}, 10", top).ok();
                }

                log::info!("  KCC PyMOL: {} (regime={}, commands: show_site0..show_site{})",
                    pp.display(), driver_regime, site_group_names.len().saturating_sub(1));
            }
        }

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

    // ── Ensemble trajectory JSON summary ──
    // Always written after multi-stream runs. Contains per-stream spike counts,
    // consensus site list, and aggregate statistics for postflight validation.
    {
        let traj_json_path = output_base.with_extension("ensemble_trajectory.json");
        let total_spikes = all_stream_spikes.len() as u64;
        let consensus_site_ids: Vec<i32> = clustered_sites.iter().map(|s| s.cluster_id).collect();
        let traj_json = serde_json::json!({
            "structure": &structure_name,
            "n_streams": n_streams,
            "total_spikes": total_spikes,
            "n_consensus_sites": clustered_sites.len(),
            "consensus_site_ids": consensus_site_ids,
            "per_stream": per_stream_stats,
        });
        if let Ok(f) = std::fs::File::create(&traj_json_path) {
            let _ = serde_json::to_writer_pretty(f, &traj_json);
            log::info!("  Ensemble trajectory JSON: {}", traj_json_path.display());
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
        // Compute stream_id for each spike from stream_spike_offsets (binary search)
        let spike_stream_id = |flat_idx: usize| -> usize {
            stream_spike_offsets.partition_point(|&off| off <= flat_idx).saturating_sub(1)
        };
        let lining_cutoff = args.lining_cutoff;
        for site in &clustered_sites {
            let site_radius = lining_cutoff + 2.0;
            let cx = site.centroid[0];
            let cy = site.centroid[1];
            let cz = site.centroid[2];
            // Collect raw spikes for this site (with flat index for stream_id lookup)
            let raw_site_spikes: Vec<_> = all_stream_spikes.iter().enumerate()
                .filter(|(_, s)| {
                    let dx = s.position[0] - cx;
                    let dy = s.position[1] - cy;
                    let dz = s.position[2] - cz;
                    (dx*dx + dy*dy + dz*dz).sqrt() <= site_radius
                })
                .collect();
            // Compute open_frequency: fraction of simulation frames with spike activity
            // Use frame_index (timestep / 1000) from actual spike data
            let unique_frames: std::collections::HashSet<i32> = raw_site_spikes.iter()
                .map(|(_, s)| s.timestep / 1000)
                .collect();
            let max_frame = raw_site_spikes.iter().map(|(_, s)| s.timestep / 1000).max().unwrap_or(0);
            let total_frames = (max_frame + 1).max(1) as f32;
            let open_frequency = unique_frames.len() as f32 / total_frames;
            let site_spikes: Vec<serde_json::Value> = raw_site_spikes.iter()
                .map(|(idx, s)| {
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
                        "spike_source": match src { 1 => "UV", 3 => "EFP", 4 => "LADD", 5 => "COFIRE", _ => "LIF" },
                        "aromatic_residue_id": arom_res,
                        "water_density": wd,
                        "vibrational_energy": ve,
                        "n_nearby_excited": nne,
                        "timestep": ts,
                        "frame_index": ts / 1000,
                        "ccns_phase": phase_label(ts),
                        "stream_id": spike_stream_id(*idx),
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

        // Stage 1: Burial-weighted centroid
        // Spikes with more nearby residues (deeper in pocket) get weight = n_residues²
        let mut weighted_centroid = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        for (_, spike) in &spikes {
            let pos = spike.position;
            let intensity = spike.intensity;

            // Burial weight: n_residues² (buried spikes count more)
            let burial_weight = (spike.n_residues as f32).max(1.0).powi(2);
            weighted_centroid[0] += pos[0] * burial_weight;
            weighted_centroid[1] += pos[1] * burial_weight;
            weighted_centroid[2] += pos[2] * burial_weight;
            total_weight += burial_weight;

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

        // Use burial-weighted centroid if burial data is present
        if total_weight > n {
            centroid[0] = weighted_centroid[0] / total_weight;
            centroid[1] = weighted_centroid[1] / total_weight;
            centroid[2] = weighted_centroid[2] / total_weight;
        } else {
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;
        }

        // Stage 2: Spike density KDE peak refinement
        // Find the local maximum of spike density near the burial centroid
        if spikes.len() >= 20 {
            let bandwidth = 2.0f32;
            let bw2 = bandwidth * bandwidth;
            let search_radius = 5.0f32;
            let grid_step = 1.0f32;
            let mut best_density = 0.0f32;
            let mut peak_pos = centroid;

            let n_steps = (search_radius / grid_step) as i32;
            for ix in -n_steps..=n_steps {
                for iy in -n_steps..=n_steps {
                    for iz in -n_steps..=n_steps {
                        let px = centroid[0] + ix as f32 * grid_step;
                        let py = centroid[1] + iy as f32 * grid_step;
                        let pz = centroid[2] + iz as f32 * grid_step;

                        let mut density = 0.0f32;
                        for (_, spike) in &spikes {
                            let dx = px - spike.position[0];
                            let dy = py - spike.position[1];
                            let dz = pz - spike.position[2];
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 < 9.0 * bw2 {
                                // Weight by burial: buried spikes contribute more to density
                                let w = (spike.n_residues as f32).max(1.0);
                                density += w * (-r2 / (2.0 * bw2)).exp();
                            }
                        }

                        if density > best_density {
                            best_density = density;
                            peak_pos = [px, py, pz];
                        }
                    }
                }
            }

            // Blend: 70% density peak + 30% burial-weighted centroid
            centroid[0] = 0.7 * peak_pos[0] + 0.3 * centroid[0];
            centroid[1] = 0.7 * peak_pos[1] + 0.3 * centroid[1];
            centroid[2] = 0.7 * peak_pos[2] + 0.3 * centroid[2];
        }

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

/// Local physics signals computed from spikes within a radius of a site centroid.
/// Used by the re-reap pass to give sub-sites genuine physics data.
#[cfg(feature = "gpu")]
struct LocalPhysics {
    burial_score: f32,
    mean_burial: f32,
    onset_score: f32,
    source_diversity: f32,
    source_entropy: f32,
    sphericity: f32,
    wd_coherence: f32,
    breathing_score: f32,
    uv_enrichment: f32,
    per_spike_quality: f32,
    n_local_spikes: usize,
    log_spike_norm: f32,
    lining_score: f32,
    frustrated_solvent_score: f32,  // ΔG_solvation proxy
    asymmetry_offset: f32,         // |CoM_spikes - centroid| — "cup" metric
    ray_escape_ratio: f32,         // Dmax/Dmin from rays — "mouth" metric
    source_counts: [u32; 4],       // [unknown, UV, LIF, EFP] raw spike counts
}

/// Compute physics signals for a site centroid from the LOCAL spike cloud.
/// Queries all spikes within `radius` of the centroid and computes:
/// burial_score, onset_score, source_diversity, source_entropy,
/// sphericity, wd_coherence, breathing_score, uv_enrichment, per_spike_quality.
#[cfg(feature = "gpu")]
fn compute_local_physics(
    centroid: [f32; 3],
    all_spikes: &[prism_nhs::fused_engine::GpuSpikeEvent],
    radius: f32,
    n_lining: usize,
    uv_burst_interval: i32,
    uv_burst_duration: i32,
) -> LocalPhysics {
    let radius_sq = radius * radius;

    // Collect local spikes within radius
    let local_spikes: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> = all_spikes.iter()
        .filter(|s| {
            let dx = s.position[0] - centroid[0];
            let dy = s.position[1] - centroid[1];
            let dz = s.position[2] - centroid[2];
            dx * dx + dy * dy + dz * dz <= radius_sq
        })
        .collect();

    let n = local_spikes.len();
    if n == 0 {
        return LocalPhysics {
            burial_score: 0.0, mean_burial: 0.0, onset_score: 0.0,
            source_diversity: 0.0, source_entropy: 0.0, sphericity: 0.0,
            wd_coherence: 0.0, breathing_score: 0.0, uv_enrichment: 0.0,
            per_spike_quality: 0.0, n_local_spikes: 0, log_spike_norm: 0.0,
            lining_score: 0.0, frustrated_solvent_score: 0.0,
            asymmetry_offset: 0.0, ray_escape_ratio: 0.0,
            source_counts: [0; 4],
        };
    }

    // ── Burial score: sigmoid(mean(n_residues), center=3.0, slope=2.0) ──
    let mean_burial: f32 = local_spikes.iter()
        .map(|s| s.n_residues as f32).sum::<f32>() / n as f32;
    let burial_score = 1.0 / (1.0 + (-2.0 * (mean_burial - 3.0)).exp());

    // ── Onset score: 1.0 - (median_timestep / max_timestep) ──
    let onset_score = {
        let mut timesteps: Vec<i32> = local_spikes.iter().map(|s| s.timestep).collect();
        timesteps.sort();
        let median_ts = timesteps[timesteps.len() / 2];
        let total_steps = timesteps.last().copied().unwrap_or(1).max(1);
        let onset_fraction = median_ts as f32 / total_steps as f32;
        (1.0 - onset_fraction).clamp(0.0, 1.0)
    };

    // ── Source diversity: UV/LIF balance ratio + EFP bonus ──
    let mut local_source_counts = [0u32; 4]; // 0=unknown, 1=UV, 2=LIF, 3=EFP
    let (source_diversity, source_entropy) = {
        for s in &local_spikes {
            let src = (s.spike_source as usize).min(3);
            local_source_counts[src] += 1;
        }
        let total = n as f32;
        let entropy: f32 = local_source_counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f32 / total;
                -p * p.ln()
            })
            .sum();

        let uv_count = local_source_counts[1] as f32;
        let lif_count = local_source_counts[2] as f32;
        let efp_count = local_source_counts[3] as f32;
        let ul_total = (uv_count + lif_count).max(1.0);
        let balance = 1.0 - ((uv_count - lif_count).abs() / ul_total);
        let efp_bonus = (efp_count / total).min(0.3) / 0.3;
        let diversity = balance * 0.7 + efp_bonus * 0.3;
        (diversity, entropy)
    };

    // ── Sphericity: eigenvalue ratio of covariance (Cardano's formula) ──
    let sphericity = if n >= 10 {
        let nf = n as f32;
        let mx = local_spikes.iter().map(|s| s.position[0]).sum::<f32>() / nf;
        let my = local_spikes.iter().map(|s| s.position[1]).sum::<f32>() / nf;
        let mz = local_spikes.iter().map(|s| s.position[2]).sum::<f32>() / nf;

        let mut cov = [0.0f32; 6]; // [xx, xy, xz, yy, yz, zz]
        for s in &local_spikes {
            let dx = s.position[0] - mx;
            let dy = s.position[1] - my;
            let dz = s.position[2] - mz;
            cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
            cov[3] += dy * dy; cov[4] += dy * dz; cov[5] += dz * dz;
        }
        for c in cov.iter_mut() { *c /= nf; }

        let a = cov[0]; let b = cov[3]; let c_val = cov[5];
        let d = cov[1]; let e = cov[2]; let f = cov[4];

        let p1 = d * d + e * e + f * f;
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

    // ── WD coherence: variance of wd_change values ──
    let wd_coherence = if n >= 10 {
        let mean_wd: f32 = local_spikes.iter()
            .map(|s| s.wd_change).sum::<f32>() / n as f32;
        let var_wd: f32 = local_spikes.iter()
            .map(|s| (s.wd_change - mean_wd).powi(2))
            .sum::<f32>() / n as f32;
        var_wd
    } else {
        0.0
    };

    // ── Breathing score: CV of per-frame burial across 200-step windows ──
    let breathing_score = if n >= 20 {
        let breath_frame_window = 200i32;
        let mut frame_burials: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
        for s in &local_spikes {
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
                (cv / 0.5).clamp(0.0, 1.0)
            } else { 0.0 }
        } else { 0.0 }
    } else {
        0.0
    };

    // ── UV enrichment: UV spike ratio during UV-on vs UV-off periods ──
    let uv_enrichment = if n > 0 && uv_burst_interval > 0 {
        let mut uv_on_count = 0u32;
        let mut uv_off_count = 0u32;
        for s in &local_spikes {
            let is_uv_source = s.spike_source == 1 || s.aromatic_type >= 0;
            let is_uv_on = (s.timestep % uv_burst_interval) < uv_burst_duration;
            if is_uv_source {
                if is_uv_on {
                    uv_on_count += 1;
                } else {
                    uv_off_count += 1;
                }
            }
        }
        let on_fraction = uv_burst_duration as f32 / uv_burst_interval as f32;
        let off_fraction = 1.0 - on_fraction;
        let uv_on_rate = if on_fraction > 0.0 { uv_on_count as f32 / on_fraction } else { 0.0 };
        let uv_off_rate = if off_fraction > 0.0 { uv_off_count as f32 / off_fraction } else { 0.0 };
        let enrichment = if uv_off_rate > 0.0 { uv_on_rate / uv_off_rate } else { 0.0 };
        (enrichment / 3.0).min(1.0)
    } else {
        0.0
    };

    // ── Per-spike quality: mean of (0.4*burial + 0.2*arom + 0.2*wd + 0.2*intensity) ──
    let per_spike_quality: f32 = {
        let sum: f32 = local_spikes.iter().map(|s| {
            let b = (s.n_residues as f32 / 6.0).min(1.0);
            let a = (s.n_nearby_excited as f32 / 3.0).min(1.0);
            let w = (s.wd_change * 20.0).min(1.0);
            let i = (s.intensity / 30.0).min(1.0);
            0.40 * b + 0.20 * a + 0.20 * w + 0.20 * i
        }).sum();
        sum / n as f32
    };

    // ── Log spike norm: (ln(n_spikes) / 14).clamp(0, 1) ──
    let log_spike_norm = ((n as f32).ln() / 14.0).clamp(0.0, 1.0);

    // ── Lining score: sigmoid(n_lining, center=12, slope=0.3) ──
    let lining_score = 1.0 / (1.0 + (-0.3 * (n_lining as f32 - 12.0)).exp());

    // ── Frustrated solvent score: ΔG_solvation proxy ──
    // High score = many frustrated water molecules displaced early in the simulation.
    // Frustrated water = significant wd_change at early timesteps via LIF/dewetting channel.
    // These are water molecules being displaced from energetically unfavorable hydration sites.
    let frustrated_solvent_score = {
        let median_ts = if !local_spikes.is_empty() {
            let mut ts: Vec<i32> = local_spikes.iter().map(|s| s.timestep).collect();
            ts.sort();
            ts[ts.len() / 2]
        } else { 0 };

        let frustrated: Vec<&prism_nhs::fused_engine::GpuSpikeEvent> = local_spikes.iter()
            .copied()
            .filter(|s| {
                s.wd_change > 0.01           // significant water displacement
                && s.timestep <= median_ts     // early onset (low barrier)
                && (s.spike_source == 2 || s.spike_source == 0) // LIF/dewetting
            })
            .collect();

        if n >= 10 {
            let frac = frustrated.len() as f32 / n as f32;
            let mean_wd: f32 = if !frustrated.is_empty() {
                frustrated.iter().map(|s| s.wd_change).sum::<f32>() / frustrated.len() as f32
            } else { 0.0 };
            // Score = fraction of frustrated spikes × mean displacement magnitude
            // Sigmoid normalize to [0, 1]
            let raw = frac * mean_wd * 100.0;
            1.0 / (1.0 + (-5.0 * (raw - 0.5)).exp())
        } else { 0.0 }
    };

    // ── Vectorial Asymmetry: |CoM_spikes - centroid| ──
    // Functional pockets ("cups") have spike CoM pulled into the protein wall.
    // Enclosed voids ("bubbles") have CoM ≈ centroid → offset ≈ 0.
    let asymmetry_offset = {
        let nf = n as f32;
        let com_x = local_spikes.iter().map(|s| s.position[0]).sum::<f32>() / nf;
        let com_y = local_spikes.iter().map(|s| s.position[1]).sum::<f32>() / nf;
        let com_z = local_spikes.iter().map(|s| s.position[2]).sum::<f32>() / nf;
        ((com_x - centroid[0]).powi(2) +
         (com_y - centroid[1]).powi(2) +
         (com_z - centroid[2]).powi(2)).sqrt()
    };

    // ── Ray-Escape Ratio: Dmax/Dmin from 26-direction rays ──
    // Cast rays from centroid along 26 directions (±x, ±y, ±z, diagonals).
    // For each ray, find distance to nearest protein atom.
    // Ratio = Dmax/Dmin. High ratio = one direction escapes (mouth). Low = enclosed.
    let ray_escape_ratio = {
        // Use spike positions as proxy for protein surface:
        // Dmin = distance to nearest spike, Dmax = distance to farthest spike
        // within the local cloud. This captures the same geometric signal.
        let mut d_min = f32::MAX;
        let mut d_max = 0.0f32;

        // 26-direction unit vectors (face + edge + corner neighbors of a cube)
        let dirs: [[f32; 3]; 26] = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0],
            [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
        ];

        // For each direction, find how far spikes extend from centroid
        // (project spike positions onto each ray direction)
        for dir in &dirs {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let ux = dir[0] / len;
            let uy = dir[1] / len;
            let uz = dir[2] / len;

            // Max projection of spikes onto this ray (how far the pocket extends)
            let mut max_proj = 0.0f32;
            for s in &local_spikes {
                let dx = s.position[0] - centroid[0];
                let dy = s.position[1] - centroid[1];
                let dz = s.position[2] - centroid[2];
                let proj = dx * ux + dy * uy + dz * uz;
                if proj > max_proj { max_proj = proj; }
            }
            if max_proj > d_max { d_max = max_proj; }
            if max_proj < d_min && max_proj > 0.0 { d_min = max_proj; }
        }

        if d_min > 0.0 && d_min < f32::MAX {
            d_max / d_min
        } else {
            1.0
        }
    };

    LocalPhysics {
        burial_score,
        mean_burial,
        onset_score,
        source_diversity,
        source_entropy,
        sphericity,
        wd_coherence,
        breathing_score,
        uv_enrichment,
        per_spike_quality,
        n_local_spikes: n,
        log_spike_norm,
        lining_score,
        frustrated_solvent_score,
        asymmetry_offset,
        ray_escape_ratio,
        source_counts: local_source_counts,
    }
}

///
/// K-means clustering on 3D positions for sub-pocket splitting.
/// Neumaier (Improved Kahan-Babuška) Summation — preserves microscopic
/// chemical signals during spike accumulation. Zero-cost abstraction.
#[inline(always)]
fn neumaier_sum<I: Iterator<Item = f32>>(iter: I) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for v in iter {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            c += (sum - t) + v;
        } else {
            c += (v - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Uses k-means++ initialization (farthest-point seeding) and runs for
/// a fixed number of iterations. Returns k centroids.
#[cfg(feature = "gpu")]
fn kmeans_split(positions: &[[f32; 3]], k: usize, iters: usize) -> Vec<[f32; 3]> {
    if positions.is_empty() || k == 0 {
        return Vec::new();
    }
    // Initialize centroids: first spike, then farthest from existing centers
    let mut centers = vec![positions[0]];
    for _ in 1..k {
        let mut best_dist = 0.0f32;
        let mut best_pos = positions[0];
        for p in positions {
            let min_d = centers.iter()
                .map(|c| ((p[0]-c[0]).powi(2) + (p[1]-c[1]).powi(2) + (p[2]-c[2]).powi(2)).sqrt())
                .fold(f32::MAX, f32::min);
            if min_d > best_dist { best_dist = min_d; best_pos = *p; }
        }
        centers.push(best_pos);
    }

    for _ in 0..iters {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for p in positions {
            let nearest = centers.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (p[0]-a[0]).powi(2) + (p[1]-a[1]).powi(2) + (p[2]-a[2]).powi(2);
                    let db = (p[0]-b[0]).powi(2) + (p[1]-b[1]).powi(2) + (p[2]-b[2]).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, _)| i).unwrap();
            sums[nearest][0] += p[0] as f64;
            sums[nearest][1] += p[1] as f64;
            sums[nearest][2] += p[2] as f64;
            counts[nearest] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centers[i] = [
                    (sums[i][0] / counts[i] as f64) as f32,
                    (sums[i][1] / counts[i] as f64) as f32,
                    (sums[i][2] / counts[i] as f64) as f32,
                ];
            }
        }
    }
    centers
}

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

        // Burial REWARD: deeper pockets are more likely to be real binding sites.
        // depth=1Å -> 0.33, depth=2Å -> 0.67, depth=3Å+ -> 1.0
        // Previous formula PENALIZED depth (exp(-(d-2)/5)), which systematically
        // ranked surface grooves above deep binding pockets — the exact opposite
        // of what drug binding sites look like.
        let surface_factor = (pocket.mean_depth / 3.0).clamp(0.3, 1.0);

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
            // Volume quality: expanded range [50, 2000] for drug-like pockets.
            // Real binding sites range from 50Å³ (fragment) to 2000Å³ (kinase cleft).
            // Previous 800Å³ cap penalized most real drug targets.
            let vol_q = if pocket.volume >= 50.0 && pocket.volume <= 2000.0 {
                1.0_f32
            } else if pocket.volume > 2000.0 {
                (2000.0 / pocket.volume).sqrt()
            } else {
                (pocket.volume / 50.0).clamp(0.1, 1.0)
            };
            0.25 * (spike_frac * 5.0).clamp(0.0, 1.0)    // spike fraction
                + 0.20 * vol_q                               // pocket-like volume (wider range)
                + 0.20 * spike_density.clamp(0.0, 10.0) / 10.0  // spike density
                + 0.15 * surface_factor                      // burial REWARD (was penalty!)
                + 0.10 * druggability.overall                // druggability
                + 0.10 * (stat.spike_count as f32).log2().clamp(0.0, 16.0) / 16.0 // log spike count
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
