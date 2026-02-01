//! [STAGE-2B] Stage 2b Processing Binary
//!
//! Processes Stage 2a outputs (raw spike events + trajectory) to generate
//! filtered, scored events for Stage 3 consumption.
//!
//! ## Pipeline Position
//! ```
//! Stage 2a (MD) → events.jsonl + trajectory
//!     ↓
//! Stage 2b (this) → processed_events.jsonl + analysis
//!     ↓
//! Stage 3 (Site Detection) → candidate_sites.json
//! ```
//!
//! ## Usage
//! ```bash
//! stage2b_process \
//!   --events stage2a/events.jsonl \
//!   --trajectory stage2a/trajectory/ \
//!   --topology topology.json \
//!   --output stage2b/processed_events.jsonl
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use prism_nhs::{
    PrismPrepTopology, RmsfCalculator, RmsfAnalysis,
    TrajectoryClusterer, ClusteringConfig, ClusteringResults,
    RtProbeAnalyzer, RtAnalysisConfig, RtAnalysisResults,
    RtProbeSnapshot,
};

#[derive(Parser, Debug)]
#[command(name = "stage2b_process")]
#[command(about = "[STAGE-2B] Process Stage 2a outputs for Stage 3 consumption")]
struct Args {
    /// Raw spike events from Stage 2a (events.jsonl)
    #[arg(long)]
    events: PathBuf,

    /// Trajectory directory from Stage 2a
    #[arg(long)]
    trajectory: PathBuf,

    /// Topology JSON file
    #[arg(long)]
    topology: PathBuf,

    /// Output processed events file
    #[arg(long)]
    output: PathBuf,

    /// Optional RT probe snapshots (rt_probes.json)
    #[arg(long)]
    rt_probes: Option<PathBuf>,

    /// RMSF convergence threshold (default: 0.8)
    #[arg(long, default_value = "0.8")]
    rmsf_threshold: f32,

    /// Minimum frames for RMSF analysis (default: 20)
    #[arg(long, default_value = "20")]
    min_frames: usize,
}

/// Processed spike event (Stage 2b output for Stage 3)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessedSpikeEvent {
    /// Original spike from Stage 2a
    #[serde(flatten)]
    spike: RawSpikeEvent,

    /// Quality score (0-1) based on trajectory analysis
    quality_score: f32,

    /// RMSF convergence at spike timestep
    rmsf_converged: bool,

    /// Cluster ID from trajectory clustering
    cluster_id: usize,

    /// Boltzmann weight of cluster
    cluster_weight: f32,

    /// RT probe void formation detected nearby
    rt_void_nearby: bool,

    /// RT probe solvation disruption detected nearby
    rt_disruption_nearby: bool,

    /// Is this a leading signal? (RT disruption before spike)
    rt_leading_signal: bool,
}

/// Raw spike event from Stage 2a
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawSpikeEvent {
    timestep: i32,
    position: [f32; 3],
    temperature: f32,
    intensity: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    log::info!("╔══════════════════════════════════════════════════════════════╗");
    log::info!("║     STAGE 2B PROCESSING                                      ║");
    log::info!("║     Trajectory Analysis → Filtered Events for Stage 3        ║");
    log::info!("╚══════════════════════════════════════════════════════════════╝");

    #[cfg(feature = "gpu")]
    {
        process_stage2b(&args)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        anyhow::bail!("GPU feature required for Stage 2b processing");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn process_stage2b(args: &Args) -> Result<()> {
    // Load topology
    log::info!("Loading topology: {}", args.topology.display());
    let topology = PrismPrepTopology::load(&args.topology)?;
    log::info!("  Atoms: {}", topology.n_atoms);

    // Load raw spike events from Stage 2a
    log::info!("Loading raw spike events: {}", args.events.display());
    let raw_spikes = load_raw_spikes(&args.events)?;
    log::info!("  Raw spikes: {}", raw_spikes.len());

    // Load trajectory frames
    log::info!("Loading trajectory frames: {}", args.trajectory.display());
    let (frames, timesteps) = load_trajectory_frames(&args.trajectory)?;
    log::info!("  Trajectory frames: {}", frames.len());

    // Step 1: RMSF convergence analysis
    log::info!("\n🔬 Step 1: RMSF Convergence Analysis");
    let rmsf_analysis = if frames.len() >= args.min_frames {
        let calculator = RmsfCalculator::new(
            &topology.atom_names,
            &topology.residue_ids,
        )?;
        let analysis = calculator.analyze_convergence(&frames)?;
        log::info!("  Pearson correlation: {:.3}", analysis.correlation);
        log::info!("  Converged: {}", if analysis.converged { "✅ YES" } else { "❌ NO" });
        Some(analysis)
    } else {
        log::warn!("  ⚠️ Insufficient frames ({} < {}), skipping RMSF", frames.len(), args.min_frames);
        None
    };

    // Step 2: Trajectory clustering
    log::info!("\n📊 Step 2: Representative Clustering");
    let clustering_config = ClusteringConfig::default();
    let clusterer = TrajectoryClusterer::new(
        clustering_config,
        &topology.atom_names,
        &topology.residue_ids,
    )?;
    let clustering = clusterer.cluster(&frames, &timesteps, None)?;
    log::info!("  Clusters: {}", clustering.num_clusters);
    log::info!("  Average cluster size: {:.1} frames", clustering.avg_cluster_size);

    // Step 3: RT probe analysis (if available)
    log::info!("\n⚡ Step 3: RT Probe Analysis");
    let rt_analysis = if let Some(ref rt_path) = args.rt_probes {
        let rt_snapshots = load_rt_snapshots(rt_path)?;
        log::info!("  RT snapshots: {}", rt_snapshots.len());

        let rt_config = RtAnalysisConfig::default();
        let analyzer = RtProbeAnalyzer::new(rt_config);
        let analysis = analyzer.analyze(&rt_snapshots)?;
        log::info!("  Void formation events: {}", analysis.void_events.len());
        log::info!("  Solvation disruption events: {}", analysis.disruption_events.len());
        Some(analysis)
    } else {
        log::warn!("  ⚠️ No RT probe data provided, skipping RT analysis");
        None
    };

    // Step 4: Process and score spike events
    log::info!("\n✨ Step 4: Scoring and Filtering Spike Events");
    let processed_spikes = process_spikes(
        &raw_spikes,
        rmsf_analysis.as_ref(),
        &clustering,
        rt_analysis.as_ref(),
    )?;
    log::info!("  Processed spikes: {}", processed_spikes.len());

    // Filter by quality score
    let filtered_spikes: Vec<_> = processed_spikes
        .into_iter()
        .filter(|s| s.quality_score > 0.5)
        .collect();
    log::info!("  High-quality spikes (score > 0.5): {}", filtered_spikes.len());

    // Step 5: Write processed events for Stage 3
    log::info!("\n💾 Step 5: Writing Processed Events");
    write_processed_spikes(&filtered_spikes, &args.output)?;
    log::info!("  Output: {}", args.output.display());

    log::info!("\n✅ Stage 2b processing complete!");
    log::info!("   Ready for Stage 3 (site detection)");

    Ok(())
}

#[cfg(feature = "gpu")]
fn load_raw_spikes(_path: &PathBuf) -> Result<Vec<RawSpikeEvent>> {
    // TODO: Implement actual JSONL parsing
    log::warn!("load_raw_spikes(): Placeholder implementation");
    Ok(Vec::new())
}

#[cfg(feature = "gpu")]
fn load_trajectory_frames(_path: &PathBuf) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    // TODO: Implement actual trajectory loading
    log::warn!("load_trajectory_frames(): Placeholder implementation");
    Ok((Vec::new(), Vec::new()))
}

#[cfg(feature = "gpu")]
fn load_rt_snapshots(_path: &PathBuf) -> Result<Vec<RtProbeSnapshot>> {
    // TODO: Implement actual RT snapshot loading
    log::warn!("load_rt_snapshots(): Placeholder implementation");
    Ok(Vec::new())
}

#[cfg(feature = "gpu")]
fn process_spikes(
    raw_spikes: &[RawSpikeEvent],
    _rmsf: Option<&RmsfAnalysis>,
    _clustering: &ClusteringResults,
    _rt: Option<&RtAnalysisResults>,
) -> Result<Vec<ProcessedSpikeEvent>> {
    // TODO: Implement actual spike processing logic
    log::warn!("process_spikes(): Placeholder implementation");

    let processed: Vec<_> = raw_spikes
        .iter()
        .map(|spike| ProcessedSpikeEvent {
            spike: spike.clone(),
            quality_score: 0.7, // Placeholder
            rmsf_converged: true,
            cluster_id: 0,
            cluster_weight: 0.1,
            rt_void_nearby: false,
            rt_disruption_nearby: false,
            rt_leading_signal: false,
        })
        .collect();

    Ok(processed)
}

#[cfg(feature = "gpu")]
fn write_processed_spikes(spikes: &[ProcessedSpikeEvent], path: &PathBuf) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let file = File::create(path)
        .context("Failed to create output file")?;
    let mut writer = std::io::BufWriter::new(file);

    for spike in spikes {
        let json = serde_json::to_string(spike)?;
        writeln!(writer, "{}", json)?;
    }

    Ok(())
}
