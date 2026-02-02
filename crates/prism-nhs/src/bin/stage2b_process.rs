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

/// Load raw spike events from Stage 2a JSONL output
///
/// Format: One JSON object per line with fields:
/// - timestep: i32
/// - position: [f32; 3]
/// - temperature: f32
/// - intensity: f32
#[cfg(feature = "gpu")]
fn load_raw_spikes(path: &PathBuf) -> Result<Vec<RawSpikeEvent>> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let file = File::open(path)
        .with_context(|| format!("Failed to open spike events file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut spikes = Vec::new();
    let mut line_num = 0;

    for line in reader.lines() {
        line_num += 1;
        let line = line.context("Failed to read line")?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        match serde_json::from_str::<RawSpikeEvent>(line) {
            Ok(spike) => spikes.push(spike),
            Err(e) => {
                log::warn!("Line {}: Failed to parse spike event: {} - {}", line_num, e, line);
            }
        }
    }

    log::info!("Loaded {} raw spike events from {}", spikes.len(), path.display());
    Ok(spikes)
}

/// Load trajectory frames from Stage 2a output directory
///
/// Expects either:
/// - Binary frame files: frame_NNNNNN.bin (flat f32 array: n_atoms * 3)
/// - JSONL trajectory: trajectory.jsonl (one frame per line)
#[cfg(feature = "gpu")]
fn load_trajectory_frames(path: &PathBuf) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    use std::io::{BufRead, BufReader, Read};
    use std::fs::{self, File};

    let mut frames = Vec::new();
    let mut timesteps = Vec::new();

    // Check for JSONL format first
    let jsonl_path = path.join("trajectory.jsonl");
    if jsonl_path.exists() {
        log::info!("Loading trajectory from JSONL: {}", jsonl_path.display());
        let file = File::open(&jsonl_path)?;
        let reader = BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse frame as flat array of positions
            #[derive(Deserialize)]
            struct TrajectoryFrame {
                timestep: i32,
                positions: Vec<f32>,
            }

            match serde_json::from_str::<TrajectoryFrame>(&line) {
                Ok(frame) => {
                    timesteps.push(frame.timestep);
                    frames.push(frame.positions);
                }
                Err(e) => {
                    log::warn!("Frame {}: parse error: {}", i, e);
                }
            }
        }

        return Ok((frames, timesteps));
    }

    // Check for binary frame files
    log::info!("Looking for binary frame files in: {}", path.display());
    let mut frame_files: Vec<_> = fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("frame_")
        })
        .filter(|e| {
            e.path().extension().map_or(false, |ext| ext == "bin")
        })
        .collect();

    // Sort by frame number
    frame_files.sort_by_key(|e| {
        let name = e.file_name();
        let name = name.to_string_lossy();
        name.strip_prefix("frame_")
            .and_then(|s| s.strip_suffix(".bin"))
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0)
    });

    for entry in &frame_files {
        let name = entry.file_name();
        let name = name.to_string_lossy();

        // Extract timestep from filename: frame_NNNNNN.bin
        let timestep: i32 = name
            .strip_prefix("frame_")
            .and_then(|s| s.strip_suffix(".bin"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(frames.len() as i32);

        let mut file = File::open(entry.path())?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Convert bytes to f32 (little-endian)
        let positions: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();

        timesteps.push(timestep);
        frames.push(positions);
    }

    if frames.is_empty() {
        log::warn!("No trajectory frames found in {}", path.display());
    } else {
        log::info!("Loaded {} trajectory frames", frames.len());
    }

    Ok((frames, timesteps))
}

/// Load RT probe snapshots from JSON file
#[cfg(feature = "gpu")]
fn load_rt_snapshots(path: &PathBuf) -> Result<Vec<RtProbeSnapshot>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)
        .with_context(|| format!("Failed to open RT probes file: {}", path.display()))?;
    let reader = BufReader::new(file);

    // Try parsing as JSON array first
    let snapshots: Vec<RtProbeSnapshot> = serde_json::from_reader(reader)
        .with_context(|| "Failed to parse RT snapshots JSON")?;

    log::info!("Loaded {} RT probe snapshots from {}", snapshots.len(), path.display());
    Ok(snapshots)
}

/// Process spike events with quality scoring
///
/// Combines RMSF convergence, clustering, and RT probe analysis
/// to assign quality scores and filter events for Stage 3.
#[cfg(feature = "gpu")]
fn process_spikes(
    raw_spikes: &[RawSpikeEvent],
    rmsf: Option<&RmsfAnalysis>,
    clustering: &ClusteringResults,
    rt: Option<&RtAnalysisResults>,
) -> Result<Vec<ProcessedSpikeEvent>> {
    let mut processed = Vec::with_capacity(raw_spikes.len());

    for spike in raw_spikes {
        // Determine RMSF convergence
        let rmsf_converged = rmsf.map_or(false, |r| r.converged);

        // Find cluster for this timestep
        let (cluster_id, cluster_weight) = find_cluster_for_timestep(
            spike.timestep,
            clustering,
        );

        // Check for nearby RT events
        let (rt_void_nearby, rt_disruption_nearby, rt_leading_signal) = if let Some(rt_analysis) = rt {
            check_rt_proximity(spike, rt_analysis)
        } else {
            (false, false, false)
        };

        // Compute quality score
        let quality_score = compute_quality_score(
            spike,
            rmsf_converged,
            cluster_weight,
            rt_void_nearby,
            rt_disruption_nearby,
            rt_leading_signal,
        );

        processed.push(ProcessedSpikeEvent {
            spike: spike.clone(),
            quality_score,
            rmsf_converged,
            cluster_id,
            cluster_weight,
            rt_void_nearby,
            rt_disruption_nearby,
            rt_leading_signal,
        });
    }

    // Sort by quality score descending
    processed.sort_by(|a, b| {
        b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(processed)
}

/// Find which cluster a timestep belongs to
///
/// Uses the representative frames to find the nearest cluster by timestep.
#[cfg(feature = "gpu")]
fn find_cluster_for_timestep(
    timestep: i32,
    clustering: &ClusteringResults,
) -> (usize, f32) {
    if clustering.representatives.is_empty() {
        return (0, 0.1);
    }

    // Find the representative closest in time to this timestep
    let mut best_cluster = 0;
    let mut best_diff = i32::MAX;
    let mut best_weight = 0.1f32;

    for (idx, rep) in clustering.representatives.iter().enumerate() {
        let diff = (rep.timestep - timestep).abs();
        if diff < best_diff {
            best_diff = diff;
            best_cluster = idx;
            best_weight = rep.boltzmann_weight;
        }
    }

    (best_cluster, best_weight)
}

/// Check if RT probe events are temporally correlated with this spike
///
/// Note: RT events don't have position data, so we use temporal correlation only.
#[cfg(feature = "gpu")]
fn check_rt_proximity(
    spike: &RawSpikeEvent,
    rt: &RtAnalysisResults,
) -> (bool, bool, bool) {
    const TIME_WINDOW: i32 = 500;         // timesteps for correlation
    const LEADING_WINDOW: i32 = 200;      // timesteps before spike for leading signal

    let spike_time = spike.timestep;

    let mut void_nearby = false;
    let mut disruption_nearby = false;
    let mut leading_signal = false;

    // Check void formation events (temporal proximity only)
    for void_event in &rt.void_events {
        let time_diff = (void_event.timestep - spike_time).abs();
        if time_diff <= TIME_WINDOW {
            void_nearby = true;

            // Check if RT event precedes spike (leading signal)
            if void_event.timestep < spike_time && (spike_time - void_event.timestep) <= LEADING_WINDOW {
                leading_signal = true;
            }
        }
    }

    // Check solvation disruption events (temporal proximity only)
    for disruption in &rt.disruption_events {
        let time_diff = (disruption.timestep - spike_time).abs();
        if time_diff <= TIME_WINDOW {
            disruption_nearby = true;

            // Solvation disruption is a strong leading indicator if it precedes spike
            if disruption.is_leading && disruption.timestep < spike_time {
                leading_signal = true;
            }
        }
    }

    (void_nearby, disruption_nearby, leading_signal)
}

/// Compute quality score for a spike event
///
/// Factors:
/// - RMSF convergence: indicates equilibrated system
/// - Cluster weight: higher weight = more thermodynamically significant
/// - RT void nearby: corroborating evidence
/// - RT leading signal: strongest indicator (RT precedes spike)
/// - Intensity: raw signal strength
#[cfg(feature = "gpu")]
fn compute_quality_score(
    spike: &RawSpikeEvent,
    rmsf_converged: bool,
    cluster_weight: f32,
    rt_void_nearby: bool,
    rt_disruption_nearby: bool,
    rt_leading_signal: bool,
) -> f32 {
    let mut score = 0.0f32;

    // Base: intensity (normalized, assume max ~100)
    score += (spike.intensity / 100.0).min(1.0) * 0.2;

    // RMSF convergence: +0.15 if converged
    if rmsf_converged {
        score += 0.15;
    }

    // Cluster weight: +0.15 * weight (favors dominant clusters)
    score += 0.15 * cluster_weight.min(1.0);

    // RT void nearby: +0.15
    if rt_void_nearby {
        score += 0.15;
    }

    // RT solvation disruption: +0.15
    if rt_disruption_nearby {
        score += 0.15;
    }

    // RT leading signal: +0.20 (strongest indicator)
    if rt_leading_signal {
        score += 0.20;
    }

    // Ensure score is in [0, 1]
    score.clamp(0.0, 1.0)
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
