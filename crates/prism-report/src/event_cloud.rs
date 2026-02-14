//! Event-Cloud Data Structures
//!
//! Sparse event representation for post-run voxelization.
//! NO dense voxel grids in-step - only event-cloud persistence.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

// =============================================================================
// ABLATION PHASE
// =============================================================================

/// Phase of the ablation protocol (which engine mode)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationPhase {
    /// Baseline: no cryo, no UV
    Baseline,
    /// Cryo-only: cryo cooling, no UV
    CryoOnly,
    /// Cryo+UV: cryo cooling with UV bias
    CryoUv,
}

impl std::fmt::Display for AblationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AblationPhase::Baseline => write!(f, "baseline"),
            AblationPhase::CryoOnly => write!(f, "cryo_only"),
            AblationPhase::CryoUv => write!(f, "cryo_uv"),
        }
    }
}

/// Temperature phase within a simulation run
/// Indicates where in the temperature schedule this event occurred
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TempPhase {
    /// Cold phase: at or near start temperature (cryogenic)
    Cold,
    /// Ramp phase: temperature transitioning from cold to warm
    Ramp,
    /// Warm phase: at or near end temperature (physiological)
    #[default]
    Warm,
}

impl std::fmt::Display for TempPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TempPhase::Cold => write!(f, "cold"),
            TempPhase::Ramp => write!(f, "ramp"),
            TempPhase::Warm => write!(f, "warm"),
        }
    }
}

// =============================================================================
// POCKET EVENT
// =============================================================================

// =============================================================================
// RAW SPIKE EVENT (For UV Enrichment Calculation)
// =============================================================================

/// Raw spike event from GPU with timestep and residue data
///
/// This preserves the essential data needed to compute UV-on vs UV-off
/// aromatic enrichment per site. Saved to spike_events.jsonl alongside events.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSpikeEvent {
    /// Timestep when spike occurred
    pub timestep: i32,
    /// Spike position [x, y, z] in Å
    pub position: [f32; 3],
    /// Nearby residue IDs (within detection radius)
    pub nearby_residues: Vec<i32>,
    /// Spike intensity
    pub intensity: f32,
    /// Ablation phase
    pub phase: AblationPhase,
    /// Replicate ID
    pub replicate_id: usize,
}

/// Single pocket opening event from the event-cloud
///
/// This is the sparse representation persisted during stepping.
/// Voxelization happens post-run from this data.
///
/// # Indexing and Grouping
///
/// Events should be grouped by `(phase, replicate_id)` before aggregation.
/// The `frame_idx` is the step index within that run, NOT a global unique ID.
/// To get a unique event identifier, use `(phase, replicate_id, frame_idx)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketEvent {
    /// Center position (Å)
    pub center_xyz: [f32; 3],

    /// Estimated pocket volume (Å³)
    pub volume_a3: f32,

    /// Number of spikes in this event
    pub spike_count: usize,

    /// Ablation phase (Baseline/CryoOnly/CryoUv)
    /// This identifies which experimental condition produced this event.
    pub phase: AblationPhase,

    /// Temperature phase (Cold/Ramp/Warm) - where in temperature schedule
    /// For Baseline runs, this is always Warm (300K constant).
    /// For Cryo runs, this reflects the explicit schedule boundaries.
    #[serde(default)]
    pub temp_phase: TempPhase,

    /// Replicate index (0-based)
    pub replicate_id: usize,

    /// Step index within this (phase, replicate_id) run.
    /// NOT a global unique ID - use (phase, replicate_id, frame_idx) tuple for uniqueness.
    /// Range: [0, total_steps_for_this_phase)
    pub frame_idx: usize,

    /// Residue indices involved
    pub residues: Vec<u32>,

    /// Confidence score [0, 1]
    #[serde(default)]
    pub confidence: f32,

    /// UV wavelength if applicable (nm)
    #[serde(default)]
    pub wavelength_nm: Option<f32>,
}

// =============================================================================
// EVENT CLOUD
// =============================================================================

/// Collection of pocket events from a pipeline run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCloud {
    /// All pocket events
    pub events: Vec<PocketEvent>,

    /// Grid origin (Å) - minimum corner
    #[serde(default)]
    pub grid_origin: [f32; 3],

    /// Grid dimensions in voxels
    #[serde(default)]
    pub grid_dims: [usize; 3],

    /// Voxel spacing (Å)
    #[serde(default = "default_spacing")]
    pub spacing: f32,
}

fn default_spacing() -> f32 {
    1.0
}

impl EventCloud {
    /// Create empty event cloud
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the cloud
    pub fn push(&mut self, event: PocketEvent) {
        self.events.push(event);
    }

    /// Number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Filter events by phase
    pub fn filter_phase(&self, phase: AblationPhase) -> Vec<&PocketEvent> {
        self.events.iter().filter(|e| e.phase == phase).collect()
    }

    /// Filter events by replicate
    pub fn filter_replicate(&self, replicate_id: usize) -> Vec<&PocketEvent> {
        self.events.iter().filter(|e| e.replicate_id == replicate_id).collect()
    }

    /// Get unique residues across all events
    pub fn all_residues(&self) -> Vec<u32> {
        let mut residues: Vec<u32> = self.events
            .iter()
            .flat_map(|e| e.residues.iter().copied())
            .collect();
        residues.sort();
        residues.dedup();
        residues
    }

    /// Compute bounding box of all events
    pub fn bounding_box(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.events.is_empty() {
            return None;
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for event in &self.events {
            for i in 0..3 {
                min[i] = min[i].min(event.center_xyz[i]);
                max[i] = max[i].max(event.center_xyz[i]);
            }
        }

        Some((min, max))
    }

    /// Set grid parameters from bounding box with padding
    pub fn set_grid_from_bounds(&mut self, padding: f32, spacing: f32) {
        if let Some((min, max)) = self.bounding_box() {
            self.spacing = spacing;
            self.grid_origin = [
                min[0] - padding,
                min[1] - padding,
                min[2] - padding,
            ];

            self.grid_dims = [
                ((max[0] - min[0] + 2.0 * padding) / spacing).ceil() as usize,
                ((max[1] - min[1] + 2.0 * padding) / spacing).ceil() as usize,
                ((max[2] - min[2] + 2.0 * padding) / spacing).ceil() as usize,
            ];
        }
    }
}

// =============================================================================
// JSONL STREAMING
// =============================================================================

/// Streaming writer for events.jsonl
pub struct EventWriter {
    writer: BufWriter<File>,
}

impl EventWriter {
    /// Create new event writer
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path.as_ref())
            .with_context(|| format!("Failed to create event file: {}", path.as_ref().display()))?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Append for resumption
    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path.as_ref())
            .with_context(|| format!("Failed to open event file: {}", path.as_ref().display()))?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Write a single event (JSONL format)
    pub fn write_event(&mut self, event: &PocketEvent) -> Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.writer, "{}", json)?;
        Ok(())
    }

    /// Flush to disk
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Read raw spike events from spike_events.jsonl for UV enrichment calculation
pub fn read_spike_events(path: impl AsRef<Path>) -> Result<Vec<RawSpikeEvent>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open spike events: {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Line {} read error", line_num + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let event: RawSpikeEvent = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse spike event on line {}", line_num + 1))?;
        events.push(event);
    }

    Ok(events)
}

// =============================================================================
// STAGE 2B PROCESSED EVENTS (Quality-Scored Spikes)
// =============================================================================

/// Processed spike event from Stage 2b with quality scoring
///
/// Stage 2b processes raw spike events from Stage 2a and adds:
/// - Quality scores based on RMSF convergence, clustering, RT correlation
/// - Cluster assignments for thermodynamic weighting
/// - RT probe correlation flags
///
/// This is the preferred input for Stage 3 when available, as it filters
/// low-quality events and provides confidence weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedSpikeEvent {
    /// Timestep when spike occurred
    pub timestep: i32,
    /// Spike position [x, y, z] in Å
    pub position: [f32; 3],
    /// Temperature at spike (K)
    #[serde(default)]
    pub temperature: f32,
    /// Spike intensity
    pub intensity: f32,

    /// Quality score (0-1) from Stage 2b analysis
    pub quality_score: f32,
    /// RMSF converged at this timestep
    pub rmsf_converged: bool,
    /// Cluster ID from trajectory clustering
    pub cluster_id: usize,
    /// Boltzmann weight of cluster
    pub cluster_weight: f32,

    /// RT probe void formation detected nearby (temporal correlation)
    pub rt_void_nearby: bool,
    /// RT probe solvation disruption detected nearby
    pub rt_disruption_nearby: bool,
    /// Is this a leading RT signal? (RT event precedes spike)
    pub rt_leading_signal: bool,
}

impl ProcessedSpikeEvent {
    /// Convert to RawSpikeEvent for backward compatibility
    pub fn to_raw(&self, phase: AblationPhase, replicate_id: usize) -> RawSpikeEvent {
        RawSpikeEvent {
            timestep: self.timestep,
            position: self.position,
            nearby_residues: Vec::new(), // Not preserved in processed events
            intensity: self.intensity,
            phase,
            replicate_id,
        }
    }

    /// Check if this event passes quality threshold
    pub fn is_high_quality(&self, threshold: f32) -> bool {
        self.quality_score >= threshold
    }
}

/// Read processed spike events from Stage 2b output (processed_events.jsonl)
///
/// These events have been filtered and scored by Stage 2b trajectory analysis.
/// Returns events sorted by quality_score descending.
pub fn read_processed_spike_events(path: impl AsRef<Path>) -> Result<Vec<ProcessedSpikeEvent>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open processed events: {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Line {} read error", line_num + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let event: ProcessedSpikeEvent = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse processed event on line {}", line_num + 1))?;
        events.push(event);
    }

    // Sort by quality score descending (best first)
    events.sort_by(|a, b| {
        b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    log::info!(
        "Loaded {} processed events from Stage 2b (min score: {:.2}, max score: {:.2})",
        events.len(),
        events.last().map_or(0.0, |e| e.quality_score),
        events.first().map_or(0.0, |e| e.quality_score)
    );

    Ok(events)
}

/// Read processed events if available, otherwise fall back to raw events
///
/// Prefers Stage 2b processed_events.jsonl for better quality filtering.
/// Falls back to raw spike_events.jsonl if processed not available.
///
/// Returns: (events, is_processed) - tuple indicating data source
pub fn read_best_available_events(
    processed_path: impl AsRef<Path>,
    raw_path: impl AsRef<Path>,
) -> Result<(Vec<ProcessedSpikeEvent>, bool)> {
    // Try processed events first (Stage 2b output)
    if processed_path.as_ref().exists() {
        match read_processed_spike_events(&processed_path) {
            Ok(events) if !events.is_empty() => {
                log::info!(
                    "Using Stage 2b processed events: {} high-quality spikes",
                    events.len()
                );
                return Ok((events, true));
            }
            Ok(_) => {
                log::warn!("Stage 2b processed_events.jsonl is empty, falling back to raw");
            }
            Err(e) => {
                log::warn!("Failed to read processed events: {}, falling back to raw", e);
            }
        }
    }

    // Fall back to raw spike events
    if raw_path.as_ref().exists() {
        let raw_events = read_spike_events(&raw_path)?;
        log::info!(
            "Using raw Stage 2a events: {} spikes (no quality scoring)",
            raw_events.len()
        );

        // Convert to ProcessedSpikeEvent with default scores
        let processed: Vec<ProcessedSpikeEvent> = raw_events
            .into_iter()
            .map(|raw| ProcessedSpikeEvent {
                timestep: raw.timestep,
                position: raw.position,
                temperature: 300.0, // Default
                intensity: raw.intensity,
                quality_score: 0.5, // Neutral score for raw events
                rmsf_converged: false,
                cluster_id: 0,
                cluster_weight: 0.1,
                rt_void_nearby: false,
                rt_disruption_nearby: false,
                rt_leading_signal: false,
            })
            .collect();

        return Ok((processed, false));
    }

    anyhow::bail!(
        "No spike events found. Expected processed_events.jsonl or spike_events.jsonl"
    );
}

/// Read events from JSONL file
pub fn read_events(path: impl AsRef<Path>) -> Result<EventCloud> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open event file: {}", path.as_ref().display()))?;

    let reader = BufReader::new(file);
    let mut cloud = EventCloud::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let event: PocketEvent = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse event at line {}", line_num + 1))?;
        cloud.push(event);
    }

    Ok(cloud)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_event_cloud_basic() {
        let mut cloud = EventCloud::new();
        assert!(cloud.is_empty());

        cloud.push(PocketEvent {
            center_xyz: [10.0, 20.0, 30.0],
            volume_a3: 150.0,
            spike_count: 5,
            phase: AblationPhase::CryoUv,
            temp_phase: TempPhase::Cold,
            replicate_id: 0,
            frame_idx: 100,
            residues: vec![10, 20, 30],
            confidence: 0.8,
            wavelength_nm: Some(280.0),
        });

        assert_eq!(cloud.len(), 1);
        assert!(!cloud.is_empty());
    }

    #[test]
    fn test_jsonl_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("events.jsonl");

        // Write events
        {
            let mut writer = EventWriter::new(&path).unwrap();
            writer.write_event(&PocketEvent {
                center_xyz: [1.0, 2.0, 3.0],
                volume_a3: 100.0,
                spike_count: 3,
                phase: AblationPhase::Baseline,
                temp_phase: TempPhase::Warm, // Baseline is always at 300K
                replicate_id: 0,
                frame_idx: 50,
                residues: vec![5, 10],
                confidence: 0.5,
                wavelength_nm: None,
            }).unwrap();
            writer.write_event(&PocketEvent {
                center_xyz: [4.0, 5.0, 6.0],
                volume_a3: 200.0,
                spike_count: 7,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Ramp,
                replicate_id: 1,
                frame_idx: 100,
                residues: vec![15, 20, 25],
                confidence: 0.9,
                wavelength_nm: None,
            }).unwrap();
            writer.flush().unwrap();
        }

        // Read back
        let cloud = read_events(&path).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.events[0].phase, AblationPhase::Baseline);
        assert_eq!(cloud.events[0].temp_phase, TempPhase::Warm);
        assert_eq!(cloud.events[1].phase, AblationPhase::CryoOnly);
        assert_eq!(cloud.events[1].temp_phase, TempPhase::Ramp);
    }

    #[test]
    fn test_bounding_box() {
        let mut cloud = EventCloud::new();
        cloud.push(PocketEvent {
            center_xyz: [0.0, 0.0, 0.0],
            volume_a3: 100.0,
            spike_count: 1,
            phase: AblationPhase::CryoUv,
            temp_phase: TempPhase::Cold,
            replicate_id: 0,
            frame_idx: 0,
            residues: vec![],
            confidence: 0.5,
            wavelength_nm: None,
        });
        cloud.push(PocketEvent {
            center_xyz: [10.0, 20.0, 30.0],
            volume_a3: 100.0,
            spike_count: 1,
            phase: AblationPhase::CryoUv,
            temp_phase: TempPhase::Warm,
            replicate_id: 0,
            frame_idx: 0,
            residues: vec![],
            confidence: 0.5,
            wavelength_nm: None,
        });

        let (min, max) = cloud.bounding_box().unwrap();
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [10.0, 20.0, 30.0]);
    }
}
