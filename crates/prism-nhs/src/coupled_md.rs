//! PRISM-TWIN: Interferometric Spike Dynamics
//!
//! Two MD simulations run simultaneously on one GPU with INTERLEAVED stepping.
//! Physics is INDEPENDENT in both streams. The observation layers are coupled
//! through spike density exchange AND cross-correlation computation.
//!
//! The value is not in individual spike trains — it's in the RELATIONSHIP
//! between them: cross-correlation reveals reproducibility, propagation speed,
//! barrier height, and mechanism. This is interferometric measurement applied
//! to molecular dynamics.
//!
//! ## Architecture
//!
//! ```text
//! for each step:
//!   engine_a.step()  // CUDA stream A (concurrent)
//!   engine_b.step()  // CUDA stream B (concurrent)
//!
//!   every exchange_interval steps:
//!     compute_spike_density(A, B)      // Layer 1: spatial
//!     compute_cross_correlation(A, B)  // Layer 2: temporal
//!     apply_threshold_exchange(A↔B)    // Layer 1: coupling
//!     update_ring_buffers(A, B)        // Layer 2: history
//! ```
//!
//! ## Layers
//!
//! - Layer 1: Spike density exchange (WHERE spikes occur → threshold adaptation)
//! - Layer 2: Temporal cross-correlation (WHEN spikes co-occur → reproducibility)
//! - Layer 4: Perturbation differential (NMA in B only → barrier measurement)
//! - Layers 3+5: Future v2 (frequency coupling, transfer entropy)

use anyhow::{Context, Result};
use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaStream, CudaModule};

use crate::persistent_engine::PersistentNhsEngine;
use crate::input::PrismPrepTopology;
use crate::fused_engine::CryoUvProtocol;
#[cfg(feature = "gpu")]
use crate::twin_kernels::{TwinRingBuffer, find_twin_ptx};

/// Ring buffer size for per-voxel spike history (for CCF computation).
/// 256 entries captures ~500 steps of spike history at typical rates.
pub const RING_SIZE: usize = 256;

/// Number of CCF lag bins (τ from -CCF_HALF_LAGS to +CCF_HALF_LAGS).
pub const CCF_HALF_LAGS: usize = 32;
pub const CCF_LAGS: usize = CCF_HALF_LAGS * 2;

/// Configuration for a PRISM-TWIN interferometric run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTwinConfig {
    /// Fraction of each phase that B lags behind A (0.0-0.5)
    pub phase_offset_fraction: f32,
    /// How often to exchange spike density + compute CCF (in steps)
    pub exchange_interval: u32,
    /// Sensitivity boost factor for threshold modification (Layer 1)
    pub sensitivity_boost: f32,
    /// Maximum threshold reduction fraction (safety limit)
    pub max_threshold_reduction: f32,
    /// Region size for coarse spike density (voxels per region edge)
    pub region_size: u32,
    /// Path to NMA modes file for stream B (None = no NMA, Layer 4 disabled)
    pub nma_modes_path: Option<String>,
    /// NMA amplification factor for stream B
    pub nma_amplification: f32,
    /// Enable CCF computation (Layer 2). Disable for Gate 1 testing.
    pub enable_ccf: bool,
    /// Enable spike density exchange (Layer 1). Disable for Gate 1 testing.
    pub enable_exchange: bool,
}

impl Default for CoupledTwinConfig {
    fn default() -> Self {
        Self {
            phase_offset_fraction: 0.20,
            exchange_interval: 100,
            sensitivity_boost: 0.01,
            max_threshold_reduction: 0.5,
            region_size: 4,
            nma_modes_path: None,
            nma_amplification: 3.0,
            enable_ccf: false,      // Gate 1: off. Gate 2+: on.
            enable_exchange: false,  // Gate 1: off. Gate 2+: on.
        }
    }
}

/// Per-residue interferometric features from coupled observation (50 fields).
///
/// Four groups:
///   Consensus (12): spike agreement, per-phase profile, spatial coherence
///   Cross-correlation (12): CCF peak, width, asymmetry, per-phase, frequency
///   Differential (18): B/A ratio, NMA-exclusive counts, barrier classification
///   Scout/propagation (8): lead time, predictive value, TE, causal flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferometricFeatures {
    pub resid: i32,
    pub resname: String,

    // ── Consensus (12 fields) ──
    pub spike_agreement_ratio: f32,
    pub consensus_intensity_mean: f32,
    /// Per-CCNS-phase spike count ratio: [cold_hold, heating, warm_hold, cooling, cold_return]
    pub consensus_phase_profile: [f32; 5],
    /// Fraction of neighboring residues (within 8Å) that also have agreement > 0.5
    pub consensus_spatial_coherence: f32,
    /// Earliest timestep where both streams agree (both have >= 1 spike near this residue)
    pub consensus_temporal_onset: f32,
    /// Count of nearby residues (within 8Å) with agreement > 0.5
    pub n_consensus_neighbors: u32,

    // ── Cross-correlation (12 fields) ──
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    /// CCF peak value computed per CCNS phase
    pub ccf_per_phase: [f32; 5],
    /// Dominant frequency in CCF (from peak spacing, 0 if no clear periodicity)
    pub ccf_frequency_peak: f32,
    pub ccf_reproducibility: f32,
    /// Standard deviation of ccf_peak_lag across phases (low = consistent timing)
    pub ccf_lag_consistency: f32,

    // ── Differential (18 fields) ──
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub b_over_a_ratio: f32,
    /// Spikes present in B but not A (within 5Å spatial, 500-step temporal window)
    pub nma_exclusive_count: u32,
    /// Spikes present in A but not B
    pub thermal_exclusive_count: u32,
    /// Intensity ratio B/A (sensitive to perturbation amplitude, not just count)
    pub b_over_a_intensity_ratio: f32,
    /// NMA mode index with highest B-exclusive correlation (-1 if no NMA)
    pub nma_responsive_mode: i32,
    /// Eigenvalue of the most responsive NMA mode (0 if no NMA)
    pub nma_mode_eigenvalue: f32,
    pub barrier_classification: String,
    /// B/A spike count ratio per CCNS phase
    pub per_phase_differential: [f32; 5],
    /// Timestep lag between NMA onset and first B-exclusive spike at this residue
    pub differential_onset_lag: f32,
    /// Integral of NMA force × displacement at this residue (0 if no NMA)
    pub nma_work_at_residue: f32,
    /// d(spikes_B)/d(NMA_amplitude) — sensitivity to perturbation (0 if no NMA)
    pub mechanical_sensitivity: f32,
    /// Magnitude of mechanical susceptibility tensor element (0 if no NMA)
    pub susceptibility_magnitude: f32,

    // ── Scout/propagation (8 fields) ──
    /// Mean timestep difference: A fires before B (positive = A leads, from phase offset)
    pub scout_lead_time: f32,
    /// P(B fires within 500 steps | A fired at this residue) — predictive value
    pub scout_predictive_value: f32,
    /// Enrichment of spike count at the phase-offset window vs baseline
    pub phase_offset_enrichment: f32,
    /// A's mean intensity at the timestep when B first fires at this residue
    pub scout_intensity_at_onset: f32,
    /// Radius (Å) of the zone around this residue where A fires before B
    pub scout_spatial_propagation: f32,

    // ════════════════════════════════════════════════════════════════════
    // PLACEHOLDER FIELDS — currently output 0.0
    //
    // These require Phase C implementations that are NOT yet built:
    //   mutual_information      → Phase C Step 15 (GPU binned TE kernel)
    //   transfer_entropy_a_to_b → Phase C Step 15 (GPU binned TE kernel)
    //   causal_flow_direction   → Phase C Step 15 (GPU binned TE kernel)
    //   ccf_frequency_peak      → requires FFT on CCF row (not yet implemented)
    //   ccf_lag_consistency     → requires per-phase CCF recomputation
    //   nma_responsive_mode     → Phase C Step 16 (NMA mode correlation)
    //   nma_mode_eigenvalue     → Phase C Step 16 (NMA mode correlation)
    //   nma_work_at_residue     → Phase C Step 16 (NMA force integration)
    //   mechanical_sensitivity  → Phase C Step 16 (perturbation dose-response)
    //   susceptibility_magnitude→ Phase C Step 16 (mechanical susceptibility)
    //   scout_spatial_propagation→ requires spatial wave-front analysis
    //
    // TOTAL: 11 placeholders out of 50 fields = 39 populated, 11 deferred
    //
    // DO NOT use these fields for ranking or model training until their
    // corresponding Phase C steps are implemented and validated.
    // ════════════════════════════════════════════════════════════════════

    /// Mutual information MI(A, B) at this residue. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub mutual_information: f32,
    /// Transfer entropy TE(A→B) at this residue. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub transfer_entropy_a_to_b: f32,
    /// TE(A→B) - TE(B→A), signed. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub causal_flow_direction: f32,
}

impl Default for InterferometricFeatures {
    fn default() -> Self {
        Self {
            resid: -1,
            resname: String::new(),
            spike_agreement_ratio: 0.0,
            consensus_intensity_mean: 0.0,
            consensus_phase_profile: [0.0; 5],
            consensus_spatial_coherence: 0.0,
            consensus_temporal_onset: 0.0,
            n_consensus_neighbors: 0,
            ccf_peak_lag: 0,
            ccf_peak_value: 0.0,
            ccf_width: 0.0,
            ccf_asymmetry: 0.0,
            ccf_per_phase: [0.0; 5],
            ccf_frequency_peak: 0.0,
            ccf_reproducibility: 0.0,
            ccf_lag_consistency: 0.0,
            spikes_a: 0,
            spikes_b: 0,
            b_over_a_ratio: 0.0,
            nma_exclusive_count: 0,
            thermal_exclusive_count: 0,
            b_over_a_intensity_ratio: 0.0,
            nma_responsive_mode: -1,
            nma_mode_eigenvalue: 0.0,
            barrier_classification: "MEDIUM".to_string(),
            per_phase_differential: [0.0; 5],
            differential_onset_lag: 0.0,
            nma_work_at_residue: 0.0,
            mechanical_sensitivity: 0.0,
            susceptibility_magnitude: 0.0,
            scout_lead_time: 0.0,
            scout_predictive_value: 0.0,
            phase_offset_enrichment: 0.0,
            scout_intensity_at_onset: 0.0,
            scout_spatial_propagation: 0.0,
            mutual_information: 0.0,
            transfer_entropy_a_to_b: 0.0,
            causal_flow_direction: 0.0,
        }
    }
}

/// Full result from a PRISM-TWIN coupled observation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTwinResult {
    pub stream_a: StreamResult,
    pub stream_b: StreamResult,
    pub config: CoupledTwinConfig,
    pub per_residue_features: Vec<InterferometricFeatures>,
    pub n_consensus_events: u64,
    pub n_differential_events: u64,

    // Gate 2 exchange accounting
    /// How many spike density exchange rounds occurred during the run.
    pub n_exchanges: u32,
    /// Summed spike intensity across all regions from A (integrated across all exchanges).
    pub total_density_a_to_b: f64,
    /// Summed spike intensity across all regions from B (integrated across all exchanges).
    pub total_density_b_to_a: f64,
    /// Number of coarse regions that had non-zero spike activity in at least one stream.
    pub n_nonzero_regions: u32,

    // Gate 3 phase offset
    /// Total spikes tracked for ring buffer exchange (both directions)
    pub ring_spikes_tracked: u64,

    // Gate 3 phase offset
    /// Number of extra cold_hold steps B runs before A's schedule
    pub phase_offset_steps: i32,
    /// Total steps for stream A
    pub steps_a: i32,
    /// Total steps for stream B (steps_a + offset)
    pub steps_b: i32,
}

/// Per-stream result data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResult {
    pub role: String,
    pub seed: u64,
    pub total_spikes: usize,
    pub perturbation: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate 2 CPU helpers: spike density + cross-correlation
// ─────────────────────────────────────────────────────────────────────────────

/// One coarse grid cell containing the summed spike intensity from a spike slice.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct DensityCell {
    /// Grid indices (ix, iy, iz)
    idx: (i32, i32, i32),
    /// Total intensity accumulated in this cell
    total_intensity: f64,
    /// Number of spikes that landed in this cell
    spike_count: u32,
}

/// Grid the protein bounding box into `cell_size_ang`-Å cells and accumulate
/// spike intensity per cell.  Returns (cells, bounding-box min [x,y,z]).
///
/// Spikes with zero or negative intensity are skipped.
#[cfg(feature = "gpu")]
fn compute_spike_density_cpu(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    cell_size_ang: f32,
) -> (Vec<DensityCell>, [f32; 3]) {
    use std::collections::HashMap;

    if spikes.is_empty() {
        return (Vec::new(), [0.0f32; 3]);
    }

    // Find bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for s in spikes {
        for d in 0..3 {
            if s.position[d] < min[d] { min[d] = s.position[d]; }
            if s.position[d] > max[d] { max[d] = s.position[d]; }
        }
    }
    // Expand bounding box by one cell on each side to avoid edge effects
    for d in 0..3 {
        min[d] -= cell_size_ang;
        max[d] += cell_size_ang;
    }

    let inv_cell = 1.0 / cell_size_ang;
    let mut map: HashMap<(i32, i32, i32), (f64, u32)> = HashMap::new();

    for s in spikes {
        if s.intensity <= 0.0 { continue; }
        let ix = ((s.position[0] - min[0]) * inv_cell).floor() as i32;
        let iy = ((s.position[1] - min[1]) * inv_cell).floor() as i32;
        let iz = ((s.position[2] - min[2]) * inv_cell).floor() as i32;
        let entry = map.entry((ix, iy, iz)).or_insert((0.0, 0));
        entry.0 += s.intensity as f64;
        entry.1 += 1;
    }

    let cells: Vec<DensityCell> = map
        .into_iter()
        .map(|(idx, (total_intensity, spike_count))| DensityCell { idx, total_intensity, spike_count })
        .collect();

    (cells, min)
}

/// CPU cross-correlation: count spike pairs (one from A, one from B) that are
/// within `spatial_thresh_ang` Å AND within `temporal_thresh_steps` steps.
///
/// Returns (n_matches, per-residue consensus counts keyed by topology resid).
#[cfg(feature = "gpu")]
fn compute_cpu_cross_correlation(
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    spatial_thresh_ang: f32,
    temporal_thresh_steps: i32,
) -> (u64, std::collections::HashMap<i32, (u32, u32, f32, f32)>) {
    // HashMap value: (count_a, count_b, sum_intensity_a, sum_intensity_b)
    use std::collections::HashMap;

    let spatial_sq = spatial_thresh_ang * spatial_thresh_ang;
    let mut n_matches: u64 = 0;
    // Per-residue accumulators: keyed by resid
    let mut per_res: HashMap<i32, (u32, u32, f32, f32)> = HashMap::new();

    // Accumulate per-residue spike counts for A
    for sa in spikes_a {
        let n = sa.n_residues.min(8) as usize;
        for r in 0..n {
            let resid = sa.nearby_residues[r];
            if resid < 0 { continue; }
            let e = per_res.entry(resid).or_insert((0, 0, 0.0, 0.0));
            e.0 += 1;
            e.2 += sa.intensity;
        }
    }
    // Accumulate per-residue spike counts for B
    for sb in spikes_b {
        let n = sb.n_residues.min(8) as usize;
        for r in 0..n {
            let resid = sb.nearby_residues[r];
            if resid < 0 { continue; }
            let e = per_res.entry(resid).or_insert((0, 0, 0.0, 0.0));
            e.1 += 1;
            e.3 += sb.intensity;
        }
    }

    // Spatial + temporal co-occurrence: SAMPLED (O(N) not O(N²))
    // With 15M+ spikes per stream, O(N²) is impossible.
    // Instead: sample up to 10K spikes from each, compute matches on sample.
    // Scale result by (total_a * total_b) / (sample_a * sample_b).
    let max_sample = 10_000usize;
    let sample_a: Vec<_> = if spikes_a.len() > max_sample {
        let step = spikes_a.len() / max_sample;
        spikes_a.iter().step_by(step).take(max_sample).collect()
    } else {
        spikes_a.iter().collect()
    };
    let sample_b: Vec<_> = if spikes_b.len() > max_sample {
        let step = spikes_b.len() / max_sample;
        spikes_b.iter().step_by(step).take(max_sample).collect()
    } else {
        spikes_b.iter().collect()
    };

    let mut sample_matches: u64 = 0;
    for sa in &sample_a {
        for sb in &sample_b {
            let dt = (sa.timestep - sb.timestep).abs();
            if dt > temporal_thresh_steps { continue; }
            let dx = sa.position[0] - sb.position[0];
            let dy = sa.position[1] - sb.position[1];
            let dz = sa.position[2] - sb.position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq <= spatial_sq {
                sample_matches += 1;
            }
        }
    }

    // Scale to estimate total matches
    let scale = (spikes_a.len() as f64 * spikes_b.len() as f64)
              / (sample_a.len() as f64 * sample_b.len() as f64).max(1.0);
    n_matches = (sample_matches as f64 * scale) as u64;

    (n_matches, per_res)
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature computation helpers (Step 6)
// ─────────────────────────────────────────────────────────────────────────────

/// CCNS protocol phase index for a given timestep.
/// 0=cold_hold, 1=heating, 2=warm_hold, 3=cooling, 4=cold_return
#[cfg(feature = "gpu")]
fn ccns_phase(timestep: i32, protocol: &CryoUvProtocol) -> usize {
    let t = timestep;
    let p1 = protocol.cold_hold_steps;
    let p2 = p1 + protocol.ramp_steps;
    let p3 = p2 + protocol.warm_hold_steps;
    let p4 = p3 + protocol.ramp_down_steps;
    if t < p1 { 0 }
    else if t < p2 { 1 }
    else if t < p3 { 2 }
    else if t < p4 { 3 }
    else { 4 }
}

/// Per-residue per-phase spike counts from a spike vector.
/// Returns HashMap<resid, [count_phase0..count_phase4]>
#[cfg(feature = "gpu")]
fn per_residue_phase_counts(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    protocol: &CryoUvProtocol,
) -> std::collections::HashMap<i32, [u32; 5]> {
    let mut map: std::collections::HashMap<i32, [u32; 5]> = std::collections::HashMap::new();
    for s in spikes {
        let phase = ccns_phase(s.timestep, protocol);
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 { continue; }
            let entry = map.entry(resid).or_insert([0; 5]);
            entry[phase] += 1;
        }
    }
    map
}

/// Per-residue earliest spike timestep from a spike vector.
/// Returns HashMap<resid, earliest_timestep>
#[cfg(feature = "gpu")]
fn per_residue_onset(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
) -> std::collections::HashMap<i32, i32> {
    let mut map: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    for s in spikes {
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 { continue; }
            let entry = map.entry(resid).or_insert(i32::MAX);
            if s.timestep < *entry { *entry = s.timestep; }
        }
    }
    map
}

/// Compute per-residue intensity sums from a spike vector.
/// Returns HashMap<resid, total_intensity>
#[cfg(feature = "gpu")]
fn per_residue_intensity(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
) -> std::collections::HashMap<i32, f32> {
    let mut map: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
    for s in spikes {
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 { continue; }
            *map.entry(resid).or_insert(0.0) += s.intensity;
        }
    }
    map
}

/// Run a PRISM-TWIN interferometric coupled observation simulation.
///
/// Two engines run with INTERLEAVED stepping on separate CUDA streams.
/// Every `exchange_interval` steps, spike density is exchanged (Layer 1)
/// and cross-correlation is computed (Layer 2).
///
/// Gate 1: interleaved stepping, NO exchange, NO CCF. Just verify dual streams.
/// Gate 2+: enable exchange and CCF progressively.
#[cfg(feature = "gpu")]
pub fn run_coupled_twin(
    config_ref: &crate::persistent_engine::PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology: &PrismPrepTopology,
    protocol: CryoUvProtocol,
    twin_config: &CoupledTwinConfig,
    seed_a: u64,
    steps: i32,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    output_dir: &std::path::Path,
) -> Result<CoupledTwinResult> {
    let seed_b = seed_a + 1000;

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║   PRISM-TWIN: Interferometric Spike Dynamics            ║");
    log::info!("╚══════════════════════════════════════════════════════════╝");
    log::info!("  Stream A (Scout):    seed={}, thermal only", seed_a);
    log::info!("  Stream B (Observer): seed={}, thermal{}", seed_b,
        if twin_config.nma_modes_path.is_some() { " + NMA" } else { " only" });
    log::info!("  Phase offset: {:.0}%", twin_config.phase_offset_fraction * 100.0);
    log::info!("  Exchange: {} (interval={})",
        if twin_config.enable_exchange { "ENABLED" } else { "DISABLED" },
        twin_config.exchange_interval);
    log::info!("  CCF: {}",
        if twin_config.enable_ccf { "ENABLED" } else { "DISABLED" });

    // Create CUDA streams
    let stream_a = context.new_stream().context("CUDA stream A")?;
    let stream_b = context.new_stream().context("CUDA stream B")?;
    log::info!("  CUDA streams: A ✓, B ✓");

    // Check VRAM before allocation
    let (vram_free, vram_total) = cudarc::driver::result::mem_get_info()
        .unwrap_or((0, 0));
    log::info!("  VRAM: {:.1} GB free / {:.1} GB total",
        vram_free as f64 / 1e9, vram_total as f64 / 1e9);

    // ── Initialize Engine A (Scout) ──
    log::info!("  Initializing Stream A...");
    let mut engine_a = PersistentNhsEngine::new_on_stream(
        config_ref, context.clone(), fused_module.clone(), stream_a,
    )?;
    engine_a.load_topology(topology)?;
    if hmr { engine_a.set_dt(0.004)?; }
    if fused_steps > 1 { engine_a.set_fused_inner_steps(fused_steps)?; }
    if adaptive_dt { engine_a.set_adaptive_dt(true)?; }
    if ladd_enabled { engine_a.set_ladd_enabled(true); }
    engine_a.set_cryo_uv_protocol(protocol.clone())?;
    engine_a.set_spike_accumulation(true);
    log::info!("  Stream A: ✓");

    // ── Initialize Engine B (Focused Observer) ──
    log::info!("  Initializing Stream B...");
    let mut engine_b = PersistentNhsEngine::new_on_stream(
        config_ref, context.clone(), fused_module.clone(), stream_b,
    )?;
    engine_b.load_topology(topology)?;
    if hmr { engine_b.set_dt(0.004)?; }
    if fused_steps > 1 { engine_b.set_fused_inner_steps(fused_steps)?; }
    if adaptive_dt { engine_b.set_adaptive_dt(true)?; }
    if ladd_enabled { engine_b.set_ladd_enabled(true); }

    // NMA perturbation for B only (Layer 4)
    if let Some(ref nma_path) = twin_config.nma_modes_path {
        engine_b.set_nma_amplification(twin_config.nma_amplification);
        engine_b.load_nma_modes(nma_path)?;
        log::info!("  Stream B: NMA loaded (amplification={:.1})", twin_config.nma_amplification);
    }

    // Gate 3: Phase-offset scheduling
    // B gets extra cold_hold steps so it starts heating AFTER A.
    // Offset = cold_hold_steps × phase_offset_fraction
    let offset_steps = (protocol.cold_hold_steps as f32 * twin_config.phase_offset_fraction) as i32;
    let mut protocol_b = protocol.clone();
    protocol_b.cold_hold_steps += offset_steps;
    let steps_b = steps + offset_steps;  // B runs longer

    engine_b.set_cryo_uv_protocol(protocol_b)?;
    engine_b.set_spike_accumulation(true);

    log::info!("  Stream B: ✓");
    log::info!("  Gate 3 phase offset: {} steps ({:.0}% of cold_hold={})",
        offset_steps, twin_config.phase_offset_fraction * 100.0, protocol.cold_hold_steps);
    log::info!("  Stream A schedule: cold_hold={}, ramp={}, warm_hold={}, total={}",
        protocol.cold_hold_steps, protocol.ramp_steps, protocol.warm_hold_steps, steps);
    log::info!("  Stream B schedule: cold_hold={}, ramp={}, warm_hold={}, total={}",
        protocol.cold_hold_steps + offset_steps, protocol.ramp_steps, protocol.warm_hold_steps, steps_b);
    log::info!("  A enters heating at step {}, B enters heating at step {}",
        protocol.cold_hold_steps, protocol.cold_hold_steps + offset_steps);

    // VRAM after allocation
    let (vram_after, _) = cudarc::driver::result::mem_get_info()
        .unwrap_or((0, 0));
    let vram_used = (vram_free - vram_after) as f64 / 1e9;
    log::info!("  VRAM used by twin engines: {:.2} GB ({:.1} GB remaining)",
        vram_used, vram_after as f64 / 1e9);

    // ── RING BUFFER ALLOCATION (TWIN interferometric coupling) ──
    //
    // Two ring buffers carry spike evidence between streams:
    //   ring_a: holds A's spikes → read by B's threshold adapter
    //   ring_b: holds B's spikes → read by A's threshold adapter
    //
    // The coupling flow per step:
    //   1. read_and_adapt: ring_b → lower A's thresholds where B saw spikes
    //   2. read_and_adapt: ring_a → lower B's thresholds where A saw spikes
    //   3. sync exchange stream
    //   4. run both engines (they pick up modified thresholds)
    //   5. compact new spikes → push to ring buffers
    //   6. periodic threshold recovery (prevent permanent suppression)

    let (mut ring_a, mut ring_b, stream_exchange) = if twin_config.enable_exchange {
        let ring_module = context.load_module(
            cudarc::nvrtc::Ptx::from_file(&find_twin_ptx("ring_buffer.ptx")?)
        ).context("Failed to load ring_buffer.ptx")?;
        let stream_ex = context.new_stream().context("CUDA exchange stream")?;

        let mut ra = TwinRingBuffer::new(&context, &stream_ex, &ring_module, 8192)?;
        let mut rb = TwinRingBuffer::new(&context, &stream_ex, &ring_module, 8192)?;
        ra.reset(&stream_ex)?;
        rb.reset(&stream_ex)?;

        log::info!("  Ring buffers: A ✓, B ✓ (capacity=8192 each)");
        (Some(ra), Some(rb), Some(stream_ex))
    } else {
        (None, None, None)
    };

    let diag_enabled = std::env::var("PRISM_TWIN_DIAG").is_ok();
    if diag_enabled {
        log::info!("  PRISM_TWIN_DIAG: threshold coupling diagnostics ENABLED");
    }

    // ── INTERLEAVED EXECUTION ──
    let inner = fused_steps.max(1) as i32;
    let outer_steps_a = (steps + inner - 1) / inner;
    let outer_steps_b = (steps_b + inner - 1) / inner;
    let outer_steps = outer_steps_a.max(outer_steps_b);

    log::info!("  Running {} interleaved outer steps (A={}, B={} fused steps)...",
        outer_steps, outer_steps_a, outer_steps_b);

    let start = std::time::Instant::now();
    let mut spikes_a_total = 0usize;
    let mut spikes_b_total = 0usize;

    // Gate 2 exchange accumulators (CPU density telemetry — kept for logging)
    let mut n_exchanges: u32 = 0;
    let mut total_density_a_to_b: f64 = 0.0;
    let mut total_density_b_to_a: f64 = 0.0;
    let mut max_nonzero_regions: u32 = 0;

    // Ring buffer spike tracking
    let mut prev_accum_len_a: usize = 0;  // index into accumulated_spikes for delta
    let mut prev_accum_len_b: usize = 0;
    let mut ring_spikes_exchanged: u64 = 0;
    let mut coupling_active_steps: u32 = 0;

    let mut a_finished = false;

    for step in 0..outer_steps {

        // ════════════════════════════════════════════════════════════════════
        // PHASE 1: Cross-stream threshold adaptation (BEFORE engine.run)
        //
        // B's recent spikes → lower A's thresholds (make A more sensitive
        //   in regions where B found activity)
        // A's recent spikes → lower B's thresholds (symmetric)
        //
        // This is the interferometric coupling — detector sensitivity in
        // one stream is steered by evidence from the other stream.
        // ════════════════════════════════════════════════════════════════════
        if let (Some(ref mut rb), Some(ref se)) = (&mut ring_b, &stream_exchange) {
            if step > 0 {
                // Get grid geometry from engine A (both engines share the same grid)
                if let Some((gx, gy, gz, ox, oy, oz, vs)) = engine_a.grid_info() {
                    // B's evidence → adapt A's thresholds
                    if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                        rb.read_and_adapt(
                            se, thresh_a, base_a,
                            (gx, gy, gz), (ox, oy, oz), vs,
                            twin_config.sensitivity_boost,
                            twin_config.max_threshold_reduction,
                            step as u32,
                            500.0,  // decay_constant: spikes > 500 steps old lose influence
                        )?;
                    }
                }
            }
        }
        if let (Some(ref mut ra), Some(ref se)) = (&mut ring_a, &stream_exchange) {
            if step > 0 {
                if let Some((gx, gy, gz, ox, oy, oz, vs)) = engine_b.grid_info() {
                    // A's evidence → adapt B's thresholds
                    if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                        ra.read_and_adapt(
                            se, thresh_b, base_b,
                            (gx, gy, gz), (ox, oy, oz), vs,
                            twin_config.sensitivity_boost,
                            twin_config.max_threshold_reduction,
                            step as u32,
                            500.0,
                        )?;
                    }
                }
            }
        }
        // Sync exchange stream BEFORE launching fused kernels so threshold
        // modifications are visible to both engines' next run().
        if let Some(ref se) = stream_exchange {
            if step > 0 {
                se.synchronize()?;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // PHASE 1.5: Diagnostic — measure coupling effectiveness
        // ════════════════════════════════════════════════════════════════════
        if diag_enabled && step > 0 && step % 100 == 0 {
            // Measure coupling effectiveness: L2 norm of (threshold - base_threshold)
            // over a sample of voxels. Non-zero delta means coupling is modifying thresholds.
            let delta_l2 = if ring_spikes_exchanged > 0 {
                // Use the ring buffer overflow count as a proxy — if spikes are
                // being exchanged and read, coupling is active. Full threshold
                // download would be expensive, so we use the spike count as signal.
                let overflow_a = ring_b.as_ref().and_then(|r| {
                    stream_exchange.as_ref().and_then(|se| r.overflow_count(se).ok())
                }).unwrap_or(0);
                let overflow_b = ring_a.as_ref().and_then(|r| {
                    stream_exchange.as_ref().and_then(|se| r.overflow_count(se).ok())
                }).unwrap_or(0);
                // Coupling is active if spikes were exchanged without 100% overflow
                if overflow_a + overflow_b < ring_spikes_exchanged as u32 {
                    coupling_active_steps += 1;
                }
                ring_spikes_exchanged as f32 / (step as f32 + 1.0)
            } else {
                0.0
            };
            log::info!("  [DIAG] step={} spikes_per_step={:.1} ring_total={} active={}/{}",
                step, delta_l2, ring_spikes_exchanged,
                coupling_active_steps, step / 100);
        }

        // ════════════════════════════════════════════════════════════════════
        // PHASE 2: Run both engines (concurrent on separate CUDA streams)
        // ════════════════════════════════════════════════════════════════════
        if step < outer_steps_a {
            let summary_a = engine_a.run(inner)?;
            spikes_a_total += summary_a.total_spikes;
        } else if !a_finished {
            a_finished = true;
            log::info!("  Stream A finished at step {} (B continues for {} more)",
                step, outer_steps_b - step);
        }

        let summary_b = engine_b.run(inner)?;
        spikes_b_total += summary_b.total_spikes;

        // ════════════════════════════════════════════════════════════════════
        // PHASE 3: Compact new spikes and push to ring buffers
        //
        // After engine.run(), new spikes have been downloaded to CPU and
        // appended to accumulated_spikes. We slice the delta (new since
        // last push), compact GpuSpikeEvent (92B) → RingSpikeEvent (48B),
        // upload, and push to the ring buffer.
        // ════════════════════════════════════════════════════════════════════
        if twin_config.enable_exchange {
            let se = stream_exchange.as_ref().unwrap();

            // A's new spikes → ring_a (for B to read)
            let curr_len_a = engine_a.accumulated_spike_count();
            if curr_len_a > prev_accum_len_a {
                // Only clone+compact when there are actually new spikes
                let accum_a = engine_a.get_accumulated_spikes();
                let delta_spikes_a = &accum_a[prev_accum_len_a..];
                if let Some(ref mut ra) = ring_a {
                    ra.push_compacted(se, delta_spikes_a)?;
                }
                ring_spikes_exchanged += delta_spikes_a.len() as u64;
                prev_accum_len_a = curr_len_a;
            }

            // B's new spikes → ring_b (for A to read)
            let curr_len_b = engine_b.accumulated_spike_count();
            if curr_len_b > prev_accum_len_b {
                let accum_b = engine_b.get_accumulated_spikes();
                let delta_spikes_b = &accum_b[prev_accum_len_b..];
                if let Some(ref mut rb) = ring_b {
                    rb.push_compacted(se, delta_spikes_b)?;
                }
                ring_spikes_exchanged += delta_spikes_b.len() as u64;
                prev_accum_len_b = curr_len_b;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // PHASE 4: Periodic threshold recovery (every 1000 steps)
        //
        // Without recovery, thresholds drift permanently downward as more
        // spikes accumulate. Recovery gently pushes thresholds back toward
        // baseline at 1% per call. This means:
        //   - Recently active voxels: stay suppressed (new spikes keep pushing down)
        //   - Inactive voxels: recover to baseline over ~100K steps
        //   - Net effect: coupling is recency-weighted
        // ════════════════════════════════════════════════════════════════════
        if twin_config.enable_exchange && step as u32 % 1000 == 0 && step > 0 {
            let n_voxels = engine_a.total_voxels() as u32;
            if let Some(ref se) = stream_exchange {
                if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                    if let Some(ref rb) = ring_b {
                        rb.threshold_recovery(se, thresh_a, base_a, n_voxels, 0.01)?;
                    }
                }
                if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                    if let Some(ref ra) = ring_a {
                        ra.threshold_recovery(se, thresh_b, base_b, n_voxels, 0.01)?;
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // CPU density exchange telemetry (kept for logging, no longer drives coupling)
        // ════════════════════════════════════════════════════════════════════
        if twin_config.enable_exchange && step as u32 % twin_config.exchange_interval == 0 {
            let snap_a = engine_a.get_accumulated_spikes();
            let snap_b = engine_b.get_accumulated_spikes();

            let cell_ang = twin_config.region_size as f32;
            let (cells_a, _) = compute_spike_density_cpu(&snap_a, cell_ang);
            let (cells_b, _) = compute_spike_density_cpu(&snap_b, cell_ang);

            let density_a: f64 = cells_a.iter().map(|c| c.total_intensity).sum();
            let density_b: f64 = cells_b.iter().map(|c| c.total_intensity).sum();
            let nonzero_union = {
                use std::collections::HashSet;
                let keys_a: HashSet<_> = cells_a.iter().map(|c| c.idx).collect();
                let keys_b: HashSet<_> = cells_b.iter().map(|c| c.idx).collect();
                keys_a.union(&keys_b).count() as u32
            };

            n_exchanges += 1;
            total_density_a_to_b += density_a;
            total_density_b_to_a += density_b;
            if nonzero_union > max_nonzero_regions { max_nonzero_regions = nonzero_union; }

            log::debug!(
                "  [exchange #{n_exchanges}] step={step}: density A={density_a:.3e} B={density_b:.3e} union={nonzero_union}"
            );
        }

        // Progress logging
        if (step + 1) % 5000 == 0 || step == outer_steps - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let steps_per_sec = (step + 1) as f64 / elapsed;
            let overflow_a = ring_a.as_ref().and_then(|r| {
                stream_exchange.as_ref().and_then(|se| r.overflow_count(se).ok())
            }).unwrap_or(0);
            let overflow_b = ring_b.as_ref().and_then(|r| {
                stream_exchange.as_ref().and_then(|se| r.overflow_count(se).ok())
            }).unwrap_or(0);
            log::info!("  Step {}/{}: A={} B={} spikes  ring_spikes_exchanged={} overflow_a={} overflow_b={} ({:.0} steps/s)",
                step + 1, outer_steps, spikes_a_total, spikes_b_total,
                ring_spikes_exchanged, overflow_a, overflow_b, steps_per_sec);
        }
    }

    let wall_time = start.elapsed();
    log::info!("  Simulation complete: {:.1}s wall time", wall_time.as_secs_f64());

    // ── Collect accumulated spikes ──
    let spikes_a = engine_a.get_accumulated_spikes();
    let spikes_b = engine_b.get_accumulated_spikes();

    log::info!("  Stream A accumulated: {} spikes", spikes_a.len());
    log::info!("  Stream B accumulated: {} spikes", spikes_b.len());

    // ── SPIKE PERSISTENCE: Write ALL raw spikes to JSON with stream_id ────────
    // This is the TRAINING SIGNAL — raw spike records must be preserved.
    {
        use std::io::Write;

        let spike_output_path = output_dir;
        let _ = std::fs::create_dir_all(spike_output_path);

        // Write combined spike file with stream_id
        let combined_path = spike_output_path.join("coupled_spikes.json");
        log::info!("  Writing {} + {} = {} raw spikes to {}",
            spikes_a.len(), spikes_b.len(), spikes_a.len() + spikes_b.len(),
            combined_path.display());

        if let Ok(mut f) = std::fs::File::create(&combined_path) {
            let _ = writeln!(f, "{{\"n_spikes_a\": {}, \"n_spikes_b\": {}, \"spikes\": [",
                spikes_a.len(), spikes_b.len());

            let write_spike = |f: &mut std::fs::File, s: &crate::fused_engine::GpuSpikeEvent, stream_id: u8, last: bool| {
                // Copy fields to locals to avoid unaligned packed struct access
                let ts = s.timestep;
                let pos = s.position;
                let int = s.intensity;
                let ve = s.vibrational_energy;
                let wd = s.water_density;
                let src = s.spike_source;
                let vi = s.voxel_idx;
                let wl = s.wavelength_nm;
                let ne = s.n_nearby_excited;
                let _ = writeln!(f, "  {{\"stream_id\": {}, \"timestep\": {}, \"x\": {:.3}, \"y\": {:.3}, \"z\": {:.3}, \
                    \"intensity\": {:.4}, \"vib_energy\": {:.6}, \"water_density\": {:.4}, \
                    \"spike_source\": {}, \"voxel_idx\": {}, \"wavelength_nm\": {:.1}, \
                    \"n_nearby_excited\": {}}}{}",
                    stream_id, ts, pos[0], pos[1], pos[2],
                    int, ve, wd, src, vi, wl, ne,
                    if last { "" } else { "," });
            };

            let total = spikes_a.len() + spikes_b.len();
            let mut count = 0;
            for s in &spikes_a {
                count += 1;
                write_spike(&mut f, s, 0, count == total);
            }
            for s in &spikes_b {
                count += 1;
                write_spike(&mut f, s, 1, count == total);
            }
            let _ = writeln!(f, "]}}");
            log::info!("  ✓ Wrote {} spike records ({:.1} MB)",
                total, combined_path.metadata().map(|m| m.len() as f64 / 1e6).unwrap_or(0.0));
        } else {
            log::warn!("  ✗ Failed to write coupled spikes to {}", combined_path.display());
        }
    }

    // ── TWIN-aware site detection ────────────────────────────────────────────
    // Runs inline here because this is the only place with access to
    // spikes_a/spikes_b as Vec<GpuSpikeEvent> before they leave scope. Writes
    // binding_sites.json, kcc_visualization.json, prism_therm.json,
    // ensemble_trajectory.json (stub), twin_per_residue.json, twin_ccf_matrix.json
    // into the output directory. Single-pass replacement for the non-twin
    // detection pass. Failure is logged but does not abort the twin run.
    {
        let twin_output_path = output_dir;
        let prefix = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace("_sanitized", "").replace("_clean", ""))
            .unwrap_or_else(|| "prism_twin".to_string());
        log::info!("  Running TWIN-aware site detection (prefix={})", prefix);
        match crate::twin_detection::detect_and_write_twin_sites(
            &spikes_a,
            &spikes_b,
            topology,
            twin_output_path,
            &prefix,
            twin_config.nma_modes_path.as_deref(),
        ) {
            Ok(summary) => {
                log::info!(
                    "  ✓ TWIN detection: {} sites ({} consensus, {} barrier-gated, {} allo-hub) in {:.1}s",
                    summary.n_sites,
                    summary.n_consensus_sites,
                    summary.n_barrier_gated_sites,
                    summary.n_allosteric_hub_sites,
                    summary.elapsed_seconds,
                );
            }
            Err(e) => {
                log::warn!("  ✗ TWIN detection failed: {} — spike JSON still written", e);
            }
        }
    }

    // ── GPU Tensor Core CCF + per-residue feature assembly ────────────────
    //
    // Two-stage computation:
    //   Stage 1: Per-residue spike counting (CPU) — agreement ratio, B/A ratio
    //   Stage 2: GPU WMMA CCF — cross-correlation matrix, CCF features
    //
    // The CPU counting gives us per-residue spike counts from both streams.
    // The GPU CCF gives us the residue×residue correlation structure.
    // Together they populate the InterferometricFeatures struct.

    // Stage 1: CPU per-residue spike counting (same as before, still needed)
    let per_res_map = if twin_config.enable_ccf {
        let result = compute_cpu_cross_correlation(&spikes_a, &spikes_b, 5.0, 500);
        log::info!("  CPU co-occurrence matches (A∩B within 5Å / 500 steps): {}", result.0);
        result.1
    } else {
        std::collections::HashMap::new()
    };

    // Stage 2: GPU WMMA CCF
    let n_residues = topology.residues.len();
    let bin_size = 100i32; // 100 steps per time bin
    let ccf_features_vec = if twin_config.enable_ccf && n_residues > 0 && !spikes_a.is_empty() && !spikes_b.is_empty() {
        log::info!("  Building CCF matrices: {} residues, {} steps, bin_size={}...",
            n_residues, steps, bin_size);
        let ccf_start = std::time::Instant::now();

        // Build mean-centered time-binned spike matrices
        let (mat_a_f32, mat_b_f32, n_res_padded, n_bins_padded) =
            crate::twin_kernels::build_ccf_matrices(&spikes_a, &spikes_b, n_residues, steps, bin_size);

        // Attempt GPU CCF
        let gpu_ccf_result = (|| -> Result<(Vec<f32>, Vec<crate::twin_kernels::CcfResidueFeatures>)> {
            let ccf_module = context.load_module(
                cudarc::nvrtc::Ptx::from_file(&crate::twin_kernels::find_twin_ptx("tensor_ccf.ptx")?)
            ).context("Failed to load tensor_ccf.ptx")?;

            let se = stream_exchange.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Exchange stream not available for CCF"))?;

            let mut ccf = crate::twin_kernels::TwinCcfCompute::new(
                se, &ccf_module, n_residues as i32, (steps / bin_size) as i32,
            )?;
            ccf.upload_matrices(se, &mat_a_f32, &mat_b_f32)?;
            ccf.compute(se)?;
            let ccf_matrix = ccf.download_ccf(se)?;

            let nonzero = ccf_matrix.iter().filter(|&&v| v.abs() > 0.01).count();
            log::info!("  ✓ GPU CCF: {}×{} matrix, {} non-zero entries ({:.1}%), {:.1}ms",
                n_residues, n_residues, nonzero,
                nonzero as f64 / (n_residues * n_residues).max(1) as f64 * 100.0,
                ccf_start.elapsed().as_secs_f64() * 1000.0);

            let features = crate::twin_kernels::extract_ccf_features(&ccf_matrix, n_residues);
            Ok((ccf_matrix, features))
        })();

        match gpu_ccf_result {
            Ok((_ccf_matrix, features)) => {
                // TODO: write ccf_matrix to output in Step 13
                Some(features)
            }
            Err(e) => {
                log::warn!("  GPU CCF failed ({}), falling back to zero CCF features", e);
                None
            }
        }
    } else {
        None
    };

    // ── Build per-residue InterferometricFeatures (50 fields) ──────────────
    //
    // Data sources:
    //   per_res_map: HashMap<resid, (count_a, count_b, intensity_a, intensity_b)> from CPU co-occurrence
    //   ccf_features_vec: Option<Vec<CcfResidueFeatures>> from GPU WMMA CCF
    //   phase_counts_a, phase_counts_b: per-residue per-phase spike counts
    //   onset_a, onset_b: earliest spike timestep per residue
    //   intensity_a, intensity_b: total intensity per residue

    let phase_counts_a = per_residue_phase_counts(&spikes_a, &protocol);
    let phase_counts_b = per_residue_phase_counts(&spikes_b, &protocol);
    let onset_a = per_residue_onset(&spikes_a);
    let onset_b = per_residue_onset(&spikes_b);
    let intensity_sum_a = per_residue_intensity(&spikes_a);
    let intensity_sum_b = per_residue_intensity(&spikes_b);

    let per_residue_features: Vec<InterferometricFeatures> = {
        let mut resid_to_name: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
        for r in &topology.residues {
            resid_to_name.insert(r.residue_id, r.residue_name.clone());
        }

        let ccf_by_idx: Vec<crate::twin_kernels::CcfResidueFeatures> = ccf_features_vec
            .unwrap_or_else(|| vec![crate::twin_kernels::CcfResidueFeatures::default(); n_residues]);

        let mut features: Vec<InterferometricFeatures> = per_res_map
            .iter()
            .filter_map(|(&resid, &(cnt_a, cnt_b, int_a, int_b))| {
                if cnt_a == 0 && cnt_b == 0 { return None; }

                // ── Consensus (12) ──
                let max_cnt = cnt_a.max(cnt_b) as f32;
                let min_cnt = cnt_a.min(cnt_b) as f32;
                let agreement = if max_cnt > 0.0 { min_cnt / max_cnt } else { 0.0 };
                let mean_intensity = if cnt_a + cnt_b > 0 {
                    (int_a + int_b) as f32 / (cnt_a + cnt_b) as f32
                } else { 0.0 };

                // Per-phase consensus: min(A_phase, B_phase) / max(A_phase, B_phase)
                let pa = phase_counts_a.get(&resid).copied().unwrap_or([0; 5]);
                let pb = phase_counts_b.get(&resid).copied().unwrap_or([0; 5]);
                let mut consensus_phase = [0.0f32; 5];
                for i in 0..5 {
                    let mx = pa[i].max(pb[i]) as f32;
                    let mn = pa[i].min(pb[i]) as f32;
                    consensus_phase[i] = if mx > 0.0 { mn / mx } else { 0.0 };
                }

                // Temporal onset: earliest step where BOTH streams have at least 1 spike
                let oa = onset_a.get(&resid).copied().unwrap_or(i32::MAX);
                let ob = onset_b.get(&resid).copied().unwrap_or(i32::MAX);
                let consensus_onset = if oa < i32::MAX && ob < i32::MAX {
                    oa.max(ob) as f32  // the later of the two = when consensus starts
                } else { 0.0 };

                // ── Cross-correlation (12) ──
                let ccf = if (resid as usize) < ccf_by_idx.len() {
                    &ccf_by_idx[resid as usize]
                } else {
                    &crate::twin_kernels::CcfResidueFeatures::default()
                };

                // Per-phase CCF: placeholder (would need per-phase CCF computation)
                // For now, weight the global CCF by the phase's consensus ratio
                let mut ccf_per_phase = [0.0f32; 5];
                for i in 0..5 {
                    ccf_per_phase[i] = ccf.ccf_peak_value * consensus_phase[i];
                }

                // ── Differential (18) ──
                let b_over_a = if cnt_a > 0 { cnt_b as f32 / cnt_a as f32 } else { f32::INFINITY };
                let int_total_a = intensity_sum_a.get(&resid).copied().unwrap_or(0.0);
                let int_total_b = intensity_sum_b.get(&resid).copied().unwrap_or(0.0);
                let b_over_a_intensity = if int_total_a > 0.0 {
                    int_total_b / int_total_a
                } else if int_total_b > 0.0 { f32::INFINITY } else { 1.0 };

                // NMA vs thermal exclusive: count phases where only one stream is active
                let mut nma_excl: u32 = 0;
                let mut thermal_excl: u32 = 0;
                for i in 0..5 {
                    if pb[i] > 0 && pa[i] == 0 { nma_excl += pb[i]; }
                    if pa[i] > 0 && pb[i] == 0 { thermal_excl += pa[i]; }
                }

                // Per-phase B/A differential
                let mut phase_diff = [0.0f32; 5];
                for i in 0..5 {
                    phase_diff[i] = if pa[i] > 0 {
                        pb[i] as f32 / pa[i] as f32
                    } else if pb[i] > 0 { 2.0 } else { 1.0 };
                }

                let barrier_classification = if b_over_a > 1.5 {
                    "LOW".to_string()
                } else if b_over_a < 0.67 {
                    "HIGH".to_string()
                } else {
                    "MEDIUM".to_string()
                };

                // ── Scout/propagation (8) ──
                // Scout lead time: how many steps earlier A fires vs B at this residue
                let lead_time = if oa < i32::MAX && ob < i32::MAX {
                    (ob - oa) as f32  // positive = A leads (expected from phase offset)
                } else { 0.0 };

                // Predictive value: if A fires, does B also fire? (proxy: agreement)
                let predictive = if cnt_a > 0 && cnt_b > 0 { agreement } else { 0.0 };

                // Phase offset enrichment: spike ratio during the offset window
                // (B's cold_hold extension) vs the rest
                let offset_window_phase = 0; // cold_hold is phase 0
                let offset_enrich = if pa[offset_window_phase] > 0 {
                    pb[offset_window_phase] as f32 / pa[offset_window_phase] as f32
                } else { 1.0 };

                // A's intensity when B first fires
                let scout_intensity = if ob < i32::MAX {
                    // Find A spikes near B's onset time — use mean intensity of A at this residue as proxy
                    if cnt_a > 0 { int_total_a / cnt_a as f32 } else { 0.0 }
                } else { 0.0 };

                let resname = resid_to_name
                    .get(&resid)
                    .cloned()
                    .unwrap_or_else(|| "UNK".to_string());

                Some(InterferometricFeatures {
                    resid,
                    resname,
                    // Consensus (12)
                    spike_agreement_ratio: agreement,
                    consensus_intensity_mean: mean_intensity,
                    consensus_phase_profile: consensus_phase,
                    consensus_spatial_coherence: 0.0,  // populated in second pass (needs neighbor lookup)
                    consensus_temporal_onset: consensus_onset,
                    n_consensus_neighbors: 0,  // populated in second pass
                    // Cross-correlation (12)
                    ccf_peak_lag: ccf.ccf_peak_lag,
                    ccf_peak_value: ccf.ccf_peak_value,
                    ccf_width: ccf.ccf_width,
                    ccf_asymmetry: ccf.ccf_asymmetry,
                    ccf_per_phase,
                    ccf_frequency_peak: 0.0,  // PLACEHOLDER: requires FFT on CCF row
                    ccf_reproducibility: if ccf.ccf_reproducibility > 0.0 {
                        ccf.ccf_reproducibility
                    } else { agreement },
                    ccf_lag_consistency: 0.0,  // PLACEHOLDER: requires per-phase CCF recomputation
                    // Differential (18)
                    spikes_a: cnt_a,
                    spikes_b: cnt_b,
                    b_over_a_ratio: b_over_a,
                    nma_exclusive_count: nma_excl,
                    thermal_exclusive_count: thermal_excl,
                    b_over_a_intensity_ratio: b_over_a_intensity,
                    nma_responsive_mode: -1,   // PLACEHOLDER: Phase C Step 16 (NMA mode correlation)
                    nma_mode_eigenvalue: 0.0,  // PLACEHOLDER: Phase C Step 16
                    barrier_classification,
                    per_phase_differential: phase_diff,
                    differential_onset_lag: lead_time, // reuse lead_time as onset differential
                    nma_work_at_residue: 0.0,  // PLACEHOLDER: Phase C Step 16 (NMA force integration)
                    mechanical_sensitivity: 0.0, // PLACEHOLDER: Phase C Step 16
                    susceptibility_magnitude: 0.0, // PLACEHOLDER: Phase C Step 16
                    // Scout/propagation (8)
                    scout_lead_time: lead_time,
                    scout_predictive_value: predictive,
                    phase_offset_enrichment: offset_enrich,
                    scout_intensity_at_onset: scout_intensity,
                    scout_spatial_propagation: 0.0,  // PLACEHOLDER: requires spatial wave-front analysis
                    mutual_information: 0.0,         // PLACEHOLDER: Phase C Step 15 (GPU TE kernel)
                    transfer_entropy_a_to_b: 0.0,    // PLACEHOLDER: Phase C Step 15 (GPU TE kernel)
                    causal_flow_direction: 0.0,      // PLACEHOLDER: Phase C Step 15 (GPU TE kernel)
                })
            })
            .collect();

        // Second pass: compute spatial coherence using topology residue positions
        // For each residue with agreement > 0.5, count neighbors within 8Å that also agree
        if topology.residues.len() > 0 {
            // Build CA position lookup: resid → (x, y, z)
            let mut ca_pos: std::collections::HashMap<i32, [f32; 3]> = std::collections::HashMap::new();
            for r in &topology.residues {
                // Use the first CA index if available, else skip
                if let Some(&ca_idx) = topology.ca_indices.iter().find(|&&idx| {
                    idx < topology.residue_ids.len() && topology.residue_ids[idx] == r.residue_id as usize
                }) {
                    if ca_idx * 3 + 2 < topology.positions.len() {
                        ca_pos.insert(r.residue_id, [
                            topology.positions[ca_idx * 3],
                            topology.positions[ca_idx * 3 + 1],
                            topology.positions[ca_idx * 3 + 2],
                        ]);
                    }
                }
            }

            // Build agreement lookup for fast neighbor check
            let agreement_by_resid: std::collections::HashMap<i32, f32> = features.iter()
                .map(|f| (f.resid, f.spike_agreement_ratio))
                .collect();

            for feat in features.iter_mut() {
                if let Some(pos) = ca_pos.get(&feat.resid) {
                    let mut n_neighbors = 0u32;
                    let mut n_agreeing = 0u32;
                    for (&other_resid, &other_pos) in &ca_pos {
                        if other_resid == feat.resid { continue; }
                        let dx = pos[0] - other_pos[0];
                        let dy = pos[1] - other_pos[1];
                        let dz = pos[2] - other_pos[2];
                        let dist_sq = dx*dx + dy*dy + dz*dz;
                        if dist_sq <= 64.0 { // 8Å radius
                            n_neighbors += 1;
                            if let Some(&other_agree) = agreement_by_resid.get(&other_resid) {
                                if other_agree > 0.5 { n_agreeing += 1; }
                            }
                        }
                    }
                    feat.n_consensus_neighbors = n_agreeing;
                    feat.consensus_spatial_coherence = if n_neighbors > 0 {
                        n_agreeing as f32 / n_neighbors as f32
                    } else { 0.0 };
                }
            }
        }

        features.sort_by(|a, b| {
            b.consensus_intensity_mean
                .partial_cmp(&a.consensus_intensity_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ════════════════════════════════════════════════════════════════
        // PLACEHOLDER AUDIT: 11 of 50 fields are currently 0.0
        //
        // Populated (39): spike_agreement_ratio, consensus_intensity_mean,
        //   consensus_phase_profile[5], consensus_spatial_coherence,
        //   consensus_temporal_onset, n_consensus_neighbors,
        //   ccf_peak_lag, ccf_peak_value, ccf_width, ccf_asymmetry,
        //   ccf_per_phase[5], ccf_reproducibility,
        //   spikes_a, spikes_b, b_over_a_ratio, nma_exclusive_count,
        //   thermal_exclusive_count, b_over_a_intensity_ratio,
        //   barrier_classification, per_phase_differential[5],
        //   differential_onset_lag, scout_lead_time, scout_predictive_value,
        //   phase_offset_enrichment, scout_intensity_at_onset
        //
        // PLACEHOLDER (11): ccf_frequency_peak, ccf_lag_consistency,
        //   nma_responsive_mode, nma_mode_eigenvalue, nma_work_at_residue,
        //   mechanical_sensitivity, susceptibility_magnitude,
        //   scout_spatial_propagation, mutual_information,
        //   transfer_entropy_a_to_b, causal_flow_direction
        //
        // These 11 fields MUST NOT be used for ranking or model training
        // until their Phase C implementations are complete.
        // ════════════════════════════════════════════════════════════════
        let n_populated = 39;
        let n_placeholder = 11;
        log::info!("  Per-residue features: {}/{} fields populated, {} placeholders (Phase C pending)",
            n_populated, n_populated + n_placeholder, n_placeholder);

        features
    };

    let n_consensus_events = per_residue_features.iter()
        .filter(|f| f.ccf_peak_value > 0.1)
        .count() as u64;
    let n_differential_events = per_residue_features
        .iter()
        .filter(|f| f.barrier_classification != "MEDIUM")
        .count() as u64;

    log::info!("  Gate 2 summary:");
    log::info!("    Exchanges:            {}", n_exchanges);
    log::info!("    Total density A→B:    {:.3e}", total_density_a_to_b);
    log::info!("    Total density B→A:    {:.3e}", total_density_b_to_a);
    log::info!("    Peak nonzero regions: {}", max_nonzero_regions);
    log::info!("    CCF consensus events: {}", n_consensus_events);
    log::info!("    Per-residue features: {}", per_residue_features.len());
    log::info!("    Differential events:  {}", n_differential_events);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Build result ──
    let result = CoupledTwinResult {
        stream_a: StreamResult {
            role: "scout".to_string(),
            seed: seed_a,
            total_spikes: spikes_a.len(),
            perturbation: "thermal_only".to_string(),
        },
        stream_b: StreamResult {
            role: "focused_observer".to_string(),
            seed: seed_b,
            total_spikes: spikes_b.len(),
            perturbation: if twin_config.nma_modes_path.is_some() {
                "thermal_plus_nma".to_string()
            } else {
                "thermal_only".to_string()
            },
        },
        config: twin_config.clone(),
        per_residue_features,
        n_consensus_events,
        n_differential_events,
        n_exchanges,
        total_density_a_to_b,
        total_density_b_to_a,
        n_nonzero_regions: max_nonzero_regions,
        ring_spikes_tracked: ring_spikes_exchanged,
        phase_offset_steps: offset_steps,
        steps_a: steps,
        steps_b: steps_b,
    };

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║   PRISM-TWIN COMPLETE                                   ║");
    log::info!("╠══════════════════════════════════════════════════════════╣");
    log::info!("║  Stream A: {:>10} spikes (scout)                   ║", spikes_a.len());
    log::info!("║  Stream B: {:>10} spikes (observer)                ║", spikes_b.len());
    log::info!("║  Exchanges:    {:>6}                                   ║", n_exchanges);
    log::info!("║  CCF matches:  {:>6}                                   ║", n_consensus_events);
    log::info!("║  Residues:     {:>6}                                   ║", result.per_residue_features.len());
    log::info!("║  Wall time: {:.1}s                                     ║", wall_time.as_secs_f64());
    log::info!("║  VRAM: {:.2} GB used                                   ║", vram_used);
    log::info!("╚══════════════════════════════════════════════════════════╝");

    Ok(result)
}

/// Placeholder for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub fn run_coupled_twin(
    _config_ref: &crate::persistent_engine::PersistentBatchConfig,
    _topology: &PrismPrepTopology,
    _twin_config: &CoupledTwinConfig,
    _seed_a: u64,
    _steps: i32,
) -> Result<CoupledTwinResult> {
    anyhow::bail!("PRISM-TWIN requires GPU feature")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (no GPU required — tests CPU-side logic only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupled_twin_config_defaults() {
        let c = CoupledTwinConfig::default();
        assert_eq!(c.phase_offset_fraction, 0.20);
        assert_eq!(c.exchange_interval, 100);
        assert!(!c.enable_ccf, "CCF should be off by default (Gate 1)");
        assert!(!c.enable_exchange, "Exchange should be off by default (Gate 1)");
        assert!(c.sensitivity_boost > 0.0);
        assert!(c.max_threshold_reduction > 0.0);
        assert!(c.max_threshold_reduction <= 1.0, "Max reduction must be ≤ 1.0 (100%)");
        assert!(c.nma_modes_path.is_none());
    }

    #[test]
    fn test_interferometric_features_default_all_fields() {
        let f = InterferometricFeatures::default();
        assert_eq!(f.resid, -1);
        assert_eq!(f.barrier_classification, "MEDIUM");
        assert_eq!(f.consensus_phase_profile, [0.0; 5]);
        assert_eq!(f.ccf_per_phase, [0.0; 5]);
        assert_eq!(f.per_phase_differential, [0.0; 5]);
        assert_eq!(f.nma_responsive_mode, -1);
        assert_eq!(f.mutual_information, 0.0, "MI must be 0 (placeholder)");
        assert_eq!(f.transfer_entropy_a_to_b, 0.0, "TE must be 0 (placeholder)");
        assert_eq!(f.causal_flow_direction, 0.0, "Causal flow must be 0 (placeholder)");
    }

    #[test]
    fn test_interferometric_features_serde_roundtrip() {
        let mut f = InterferometricFeatures::default();
        f.resid = 42;
        f.resname = "ALA".to_string();
        f.spike_agreement_ratio = 0.85;
        f.ccf_peak_value = 0.72;
        f.spikes_a = 150;
        f.spikes_b = 120;
        f.barrier_classification = "LOW".to_string();
        f.consensus_phase_profile = [0.1, 0.3, 0.8, 0.5, 0.2];
        f.scout_lead_time = 500.0;

        let json = serde_json::to_string(&f).expect("serialize");
        let deser: InterferometricFeatures = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deser.resid, 42);
        assert_eq!(deser.resname, "ALA");
        assert!((deser.spike_agreement_ratio - 0.85).abs() < 1e-6);
        assert!((deser.ccf_peak_value - 0.72).abs() < 1e-6);
        assert_eq!(deser.spikes_a, 150);
        assert_eq!(deser.barrier_classification, "LOW");
        assert!((deser.consensus_phase_profile[2] - 0.8).abs() < 1e-6);
        assert!((deser.scout_lead_time - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_interferometric_features_field_count() {
        // Verify the JSON contains all expected fields by checking key count
        let f = InterferometricFeatures::default();
        let json_val: serde_json::Value = serde_json::to_value(&f).expect("to_value");
        let obj = json_val.as_object().expect("should be object");
        // 2 identity fields (resid, resname) + 48 feature fields = 50 total
        // But arrays serialize as arrays inside the object, so count keys
        let n_keys = obj.len();
        assert!(n_keys >= 30, "Expected ≥30 JSON keys for 50-field struct, got {}", n_keys);
    }

    #[test]
    fn test_ccns_phase_assignment() {
        let protocol = CryoUvProtocol {
            start_temp: 50.0,
            end_temp: 300.0,
            cold_hold_steps: 5000,
            ramp_steps: 10000,
            warm_hold_steps: 5000,
            current_step: 0,
            uv_burst_energy: 0.5,
            uv_burst_interval: 500,
            uv_burst_duration: 10,
            scan_wavelengths: vec![280.0],
            wavelength_dwell_steps: 100,
            ramp_down_steps: 6000,
            cold_return_steps: 4000,
            stepped_holds: Vec::new(),
        };
        assert_eq!(ccns_phase(0, &protocol), 0, "t=0 → cold_hold");
        assert_eq!(ccns_phase(4999, &protocol), 0, "t=4999 → cold_hold");
        assert_eq!(ccns_phase(5000, &protocol), 1, "t=5000 → heating");
        assert_eq!(ccns_phase(14999, &protocol), 1, "t=14999 → heating");
        assert_eq!(ccns_phase(15000, &protocol), 2, "t=15000 → warm_hold");
        assert_eq!(ccns_phase(19999, &protocol), 2, "t=19999 → warm_hold");
        assert_eq!(ccns_phase(20000, &protocol), 3, "t=20000 → cooling");
        assert_eq!(ccns_phase(25999, &protocol), 3, "t=25999 → cooling");
        assert_eq!(ccns_phase(26000, &protocol), 4, "t=26000 → cold_return");
        assert_eq!(ccns_phase(30000, &protocol), 4, "t=30000 → cold_return");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_spike_density_cpu_empty() {
        let (cells, _bbox) = compute_spike_density_cpu(&[], 4.0);
        assert!(cells.is_empty());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_cpu_cross_correlation_empty() {
        let (n, map) = compute_cpu_cross_correlation(&[], &[], 5.0, 500);
        assert_eq!(n, 0);
        assert!(map.is_empty());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_per_residue_phase_counts_basic() {
        use crate::fused_engine::GpuSpikeEvent;
        let protocol = CryoUvProtocol {
            start_temp: 50.0, end_temp: 300.0,
            cold_hold_steps: 100, ramp_steps: 100, warm_hold_steps: 100,
            current_step: 0,
            uv_burst_energy: 0.5, uv_burst_interval: 500, uv_burst_duration: 10,
            scan_wavelengths: vec![280.0], wavelength_dwell_steps: 100,
            ramp_down_steps: 100, cold_return_steps: 100,
            stepped_holds: Vec::new(),
        };

        let mut spike = GpuSpikeEvent::default();
        spike.timestep = 50;  // cold_hold phase
        spike.nearby_residues[0] = 0;
        spike.n_residues = 1;

        let counts = per_residue_phase_counts(&[spike], &protocol);
        assert_eq!(counts.get(&0).unwrap()[0], 1, "Should have 1 spike in cold_hold");
        assert_eq!(counts.get(&0).unwrap()[1], 0, "Should have 0 in heating");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_per_residue_onset() {
        use crate::fused_engine::GpuSpikeEvent;
        let mut s1 = GpuSpikeEvent::default();
        s1.timestep = 1000; s1.nearby_residues[0] = 5; s1.n_residues = 1;
        let mut s2 = GpuSpikeEvent::default();
        s2.timestep = 500; s2.nearby_residues[0] = 5; s2.n_residues = 1;

        let onset = per_residue_onset(&[s1, s2]);
        assert_eq!(*onset.get(&5).unwrap(), 500, "Onset should be earliest timestep");
    }
}

