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

/// Per-residue interferometric features from coupled observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferometricFeatures {
    pub resid: i32,
    pub resname: String,

    // Consensus
    pub spike_agreement_ratio: f32,
    pub consensus_intensity_mean: f32,

    // Cross-correlation (Layer 2)
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    pub ccf_reproducibility: f32,

    // Differential (Layer 4)
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub b_over_a_ratio: f32,
    pub barrier_classification: String,  // "LOW", "MEDIUM", "HIGH"
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

    // ── INTERLEAVED EXECUTION ──
    let inner = fused_steps.max(1) as i32;
    let outer_steps_a = (steps + inner - 1) / inner;
    let outer_steps_b = (steps_b + inner - 1) / inner;
    let outer_steps = outer_steps_a.max(outer_steps_b);  // run until B finishes

    log::info!("  Running {} interleaved outer steps (A={}, B={} fused steps)...",
        outer_steps, outer_steps_a, outer_steps_b);

    let start = std::time::Instant::now();
    let mut spikes_a_total = 0usize;
    let mut spikes_b_total = 0usize;

    // Gate 2 exchange accumulators
    let mut n_exchanges: u32 = 0;
    let mut total_density_a_to_b: f64 = 0.0;
    let mut total_density_b_to_a: f64 = 0.0;
    let mut max_nonzero_regions: u32 = 0;

    let mut a_finished = false;

    for step in 0..outer_steps {
        // Step both engines. A stops after outer_steps_a, B continues to outer_steps_b.
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

        // ── Gate 2: Spike density exchange ────────────────────────────────────
        if twin_config.enable_exchange && step as u32 % twin_config.exchange_interval == 0 {
            // Snapshot accumulated spikes from both engines mid-run.
            // These are *cumulative* from the start, so the density represents
            // the history up to this exchange point.
            let snap_a = engine_a.get_accumulated_spikes();
            let snap_b = engine_b.get_accumulated_spikes();

            // Compute coarse 4Å density grids on CPU
            let cell_ang = twin_config.region_size as f32; // region_size is in Å
            let (cells_a, _bbox_a) = compute_spike_density_cpu(&snap_a, cell_ang);
            let (cells_b, _bbox_b) = compute_spike_density_cpu(&snap_b, cell_ang);

            // Summarise: total intensity and non-zero region count
            let density_a: f64 = cells_a.iter().map(|c| c.total_intensity).sum();
            let density_b: f64 = cells_b.iter().map(|c| c.total_intensity).sum();
            let nonzero_a = cells_a.len() as u32;
            let nonzero_b = cells_b.len() as u32;
            let nonzero_union = {
                // Union of active regions: use a HashSet on the grid keys
                use std::collections::HashSet;
                let keys_a: HashSet<_> = cells_a.iter().map(|c| c.idx).collect();
                let keys_b: HashSet<_> = cells_b.iter().map(|c| c.idx).collect();
                (keys_a.union(&keys_b).count()) as u32
            };

            // Accumulate exchange statistics
            n_exchanges += 1;
            total_density_a_to_b += density_a;
            total_density_b_to_a += density_b;
            if nonzero_union > max_nonzero_regions {
                max_nonzero_regions = nonzero_union;
            }

            log::debug!(
                "  [exchange #{n_exchanges}] step={step}: \
                 A regions={nonzero_a} density={density_a:.3e} | \
                 B regions={nonzero_b} density={density_b:.3e} | \
                 union={nonzero_union}"
            );
        }
        // ─────────────────────────────────────────────────────────────────────

        // Progress logging
        if (step + 1) % 5000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let steps_per_sec = (step + 1) as f64 / elapsed;
            log::info!("  Step {}/{}: A={} B={} spikes  exchanges={n_exchanges} ({:.0} steps/s)",
                step + 1, outer_steps, spikes_a_total, spikes_b_total, steps_per_sec);
        }
    }

    let wall_time = start.elapsed();
    log::info!("  Simulation complete: {:.1}s wall time", wall_time.as_secs_f64());

    // ── Collect accumulated spikes ──
    let spikes_a = engine_a.get_accumulated_spikes();
    let spikes_b = engine_b.get_accumulated_spikes();

    log::info!("  Stream A accumulated: {} spikes", spikes_a.len());
    log::info!("  Stream B accumulated: {} spikes", spikes_b.len());

    // ── Gate 2: CPU cross-correlation + per-residue feature assembly ──────────
    let (n_ccf_matches, per_res_map) = if twin_config.enable_ccf {
        log::info!("  Computing CPU cross-correlation (spatial=5Å, temporal=500 steps)...");
        let result = compute_cpu_cross_correlation(&spikes_a, &spikes_b, 5.0, 500);
        log::info!("  CCF matches (A∩B within 5Å / 500 steps): {}", result.0);
        result
    } else {
        (0u64, std::collections::HashMap::new())
    };

    // Build per-residue features from consensus spike counts.
    // For Gate 2 we populate the residue list using topology residue metadata,
    // with spike_agreement_ratio = min(count_a, count_b) / max(count_a, count_b),
    // and consensus_intensity_mean from the averaged intensity.
    // CCF fields are left at 0 until the GPU CCF kernel is integrated (Gate 3).
    let per_residue_features: Vec<InterferometricFeatures> = {
        // Build a lookup: resid → (resname) from topology
        let mut resid_to_name: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
        for r in &topology.residues {
            resid_to_name.insert(r.residue_id, r.residue_name.clone());
        }

        let mut features: Vec<InterferometricFeatures> = per_res_map
            .iter()
            .filter_map(|(&resid, &(cnt_a, cnt_b, int_a, int_b))| {
                // Only emit residues that appeared in at least one stream
                if cnt_a == 0 && cnt_b == 0 { return None; }
                let max_cnt = cnt_a.max(cnt_b) as f32;
                let min_cnt = cnt_a.min(cnt_b) as f32;
                let agreement = if max_cnt > 0.0 { min_cnt / max_cnt } else { 0.0 };
                let mean_intensity = if cnt_a + cnt_b > 0 {
                    (int_a + int_b) as f32 / (cnt_a + cnt_b) as f32
                } else {
                    0.0
                };
                let b_over_a = if cnt_a > 0 {
                    cnt_b as f32 / cnt_a as f32
                } else {
                    f32::INFINITY
                };
                // Simple barrier classification by B/A ratio (Layer 4 placeholder)
                let barrier_classification = if b_over_a > 1.5 {
                    "LOW".to_string()      // NMA-enhanced B sees more → low barrier
                } else if b_over_a < 0.67 {
                    "HIGH".to_string()     // B sees fewer spikes → high barrier
                } else {
                    "MEDIUM".to_string()
                };
                let resname = resid_to_name
                    .get(&resid)
                    .cloned()
                    .unwrap_or_else(|| "UNK".to_string());
                Some(InterferometricFeatures {
                    resid,
                    resname,
                    spike_agreement_ratio: agreement,
                    consensus_intensity_mean: mean_intensity,
                    // CCF fields: Gate 3+ (GPU kernel integration)
                    ccf_peak_lag: 0,
                    ccf_peak_value: 0.0,
                    ccf_width: 0.0,
                    ccf_asymmetry: 0.0,
                    ccf_reproducibility: agreement,  // proxy until real CCF
                    spikes_a: cnt_a,
                    spikes_b: cnt_b,
                    b_over_a_ratio: b_over_a,
                    barrier_classification,
                })
            })
            .collect();

        // Sort by consensus intensity descending for deterministic output
        features.sort_by(|a, b| {
            b.consensus_intensity_mean
                .partial_cmp(&a.consensus_intensity_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        features
    };

    let n_consensus_events = n_ccf_matches;
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

