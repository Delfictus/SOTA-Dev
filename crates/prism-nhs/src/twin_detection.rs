//! # PRISM-TWIN site detection
//!
//! First-class TWIN-aware site detection that uses the interferometric data
//! produced by `coupled_md::run_coupled_twin`. This is NOT a wrapper around
//! single-stream detection — it consumes signals that only coupled observation
//! can produce:
//!
//! 1. **Consensus** — per-residue agreement between Stream A (scout) and
//!    Stream B (observer). High consensus = reproducible pocket dynamics,
//!    detectable only because we have two independent observations.
//! 2. **Differential** — residues that light up in Stream B but not Stream A.
//!    Stream B carries NMA-biased perturbation during warm_hold; residues
//!    active only in B are BARRIER-GATED (invisible to thermal-only MD).
//! 3. **Cross-stream CCF** — mean-centered cross-correlation between the two
//!    streams' per-residue spike matrices. High distant-pair CCF = allosteric
//!    coupling. One site opening causes another to open.
//! 4. **Phase structure** — per-phase spike counts per stream from the CCNS
//!    protocol (cold_hold, heating, warm_hold, cooling, cold_return) for
//!    thermal vs barrier-gated classification.
//!
//! Pipeline:
//!
//! ```text
//! spikes_a + spikes_b + topology
//!   │
//!   ├─ map spikes to nearest CA (per stream, per phase)
//!   │
//!   ├─ consensus vector (both streams agree)
//!   ├─ differential vector (B-only: NMA-exclusive)
//!   ├─ mean-centered CCF matrix (allosteric)
//!   │
//!   ├─ candidate residues = union of
//!   │    - top consensus
//!   │    - top differential
//!   │    - top CCF centrality
//!   │
//!   ├─ spatial cluster candidates → raw sites
//!   │
//!   ├─ classify each site:
//!   │    CONSENSUS_CRYPTIC | BARRIER_GATED | ALLOSTERIC_HUB
//!   │    COOPERATIVE_NETWORK | NMA_RESPONSIVE | THERMAL_TRANSIENT
//!   │    PREFORMED_STABLE
//!   │
//!   ├─ twin rank score (6 signals, interferometric)
//!   │
//!   └─ write {prefix}.binding_sites.json (twin schema)
//!         {prefix}.kcc_visualization.json (CCF-derived)
//!         {prefix}.topology.prism_therm.json (phase-derived)
//!         {prefix}.ensemble_trajectory.json (stub)
//!         {prefix}.twin_per_residue.json (48+ TWIN fields per residue)
//!         {prefix}.twin_ccf_matrix.json (full CCF matrix for downstream)
//! ```

#![cfg(feature = "gpu")]

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;

use crate::fused_engine::GpuSpikeEvent;
use crate::input::PrismPrepTopology;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — tunable detection parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial clustering radius for site construction (Å).
const CLUSTER_EPS_ANGSTROM: f32 = 8.0;

/// Minimum candidate residues per cluster to be emitted as a site.
const CLUSTER_MIN_PTS: usize = 3;

/// Nearest-CA cutoff for spike→residue assignment (Å).
const SPIKE_TO_CA_CUTOFF: f32 = 12.0;

/// Number of time bins for CCF matrix.
const CCF_TIME_BINS: usize = 200;

/// Maximum spikes per stream to use for CCF (downsampled if exceeded).
const CCF_SAMPLE_LIMIT: usize = 600_000;

/// Allosteric threshold: pairs with |i-j| > this AND ccf > threshold are allosteric.
const ALLOSTERIC_MIN_SEPARATION: usize = 20;

/// Protocol phase boundaries as fractions of total timesteps (--fast --hysteresis).
/// Matches the engine's CCNS 5-phase protocol exactly.
const PHASE_BOUNDS_FAST_HYSTERESIS: [(f32, f32, &str); 5] = [
    (0.000, 0.311, "cold_hold"),
    (0.311, 0.444, "heating"),
    (0.444, 0.778, "warm_hold"),
    (0.778, 0.889, "cooling"),
    (0.889, 1.000, "cold_return"),
];

/// TWIN rank score weights. All must sum to 1.0.
const RANK_W_CONSENSUS:      f32 = 0.30;
const RANK_W_DIFFERENTIAL:   f32 = 0.20;
const RANK_W_ALLOSTERIC:     f32 = 0.15;
const RANK_W_DRUGGABILITY:   f32 = 0.15;
const RANK_W_HYSTERESIS:     f32 = 0.10;
const RANK_W_SPIKE_DENSITY:  f32 = 0.10;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of the twin detection run. Written into coupled_twin_result.json
/// by the caller to keep a single source of truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinDetectionSummary {
    pub n_sites: usize,
    pub n_consensus_sites: usize,
    pub n_barrier_gated_sites: usize,
    pub n_allosteric_hub_sites: usize,
    pub n_cooperative_network_sites: usize,
    pub n_nma_responsive_sites: usize,
    pub n_thermal_transient_sites: usize,
    pub n_preformed_stable_sites: usize,
    pub ccf_distant_pair_mean: f32,
    pub ccf_max_offdiag: f32,
    pub n_candidate_residues: usize,
    pub n_ca_atoms: usize,
    pub elapsed_seconds: f64,
    pub files_written: Vec<String>,
}

/// The full twin detection entry point. Call AFTER `run_coupled_twin` has
/// returned, passing the two spike streams plus metadata. Writes all twin
/// detection artifacts to `output_dir` under `{prefix}.*` naming.
///
/// This function does NOT require a GPU (all CPU), but is gated on the gpu
/// feature because it consumes `GpuSpikeEvent`.
pub fn detect_and_write_twin_sites(
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    topology: &PrismPrepTopology,
    output_dir: &Path,
    prefix: &str,
    nma_modes_path: Option<&str>,
) -> Result<TwinDetectionSummary> {
    let t0 = std::time::Instant::now();
    std::fs::create_dir_all(output_dir)?;

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  PRISM-TWIN INTERFEROMETRIC SITE DETECTION                     ║");
    log::info!("║  Consumes: consensus + differential + CCF + phase structure   ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // ── Structure ──
    let ca_xyz = extract_ca_positions(topology);
    let n_res = ca_xyz.len();
    log::info!("  Structure: {} CA atoms", n_res);
    log::info!("  Stream A: {} spikes (scout)", spikes_a.len());
    log::info!("  Stream B: {} spikes (observer, thermal+NMA)", spikes_b.len());

    // ── Step 1: per-residue per-phase spike counts ──
    let counts_a = assign_spikes_to_residues(spikes_a, &ca_xyz);
    let counts_b = assign_spikes_to_residues(spikes_b, &ca_xyz);

    let max_ts = spikes_a.iter().chain(spikes_b.iter())
        .map(|s| s.timestep).max().unwrap_or(1) as f32;
    let phase_counts_a = per_phase_spike_counts(spikes_a, &ca_xyz, max_ts, n_res);
    let phase_counts_b = per_phase_spike_counts(spikes_b, &ca_xyz, max_ts, n_res);

    // ── Step 2: consensus vector ──
    let consensus = compute_consensus_vector(&counts_a, &counts_b);
    let n_consensus = consensus.iter().filter(|&&c| c > 0.0).count();
    log::info!("  Consensus residues (both streams active): {}/{}", n_consensus, n_res);

    // ── Step 3: differential vector ──
    let differential = compute_differential_vector(&counts_a, &counts_b);
    let n_differential = differential.iter().filter(|&&d| d > 0.0).count();
    log::info!("  Differential residues (B-only, NMA-exclusive): {}/{}", n_differential, n_res);

    // ── Step 4: mean-centered CCF matrix ──
    let (ccf, ccf_stats) = compute_ccf_matrix(spikes_a, spikes_b, &ca_xyz, n_res);
    log::info!("  CCF: distant-pair mean={:.4} max_offdiag={:.4} (< 0.15 ok)",
        ccf_stats.distant_pair_mean, ccf_stats.max_offdiag);

    // ── Step 5: candidate residues (union) ──
    let candidate_residues = build_candidate_residues(&consensus, &differential, &ccf);
    log::info!("  Candidate residues (union top-consensus / top-differential / top-CCF-centrality): {}",
        candidate_residues.len());

    // ── Step 6: spatial cluster → raw sites ──
    let raw_sites = cluster_candidates(&candidate_residues, &ca_xyz);
    log::info!("  Raw spatial clusters: {}", raw_sites.len());

    // ── Step 7: per-site aggregate + classify + rank ──
    let mut sites = build_twin_sites(
        &raw_sites, &ca_xyz, topology,
        &counts_a, &counts_b,
        &phase_counts_a, &phase_counts_b,
        &consensus, &differential,
        &ccf,
    );

    // Classification, NMA responsive modes, twin rank
    for site in sites.iter_mut() {
        classify_twin_site(site);
    }
    compute_twin_ranks(&mut sites);
    sites.sort_by(|a, b| {
        b.twin_rank_score.partial_cmp(&a.twin_rank_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, site) in sites.iter_mut().enumerate() {
        site.rank = (rank + 1) as u32;
    }

    let class_counts = tally_classifications(&sites);
    log::info!("  Sites detected: {} total", sites.len());
    log::info!("    CONSENSUS_CRYPTIC:    {}", class_counts.consensus_cryptic);
    log::info!("    BARRIER_GATED:        {}", class_counts.barrier_gated);
    log::info!("    ALLOSTERIC_HUB:       {}", class_counts.allosteric_hub);
    log::info!("    COOPERATIVE_NETWORK:  {}", class_counts.cooperative_network);
    log::info!("    NMA_RESPONSIVE:       {}", class_counts.nma_responsive);
    log::info!("    THERMAL_TRANSIENT:    {}", class_counts.thermal_transient);
    log::info!("    PREFORMED_STABLE:     {}", class_counts.preformed_stable);

    // ── Step 8: Build per-residue twin features ──
    let per_residue = build_per_residue_twin_features(
        &ca_xyz, topology,
        &counts_a, &counts_b,
        &phase_counts_a, &phase_counts_b,
        &consensus, &differential,
        &ccf,
    );

    // ── Step 9: Write outputs ──
    let mut files_written = Vec::new();

    // 1. binding_sites.json (twin schema, drop-in compatible with postprocess_twin.py)
    let bs_path = output_dir.join(format!("{prefix}.binding_sites.json"));
    write_binding_sites_json(&bs_path, &sites, topology)?;
    files_written.push(bs_path.display().to_string());

    // 2. kcc_visualization.json (CCF-derived)
    let kcc_path = output_dir.join(format!("{prefix}.kcc_visualization.json"));
    write_kcc_json(&kcc_path, &sites, &ccf, topology)?;
    files_written.push(kcc_path.display().to_string());

    // 3. prism_therm.json (phase-derived)
    let therm_path = output_dir.join(format!("{prefix}.topology.prism_therm.json"));
    write_prism_therm_json(&therm_path, &sites)?;
    files_written.push(therm_path.display().to_string());

    // 4. ensemble_trajectory.json (stub — empty list, twin doesn't track per-replica ensemble)
    let ens_path = output_dir.join(format!("{prefix}.ensemble_trajectory.json"));
    std::fs::write(&ens_path, r#"{"replicas": [], "note": "TWIN mode: ensemble trajectory is captured in coupled_spikes.json"}"#)?;
    files_written.push(ens_path.display().to_string());

    // 5. Per-residue twin features
    let pr_path = output_dir.join(format!("{prefix}.twin_per_residue.json"));
    let pr_json = serde_json::to_string_pretty(&per_residue)?;
    std::fs::write(&pr_path, pr_json)?;
    files_written.push(pr_path.display().to_string());

    // 6. CCF matrix (flat row-major JSON for downstream consumption)
    let ccf_path = output_dir.join(format!("{prefix}.twin_ccf_matrix.json"));
    let ccf_flat: Vec<f32> = ccf.iter().flatten().copied().collect();
    let ccf_wire = serde_json::json!({
        "n_residues": n_res,
        "shape": [n_res, n_res],
        "ccf_row_major": ccf_flat,
        "distant_pair_mean": ccf_stats.distant_pair_mean,
        "max_offdiag": ccf_stats.max_offdiag,
        "mean_centered": true,
        "method": "mean_centered_per_residue_unit_norm_double_centering",
    });
    std::fs::write(&ccf_path, serde_json::to_string(&ccf_wire)?)?;
    files_written.push(ccf_path.display().to_string());

    let summary = TwinDetectionSummary {
        n_sites: sites.len(),
        n_consensus_sites: class_counts.consensus_cryptic,
        n_barrier_gated_sites: class_counts.barrier_gated,
        n_allosteric_hub_sites: class_counts.allosteric_hub,
        n_cooperative_network_sites: class_counts.cooperative_network,
        n_nma_responsive_sites: class_counts.nma_responsive,
        n_thermal_transient_sites: class_counts.thermal_transient,
        n_preformed_stable_sites: class_counts.preformed_stable,
        ccf_distant_pair_mean: ccf_stats.distant_pair_mean,
        ccf_max_offdiag: ccf_stats.max_offdiag,
        n_candidate_residues: candidate_residues.len(),
        n_ca_atoms: n_res,
        elapsed_seconds: t0.elapsed().as_secs_f64(),
        files_written: files_written.iter().map(|p| {
            Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
        }).collect(),
    };

    log::info!("  Twin detection: {} sites in {:.1}s",
        sites.len(), summary.elapsed_seconds);

    let _ = nma_modes_path;  // Reserved for future per-mode attribution (Gate 4)

    Ok(summary)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TwinSiteClass {
    ConsensusCryptic,
    BarrierGated,
    AllostericHub,
    CooperativeNetwork,
    NmaResponsive,
    ThermalTransient,
    PreformedStable,
}

impl TwinSiteClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConsensusCryptic    => "CONSENSUS_CRYPTIC",
            Self::BarrierGated        => "BARRIER_GATED",
            Self::AllostericHub       => "ALLOSTERIC_HUB",
            Self::CooperativeNetwork  => "COOPERATIVE_NETWORK",
            Self::NmaResponsive       => "NMA_RESPONSIVE",
            Self::ThermalTransient    => "THERMAL_TRANSIENT",
            Self::PreformedStable     => "PREFORMED_STABLE",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinSite {
    pub id: u32,
    pub rank: u32,
    pub classification: String,      // stringified TwinSiteClass
    pub twin_site_class: String,     // same, redundant for compat
    pub centroid: [f32; 3],
    pub volume_angstrom3: f32,       // convex hull estimate
    pub n_lining_residues: usize,
    pub residue_ids: Vec<i32>,       // topology resids (0-based in topology space)
    pub lining_residues: Vec<LiningResidue>,

    // Raw totals (over both streams)
    pub spike_count: u64,
    pub spikes_a: u64,
    pub spikes_b: u64,

    // Twin signals (in [0,1] unless noted)
    pub consensus_fraction: f32,     // fraction of lining residues active in BOTH streams
    pub differential_signal: f32,    // fraction of lining residues active ONLY in B (NMA-exclusive)
    pub ccf_centrality: f32,         // mean CCF to residues outside the site
    pub ccf_allosteric_score: f32,   // normalized count of strong distant CCF pairs
    pub hysteresis_asymmetry: f32,   // |warm_hold_frac - cold_hold_frac|
    pub druggability: f32,           // heuristic from lining residue chemistry
    pub spike_density: f32,          // spikes per lining residue per 1000 steps, normalized

    // Phase structure
    pub warm_hold_fraction: f32,
    pub cold_hold_fraction: f32,
    pub heating_fraction: f32,
    pub cooling_fraction: f32,
    pub cold_return_fraction: f32,

    // Engine compat fields (read by postprocess_twin.py)
    pub quality_score: f32,          // alias for twin_rank_score, for compat
    pub druggability_score: f32,     // duplicate of druggability for compat
    pub classification_engine: String,  // LIGSITE-like label: "Cryptic", "Druggable", etc
    pub therm_class: String,         // "CRYPTIC", "RESPONSIVE", "DYNAMIC", "INERT"
    pub aromatic_score: f32,

    // Twin ranking output
    pub twin_rank_score: f32,
    pub rank_components: RankComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiningResidue {
    pub resid: i32,
    pub resname: String,
    pub min_distance_angstrom: f32,
    pub aa1: String,
    pub in_consensus: bool,
    pub in_differential: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankComponents {
    pub consensus_component: f32,
    pub differential_component: f32,
    pub allosteric_component: f32,
    pub druggability_component: f32,
    pub hysteresis_component: f32,
    pub spike_density_component: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinPerResidue {
    pub resid: i32,
    pub resname: String,
    pub aa1: String,
    pub ca_xyz: [f32; 3],

    // Consensus / differential
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub total_spikes: u32,
    pub consensus_score: f32,
    pub differential_score: f32,
    pub b_over_a_ratio: f32,

    // CCF
    pub ccf_mean: f32,
    pub ccf_max: f32,
    pub ccf_n_distant_above_0p5: u32,
    pub ccf_centrality: f32,

    // Phase structure
    pub phase_cold_hold: u32,
    pub phase_heating: u32,
    pub phase_warm_hold: u32,
    pub phase_cooling: u32,
    pub phase_cold_return: u32,
    pub warm_hold_fraction: f32,

    // Barrier
    pub barrier_class: String,   // LOW / MEDIUM / HIGH / UNKNOWN
}

struct CcfStats {
    distant_pair_mean: f32,
    max_offdiag: f32,
}

struct ClassTally {
    consensus_cryptic: usize,
    barrier_gated: usize,
    allosteric_hub: usize,
    cooperative_network: usize,
    nma_responsive: usize,
    thermal_transient: usize,
    preformed_stable: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Structure helpers
// ─────────────────────────────────────────────────────────────────────────────

fn extract_ca_positions(topo: &PrismPrepTopology) -> Vec<[f32; 3]> {
    topo.ca_indices.iter().map(|&ai| {
        let base = ai * 3;
        [topo.positions[base], topo.positions[base + 1], topo.positions[base + 2]]
    }).collect()
}

fn assign_spikes_to_residues(
    spikes: &[GpuSpikeEvent],
    ca_xyz: &[[f32; 3]],
) -> Vec<u32> {
    let n = ca_xyz.len();
    let mut counts = vec![0u32; n];
    if n == 0 { return counts; }
    let cutoff2 = SPIKE_TO_CA_CUTOFF * SPIKE_TO_CA_CUTOFF;
    for s in spikes {
        let pos = s.position;  // Packed struct copy
        let mut best_idx = usize::MAX;
        let mut best_d2 = cutoff2;
        for (i, ca) in ca_xyz.iter().enumerate() {
            let dx = pos[0] - ca[0];
            let dy = pos[1] - ca[1];
            let dz = pos[2] - ca[2];
            let d2 = dx*dx + dy*dy + dz*dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = i;
            }
        }
        if best_idx != usize::MAX {
            counts[best_idx] += 1;
        }
    }
    counts
}

fn per_phase_spike_counts(
    spikes: &[GpuSpikeEvent],
    ca_xyz: &[[f32; 3]],
    max_ts: f32,
    n_res: usize,
) -> [Vec<u32>; 5] {
    let mut phase_counts: [Vec<u32>; 5] = [
        vec![0; n_res], vec![0; n_res], vec![0; n_res],
        vec![0; n_res], vec![0; n_res],
    ];
    if n_res == 0 || max_ts <= 0.0 { return phase_counts; }
    let cutoff2 = SPIKE_TO_CA_CUTOFF * SPIKE_TO_CA_CUTOFF;
    for s in spikes {
        let pos = s.position;
        let ts_f = s.timestep as f32;
        let frac = ts_f / max_ts;
        let phase_idx = PHASE_BOUNDS_FAST_HYSTERESIS.iter().position(|&(lo, hi, _)| {
            frac >= lo && frac < hi
        }).unwrap_or(4);
        // nearest CA
        let mut best_idx = usize::MAX;
        let mut best_d2 = cutoff2;
        for (i, ca) in ca_xyz.iter().enumerate() {
            let dx = pos[0] - ca[0];
            let dy = pos[1] - ca[1];
            let dz = pos[2] - ca[2];
            let d2 = dx*dx + dy*dy + dz*dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = i;
            }
        }
        if best_idx != usize::MAX {
            phase_counts[phase_idx][best_idx] += 1;
        }
    }
    phase_counts
}

// ─────────────────────────────────────────────────────────────────────────────
// Consensus / differential / CCF
// ─────────────────────────────────────────────────────────────────────────────

fn compute_consensus_vector(counts_a: &[u32], counts_b: &[u32]) -> Vec<f32> {
    let n = counts_a.len().min(counts_b.len());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let a = counts_a[i] as f32;
        let b = counts_b[i] as f32;
        if a > 0.0 && b > 0.0 {
            let mn = a.min(b);
            let mx = a.max(b);
            // Consensus = (geometric mean of counts) * (agreement ratio)
            out[i] = (a * b).sqrt() * (mn / mx);
        }
    }
    out
}

fn compute_differential_vector(counts_a: &[u32], counts_b: &[u32]) -> Vec<f32> {
    let n = counts_a.len().min(counts_b.len());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let a = counts_a[i] as f32;
        let b = counts_b[i] as f32;
        // Differential = Stream B excess over Stream A (NMA-exclusive signal)
        if b > a {
            // A small a with large b → large differential.
            // A large a with slightly larger b → small differential.
            let excess = b - a;
            let normalization = 1.0 + a;   // penalize sites that are already active in A
            out[i] = excess / normalization;
        }
    }
    out
}

fn compute_ccf_matrix(
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    ca_xyz: &[[f32; 3]],
    n_res: usize,
) -> (Vec<Vec<f32>>, CcfStats) {
    // Time-binned per-residue spike matrices for A and B, then mean-center
    // along BOTH axes (remove the global CCNS protocol envelope AND the
    // per-residue mean), then unit-normalize rows, then correlate A with B.
    let mut mat_a = vec![vec![0.0f32; CCF_TIME_BINS]; n_res];
    let mut mat_b = vec![vec![0.0f32; CCF_TIME_BINS]; n_res];

    if n_res == 0 {
        let empty = vec![vec![0.0f32; 0]; 0];
        return (empty, CcfStats { distant_pair_mean: 0.0, max_offdiag: 0.0 });
    }

    let max_ts = spikes_a.iter().chain(spikes_b.iter())
        .map(|s| s.timestep).max().unwrap_or(1).max(1) as f32;
    let bin_size = (max_ts / CCF_TIME_BINS as f32).max(1.0);

    // Downsample if needed
    let stride_a = (spikes_a.len() / CCF_SAMPLE_LIMIT).max(1);
    let stride_b = (spikes_b.len() / CCF_SAMPLE_LIMIT).max(1);

    let cutoff2 = SPIKE_TO_CA_CUTOFF * SPIKE_TO_CA_CUTOFF;
    let mut accumulate = |spikes: &[GpuSpikeEvent], stride: usize, mat: &mut [Vec<f32>]| {
        for (k, s) in spikes.iter().enumerate() {
            if k % stride != 0 { continue; }
            let pos = s.position;
            let mut best_idx = usize::MAX;
            let mut best_d2 = cutoff2;
            for (i, ca) in ca_xyz.iter().enumerate() {
                let dx = pos[0] - ca[0];
                let dy = pos[1] - ca[1];
                let dz = pos[2] - ca[2];
                let d2 = dx*dx + dy*dy + dz*dz;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = i;
                }
            }
            if best_idx == usize::MAX { continue; }
            let bin = ((s.timestep as f32 / bin_size) as usize).min(CCF_TIME_BINS - 1);
            mat[best_idx][bin] += 1.0;
        }
    };
    accumulate(spikes_a, stride_a, &mut mat_a);
    accumulate(spikes_b, stride_b, &mut mat_b);

    // Double-centering: subtract column mean (across residues, per time bin)
    // then row mean (across time, per residue). This kills the CCNS envelope.
    double_center(&mut mat_a);
    double_center(&mut mat_b);

    // Unit-normalize rows
    unit_normalize_rows(&mut mat_a);
    unit_normalize_rows(&mut mat_b);

    // CCF[i][j] = <mat_a[i], mat_b[j]> (dot product of unit-normalized rows)
    // Symmetrize: (A⋅B^T + B⋅A^T) / 2
    let mut ccf = vec![vec![0.0f32; n_res]; n_res];
    for i in 0..n_res {
        for j in 0..n_res {
            let mut dot_ab = 0.0f32;
            for t in 0..CCF_TIME_BINS {
                dot_ab += mat_a[i][t] * mat_b[j][t];
            }
            ccf[i][j] = dot_ab;
        }
    }
    // Symmetrize
    for i in 0..n_res {
        for j in (i + 1)..n_res {
            let avg = 0.5 * (ccf[i][j] + ccf[j][i]);
            ccf[i][j] = avg;
            ccf[j][i] = avg;
        }
    }
    // Set diagonal to self-correlation = 1 (unit-normalized)
    for i in 0..n_res {
        ccf[i][i] = 1.0;
    }

    // Stats
    let mut distant_sum = 0.0f64;
    let mut distant_n = 0u64;
    let mut max_offdiag = 0.0f32;
    for i in 0..n_res {
        for j in (i + 1)..n_res {
            let v = ccf[i][j];
            if v.abs() > max_offdiag.abs() { max_offdiag = v; }
            if (j - i) > ALLOSTERIC_MIN_SEPARATION {
                distant_sum += v as f64;
                distant_n += 1;
            }
        }
    }
    let distant_pair_mean = if distant_n > 0 {
        (distant_sum / distant_n as f64) as f32
    } else { 0.0 };

    (ccf, CcfStats { distant_pair_mean, max_offdiag })
}

fn double_center(mat: &mut [Vec<f32>]) {
    let n_res = mat.len();
    if n_res == 0 { return; }
    let n_bins = mat[0].len();
    if n_bins == 0 { return; }
    // Column mean (per time bin, across residues)
    let mut col_mean = vec![0.0f32; n_bins];
    for row in mat.iter() {
        for (t, v) in row.iter().enumerate() {
            col_mean[t] += *v;
        }
    }
    for v in col_mean.iter_mut() { *v /= n_res as f32; }
    for row in mat.iter_mut() {
        for (t, v) in row.iter_mut().enumerate() {
            *v -= col_mean[t];
        }
    }
    // Row mean (per residue)
    for row in mat.iter_mut() {
        let m: f32 = row.iter().copied().sum::<f32>() / n_bins as f32;
        for v in row.iter_mut() { *v -= m; }
    }
}

fn unit_normalize_rows(mat: &mut [Vec<f32>]) {
    for row in mat.iter_mut() {
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for v in row.iter_mut() { *v /= norm; }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate residues + clustering
// ─────────────────────────────────────────────────────────────────────────────

fn build_candidate_residues(
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<usize> {
    let n = consensus.len();
    if n == 0 { return Vec::new(); }

    // Top 20% consensus residues
    let mut sorted_c: Vec<(usize, f32)> = consensus.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_c: Vec<usize> = sorted_c.iter().take(n / 5 + 1)
        .filter(|&(_, v)| *v > 0.0)
        .map(|&(i, _)| i).collect();

    // Top 20% differential residues
    let mut sorted_d: Vec<(usize, f32)> = differential.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_d.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_d: Vec<usize> = sorted_d.iter().take(n / 5 + 1)
        .filter(|&(_, v)| *v > 0.0)
        .map(|&(i, _)| i).collect();

    // Top 10% CCF centrality (mean off-diagonal CCF)
    let mut centrality: Vec<(usize, f32)> = (0..n).map(|i| {
        let mut s = 0.0f32;
        for j in 0..n {
            if i != j { s += ccf[i][j].abs(); }
        }
        let mean = if n > 1 { s / (n as f32 - 1.0) } else { 0.0 };
        (i, mean)
    }).collect();
    centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ccf: Vec<usize> = centrality.iter().take(n / 10 + 1)
        .filter(|&(_, v)| *v > 0.05)
        .map(|&(i, _)| i).collect();

    let mut set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for i in &top_c { set.insert(*i); }
    for i in &top_d { set.insert(*i); }
    for i in &top_ccf { set.insert(*i); }
    set.into_iter().collect()
}

fn cluster_candidates(
    candidates: &[usize],
    ca_xyz: &[[f32; 3]],
) -> Vec<Vec<usize>> {
    // Simple DBSCAN over candidate CA positions
    let n = candidates.len();
    let eps2 = CLUSTER_EPS_ANGSTROM * CLUSTER_EPS_ANGSTROM;
    let mut visited = vec![false; n];
    let mut cluster_id = vec![-1i32; n];
    let mut next_cluster = 0i32;

    let neighbors = |idx: usize| -> Vec<usize> {
        let mut neigh = Vec::new();
        let p1 = ca_xyz[candidates[idx]];
        for j in 0..n {
            if j == idx { continue; }
            let p2 = ca_xyz[candidates[j]];
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            let dz = p1[2] - p2[2];
            if dx*dx + dy*dy + dz*dz <= eps2 {
                neigh.push(j);
            }
        }
        neigh
    };

    for i in 0..n {
        if visited[i] { continue; }
        visited[i] = true;
        let neigh = neighbors(i);
        if neigh.len() + 1 < CLUSTER_MIN_PTS { continue; }
        let cid = next_cluster;
        next_cluster += 1;
        cluster_id[i] = cid;
        let mut stack = neigh;
        while let Some(j) = stack.pop() {
            if !visited[j] {
                visited[j] = true;
                let nj = neighbors(j);
                if nj.len() + 1 >= CLUSTER_MIN_PTS {
                    for k in nj { stack.push(k); }
                }
            }
            if cluster_id[j] == -1 {
                cluster_id[j] = cid;
            }
        }
    }

    let mut clusters: Vec<Vec<usize>> = (0..next_cluster).map(|_| Vec::new()).collect();
    for (i, &cid) in cluster_id.iter().enumerate() {
        if cid >= 0 {
            clusters[cid as usize].push(candidates[i]);
        }
    }
    clusters.retain(|c| c.len() >= CLUSTER_MIN_PTS);
    clusters
}

// ─────────────────────────────────────────────────────────────────────────────
// Site construction
// ─────────────────────────────────────────────────────────────────────────────

fn build_twin_sites(
    raw_clusters: &[Vec<usize>],
    ca_xyz: &[[f32; 3]],
    topology: &PrismPrepTopology,
    counts_a: &[u32],
    counts_b: &[u32],
    phase_counts_a: &[Vec<u32>; 5],
    phase_counts_b: &[Vec<u32>; 5],
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<TwinSite> {
    let mut sites = Vec::with_capacity(raw_clusters.len());
    let n_res = ca_xyz.len();
    // Percentile thresholds so "in_consensus"/"in_differential" mean
    // "strongly consensus/differential" — not "had at least one spike"
    // (which is trivially true for most residues in a 30M spike run).
    let consensus_thresh = percentile_gt0(consensus, 0.75);
    let differential_thresh = percentile_gt0(differential, 0.75);

    for (ci, cluster) in raw_clusters.iter().enumerate() {
        if cluster.is_empty() { continue; }
        // Centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for &ri in cluster {
            cx += ca_xyz[ri][0];
            cy += ca_xyz[ri][1];
            cz += ca_xyz[ri][2];
        }
        let n_c = cluster.len() as f32;
        let centroid = [cx / n_c, cy / n_c, cz / n_c];

        // Lining residues: any CA within 8 Å of the centroid
        let mut lining: Vec<LiningResidue> = Vec::new();
        let mut lining_ids: Vec<i32> = Vec::new();
        let lining_cutoff2 = 8.0 * 8.0;
        for i in 0..n_res {
            let dx = ca_xyz[i][0] - centroid[0];
            let dy = ca_xyz[i][1] - centroid[1];
            let dz = ca_xyz[i][2] - centroid[2];
            let d2 = dx*dx + dy*dy + dz*dz;
            if d2 < lining_cutoff2 {
                let resname = topology.residue_names.get(topology.ca_indices[i])
                    .cloned().unwrap_or_else(|| "UNK".to_string());
                let aa1 = aa3_to_aa1(&resname);
                    lining.push(LiningResidue {
                    resid: i as i32,
                    resname,
                    aa1,
                    min_distance_angstrom: d2.sqrt(),
                    in_consensus: consensus[i] > consensus_thresh,
                    in_differential: differential[i] > differential_thresh,
                });
                lining_ids.push(i as i32);
            }
        }

        // Spike totals over lining residues
        let mut sa: u64 = 0;
        let mut sb: u64 = 0;
        for lr in &lining {
            sa += counts_a[lr.resid as usize] as u64;
            sb += counts_b[lr.resid as usize] as u64;
        }
        let total = sa + sb;

        // Consensus / differential fractions over lining
        let n_consensus = lining.iter().filter(|r| r.in_consensus).count();
        let n_differential = lining.iter().filter(|r| r.in_differential).count();
        let consensus_fraction = if lining.is_empty() { 0.0 } else { n_consensus as f32 / lining.len() as f32 };
        let differential_signal = if lining.is_empty() { 0.0 } else { n_differential as f32 / lining.len() as f32 };

        // CCF centrality: mean |CCF| from lining residues to residues OUTSIDE the site
        let lining_set: std::collections::HashSet<usize> = lining.iter().map(|r| r.resid as usize).collect();
        let mut ccf_sum = 0.0f32;
        let mut ccf_n = 0u32;
        let mut strong_distant_pairs: u32 = 0;
        for &li in &lining_set {
            for j in 0..n_res {
                if lining_set.contains(&j) { continue; }
                let v = ccf[li][j].abs();
                ccf_sum += v;
                ccf_n += 1;
                if v > 0.5 && (j as i32 - li as i32).unsigned_abs() as usize > ALLOSTERIC_MIN_SEPARATION {
                    strong_distant_pairs += 1;
                }
            }
        }
        let ccf_centrality = if ccf_n > 0 { ccf_sum / ccf_n as f32 } else { 0.0 };
        let ccf_allosteric_score = (strong_distant_pairs as f32 / lining.len().max(1) as f32).min(1.0);

        // Phase fractions over lining residues
        let mut p_sum = [0u64; 5];
        for lr in &lining {
            for phase in 0..5 {
                p_sum[phase] += phase_counts_a[phase][lr.resid as usize] as u64;
                p_sum[phase] += phase_counts_b[phase][lr.resid as usize] as u64;
            }
        }
        let phase_total = p_sum.iter().sum::<u64>().max(1) as f32;
        let cold_hold_fraction   = p_sum[0] as f32 / phase_total;
        let heating_fraction     = p_sum[1] as f32 / phase_total;
        let warm_hold_fraction   = p_sum[2] as f32 / phase_total;
        let cooling_fraction     = p_sum[3] as f32 / phase_total;
        let cold_return_fraction = p_sum[4] as f32 / phase_total;
        let hysteresis_asymmetry = (warm_hold_fraction - cold_hold_fraction).abs();

        // Druggability heuristic from lining residue chemistry
        let druggability = simple_druggability(&lining);

        // Aromatic score
        let aromatic_score = lining.iter().filter(|r| is_aromatic(&r.aa1)).count() as f32
            / lining.len().max(1) as f32;

        // Spike density: (sa + sb) / n_lining, normalized to [0, 1] by a soft ceiling
        let raw_density = (total as f32 / lining.len().max(1) as f32) / 1000.0;
        let spike_density = (raw_density / (1.0 + raw_density)).min(1.0);

        // Convex-hull volume estimate (bounding sphere proxy — cheap)
        let mut rmax = 0.0f32;
        for lr in &lining {
            let ca = ca_xyz[lr.resid as usize];
            let dx = ca[0] - centroid[0];
            let dy = ca[1] - centroid[1];
            let dz = ca[2] - centroid[2];
            let d2 = dx*dx + dy*dy + dz*dz;
            if d2 > rmax { rmax = d2; }
        }
        let r = rmax.sqrt();
        let volume = (4.0 / 3.0) * std::f32::consts::PI * r * r * r * 0.5; // × 0.5 for pocket-like

        let site = TwinSite {
            id: ci as u32,
            rank: 0,                  // filled after ranking
            classification: String::new(),       // filled after classify
            twin_site_class: String::new(),
            centroid,
            volume_angstrom3: volume,
            n_lining_residues: lining.len(),
            residue_ids: lining_ids,
            lining_residues: lining,
            spike_count: total,
            spikes_a: sa,
            spikes_b: sb,
            consensus_fraction,
            differential_signal,
            ccf_centrality,
            ccf_allosteric_score,
            hysteresis_asymmetry,
            druggability,
            spike_density,
            warm_hold_fraction,
            cold_hold_fraction,
            heating_fraction,
            cooling_fraction,
            cold_return_fraction,
            quality_score: 0.0,                  // alias of twin_rank_score, filled later
            druggability_score: druggability,
            classification_engine: String::new(),
            therm_class: String::new(),
            aromatic_score,
            twin_rank_score: 0.0,
            rank_components: RankComponents {
                consensus_component: 0.0,
                differential_component: 0.0,
                allosteric_component: 0.0,
                druggability_component: 0.0,
                hysteresis_component: 0.0,
                spike_density_component: 0.0,
            },
        };
        sites.push(site);
    }

    sites
}

/// Returns the p-th percentile of nonzero values in the vector (p in [0, 1]).
/// Falls back to 0.0 if the vector has no positive values.
fn percentile_gt0(values: &[f32], p: f32) -> f32 {
    let mut nonzero: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if nonzero.is_empty() { return 0.0; }
    nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((nonzero.len() as f32) * p).floor() as usize;
    nonzero[idx.min(nonzero.len() - 1)]
}

fn simple_druggability(lining: &[LiningResidue]) -> f32 {
    if lining.is_empty() { return 0.0; }
    let n = lining.len() as f32;
    let hydrophobic = lining.iter().filter(|r| {
        matches!(r.aa1.as_str(), "A" | "V" | "L" | "I" | "M" | "F" | "W" | "Y" | "P" | "C")
    }).count() as f32 / n;
    let aromatic = lining.iter().filter(|r| is_aromatic(&r.aa1)).count() as f32 / n;
    let polar = lining.iter().filter(|r| {
        matches!(r.aa1.as_str(), "S" | "T" | "N" | "Q" | "H")
    }).count() as f32 / n;
    // Simple heuristic: prefer hydrophobic + aromatic, modest polar
    (0.5 * hydrophobic + 0.3 * aromatic + 0.2 * polar).min(1.0)
}

fn is_aromatic(aa1: &str) -> bool {
    matches!(aa1, "F" | "W" | "Y" | "H")
}

fn aa3_to_aa1(resname: &str) -> String {
    match resname {
        "ALA" => "A", "ARG" => "R", "ASN" => "N", "ASP" | "ASH" => "D",
        "CYS" | "CYX" | "CYM" => "C", "GLN" => "Q", "GLU" | "GLH" => "E",
        "GLY" => "G", "HIS" | "HID" | "HIE" | "HIP" => "H",
        "ILE" => "I", "LEU" => "L", "LYS" | "LYN" => "K", "MET" => "M",
        "PHE" => "F", "PRO" => "P", "SER" => "S", "THR" => "T",
        "TRP" => "W", "TYR" => "Y", "VAL" => "V",
        _ => "X",
    }.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification + ranking
// ─────────────────────────────────────────────────────────────────────────────

fn classify_twin_site(site: &mut TwinSite) {
    // The classification order matters: earlier branches are more specific.
    let class = if site.consensus_fraction > 0.7 && site.hysteresis_asymmetry > 0.15 {
        TwinSiteClass::ConsensusCryptic
    } else if site.differential_signal > 0.5 && site.consensus_fraction < 0.4 {
        TwinSiteClass::BarrierGated
    } else if site.ccf_allosteric_score > 0.3 && site.ccf_centrality > 0.15 {
        TwinSiteClass::AllostericHub
    } else if site.differential_signal > 0.25 && site.ccf_allosteric_score > 0.1 {
        TwinSiteClass::NmaResponsive
    } else if site.consensus_fraction > 0.5 && site.cold_hold_fraction > 0.25 {
        TwinSiteClass::PreformedStable
    } else if site.ccf_centrality > 0.1 {
        TwinSiteClass::CooperativeNetwork
    } else {
        TwinSiteClass::ThermalTransient
    };

    let class_str = class.as_str().to_string();
    site.classification = class_str.clone();
    site.twin_site_class = class_str;

    // Compat fields for postprocess_twin.py (which expects LIGSITE-like labels)
    site.classification_engine = match class {
        TwinSiteClass::ConsensusCryptic   => "Cryptic".to_string(),
        TwinSiteClass::BarrierGated       => "Cryptic".to_string(),
        TwinSiteClass::AllostericHub      => "Allosteric".to_string(),
        TwinSiteClass::CooperativeNetwork => "Allosteric".to_string(),
        TwinSiteClass::NmaResponsive      => "Cryptic".to_string(),
        TwinSiteClass::PreformedStable    => "Druggable".to_string(),
        TwinSiteClass::ThermalTransient   => "Unknown".to_string(),
    };

    site.therm_class = match class {
        TwinSiteClass::ConsensusCryptic   => "CRYPTIC",
        TwinSiteClass::BarrierGated       => "DYNAMIC",
        TwinSiteClass::AllostericHub      => "RESPONSIVE",
        TwinSiteClass::CooperativeNetwork => "RESPONSIVE",
        TwinSiteClass::NmaResponsive      => "DYNAMIC",
        TwinSiteClass::PreformedStable    => "INERT",
        TwinSiteClass::ThermalTransient   => "INERT",
    }.to_string();
}

fn compute_twin_ranks(sites: &mut [TwinSite]) {
    for site in sites.iter_mut() {
        let c = RANK_W_CONSENSUS     * site.consensus_fraction;
        let d = RANK_W_DIFFERENTIAL  * site.differential_signal;
        let a = RANK_W_ALLOSTERIC    * site.ccf_allosteric_score;
        let dr = RANK_W_DRUGGABILITY * site.druggability;
        let h = RANK_W_HYSTERESIS    * site.hysteresis_asymmetry.min(1.0);
        let sd = RANK_W_SPIKE_DENSITY * site.spike_density;
        let total = c + d + a + dr + h + sd;
        site.twin_rank_score = total;
        site.quality_score = total;  // alias for compat
        site.rank_components = RankComponents {
            consensus_component: c,
            differential_component: d,
            allosteric_component: a,
            druggability_component: dr,
            hysteresis_component: h,
            spike_density_component: sd,
        };
    }
}

fn tally_classifications(sites: &[TwinSite]) -> ClassTally {
    let mut t = ClassTally {
        consensus_cryptic: 0, barrier_gated: 0, allosteric_hub: 0,
        cooperative_network: 0, nma_responsive: 0, thermal_transient: 0,
        preformed_stable: 0,
    };
    for s in sites {
        match s.classification.as_str() {
            "CONSENSUS_CRYPTIC"    => t.consensus_cryptic += 1,
            "BARRIER_GATED"        => t.barrier_gated += 1,
            "ALLOSTERIC_HUB"       => t.allosteric_hub += 1,
            "COOPERATIVE_NETWORK"  => t.cooperative_network += 1,
            "NMA_RESPONSIVE"       => t.nma_responsive += 1,
            "THERMAL_TRANSIENT"    => t.thermal_transient += 1,
            "PREFORMED_STABLE"     => t.preformed_stable += 1,
            _ => {}
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-residue features
// ─────────────────────────────────────────────────────────────────────────────

fn build_per_residue_twin_features(
    ca_xyz: &[[f32; 3]],
    topology: &PrismPrepTopology,
    counts_a: &[u32],
    counts_b: &[u32],
    phase_counts_a: &[Vec<u32>; 5],
    phase_counts_b: &[Vec<u32>; 5],
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<TwinPerResidue> {
    let n = ca_xyz.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let sa = counts_a[i];
        let sb = counts_b[i];
        let total = sa + sb;
        let b_over_a = if sa > 0 { sb as f32 / sa as f32 } else if sb > 0 { f32::INFINITY } else { 0.0 };

        let p_cold_hold = phase_counts_a[0][i] + phase_counts_b[0][i];
        let p_heating = phase_counts_a[1][i] + phase_counts_b[1][i];
        let p_warm_hold = phase_counts_a[2][i] + phase_counts_b[2][i];
        let p_cooling = phase_counts_a[3][i] + phase_counts_b[3][i];
        let p_cold_return = phase_counts_a[4][i] + phase_counts_b[4][i];
        let warm_hold_fraction = if total > 0 {
            p_warm_hold as f32 / total as f32
        } else { 0.0 };

        // CCF features
        let mut ccf_sum = 0.0f32;
        let mut ccf_max = 0.0f32;
        let mut n_distant_strong: u32 = 0;
        for j in 0..n {
            if i == j { continue; }
            let v = ccf[i][j];
            let abs_v = v.abs();
            ccf_sum += abs_v;
            if abs_v > ccf_max { ccf_max = abs_v; }
            if abs_v > 0.5 && (j as i32 - i as i32).unsigned_abs() as usize > ALLOSTERIC_MIN_SEPARATION {
                n_distant_strong += 1;
            }
        }
        let ccf_mean = if n > 1 { ccf_sum / (n - 1) as f32 } else { 0.0 };
        let ccf_centrality = ccf_mean;

        let barrier_class = classify_barrier(sa, sb, warm_hold_fraction);
        let resname = topology.residue_names.get(topology.ca_indices[i])
            .cloned().unwrap_or_else(|| "UNK".to_string());
        let aa1 = aa3_to_aa1(&resname);

        out.push(TwinPerResidue {
            resid: i as i32,
            resname,
            aa1,
            ca_xyz: ca_xyz[i],
            spikes_a: sa,
            spikes_b: sb,
            total_spikes: total,
            consensus_score: consensus[i],
            differential_score: differential[i],
            b_over_a_ratio: b_over_a,
            ccf_mean,
            ccf_max,
            ccf_n_distant_above_0p5: n_distant_strong,
            ccf_centrality,
            phase_cold_hold: p_cold_hold,
            phase_heating: p_heating,
            phase_warm_hold: p_warm_hold,
            phase_cooling: p_cooling,
            phase_cold_return: p_cold_return,
            warm_hold_fraction,
            barrier_class,
        });
    }
    out
}

fn classify_barrier(sa: u32, sb: u32, warm_hold_frac: f32) -> String {
    if sa == 0 && sb == 0 {
        return "UNKNOWN".to_string();
    }
    let b_over_a = if sa > 0 { sb as f32 / sa as f32 } else { f32::INFINITY };
    if b_over_a > 1.5 && warm_hold_frac > 0.4 {
        "LOW".to_string()
    } else if b_over_a < 0.67 {
        "HIGH".to_string()
    } else {
        "MEDIUM".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output writers — JSON formats compatible with postprocess_twin.py
// ─────────────────────────────────────────────────────────────────────────────

fn write_binding_sites_json(
    path: &Path,
    sites: &[TwinSite],
    topology: &PrismPrepTopology,
) -> Result<()> {
    // Mirror the non-twin binding_sites.json schema that postprocess_twin.py consumes,
    // with additional twin_* fields appended to each site record.
    let sites_json: Vec<serde_json::Value> = sites.iter().map(|s| {
        let lining_resids: Vec<i32> = s.lining_residues.iter().map(|r| r.resid).collect();
        let lining_for_json: Vec<serde_json::Value> = s.lining_residues.iter().map(|lr| {
            serde_json::json!({
                "resid": lr.resid,
                "resname": lr.resname,
                "chain": "A",
                "is_catalytic": false,
                "min_distance": lr.min_distance_angstrom,
                "n_atoms": 8,  // placeholder; postprocess_twin.py doesn't need exact value
            })
        }).collect();

        serde_json::json!({
            "id": s.id,
            "rank": s.rank,
            "rank_score": s.twin_rank_score,   // alias expected by prism-postflight.py
            "centroid": s.centroid,
            "volume": s.volume_angstrom3,
            "classification": s.classification_engine,
            "therm_class": s.therm_class,
            "quality_score": s.quality_score,
            "druggability": s.druggability_score,
            "is_druggable": s.druggability_score > 0.4,
            "aromatic_score": s.aromatic_score,
            "spike_count": s.spike_count,
            "residue_ids": lining_resids,
            "lining_residues": lining_for_json,
            "ccns_tau": 0.0,
            "hysteresis_asymmetry": s.hysteresis_asymmetry,
            "relative_asymmetry": s.cold_return_fraction - s.cold_hold_fraction,
            "catalytic_residue_count": 0,
            // TWIN-exclusive fields
            "twin_site_class": s.twin_site_class,
            "twin_rank_score": s.twin_rank_score,
            "twin_consensus_fraction": s.consensus_fraction,
            "twin_differential_signal": s.differential_signal,
            "twin_ccf_centrality": s.ccf_centrality,
            "twin_ccf_allosteric_score": s.ccf_allosteric_score,
            "twin_spike_density": s.spike_density,
            "twin_spikes_a": s.spikes_a,
            "twin_spikes_b": s.spikes_b,
            "twin_phase_warm_hold_fraction": s.warm_hold_fraction,
            "twin_phase_cold_hold_fraction": s.cold_hold_fraction,
            "twin_rank_components": s.rank_components,
        })
    }).collect();

    // cryptic_sites and all_pockets MUST be lists (not ints) — prism-postflight.py
    // iterates over them and takes len(), so emitting counts here crashes the wrapper.
    let cryptic_sites_list: Vec<serde_json::Value> = sites.iter()
        .filter(|s| s.therm_class == "CRYPTIC" || s.therm_class == "DYNAMIC")
        .map(|s| serde_json::json!({
            "id": s.id,
            "centroid": s.centroid,
            "classification": s.classification_engine,
            "twin_site_class": s.twin_site_class,
        })).collect();

    let root = serde_json::json!({
        "mode": "prism-twin",
        "detection_method": "twin-aware: consensus + differential + CCF + phase",
        "n_streams": 2,
        "structure": {
            "n_residues": topology.n_residues,
            "n_atoms": topology.n_atoms,
        },
        "sites": sites_json,
        "binding_sites": sites.len(),
        "druggable_sites": sites.iter().filter(|s| s.druggability_score > 0.4).count(),
        "cryptic_sites": cryptic_sites_list,
        "all_pockets": sites_json.iter().map(|s| serde_json::json!({
            "id": s.get("id"),
            "centroid": s.get("centroid"),
        })).collect::<Vec<_>>(),
    });

    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn write_kcc_json(
    path: &Path,
    sites: &[TwinSite],
    ccf: &[Vec<f32>],
    topology: &PrismPrepTopology,
) -> Result<()> {
    // KCC visualization from CCF: top CCF pairs are "causally coupled".
    let n = ccf.len();
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = ccf[i][j].abs();
            if v > 0.5 && (j - i) > ALLOSTERIC_MIN_SEPARATION {
                edges.push((i, j, v));
            }
        }
    }
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    edges.truncate(500);

    let mut residue_rows: Vec<serde_json::Value> = Vec::with_capacity(n);
    for i in 0..n {
        let resname = topology.residue_names.get(topology.ca_indices[i])
            .cloned().unwrap_or_else(|| "UNK".to_string());
        let degree = edges.iter().filter(|(a, b, _)| *a == i || *b == i).count();
        residue_rows.push(serde_json::json!({
            "resid": i,
            "resname": resname,
            "ca_position": [0.0, 0.0, 0.0],  // not used by postprocess_twin.py
            "degree": degree,
            "ccf_centrality": degree as f32 / n.max(1) as f32,
        }));
    }

    let sites_rows: Vec<serde_json::Value> = sites.iter().map(|s| {
        serde_json::json!({
            "id": s.id,
            "centroid": s.centroid,
            "gtck_rank": s.rank,
            "rank_score": s.twin_rank_score,
            "kcc_confidence": s.ccf_centrality,
            "driver_residue_topo": s.residue_ids.first().copied().unwrap_or(-1),
            "site_causal_lag_steps": 0.0,
            "site_burst_motion": s.differential_signal,
            "site_lag_corr_peak": s.ccf_centrality,
            "site_local_cov": s.consensus_fraction,
        })
    }).collect();

    let root = serde_json::json!({
        "mode": "prism-twin",
        "semantics": {
            "kcc": "Cross-stream CCF from TWIN interferometric observation",
            "vector": "Allosteric coupling direction",
            "color_mapping": "CCF magnitude",
            "weight": "rank by TWIN score",
            "site": "TWIN-detected pocket",
        },
        "interpretation_guidelines": {
            "persistent_site": "high consensus_fraction",
            "transient_site": "THERMAL_TRANSIENT class",
            "high_value_residue": "high ccf_centrality",
            "noise_pattern": "low consensus with differential signal",
        },
        "residues": residue_rows,
        "sites": sites_rows,
        "vector_field_definition": "TWIN CCF-derived coupling",
        "pdb_source": "from topology",
        "ccf_edges_top500": edges.iter().map(|(i, j, v)| serde_json::json!([i, j, v])).collect::<Vec<_>>(),
    });

    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn write_prism_therm_json(path: &Path, sites: &[TwinSite]) -> Result<()> {
    let sites_rows: Vec<serde_json::Value> = sites.iter().map(|s| {
        serde_json::json!({
            "centroid_angstrom": s.centroid,
            "asymmetry_score": s.hysteresis_asymmetry,
            "ccns_classification": s.therm_class,
            "druggability": s.druggability,
            "cooling_spike_count": 0,
            "cooling_spike_rate": 0.0,
            "heating_spike_count": 0,
            "heating_spike_rate": 0.0,
            "twin_site_class": s.twin_site_class,
            "twin_rank_score": s.twin_rank_score,
            "tide_trigger_residues": s.residue_ids.iter().take(5).copied().collect::<Vec<_>>(),
        })
    }).collect();

    let hysteretic_sites = sites.iter().filter(|s| s.hysteresis_asymmetry > 0.15).count();

    let root = serde_json::json!({
        "mode": "prism-twin",
        "source": "TWIN phase-resolved spike counts per stream per residue",
        "global_pockets": [],
        "hysteresis_threshold": 0.15,
        "hysteretic_site_count": hysteretic_sites,
        "sdst_event_count": sites.iter().map(|s| s.spike_count).sum::<u64>(),
        "total_avalanches": 0,
        "sites": sites_rows,
        "tide_residues_mapped": sites.iter().flat_map(|s| s.residue_ids.iter().take(3).copied()).collect::<Vec<_>>(),
    });

    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

// Silence unused imports helper
#[allow(dead_code)]
fn _silence() -> HashMap<String, String> { HashMap::new() }
