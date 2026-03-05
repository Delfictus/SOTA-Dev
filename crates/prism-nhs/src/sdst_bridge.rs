//! PRISM-Therm: SDST (Spike-Driven Sparse Temporal Hash) integration bridge.
//!
//! Converts NHS spike events into SDST format, feeds them into the
//! thermodynamic hash table, and runs the 5-phase hysteresis + CCNS + TIDE
//! analysis pipeline that distinguishes cryptic binding sites from thermal noise.
//!
//! # Pipeline
//! ```text
//! GpuSpikeEvent[] (NHS output, all phases 0-4)
//!        ↓  ingest_all_spikes()  (sorted by timestep)
//! SDST hash table  (events tagged: phase_id, local_temp, water_density, vibrational_energy)
//!        ↓  analyze()
//! ┌─ hysteresis_region() per NHS site  →  asymmetry_score, is_hysteretic
//! ├─ ccns_region()       per NHS site  →  tau, druggability classification
//! ├─ tide_decomposition() per site     →  per-residue causal ΔG (top 20 by TE)
//! └─ ccns_all_pockets()  global scan   →  independent pocket list from SDST
//! ```
//!
//! # Phase boundaries
//! Computed at runtime from the live `CryoUvProtocol` instance — never hardcoded:
//! ```text
//! p0=0,  p1=cold_hold_steps,  p2+=ramp_steps,
//! p3+=warm_hold_steps,  p4+=ramp_down_steps,  p5+=cold_return_steps
//! ```
//! For `fast_35k().with_hysteresis()` this gives [0, 14000, 20000, 35000, 41000, 55000].
//! For `standard().with_hysteresis()` this gives [0, 5000, 15000, 20000, 30000, 35000].

#![allow(clippy::needless_pass_by_value)]

use anyhow::Result;
use serde::Serialize;

use crate::fused_engine::{CryoUvProtocol, GpuSpikeEvent};
use crate::input::PrismPrepTopology;
use crate::persistent_engine::ClusteredBindingSite;

// Re-exports from the sdst crate that we need
use sdst::{Sdst, SdstConfig, CcnsResult, HysteresisResult, SpatialRegion, CcnsClass};

const GRID_DIM: usize = 128;
const GRID_SPACING: f32 = 0.75;
const GRID_PADDING: f32 = 5.0;
const HYSTERESIS_THRESHOLD: f32 = 0.2;
/// Painting radius (voxels) for residue map construction.
/// 2 voxels × 0.75Å = 1.5Å — covers van der Waals radius of most atoms.
const RESMAP_PAINT_RADIUS: i32 = 2;
/// Maximum TIDE residues to report per site (top by transfer entropy).
const TIDE_TOP_RESIDUES: usize = 20;

// ---------------------------------------------------------------------------
// Output structs (serialized into the binding_sites.json "prism_therm" key)
// ---------------------------------------------------------------------------

fn ccns_class_name(cls: CcnsClass) -> &'static str {
    match cls {
        CcnsClass::Soc => "Soc",
        CcnsClass::NearCritical => "NearCritical",
        CcnsClass::Barrier => "Barrier",
    }
}

/// Per-residue TIDE (causal ΔG decomposition) result.
#[derive(Debug, Clone, Serialize)]
pub struct TideResidueResult {
    pub residue_id: u32,
    /// Transfer entropy-weighted ΔG contribution (kcal/mol proxy).
    /// Negative = stabilizing pocket, positive = destabilizing.
    pub causal_dg: f32,
    /// Transfer entropy: causal information flow from this residue to pocket.
    /// Higher = more causal influence. Drug design leverage point.
    pub transfer_entropy: f32,
    /// Fisher information: sensitivity of pocket to perturbations at this residue.
    /// High Fisher info = high-leverage mutation/modification target.
    pub fisher_info: f32,
    /// KL divergence between heating/cooling spike distributions.
    /// High KL = conformational reorganization cost for this residue.
    pub kl_divergence: f32,
    /// Number of spikes from this residue causally connected to pocket.
    pub n_causal_spikes: u32,
}

/// Per-site PRISM-Therm analysis result.
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermSiteResult {
    pub site_id: i32,
    /// Heating/cooling spike rate asymmetry in [0, 1].
    /// > 0.2 typically indicates a cryptic (hysteretic) pocket.
    pub asymmetry_score: f32,
    pub is_hysteretic: bool,
    pub heating_spike_rate: f32,
    pub cooling_spike_rate: f32,
    pub heating_spike_count: u32,
    pub cooling_spike_count: u32,
    /// CCNS power-law exponent. tau < 1.5 = SOC (most druggable).
    pub tau: f32,
    pub tau_stderr: f32,
    pub ccns_classification: String,
    pub druggability: f32,
    pub n_avalanches: u32,
    /// TIDE: top residues by transfer entropy (causal ΔG decomposition).
    /// Up to 20 residues, sorted by TE descending.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tide_decomposition: Vec<TideResidueResult>,
}

/// One globally-detected hysteretic pocket (from SDST's own scan).
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermGlobalPocket {
    pub grid_region: [u32; 6],  // [x_min, x_max, y_min, y_max, z_min, z_max]
    pub tau: f32,
    pub ccns_classification: String,
    pub druggability: f32,
    /// Approximate centroid in Ångstroms.
    pub centroid_angstrom: [f32; 3],
}

/// Complete PRISM-Therm output for one structure run.
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermAnalysis {
    /// Total events stored in SDST.
    pub sdst_event_count: u32,
    /// Hysteresis threshold applied (default 0.2).
    pub hysteresis_threshold: f32,
    /// Number of NHS sites classified as hysteretic.
    pub hysteretic_site_count: usize,
    /// Per-NHS-site thermodynamic analysis.
    pub sites: Vec<PrismThermSiteResult>,
    /// Pockets found by SDST's own global scan (may include novel sites).
    pub global_pockets: Vec<PrismThermGlobalPocket>,
    /// Total unique avalanches detected across the full simulation.
    pub total_avalanches: usize,
    /// Number of residues mapped in the TIDE residue map.
    pub tide_residues_mapped: usize,
}

// ---------------------------------------------------------------------------
// SdstBridge
// ---------------------------------------------------------------------------

/// PRISM-Therm integration bridge between the NHS engine and the SDST library.
///
/// Instantiate with the live `protocol` (after `with_hysteresis()` has been
/// applied) and the topology, so phase boundaries are always exact.
pub struct SdstBridge {
    sdst: Sdst,
    protocol: CryoUvProtocol,
    grid_origin: [f32; 3],
    /// Dense linear-voxel-indexed residue map (size = GRID_DIM³).
    /// Index = x + y*GRID_DIM + z*GRID_DIM². Value = residue_id or u32::MAX.
    residue_map: Vec<u32>,
    /// Total number of unique residues in the topology.
    n_residues: u32,
}

impl SdstBridge {
    /// Create a bridge configured from the live protocol and topology.
    ///
    /// `expected_spike_count` is used to size the SDST event buffer with a
    /// 1.5× safety margin (minimum 2M events).
    pub fn new(
        topology: &PrismPrepTopology,
        protocol: &CryoUvProtocol,
        expected_spike_count: usize,
    ) -> Result<Self> {
        let grid_origin = topology.grid_origin(GRID_PADDING);
        let config = Self::make_config(protocol, expected_spike_count);

        let sdst = Sdst::new(&config)
            .map_err(|e| anyhow::anyhow!("SDST create failed: {:?}", e))?;

        let residue_map = Self::build_residue_map(topology, &grid_origin);
        let n_residues = topology.n_residues as u32;

        let mapped_count = residue_map.iter().filter(|&&v| v != u32::MAX).count();
        log::info!("  PRISM-Therm: residue map built — {} voxels mapped ({} residues)",
            mapped_count, n_residues);

        Ok(Self {
            sdst,
            protocol: protocol.clone(),
            grid_origin,
            residue_map,
            n_residues,
        })
    }

    /// Build SdstConfig from the live CryoUvProtocol.
    /// No step counts are hardcoded — all come from the protocol instance.
    fn make_config(protocol: &CryoUvProtocol, expected_spike_count: usize) -> SdstConfig {
        // Get base defaults from the C library (128³ grid, 0.75Å, 4M hash, etc.)
        let mut cfg: SdstConfig = unsafe { sdst::sdst_default_config() };

        // Grid: matches NHS engine exactly
        cfg.grid_nx = GRID_DIM as u32;
        cfg.grid_ny = GRID_DIM as u32;
        cfg.grid_nz = GRID_DIM as u32;
        cfg.grid_spacing = GRID_SPACING;

        // Phase boundaries: cumulative sums from the live protocol fields.
        // with_hysteresis() mirrors ramp and cold-hold into the cooling phases,
        // but we always read the actual field values — never assume defaults.
        let p0 = 0u32;
        let p1 = p0 + protocol.cold_hold_steps.max(0) as u32;
        let p2 = p1 + protocol.ramp_steps.max(0) as u32;
        let p3 = p2 + protocol.warm_hold_steps.max(0) as u32;
        let p4 = p3 + protocol.ramp_down_steps.max(0) as u32;
        let p5 = p4 + protocol.cold_return_steps.max(0) as u32;
        cfg.phase_boundaries = [p0, p1, p2, p3, p4, p5];

        // Event buffer: scale to actual spike count. At 36 bytes/event,
        // 28M events = 1GB -- safe ceiling for 16GB VRAM.
        cfg.max_spike_events = (expected_spike_count as u32)
            .max(1_000_000)
            .min(28_000_000);

        cfg
    }

    /// Build a dense linear-voxel-indexed residue map from the topology.
    ///
    /// For each protein atom, paints a sphere of radius `RESMAP_PAINT_RADIUS`
    /// voxels centered at the atom's grid position. When multiple atoms compete
    /// for a voxel, the closest atom (by Euclidean distance) wins.
    ///
    /// Size = `GRID_DIM³` (128³ = 2,097,152 entries, 8MB).
    /// Index = `x + y*GRID_DIM + z*GRID_DIM²`.
    /// Empty voxels = `u32::MAX`.
    fn build_residue_map(topology: &PrismPrepTopology, grid_origin: &[f32; 3]) -> Vec<u32> {
        let dim = GRID_DIM;
        let total = dim * dim * dim;
        let mut resmap = vec![u32::MAX; total];
        let mut dist_sq_map = vec![f32::MAX; total];

        let r = RESMAP_PAINT_RADIUS;

        for atom_idx in 0..topology.n_atoms {
            let px = topology.positions[atom_idx * 3];
            let py = topology.positions[atom_idx * 3 + 1];
            let pz = topology.positions[atom_idx * 3 + 2];
            let res_id = topology.residue_ids[atom_idx] as u32;

            // Atom's voxel coordinate (continuous → integer)
            let vx = ((px - grid_origin[0]) / GRID_SPACING) as i32;
            let vy = ((py - grid_origin[1]) / GRID_SPACING) as i32;
            let vz = ((pz - grid_origin[2]) / GRID_SPACING) as i32;

            // Paint a (2r+1)³ cube, keeping only the closest atom per voxel
            for dz in -r..=r {
                let nz = vz + dz;
                if nz < 0 || nz >= dim as i32 { continue; }
                for dy in -r..=r {
                    let ny = vy + dy;
                    if ny < 0 || ny >= dim as i32 { continue; }
                    for dx in -r..=r {
                        let nx = vx + dx;
                        if nx < 0 || nx >= dim as i32 { continue; }

                        let linear = nx as usize + ny as usize * dim + nz as usize * dim * dim;

                        // Euclidean distance² from voxel center to atom
                        let vox_x = nx as f32 * GRID_SPACING + grid_origin[0];
                        let vox_y = ny as f32 * GRID_SPACING + grid_origin[1];
                        let vox_z = nz as f32 * GRID_SPACING + grid_origin[2];
                        let d2 = (vox_x - px).powi(2) + (vox_y - py).powi(2) + (vox_z - pz).powi(2);

                        if d2 < dist_sq_map[linear] {
                            dist_sq_map[linear] = d2;
                            resmap[linear] = res_id;
                        }
                    }
                }
            }
        }

        resmap
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Convert and insert accumulated NHS spike events into SDST.
    ///
    /// **Spike cap:** SDST's `kernel_detect_parents` is O(N × neighborhood),
    /// so feeding >500K spikes causes multi-minute GPU stalls. We cap at
    /// Takes all spikes (up to 1GB VRAM cap) to preserve temporal distribution.
    /// the most thermodynamic signal). The full multi-stream pool can exceed
    /// 10M spikes; we subsample before SDST insertion.
    ///
    /// Spikes are then sorted by timestep so parent-detection ordering is correct.
    pub fn ingest_all_spikes(&self, spikes: &[GpuSpikeEvent]) -> Result<u32> {
        if spikes.is_empty() {
            return self.sdst.event_count()
                .map_err(|e| anyhow::anyhow!("SDST event_count: {:?}", e));
        }
        // VRAM safety cap: 1GB / 36 bytes = ~27.7M events.
        // Below that, take EVERY spike to preserve temporal phase distribution.
        // The old 500K intensity-based cap destroyed hysteresis signal.
        const MAX_SDST_MEMORY_BYTES: usize = 1_024 * 1_024 * 1_024;
        const BYTES_PER_EVENT: usize = 36;
        const ABSOLUTE_MAX_EVENTS: usize = MAX_SDST_MEMORY_BYTES / BYTES_PER_EVENT;

        let working: Vec<&GpuSpikeEvent> = if spikes.len() > ABSOLUTE_MAX_EVENTS {
            let stride = spikes.len() / ABSOLUTE_MAX_EVENTS;
            log::warn!("  PRISM-Therm: {} spikes exceeds 1GB cap, stride-sampling to {}",
                        spikes.len(), ABSOLUTE_MAX_EVENTS);
            let mut sampled: Vec<&GpuSpikeEvent> = spikes.iter()
                .step_by(stride)
                .take(ABSOLUTE_MAX_EVENTS)
                .collect();
            sampled.sort_unstable_by_key(|s| s.timestep);
            sampled
        } else {
            let mut sorted: Vec<&GpuSpikeEvent> = spikes.iter().collect();
            sorted.sort_unstable_by_key(|s| s.timestep);
            log::info!("  PRISM-Therm: ingesting all {} spikes into SDST", spikes.len());
            sorted
        };
        let inputs: Vec<sdst::SpikeInput> = working.iter()
            .map(|s| self.convert_spike(s))
            .collect();

        self.sdst.insert_raw(&inputs)
            .map_err(|e| anyhow::anyhow!("SDST insert_raw failed: {:?}", e))?;

        self.sdst.event_count()
            .map_err(|e| anyhow::anyhow!("SDST event_count: {:?}", e))
    }

    /// Run the full hysteresis + CCNS + TIDE analysis pipeline.
    ///
    /// 1. Per-site: `hysteresis_region` + `ccns_region` for each NHS site
    /// 2. Per-site: `tide_decomposition` — causal ΔG per residue (top 20 by TE)
    /// 3. Global:   `ccns_all_pockets` finds pockets SDST detected independently
    ///              (may reveal cryptic sites NHS missed or confirm NHS results)
    /// 4. Avalanche statistics summary
    pub fn analyze(
        &self,
        sites: &[ClusteredBindingSite],
    ) -> Result<PrismThermAnalysis> {
        let event_count = self.sdst.event_count()
            .map_err(|e| anyhow::anyhow!("SDST event_count: {:?}", e))?;

        // Per-site analysis (hysteresis + CCNS + TIDE)
        let site_results: Vec<PrismThermSiteResult> = sites.iter()
            .map(|site| {
                let region = self.centroid_to_region(&site.centroid);
                self.analyze_site(site.cluster_id, &region)
            })
            .collect();

        let hysteretic_count = site_results.iter().filter(|r| r.is_hysteretic).count();

        // Log TIDE summary
        for sr in &site_results {
            if !sr.tide_decomposition.is_empty() {
                log::info!("  TIDE Site {}: {} active residues, top TE={:.4}, top dG={:.3}",
                    sr.site_id,
                    sr.tide_decomposition.len(),
                    sr.tide_decomposition[0].transfer_entropy,
                    sr.tide_decomposition[0].causal_dg);
            }
        }

        // Global SDST pocket scan (runs sdst_hysteresis_scan + sdst_ccns internally)
        let global_pockets = self.run_global_scan();

        // Avalanche count: proxy for total observed conformational events
        let total_avalanches = self.sdst.avalanche_stats(-1)
            .map(|v| v.len())
            .unwrap_or(0);

        let mapped = self.residue_map.iter().filter(|&&v| v != u32::MAX).count();

        Ok(PrismThermAnalysis {
            sdst_event_count: event_count,
            hysteresis_threshold: HYSTERESIS_THRESHOLD,
            hysteretic_site_count: hysteretic_count,
            sites: site_results,
            global_pockets,
            total_avalanches,
            tide_residues_mapped: mapped,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Convert one `GpuSpikeEvent` to `SpikeInput` for SDST ingestion.
    fn convert_spike(&self, s: &GpuSpikeEvent) -> sdst::SpikeInput {
        let ts = s.timestep.max(0) as u32;

        // Decode grid coordinates from packed voxel_idx
        // NHS linear layout: voxel_idx = vx + vy*128 + vz*128*128
        let idx = s.voxel_idx.max(0) as usize;
        let vx = (idx % GRID_DIM) as u32;
        let vy = ((idx / GRID_DIM) % GRID_DIM) as u32;
        let vz = (idx / (GRID_DIM * GRID_DIM)) as u32;
        let dim = GRID_DIM as u32 - 1;

        // Phase 9: Real thermodynamic fields (not proxies).
        //
        // energy_gradient = |∂WD/∂t| — the water density change that drove the LIF
        //   spike. This is the actual signal the neuron thresholds against, so it
        //   directly measures thermodynamic barrier crossing.
        //
        // local_temp = T_protocol × (1 + 0.5 × (1 − WD)) — spatially modulated
        //   temperature. Buried voxels (WD ≈ 0) see 1.5× protocol temp; fully
        //   solvated voxels (WD ≈ WD_bulk) see ~1× protocol temp. This gives SDST
        //   spatial temperature resolution without per-voxel KE computation.
        //
        // solvent_exposure = water_density — direct per-voxel NHS measurement.
        let wd_clamped = s.water_density.clamp(0.0, 1.0);
        let t_protocol = self.temp_at(ts);
        let local_temp = t_protocol * (1.0 + 0.5 * (1.0 - wd_clamped));

        sdst::SpikeInput {
            voxel_x: vx.min(dim),
            voxel_y: vy.min(dim),
            voxel_z: vz.min(dim),
            timestamp: ts,
            amplitude:        s.intensity,
            local_temp:       local_temp,
            energy_gradient:  s.wd_change,       // |∂WD/∂t| from kernel
            solvent_exposure: s.water_density,   // NHS water density field
            phase_id:         self.phase_at(ts),
        }
    }

    /// Compute which of the 5 hysteresis phases (0-4) contains `step`.
    fn phase_at(&self, step: u32) -> u8 {
        let p = &self.protocol;
        let p1 = p.cold_hold_steps.max(0) as u32;
        let p2 = p1 + p.ramp_steps.max(0) as u32;
        let p3 = p2 + p.warm_hold_steps.max(0) as u32;
        let p4 = p3 + p.ramp_down_steps.max(0) as u32;
        if step < p1 { 0 }
        else if step < p2 { 1 }
        else if step < p3 { 2 }
        else if step < p4 { 3 }
        else { 4 }
    }

    /// Interpolate the protocol temperature (K) at a given timestep.
    fn temp_at(&self, step: u32) -> f32 {
        let p = &self.protocol;
        let s = step as i32;
        let p1 = p.cold_hold_steps;
        let p2 = p1 + p.ramp_steps;
        let p3 = p2 + p.warm_hold_steps;
        let p4 = p3 + p.ramp_down_steps;

        if s < p1 {
            p.start_temp
        } else if s < p2 {
            let frac = (s - p1) as f32 / p.ramp_steps.max(1) as f32;
            p.start_temp + frac * (p.end_temp - p.start_temp)
        } else if s < p3 {
            p.end_temp
        } else if s < p4 {
            // Phase 4: ramp down (cooling)
            let frac = (s - p3) as f32 / p.ramp_down_steps.max(1) as f32;
            p.end_temp - frac * (p.end_temp - p.start_temp)
        } else {
            // Phase 5: cold return hold
            p.start_temp
        }
    }

    /// Convert a site centroid (Å) to an SDST SpatialRegion in grid voxels.
    /// Radius = 8 voxels ≈ 6 Å — matches the default lining residue cutoff.
    fn centroid_to_region(&self, centroid: &[f32; 3]) -> SpatialRegion {
        let radius = 8u32;
        let dim = GRID_DIM as u32;

        let to_voxel = |pos: f32, origin: f32| -> u32 {
            let v = ((pos - origin) / GRID_SPACING).max(0.0) as u32;
            v.min(dim - 1)
        };

        let cx = to_voxel(centroid[0], self.grid_origin[0]);
        let cy = to_voxel(centroid[1], self.grid_origin[1]);
        let cz = to_voxel(centroid[2], self.grid_origin[2]);

        SpatialRegion {
            x_min: cx.saturating_sub(radius),
            x_max: (cx + radius).min(dim - 1),
            y_min: cy.saturating_sub(radius),
            y_max: (cy + radius).min(dim - 1),
            z_min: cz.saturating_sub(radius),
            z_max: (cz + radius).min(dim - 1),
        }
    }

    /// Analyze one NHS binding site: hysteresis + CCNS + TIDE.
    fn analyze_site(&self, site_id: i32, region: &SpatialRegion) -> PrismThermSiteResult {
        let hyst = self.sdst
            .hysteresis_region(region, HYSTERESIS_THRESHOLD)
            .unwrap_or_else(|_| HysteresisResult {
                heating_spike_rate: 0.0,
                cooling_spike_rate: 0.0,
                asymmetry_score: 0.0,
                avalanche_size_ratio: 1.0,
                wavefront_coherence_ratio: 1.0,
                heating_spike_count: 0,
                cooling_spike_count: 0,
                is_hysteretic: false,
            });

        let ccns = self.sdst
            .ccns_region(region)
            .unwrap_or_else(|_| CcnsResult {
                tau: 0.0,
                classification: CcnsClass::Soc,
                tau_stderr: 0.0,
                n_avalanches: 0,
                druggability: 0.0,
            });

        // TIDE decomposition: per-residue causal ΔG
        let tide = self.run_tide(region);

        PrismThermSiteResult {
            site_id,
            asymmetry_score:     hyst.asymmetry_score,
            is_hysteretic:       hyst.is_hysteretic,
            heating_spike_rate:  hyst.heating_spike_rate,
            cooling_spike_rate:  hyst.cooling_spike_rate,
            heating_spike_count: hyst.heating_spike_count,
            cooling_spike_count: hyst.cooling_spike_count,
            tau:                 ccns.tau,
            tau_stderr:          ccns.tau_stderr,
            ccns_classification: ccns_class_name(ccns.classification).to_string(),
            druggability:        ccns.druggability,
            n_avalanches:        ccns.n_avalanches,
            tide_decomposition:  tide,
        }
    }

    /// Run TIDE decomposition for a pocket region.
    ///
    /// Returns up to `TIDE_TOP_RESIDUES` (20) residues sorted by transfer
    /// entropy descending. These are the residues with strongest causal
    /// influence on the pocket — the drug design leverage points.
    fn run_tide(&self, region: &SpatialRegion) -> Vec<TideResidueResult> {
        let decomp = match self.sdst.tide_decomposition(
            region,
            &self.residue_map,
            self.n_residues,
        ) {
            Ok(d) => d,
            Err(e) => {
                log::debug!("  TIDE: decomposition failed: {:?}", e);
                return Vec::new();
            }
        };

        if decomp.is_empty() {
            return Vec::new();
        }

        // Convert to output format, sort by TE descending, take top N
        let mut results: Vec<TideResidueResult> = decomp.into_iter()
            .filter(|d| d.transfer_entropy > 0.0 || d.n_causal_spikes > 0)
            .map(|d| TideResidueResult {
                residue_id: d.residue_id,
                causal_dg: d.causal_dg,
                transfer_entropy: d.transfer_entropy,
                fisher_info: d.fisher_info,
                kl_divergence: d.kl_divergence,
                n_causal_spikes: d.n_causal_spikes,
            })
            .collect();

        results.sort_unstable_by(|a, b|
            b.transfer_entropy.partial_cmp(&a.transfer_entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        );
        results.truncate(TIDE_TOP_RESIDUES);
        results
    }

    /// Run the global SDST pocket scan via `ccns_all_pockets()`.
    ///
    /// `ccns_all_pockets` internally calls `sdst_hysteresis_scan` to find
    /// candidate regions, then runs `sdst_ccns_region` on each. We convert
    /// the region centre to Å for the output JSON.
    fn run_global_scan(&self) -> Vec<PrismThermGlobalPocket> {
        let pockets = match self.sdst.ccns_all_pockets() {
            Ok(v) => v,
            Err(e) => {
                log::warn!("PRISM-Therm: ccns_all_pockets failed: {:?}", e);
                return Vec::new();
            }
        };

        pockets
            .into_iter()
            .take(50) // Limit to top 50 for JSON size
            .map(|(ccns, region)| {
                // Approximate centroid in Å
                let cx = ((region.x_min + region.x_max) as f32 / 2.0) * GRID_SPACING + self.grid_origin[0];
                let cy = ((region.y_min + region.y_max) as f32 / 2.0) * GRID_SPACING + self.grid_origin[1];
                let cz = ((region.z_min + region.z_max) as f32 / 2.0) * GRID_SPACING + self.grid_origin[2];

                PrismThermGlobalPocket {
                    grid_region: [
                        region.x_min, region.x_max,
                        region.y_min, region.y_max,
                        region.z_min, region.z_max,
                    ],
                    tau: ccns.tau,
                    ccns_classification: ccns_class_name(ccns.classification).to_string(),
                    druggability: ccns.druggability,
                    centroid_angstrom: [cx, cy, cz],
                }
            })
            .collect()
    }
}
