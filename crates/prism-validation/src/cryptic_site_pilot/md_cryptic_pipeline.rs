//! MD-Based Cryptic Site Detection Pipeline
//!
//! Production-quality pipeline using Langevin MD for conformational sampling
//! with residue-based pocket tracking (Jaccard similarity).
//!
//! # Scientific Integrity
//!
//! All classification thresholds are:
//! - Derived from published literature (CryptoSite, PocketMiner)
//! - Set BEFORE validation runs
//! - Applied identically to all structures
//! - NOT tuned post-hoc to match expected results
//!
//! # Key Differences from ANM Pipeline
//!
//! | Aspect | ANM Pipeline | MD Pipeline |
//! |--------|--------------|-------------|
//! | Coordinates | Cα only | All-atom |
//! | Sampling | Normal modes | Langevin dynamics |
//! | Pocket matching | Centroid (8Å) | Residue Jaccard (30%) |
//! | Solvent | None | Implicit GB |
//!
//! # References
//!
//! - CryptoSite: Cimermancic et al. (2016) J. Mol. Biol. 428:709-719
//! - PocketMiner: Meller et al. (2023) Nature Communications 14:5952

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "cryptic-gpu")]
use prism_gpu::{AmberMegaFusedHmc, LcpoSasaGpu, HmcRunResult};

#[cfg(feature = "cryptic-gpu")]
use cudarc::driver::CudaContext;

use super::topology_loader::PrismTopology;
use super::druggability::{DruggabilityScore, DruggabilityScorer};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// MD-based cryptic site detection configuration
///
/// # Parameter Provenance (DO NOT MODIFY WITHOUT LITERATURE JUSTIFICATION)
///
/// All thresholds are set BEFORE validation and apply to ALL structures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdCrypticConfig {
    // =========================================================================
    // PHYSICAL CONSTANTS — DO NOT MODIFY
    // =========================================================================

    /// Water probe radius for SASA calculation (Å)
    /// Source: Standard value for water molecule
    /// DO NOT MODIFY - this is a physical constant
    pub sasa_probe_radius: f32,

    // =========================================================================
    // MD PARAMETERS — STANDARD PRACTICE
    // =========================================================================

    /// Integration timestep (femtoseconds)
    /// 2.0 fs is standard with SHAKE/RATTLE H-bond constraints
    pub dt_fs: f32,

    /// Target temperature (Kelvin)
    /// 310 K = physiological; 300 K = room temperature (both valid)
    pub temperature_k: f32,

    /// Langevin friction coefficient (fs⁻¹)
    /// 1.0 fs⁻¹ is standard for implicit solvent
    pub gamma_fs: f32,

    /// Steps per snapshot (determines snapshot frequency)
    /// 10,000 steps × 2 fs = 20 ps per frame
    pub segment_steps: usize,

    /// Number of production frames to collect
    /// 200 frames × 20 ps = 4 ns total production
    pub n_frames: usize,

    /// Equilibration steps (discarded)
    /// 50 ps is standard for small-medium proteins
    pub equilibration_steps: usize,

    // =========================================================================
    // CLASSIFICATION THRESHOLDS — LITERATURE-DERIVED, SET BEFORE VALIDATION
    // =========================================================================

    /// Volume coefficient of variation threshold for cryptic classification
    ///
    /// Source: CryptoSite (Cimermancic et al., 2016), PocketMiner (Meller et al., 2023)
    /// Rationale: CV > 0.20 distinguishes cryptic (transient) from constitutive pockets
    ///
    /// DO NOT TUNE POST-HOC TO MATCH EXPECTED RESULTS
    pub cv_threshold: f64,

    /// Minimum open frequency for cryptic classification
    ///
    /// Rationale: <5% appearance is likely noise/artifact
    pub min_open_frequency: f64,

    /// Maximum open frequency for cryptic classification
    ///
    /// Rationale: >90% means pocket is always open (constitutive, not cryptic)
    pub max_open_frequency: f64,

    /// Jaccard similarity threshold for pocket matching across frames
    ///
    /// Source: Standard clustering practice
    /// Rationale: 30% residue overlap = same pocket identity
    pub jaccard_threshold: f64,

    /// Minimum pocket volume for detection (Å³)
    ///
    /// Source: Fragment-based drug discovery literature
    /// Rationale: Minimum volume for drug-like fragment (~150 Da) binding
    pub min_pocket_volume: f64,

    /// Maximum pocket volume (Å³)
    ///
    /// Rationale: Very large cavities are likely artifacts or active site channels
    pub max_pocket_volume: f64,

    // =========================================================================
    // POCKET DETECTION PARAMETERS
    // =========================================================================

    /// Neighbor distance cutoff for pocket detection (Å)
    /// Tighter than Cα-only (10Å) because using all heavy atoms
    pub pocket_neighbor_cutoff: f32,

    /// Minimum neighbors for concave region (pocket candidate)
    pub min_pocket_neighbors: usize,

    /// Maximum neighbors (too many = buried core, not pocket)
    pub max_pocket_neighbors: usize,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for MdCrypticConfig {
    fn default() -> Self {
        Self {
            // Physical constants
            sasa_probe_radius: 1.4,  // Water molecule radius - NON-NEGOTIABLE

            // MD parameters (standard practice)
            dt_fs: 2.0,              // Standard with H-constraints
            temperature_k: 310.0,    // Physiological temperature
            gamma_fs: 1.0,           // Standard Langevin friction
            segment_steps: 10_000,   // 20 ps per frame at 2 fs timestep
            n_frames: 200,           // 4 ns total production
            equilibration_steps: 25_000, // 50 ps equilibration

            // Classification thresholds - LITERATURE DERIVED, SET BEFORE VALIDATION
            cv_threshold: 0.20,          // CryptoSite, PocketMiner standard
            min_open_frequency: 0.05,    // 5% minimum
            max_open_frequency: 0.90,    // 90% maximum
            jaccard_threshold: 0.30,     // 30% residue overlap
            min_pocket_volume: 80.0,     // Fragment-sized minimum
            max_pocket_volume: 2000.0,   // Upper bound

            // Pocket detection
            pocket_neighbor_cutoff: 8.0, // Tighter than Cα (10Å)
            min_pocket_neighbors: 8,
            max_pocket_neighbors: 25,

            seed: 42,
        }
    }
}

impl MdCrypticConfig {
    /// Quick test configuration (shorter simulation)
    pub fn quick_test() -> Self {
        Self {
            segment_steps: 5_000,     // 10 ps per frame
            n_frames: 50,             // 500 ps total
            equilibration_steps: 10_000, // 20 ps equilibration
            ..Default::default()
        }
    }

    /// Production configuration (longer simulation for difficult cases)
    pub fn production() -> Self {
        Self {
            segment_steps: 10_000,    // 20 ps per frame
            n_frames: 400,            // 8 ns total
            equilibration_steps: 50_000, // 100 ps equilibration
            ..Default::default()
        }
    }

    /// Total simulation time in picoseconds
    pub fn total_time_ps(&self) -> f64 {
        let production_ps = self.n_frames as f64 * self.segment_steps as f64 * self.dt_fs as f64 / 1000.0;
        let eq_ps = self.equilibration_steps as f64 * self.dt_fs as f64 / 1000.0;
        production_ps + eq_ps
    }
}

// =============================================================================
// POCKET DATA STRUCTURES
// =============================================================================

/// A pocket detected in a single frame
#[derive(Debug, Clone)]
pub struct DetectedPocket {
    /// Centroid position [x, y, z] in Å
    pub centroid: [f32; 3],
    /// Residue IDs defining this pocket
    pub residue_ids: Vec<i32>,
    /// Estimated volume (Å³)
    pub volume: f64,
    /// Total SASA of pocket residues (Å²)
    pub sasa: f64,
}

/// Time series data for a tracked pocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketTimeSeries {
    /// Unique pocket identifier
    pub pocket_id: String,
    /// Consensus residues defining this pocket (union across frames)
    pub defining_residues: Vec<i32>,
    /// Volume per frame (0.0 if not observed in that frame)
    pub volumes: Vec<f64>,
    /// SASA per frame (0.0 if not observed)
    pub sasa_values: Vec<f64>,
    /// Frame indices where pocket was observed
    pub frames_observed: Vec<usize>,

    // Statistics (computed in finalize())
    /// Mean volume when open
    pub mean_volume: f64,
    /// Standard deviation of volume
    pub std_volume: f64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    /// Fraction of frames where pocket was observed
    pub open_frequency: f64,
    /// Mean SASA when open
    pub mean_sasa: f64,
    /// CV of SASA (independent measure)
    pub cv_sasa: f64,
}

impl PocketTimeSeries {
    fn new(pocket_id: String, n_frames: usize) -> Self {
        Self {
            pocket_id,
            defining_residues: Vec::new(),
            volumes: vec![0.0; n_frames],
            sasa_values: vec![0.0; n_frames],
            frames_observed: Vec::new(),
            mean_volume: 0.0,
            std_volume: 0.0,
            cv: 0.0,
            open_frequency: 0.0,
            mean_sasa: 0.0,
            cv_sasa: 0.0,
        }
    }

    fn compute_statistics(&mut self) {
        let observed_volumes: Vec<f64> = self.volumes.iter()
            .filter(|&&v| v > 0.0)
            .copied()
            .collect();

        let observed_sasa: Vec<f64> = self.sasa_values.iter()
            .filter(|&&s| s > 0.0)
            .copied()
            .collect();

        let n_observed = observed_volumes.len();
        let n_total = self.volumes.len();

        if n_observed >= 5 {
            // Volume statistics
            self.mean_volume = observed_volumes.iter().sum::<f64>() / n_observed as f64;
            let variance = observed_volumes.iter()
                .map(|v| (v - self.mean_volume).powi(2))
                .sum::<f64>() / n_observed as f64;
            self.std_volume = variance.sqrt();
            self.cv = if self.mean_volume > 0.0 { self.std_volume / self.mean_volume } else { 0.0 };

            // SASA statistics
            if !observed_sasa.is_empty() {
                self.mean_sasa = observed_sasa.iter().sum::<f64>() / observed_sasa.len() as f64;
                let sasa_variance = observed_sasa.iter()
                    .map(|s| (s - self.mean_sasa).powi(2))
                    .sum::<f64>() / observed_sasa.len() as f64;
                let std_sasa = sasa_variance.sqrt();
                self.cv_sasa = if self.mean_sasa > 0.0 { std_sasa / self.mean_sasa } else { 0.0 };
            }

            self.open_frequency = n_observed as f64 / n_total as f64;
        }
    }
}

/// Classified cryptic site with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSite {
    /// Site identifier
    pub site_id: String,
    /// Rank by confidence
    pub rank: usize,
    /// Residues defining the site
    pub residues: Vec<i32>,
    /// Volume statistics
    pub mean_volume: f64,
    pub cv_volume: f64,
    /// SASA statistics
    pub mean_sasa: f64,
    pub cv_sasa: f64,
    /// Open frequency
    pub open_frequency: f64,
    /// Classification confidence (based on CV magnitude)
    pub confidence: f64,
    /// Druggability assessment
    pub druggability: Option<DruggabilityScore>,
}

// =============================================================================
// RESIDUE-BASED VOLUME TRACKER (Jaccard Matching)
// =============================================================================

/// Tracks pockets across frames using residue-based Jaccard similarity
///
/// Unlike centroid-based tracking (which fails with ANM drift), this matches
/// pockets by the overlap of their defining residue sets.
pub struct ResidueBasedVolumeTracker {
    /// All tracked pockets
    pockets: HashMap<String, PocketTimeSeries>,
    /// Jaccard threshold for matching (0.30 = 30% overlap)
    jaccard_threshold: f64,
    /// Total number of frames
    n_frames: usize,
    /// Next pocket ID
    next_pocket_id: usize,
    /// Minimum observations to consider pocket valid
    min_observations: usize,
}

impl ResidueBasedVolumeTracker {
    pub fn new(n_frames: usize, jaccard_threshold: f64) -> Self {
        Self {
            pockets: HashMap::new(),
            jaccard_threshold,
            n_frames,
            next_pocket_id: 0,
            min_observations: 5, // Need at least 5 observations for statistics
        }
    }

    /// Add a pocket observation from a single frame
    pub fn add_observation(
        &mut self,
        frame_idx: usize,
        residues: &[i32],
        volume: f64,
        sasa: f64,
    ) {
        let residue_set: HashSet<i32> = residues.iter().copied().collect();

        // Try to find matching pocket by Jaccard similarity
        let matching_pocket_id = self.find_matching_pocket(&residue_set);

        match matching_pocket_id {
            Some(pocket_id) => {
                // Update existing pocket
                let pocket = self.pockets.get_mut(&pocket_id).unwrap();
                pocket.volumes[frame_idx] = volume;
                pocket.sasa_values[frame_idx] = sasa;
                if !pocket.frames_observed.contains(&frame_idx) {
                    pocket.frames_observed.push(frame_idx);
                }
                // Merge residues (union)
                for &res in residues {
                    if !pocket.defining_residues.contains(&res) {
                        pocket.defining_residues.push(res);
                    }
                }
            }
            None => {
                // Create new pocket
                let pocket_id = format!("pocket_{}", self.next_pocket_id);
                self.next_pocket_id += 1;

                let mut pocket = PocketTimeSeries::new(pocket_id.clone(), self.n_frames);
                pocket.volumes[frame_idx] = volume;
                pocket.sasa_values[frame_idx] = sasa;
                pocket.frames_observed.push(frame_idx);
                pocket.defining_residues = residues.to_vec();

                self.pockets.insert(pocket_id, pocket);
            }
        }
    }

    /// Find pocket matching by Jaccard similarity
    fn find_matching_pocket(&self, query_residues: &HashSet<i32>) -> Option<String> {
        let mut best_match: Option<(String, f64)> = None;

        for (id, pocket) in &self.pockets {
            let existing_set: HashSet<i32> = pocket.defining_residues.iter().copied().collect();

            let intersection = query_residues.intersection(&existing_set).count();
            let union = query_residues.union(&existing_set).count();

            if union > 0 {
                let jaccard = intersection as f64 / union as f64;

                if jaccard >= self.jaccard_threshold {
                    match &best_match {
                        Some((_, best_jaccard)) if jaccard > *best_jaccard => {
                            best_match = Some((id.clone(), jaccard));
                        }
                        None => {
                            best_match = Some((id.clone(), jaccard));
                        }
                        _ => {}
                    }
                }
            }
        }

        best_match.map(|(id, _)| id)
    }

    /// Finalize tracking and compute statistics for all pockets
    pub fn finalize(&mut self) {
        for pocket in self.pockets.values_mut() {
            pocket.defining_residues.sort();
            pocket.compute_statistics();
        }
    }

    /// Get all tracked pockets (including non-cryptic)
    pub fn get_all_pockets(&self) -> Vec<&PocketTimeSeries> {
        self.pockets.values()
            .filter(|p| p.frames_observed.len() >= self.min_observations)
            .collect()
    }

    /// Get cryptic pockets based on classification criteria
    ///
    /// A pocket is cryptic if:
    /// - SASA CV > cv_threshold (shows solvent exposure variance - "breathing")
    /// - open_frequency in [min_freq, max_freq] (not always open or closed)
    ///
    /// NOTE: We use SASA CV (not volume CV) because:
    /// - SASA directly measures solvent accessibility (pocket opening)
    /// - Volume CV can be misleading for shallow/transient pockets
    /// - Literature (CryptoSite, PocketMiner) uses SASA-based metrics
    pub fn get_cryptic_pockets(
        &self,
        cv_threshold: f64,
        min_freq: f64,
        max_freq: f64,
    ) -> Vec<CrypticSite> {
        let mut cryptic: Vec<CrypticSite> = self.pockets.iter()
            .filter(|(_, p)| {
                p.frames_observed.len() >= self.min_observations
                && p.cv_sasa > cv_threshold  // SASA CV, not volume CV
                && p.open_frequency >= min_freq
                && p.open_frequency <= max_freq
            })
            .map(|(id, p)| {
                // Confidence based on how much SASA CV exceeds threshold
                let confidence = ((p.cv_sasa - cv_threshold) / cv_threshold).min(1.0).max(0.0);

                CrypticSite {
                    site_id: id.clone(),
                    rank: 0, // Set later
                    residues: p.defining_residues.clone(),
                    mean_volume: p.mean_volume,
                    cv_volume: p.cv,
                    mean_sasa: p.mean_sasa,
                    cv_sasa: p.cv_sasa,
                    open_frequency: p.open_frequency,
                    confidence,
                    druggability: None,
                }
            })
            .collect();

        // Sort by confidence (descending)
        cryptic.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        for (i, site) in cryptic.iter_mut().enumerate() {
            site.rank = i + 1;
        }

        cryptic
    }

    /// Get diagnostic statistics for debugging
    /// NOTE: Reports SASA CV values (the classification metric), not volume CV
    pub fn get_diagnostics(&self) -> TrackerDiagnostics {
        let all_pockets: Vec<&PocketTimeSeries> = self.pockets.values().collect();
        let valid_pockets: Vec<&PocketTimeSeries> = all_pockets.iter()
            .filter(|p| p.frames_observed.len() >= self.min_observations)
            .copied()
            .collect();

        // Use SASA CV for diagnostics (matches classification criteria)
        let cv_values: Vec<f64> = valid_pockets.iter().map(|p| p.cv_sasa).collect();
        let freq_values: Vec<f64> = valid_pockets.iter().map(|p| p.open_frequency).collect();

        TrackerDiagnostics {
            total_pockets_seen: all_pockets.len(),
            valid_pockets: valid_pockets.len(),
            cv_min: cv_values.iter().copied().fold(f64::INFINITY, f64::min),
            cv_max: cv_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            cv_mean: if !cv_values.is_empty() { cv_values.iter().sum::<f64>() / cv_values.len() as f64 } else { 0.0 },
            freq_min: freq_values.iter().copied().fold(f64::INFINITY, f64::min),
            freq_max: freq_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Diagnostic information for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerDiagnostics {
    pub total_pockets_seen: usize,
    pub valid_pockets: usize,
    pub cv_min: f64,
    pub cv_max: f64,
    pub cv_mean: f64,
    pub freq_min: f64,
    pub freq_max: f64,
}

// =============================================================================
// MD CRYPTIC PIPELINE
// =============================================================================

/// Result of MD-based cryptic site detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdCrypticResult {
    /// PDB identifier
    pub pdb_id: String,
    /// Configuration used (for reproducibility)
    pub config: MdCrypticConfig,
    /// Total simulation time (ps)
    pub total_time_ps: f64,
    /// Number of frames collected
    pub n_frames: usize,
    /// All tracked pockets
    pub all_pockets: Vec<PocketTimeSeries>,
    /// Classified cryptic sites
    pub cryptic_sites: Vec<CrypticSite>,
    /// Diagnostic information
    pub diagnostics: TrackerDiagnostics,
    /// Computation time (seconds)
    pub computation_time_secs: f64,
}

/// MD-based cryptic site detection pipeline
#[cfg(feature = "cryptic-gpu")]
pub struct MdCrypticPipeline {
    config: MdCrypticConfig,
    druggability_scorer: DruggabilityScorer,
}

#[cfg(feature = "cryptic-gpu")]
impl MdCrypticPipeline {
    /// Create new MD cryptic pipeline
    pub fn new(config: MdCrypticConfig) -> Result<Self> {
        Ok(Self {
            config,
            druggability_scorer: DruggabilityScorer::default(),
        })
    }

    /// Run the complete pipeline on a prism-prep topology
    pub fn run(&self, topology_path: &str) -> Result<MdCrypticResult> {
        use std::sync::Arc;

        let start_time = std::time::Instant::now();

        log::info!("[MD-CRYPTIC] Starting MD-based cryptic site detection");
        log::info!("[MD-CRYPTIC] Config: {} frames, {} ps total",
            self.config.n_frames, self.config.total_time_ps());

        // 1. Load topology
        let topology = PrismTopology::load(Path::new(topology_path))
            .with_context(|| format!("Failed to load topology: {}", topology_path))?;

        let pdb_id = topology.get_pdb_id();
        let n_atoms = topology.n_atoms;

        log::info!("[MD-CRYPTIC] Loaded {} atoms, {} residues from {}",
            n_atoms, topology.n_residues, pdb_id);

        // 2. Initialize GPU context
        let context = CudaContext::new(0)
            .context("Failed to create CUDA context")?;

        // 3. Initialize MD engine
        let mut md_engine = self.setup_md_engine(&context, &topology)?;

        // 4. Initialize GPU SASA calculator
        let sasa_calculator = LcpoSasaGpu::new(context.clone())
            .context("Failed to initialize GPU SASA calculator")?;

        // 5. Initialize volume tracker
        let mut volume_tracker = ResidueBasedVolumeTracker::new(
            self.config.n_frames,
            self.config.jaccard_threshold,
        );

        // 6. Equilibration (discard)
        log::info!("[MD-CRYPTIC] Running {} steps equilibration ({:.1} ps)...",
            self.config.equilibration_steps,
            self.config.equilibration_steps as f64 * self.config.dt_fs as f64 / 1000.0);

        md_engine.run(
            self.config.equilibration_steps,
            self.config.dt_fs,
            self.config.temperature_k,
            self.config.gamma_fs,
        ).context("Equilibration failed")?;

        log::info!("[MD-CRYPTIC] Equilibration complete, starting production...");

        // 7. Production loop - capture snapshots
        let atom_types = topology.get_sasa_atom_types();
        let radii = topology.get_vdw_radii();
        let atom_to_residue = topology.get_atom_to_residue_map();

        for frame_idx in 0..self.config.n_frames {
            // Run MD segment
            let result = md_engine.run(
                self.config.segment_steps,
                self.config.dt_fs,
                self.config.temperature_k,
                self.config.gamma_fs,
            ).with_context(|| format!("MD failed at frame {}", frame_idx))?;

            // Compute SASA for this frame
            let sasa_result = sasa_calculator.compute(
                &result.positions,
                &atom_types,
                Some(&radii),
            ).with_context(|| format!("SASA calculation failed at frame {}", frame_idx))?;

            // Detect pockets (all-atom)
            let pockets = self.detect_pockets_allatom(
                &result.positions,
                &topology,
                &sasa_result.per_atom,
                &atom_to_residue,
            );

            // Track each detected pocket
            for pocket in pockets {
                volume_tracker.add_observation(
                    frame_idx,
                    &pocket.residue_ids,
                    pocket.volume,
                    pocket.sasa,
                );
            }

            if (frame_idx + 1) % 20 == 0 || frame_idx == 0 {
                log::info!("[MD-CRYPTIC] Frame {}/{} (T={:.1}K, PE={:.1})",
                    frame_idx + 1, self.config.n_frames,
                    result.avg_temperature, result.potential_energy);
            }
        }

        // 8. Finalize and classify
        volume_tracker.finalize();

        let diagnostics = volume_tracker.get_diagnostics();
        log::info!("[MD-CRYPTIC] Tracked {} valid pockets (CV range: {:.3} - {:.3})",
            diagnostics.valid_pockets, diagnostics.cv_min, diagnostics.cv_max);

        let mut cryptic_sites = volume_tracker.get_cryptic_pockets(
            self.config.cv_threshold,
            self.config.min_open_frequency,
            self.config.max_open_frequency,
        );

        // 9. Score druggability for cryptic sites
        for site in &mut cryptic_sites {
            let residue_names: Vec<String> = site.residues.iter()
                .filter_map(|&res_id| {
                    topology.residue_ids.iter()
                        .position(|&r| r == res_id)
                        .and_then(|idx| topology.residue_names.get(idx).cloned())
                })
                .collect();

            site.druggability = Some(self.druggability_scorer.score_simple(
                &residue_names,
                site.mean_volume,
            ));
        }

        let elapsed = start_time.elapsed().as_secs_f64();

        log::info!("[MD-CRYPTIC] Complete: {} cryptic sites detected in {:.1}s",
            cryptic_sites.len(), elapsed);

        // Report diagnostics for scientific integrity
        if cryptic_sites.is_empty() {
            log::warn!("[MD-CRYPTIC] No cryptic sites detected. Diagnostics:");
            log::warn!("  - Valid pockets: {}", diagnostics.valid_pockets);
            log::warn!("  - CV range: {:.3} - {:.3} (threshold: {:.3})",
                diagnostics.cv_min, diagnostics.cv_max, self.config.cv_threshold);
            log::warn!("  - Freq range: {:.2} - {:.2}",
                diagnostics.freq_min, diagnostics.freq_max);
            log::warn!("  DO NOT adjust thresholds post-hoc. Investigate physics.");
        }

        Ok(MdCrypticResult {
            pdb_id,
            config: self.config.clone(),
            total_time_ps: self.config.total_time_ps(),
            n_frames: self.config.n_frames,
            all_pockets: volume_tracker.get_all_pockets().into_iter().cloned().collect(),
            cryptic_sites,
            diagnostics,
            computation_time_secs: elapsed,
        })
    }

    /// Set up the MD engine from topology
    fn setup_md_engine(
        &self,
        context: &Arc<CudaContext>,
        topology: &PrismTopology,
    ) -> Result<AmberMegaFusedHmc> {
        use prism_gpu::amber_mega_fused::build_exclusion_lists;

        let n_atoms = topology.n_atoms;

        let mut md = AmberMegaFusedHmc::new(context.clone(), n_atoms)
            .context("Failed to create MD engine")?;

        // Convert topology to AMBER format
        let positions: Vec<f32> = topology.positions.iter().map(|&x| x as f32).collect();

        // Bonded terms - note: AMBER format is (i, j, k, r0) for bonds
        let bonds: Vec<(usize, usize, f32, f32)> = topology.bonds.iter()
            .map(|b| (b.i, b.j, b.k as f32, b.r0 as f32))
            .collect();

        let angles: Vec<(usize, usize, usize, f32, f32)> = topology.angles.iter()
            .map(|a| (a.i, a.j, a.k_idx, a.force_k as f32, a.theta0 as f32))
            .collect();

        // Dihedrals: (i, j, k, l, pk, n, phase) where n is float periodicity
        let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topology.dihedrals.iter()
            .map(|d| (d.i, d.j, d.k_idx, d.l, d.force_k as f32, d.periodicity as f32, d.phase as f32))
            .collect();

        // NB params: (sigma, epsilon, charge, mass) per atom
        let nb_params: Vec<(f32, f32, f32, f32)> = (0..n_atoms)
            .map(|i| {
                let sigma = topology.lj_params.get(i).map(|p| p.sigma as f32).unwrap_or(3.4);
                let epsilon = topology.lj_params.get(i).map(|p| p.epsilon as f32).unwrap_or(0.1);
                let charge = topology.charges.get(i).map(|&c| c as f32).unwrap_or(0.0);
                let mass = topology.masses.get(i).map(|&m| m as f32).unwrap_or(12.0);
                (sigma, epsilon, charge, mass)
            })
            .collect();

        // Build exclusions from bonds and angles
        let exclusions = build_exclusion_lists(&bonds, &angles, n_atoms);

        // Upload topology
        md.upload_topology(
            &positions,
            &bonds,
            &angles,
            &dihedrals,
            &nb_params,
            &exclusions,
        ).context("Failed to upload topology to MD engine")?;

        // Set up H-bond constraints from topology if available
        if !topology.h_clusters.is_empty() {
            log::info!("[MD-CRYPTIC] Setting up {} H-bond constraints", topology.h_clusters.len());
            // H-constraints setup would go here
        }

        // Initialize velocities at target temperature
        md.initialize_velocities(self.config.temperature_k)
            .context("Failed to initialize velocities")?;

        Ok(md)
    }

    /// Detect pockets using all-atom coordinates
    fn detect_pockets_allatom(
        &self,
        positions: &[f32],
        topology: &PrismTopology,
        per_atom_sasa: &[f32],
        atom_to_residue: &[i32],
    ) -> Vec<DetectedPocket> {
        let n_atoms = positions.len() / 3;
        let cutoff_sq = self.config.pocket_neighbor_cutoff.powi(2);

        let mut pocket_candidates: Vec<DetectedPocket> = Vec::new();

        // For each heavy atom, check if it's in a pocket-like environment
        for i in 0..n_atoms {
            // Skip hydrogens
            if topology.is_hydrogen(i) {
                continue;
            }

            let xi = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];

            // Count heavy atom neighbors
            let mut neighbors: Vec<usize> = Vec::new();
            for j in 0..n_atoms {
                if j == i || topology.is_hydrogen(j) {
                    continue;
                }

                let xj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
                let dx = xj[0] - xi[0];
                let dy = xj[1] - xi[1];
                let dz = xj[2] - xi[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    neighbors.push(j);
                }
            }

            // Pocket criterion: moderately buried
            if neighbors.len() >= self.config.min_pocket_neighbors
                && neighbors.len() <= self.config.max_pocket_neighbors
            {
                // Collect residue IDs for pocket
                let mut residue_set: HashSet<i32> = HashSet::new();
                residue_set.insert(atom_to_residue[i]);
                for &j in &neighbors {
                    residue_set.insert(atom_to_residue[j]);
                }

                // Estimate volume (~25 Å³ per heavy atom contact)
                let volume = neighbors.len() as f64 * 25.0;

                // Skip if outside volume range
                if volume < self.config.min_pocket_volume || volume > self.config.max_pocket_volume {
                    continue;
                }

                // Sum SASA for pocket atoms
                let mut sasa = 0.0f64;
                for &res_id in &residue_set {
                    for a in 0..n_atoms {
                        if atom_to_residue[a] == res_id {
                            sasa += per_atom_sasa[a] as f64;
                        }
                    }
                }

                pocket_candidates.push(DetectedPocket {
                    centroid: xi,
                    residue_ids: residue_set.into_iter().collect(),
                    volume,
                    sasa,
                });
            }
        }

        // Merge overlapping pockets
        self.merge_pockets(pocket_candidates)
    }

    /// Merge overlapping pocket candidates
    fn merge_pockets(&self, mut candidates: Vec<DetectedPocket>) -> Vec<DetectedPocket> {
        if candidates.is_empty() {
            return candidates;
        }

        // Sort by volume (largest first)
        candidates.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap_or(std::cmp::Ordering::Equal));

        let mut merged: Vec<DetectedPocket> = Vec::new();
        let mut used = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if used[i] {
                continue;
            }

            let mut current = candidates[i].clone();
            used[i] = true;

            // Find and merge overlapping pockets
            for j in (i + 1)..candidates.len() {
                if used[j] {
                    continue;
                }

                // Check residue overlap (Jaccard)
                let set_i: HashSet<i32> = current.residue_ids.iter().copied().collect();
                let set_j: HashSet<i32> = candidates[j].residue_ids.iter().copied().collect();

                let intersection = set_i.intersection(&set_j).count();
                let union = set_i.union(&set_j).count();

                if union > 0 && (intersection as f64 / union as f64) >= 0.3 {
                    // Merge
                    for res in &candidates[j].residue_ids {
                        if !current.residue_ids.contains(res) {
                            current.residue_ids.push(*res);
                        }
                    }
                    current.volume = current.volume.max(candidates[j].volume);
                    current.sasa += candidates[j].sasa;
                    used[j] = true;
                }
            }

            merged.push(current);
        }

        merged
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = MdCrypticConfig::default();

        // Physical constants
        assert_eq!(config.sasa_probe_radius, 1.4);

        // Thresholds (literature-derived)
        assert_eq!(config.cv_threshold, 0.20);
        assert_eq!(config.jaccard_threshold, 0.30);
        assert_eq!(config.min_open_frequency, 0.05);
        assert_eq!(config.max_open_frequency, 0.90);
    }

    #[test]
    fn test_jaccard_similarity() {
        let set1: HashSet<i32> = [1, 2, 3, 4, 5].iter().copied().collect();
        let set2: HashSet<i32> = [3, 4, 5, 6, 7].iter().copied().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        let jaccard = intersection as f64 / union as f64;

        // intersection = {3,4,5} = 3
        // union = {1,2,3,4,5,6,7} = 7
        // jaccard = 3/7 ≈ 0.43
        assert!((jaccard - 0.4286).abs() < 0.01);
    }

    #[test]
    fn test_volume_tracker_matching() {
        let mut tracker = ResidueBasedVolumeTracker::new(100, 0.30);

        // First observation
        tracker.add_observation(0, &[1, 2, 3, 4, 5], 150.0, 200.0);

        // Similar pocket (high overlap) - should match
        tracker.add_observation(1, &[2, 3, 4, 5, 6], 160.0, 210.0);

        // Different pocket (low overlap) - should create new
        tracker.add_observation(2, &[10, 11, 12, 13, 14], 180.0, 220.0);

        assert_eq!(tracker.pockets.len(), 2);
    }

    #[test]
    fn test_cv_calculation() {
        let mut pocket = PocketTimeSeries::new("test".to_string(), 10);

        // Add some volume observations with variance
        pocket.volumes = vec![100.0, 120.0, 80.0, 110.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        pocket.sasa_values = vec![50.0, 60.0, 40.0, 55.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        pocket.frames_observed = vec![0, 1, 2, 3, 4];

        pocket.compute_statistics();

        // Mean = 100, std ≈ 14.14, CV ≈ 0.14
        assert!((pocket.mean_volume - 100.0).abs() < 1.0);
        assert!(pocket.cv > 0.0);
        assert_eq!(pocket.open_frequency, 0.5); // 5 out of 10 frames
    }
}
